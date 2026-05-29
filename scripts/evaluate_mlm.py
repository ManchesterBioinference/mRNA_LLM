"""
MLM evaluation script: reports top-1/5/10 token accuracy and perplexity
on a held-out FASTA file for the extended pre-trained genomic LM.

Addresses reviewer question:
  "What is the performance of the extended pre-trained masked learning model
   when predicting masked tokens? Report as percentage of tokens correctly predicted."

Usage:
  python scripts/evaluate_mlm.py --params params.yaml
  python scripts/evaluate_mlm.py --model_dir output/flyTrained/best_checkpoint \
      --data_dir output/data/flyUTRs --split test.fasta  # Also evaluate base GENA-LM for comparison:
  python scripts/evaluate_mlm.py --compare_base"""

import argparse
import json
import logging
import math
import os

import torch
import yaml
from Bio import SeqIO
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_data(data_dir, split, tokenizer, max_seq_length):
    """
    Replicate load_data() from trainToFlyUTRs.py exactly.
    FASTA format: >header\\n5utr_seq,3utr_seq
    """
    path = os.path.join(data_dir, split)
    records = list(SeqIO.parse(path, "fasta"))
    seqs = []
    attention_masks = []
    skipped = 0

    for record in records:
        utr5, utr3 = record.seq.split(",")
        utr5_ids = tokenizer.encode(
            str(utr5).replace("U", "T"), add_special_tokens=True
        )
        utr3_ids = tokenizer.encode(
            str(utr3).replace("U", "T"), add_special_tokens=True
        )
        # Concatenate: utr5 [CLS]...[SEP] + utr3 ...[SEP]  (skip duplicate [CLS])
        s = utr5_ids + utr3_ids[1:]

        if len(s) < 10 or len(s) > max_seq_length:
            skipped += 1
            continue

        am = [1] * len(s) + [0] * (max_seq_length - len(s))
        s = s + [tokenizer.pad_token_id] * (max_seq_length - len(s))

        if s not in seqs:  # prevent duplicates (matches training behaviour)
            seqs.append(s)
            attention_masks.append(am)

    logger.info(
        "Loaded %d sequences from %s (skipped %d: too short / too long / duplicate)",
        len(seqs),
        split,
        skipped,
    )
    return TensorDataset(torch.tensor(seqs), torch.tensor(attention_masks))


def _majority_class_baseline(dataset, tokenizer, max_seq_length):
    """
    Compute the majority-class (most-frequent token) baseline.
    Scans all non-special, non-padding token positions and finds the single
    most common token.  That token is always predicted; accuracy = its frequency.
    Also returns uniform-random accuracy = k / vocab_size for k in {1,5,10}.
    """
    from collections import Counter

    special_ids = set(tokenizer.all_special_ids)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    counts: Counter = Counter()
    for i in range(len(dataset)):
        ids = dataset[i][0].tolist()        # input_ids
        am  = dataset[i][1].tolist()        # attention_mask
        for tok, am_val in zip(ids, am):
            if am_val == 1 and tok not in special_ids and tok != pad_id:
                counts[tok] += 1

    total = sum(counts.values())
    if total == 0:
        return {}

    top1_tok_freq   = counts.most_common(1)[0][1]
    top5_tok_freq   = sum(v for _, v in counts.most_common(5))
    top10_tok_freq  = sum(v for _, v in counts.most_common(10))
    top100_tok_freq = sum(v for _, v in counts.most_common(min(100, len(counts))))
    top1000_tok_freq = sum(v for _, v in counts.most_common(min(1000, len(counts))))

    return {
        "random_top1":     round(1 / vocab_size * 100, 4),
        "random_top5":     round(5 / vocab_size * 100, 4),
        "random_top10":    round(10 / vocab_size * 100, 4),
        "random_top100":   round(min(100, vocab_size) / vocab_size * 100, 4),
        "random_top1000":  round(min(1000, vocab_size) / vocab_size * 100, 4),
        "random_perplexity": vocab_size,
        "majority_top1":   round(top1_tok_freq / total * 100, 4),
        "majority_top5":   round(top5_tok_freq / total * 100, 4),
        "majority_top10":  round(top10_tok_freq / total * 100, 4),
        "majority_top100": round(top100_tok_freq / total * 100, 4),
        "majority_top1000": round(top1000_tok_freq / total * 100, 4),
        "vocab_size":      vocab_size,
        "n_eligible_tokens": total,
    }


def _run_model_eval(model, device, dataloader, data_collator, seed):
    """Run one full evaluation pass; return (top1, top5, top10, top100, top1000, perplexity, n_masked)."""
    torch.manual_seed(seed)

    total_masked = 0
    top1_correct = top5_correct = top10_correct = 0
    top100_correct = top1000_correct = 0
    total_loss_sum = 0.0

    for batch in tqdm(dataloader, desc=f"  {type(model).__name__}", leave=False):
        input_ids_batch   = batch[0]
        attention_mask_batch = batch[1]

        collated = data_collator(list(input_ids_batch))
        collated["attention_mask"] = attention_mask_batch
        collated = {k: v.to(device) for k, v in collated.items()}

        labels = collated["labels"]
        mask   = (labels != -100) & (collated["attention_mask"] == 1)
        n_masked = mask.sum().item()
        if n_masked == 0:
            continue

        with torch.no_grad():
            outputs = model(**collated)

        loss   = outputs[0]
        logits = outputs[1]

        total_loss_sum += loss.item() * n_masked

        true_ids      = labels[mask]
        logits_masked = logits[mask]
        vocab_k = logits_masked.size(-1)

        top1_correct  += (logits_masked.argmax(dim=-1) == true_ids).sum().item()
        
        k5 = min(5, vocab_k)
        top5_indices   = logits_masked.topk(k5, dim=-1).indices
        top5_correct  += (top5_indices  == true_ids.unsqueeze(-1)).any(dim=-1).sum().item()
        
        k10 = min(10, vocab_k)
        top10_indices  = logits_masked.topk(k10, dim=-1).indices
        top10_correct += (top10_indices == true_ids.unsqueeze(-1)).any(dim=-1).sum().item()
        
        k100 = min(100, vocab_k)
        top100_indices = logits_masked.topk(k100, dim=-1).indices
        top100_correct += (top100_indices == true_ids.unsqueeze(-1)).any(dim=-1).sum().item()
        
        k1000 = min(1000, vocab_k)
        top1000_indices = logits_masked.topk(k1000, dim=-1).indices
        top1000_correct += (top1000_indices == true_ids.unsqueeze(-1)).any(dim=-1).sum().item()
        
        total_masked  += n_masked

    if total_masked == 0:
        raise RuntimeError("No masked tokens found.")

    return (
        top1_correct  / total_masked,
        top5_correct  / total_masked,
        top10_correct / total_masked,
        top100_correct / total_masked,
        top1000_correct / total_masked,
        math.exp(total_loss_sum / total_masked),
        total_masked,
    )


def evaluate(args):
    # ── Device ────────────────────────────────────────────────────────────
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    logger.info("Using device: %s", device)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info("Loading tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # ── Data ──────────────────────────────────────────────────────────────
    dataset  = load_data(args.data_dir, args.split, tokenizer, args.max_seq_length)
    sampler  = SequentialSampler(dataset)
    # Same collator as training (mlm_probability=0.15)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # ── Baselines (no model required) ─────────────────────────────────────
    logger.info("Computing baselines (random + majority-class) ...")
    baselines = _majority_class_baseline(dataset, tokenizer, args.max_seq_length)

    # ── Extended pre-trained model ────────────────────────────────────────
    # Use the same loading pattern as trainToFlyUTRs.py so the GENA-LM
    # BertForMaskedLM custom class is loaded via auto_map / trust_remote_code.
    logger.info("Loading extended pre-trained model from %s", args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    top1, top5, top10, top100, top1000, ppl, n_masked = _run_model_eval(
        model, device, dataloader, data_collator, args.seed
    )
    results = {
        "n_sequences":    len(dataset),
        "n_masked_tokens": n_masked,
        "extended_pretrained": {
            "top1_accuracy":    round(top1, 6),
            "top5_accuracy":    round(top5, 6),
            "top10_accuracy":   round(top10, 6),
            "top100_accuracy":  round(top100, 6),
            "top1000_accuracy": round(top1000, 6),
            "perplexity":       round(ppl, 4),
        },
        "baselines": baselines,
    }
    del model  # free GPU memory before optional second run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Optional: base GENA-LM (before extended pre-training) ─────────────
    base_metrics = None
    if getattr(args, "compare_base", False):
        base_model_id = "AIRI-Institute/gena-lm-bert-base-fly"
        logger.info("Loading base GENA-LM from %s for comparison ...", base_model_id)
        base_model = AutoModel.from_pretrained(base_model_id, trust_remote_code=True)
        base_model.to(device)
        base_model.eval()
        b1, b5, b10, b100, b1000, bppl, _ = _run_model_eval(
            base_model, device, dataloader, data_collator, args.seed
        )
        base_metrics = {
            "top1_accuracy":    round(b1, 6),
            "top5_accuracy":    round(b5, 6),
            "top10_accuracy":   round(b10, 6),
            "top100_accuracy":  round(b100, 6),
            "top1000_accuracy": round(b1000, 6),
            "perplexity":       round(bppl, 4),
        }
        results["base_gena_lm"] = base_metrics
        del base_model

    # ── Print comparison table ─────────────────────────────────────────────
    vocab_size = baselines.get("vocab_size", tokenizer.vocab_size)
    logger.info("")
    logger.info("=" * 72)
    logger.info("MLM Evaluation  [%s / %s]  |  %d seqs  |  %d masked tokens",
                args.data_dir, args.split, len(dataset), n_masked)
    logger.info("=" * 72)
    logger.info("%-35s  %10s  %10s  %10s  %10s  %10s  %10s",
                "Model", "Top-1 (%)", "Top-5 (%)", "Top-10 (%)", "Top-100 (%)", "Top-1000 (%)", "Perplexity")
    logger.info("-" * 72)
    logger.info("%-35s  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.1f",
                f"Uniform random (1/{vocab_size:,})",
                baselines["random_top1"], baselines["random_top5"],
                baselines["random_top10"], baselines["random_top100"],
                baselines["random_top1000"], baselines["random_perplexity"])
    logger.info("%-35s  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10s",
                "Majority-class (most-freq token)",
                baselines["majority_top1"], baselines["majority_top5"],
                baselines["majority_top10"], baselines["majority_top100"],
                baselines["majority_top1000"], "—")
    if base_metrics:
        logger.info("%-35s  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.1f",
                    "Base GENA-LM (no extension)",
                    base_metrics["top1_accuracy"] * 100,
                    base_metrics["top5_accuracy"] * 100,
                    base_metrics["top10_accuracy"] * 100,
                    base_metrics["top100_accuracy"] * 100,
                    base_metrics["top1000_accuracy"] * 100,
                    base_metrics["perplexity"])
    logger.info("%-35s  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.1f",
                "Extended pre-trained (ours)",
                top1 * 100, top5 * 100, top10 * 100, top100 * 100, top1000 * 100, ppl)
    logger.info("=" * 72)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to %s", args.output_file)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLM masked-token prediction accuracy."
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        type=str,
        help="Path to the YAML file containing parameters.",
    )
    parser.add_argument(
        "--model_dir",
        default="output/flyTrained/best_checkpoint",
        type=str,
        help="Directory containing the fine-tuned model (pytorch_model.bin + tokenizer).",
    )
    parser.add_argument(
        "--data_dir",
        default="output/data/flyUTRs",
        type=str,
        help="Directory containing the FASTA data files.",
    )
    parser.add_argument(
        "--split",
        default="test.fasta",
        type=str,
        help="FASTA file within data_dir to evaluate on.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="Maximum total sequence length (must match training).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducible masking.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA.",
    )
    parser.add_argument(
        "--output_file",
        default="dvclive/flyUTRs/mlm_eval_metrics.json",
        type=str,
        help="Path to write JSON metrics.",
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="Also evaluate the original AIRI-Institute/gena-lm-bert-base-fly model "
             "(before extended pre-training) and print a side-by-side comparison.",
    )

    # Step 1: get --params path only
    args = parser.parse_known_args()[0]

    # Step 2: load YAML and set as parser defaults (CLI will override)
    if args.params and os.path.exists(args.params):
        with open(args.params, "r") as fh:
            yaml_params = yaml.safe_load(fh)
        if "evaluateMLM" in yaml_params:
            parser.set_defaults(**yaml_params["evaluateMLM"])

    # Step 3: full parse; CLI > YAML > argparse defaults
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
