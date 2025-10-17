# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# run this command:
# python scripts/manualTokenShuffle.py --params params.yaml --model_name_or_path output/ftModel/best_spearmanr/ --data_dir output/data/decay --output_dir output/predict --scaler output/ftModel/best_spearmanr/scaler.joblib --label tokenShuffle --extraFeatures output/data/codons/extraFeatures.csv --mfe output/data/codons/vienna_features.csv

import argparse
import glob
import json
import yaml
import logging
import os
import random
import re
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from typing import List
import pandas as pd
import joblib


import numpy as np
import torch
from decouple import config
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from data_loaders import load_and_cache_examples_3utr as load_and_cache_examples
from data_loaders import visualize 


from __init__ import glue_compute_metrics as compute_metrics

from transformers import (
    AutoTokenizer,
) 

from GenaLMWithExtraFeatures import GenaLMWithExtraFeatures
logger = logging.getLogger(__name__)

TOKEN_ID_GROUP = ["bert", "3utrlong", "3utrlongcat", "xlnet", "albert"]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def plotPredictions(preds, out_label_ids, results, label=None, eval_output_dir=None):
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    
    plt.figure(figsize=(5,5))
    
    # Create a scatter plot
    plt.scatter(out_label_ids, preds, alpha=0.5)
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Labels')
    plt.xlim(min(out_label_ids.min(), preds.min()), max(out_label_ids.max(), preds.max()))
    plt.ylim(min(out_label_ids.min(), preds.min()), max(out_label_ids.max(), preds.max()))
    
    # Add a red dashed line for x=y
    plt.plot([min(out_label_ids.min(), preds.min()), max(out_label_ids.max(), preds.max())], 
             [min(out_label_ids.min(), preds.min()), max(out_label_ids.max(), preds.max())], 
             'r--')
    
    # Calculate Pearson and Spearman correlation coefficients
    pearson_corr = results.get('pearson', pearsonr(out_label_ids, preds)[0])
    spearman_corr = results.get('spearmanr', spearmanr(out_label_ids, preds)[0])
    
    # Annotate the plot with the correlation coefficients
    plt.annotate(f'Pearson: {pearson_corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.annotate(f'Spearman: {spearman_corr:.2f}', xy=(0.05, 0.90), xycoords='axes fraction')
    plt.savefig(os.path.join(eval_output_dir, f'predictions_vs_true_labels{label}.svg'))
    pd.DataFrame({'True Labels': out_label_ids, 'Predictions': preds}).to_csv(os.path.join(eval_output_dir, f'predictions_vs_true_labels{label}.csv'), index=False)

def load_data(args, tokenizer, test_run=False, split='train.fasta'):
    from Bio import SeqIO
    data = list(SeqIO.parse(os.path.join(args.data_dir, split), 'fasta'))
    labels = []
    seqs = []
    attention_masks = []
    tr_ids = []
    for record in data:
        utr5, utr3 = record.seq.split(',')
        utr5 = tokenizer.encode(str(utr5).replace('U',"T"), add_special_tokens=True)
        utr3 = tokenizer.encode(str(utr3).replace('U',"T"), add_special_tokens=True) #, max_length=args.max_seq_length-len(utr5)+1, pad_to_max_length=False, truncation=True)
        s = utr5 + utr3[1:]
        if len(s) < 10 or len(s) > args.max_seq_length:
            continue
        am = [1] * len(s) + [0] * (args.max_seq_length - len(s))  # attention mask
        # pad s to max_seq_length
        s = s + [tokenizer.pad_token_id]*(args.max_seq_length-len(s))
        if s not in seqs: # prevent duplicates
            seqs.append(s)
            attention_masks.append(am)
            labels.append(float(record.id))
            tr_ids.append(record.description.split(' ')[1])

    return torch.tensor(labels), torch.tensor(seqs), torch.tensor(attention_masks), tr_ids

def token_shuffle_analysis(args, model, tokenizer, extraFeatures=None, scaler=None):
    """
    Find the shortest test sequence and systematically replace each token position
    with all possible tokens from the vocabulary to analyze prediction changes.
    """
    eval_output_dir = args.output_dir
    
    # Load test data
    labels, seqs, atten_masks, tr_ids = load_data(args, tokenizer, split='test.fasta')
    
    # Find the shortest sequence (non-padding tokens)
    seq_lengths = []
    for seq in seqs:
        # Count non-padding tokens
        length = (seq != tokenizer.pad_token_id).sum().item()
        seq_lengths.append(length)
    
    shortest_idx = np.argmin(seq_lengths)
    shortest_seq = seqs[shortest_idx].clone()
    shortest_label = labels[shortest_idx]
    shortest_tr_id = tr_ids[shortest_idx]
    shortest_length = seq_lengths[shortest_idx]
    
    logger.info(f"Shortest sequence: ID={shortest_tr_id}, Length={shortest_length}, Label={shortest_label}")
    logger.info(f"Original tokens: {shortest_seq[:shortest_length].tolist()}")
    
    # Get vocabulary size
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Prepare extra features if available
    if extraFeatures is not None:
        if not pd.api.types.is_string_dtype(extraFeatures.index):
            extraFeatures.index = extraFeatures.index.astype(str)
        
        try:
            ef_index_list_str = extraFeatures.index.tolist()
            tr_id_idx = ef_index_list_str.index(shortest_tr_id)
            extra_feat = extraFeatures.iloc[[tr_id_idx]].to_numpy()
            extra_feat_tensor = torch.tensor(extra_feat, dtype=torch.float32).to(args.device)
        except ValueError:
            logger.warning(f"Transcript ID '{shortest_tr_id}' not found in extraFeatures. Using zeros.")
            extra_feat_tensor = None
    else:
        extra_feat_tensor = None
    
    # Store results
    results_list = []
    
    # Get original prediction
    model.eval()
    with torch.no_grad():
        input_seq = shortest_seq.unsqueeze(0).to(args.device)
        attention_mask = atten_masks[shortest_idx].unsqueeze(0).to(args.device)
        
        inputs = {
            "input_ids": input_seq,
            "attention_mask": attention_mask,
            "extra_features": extra_feat_tensor
        }
        outputs = model(**inputs)
        original_pred = outputs['logits'].detach().cpu().numpy()[0, 0]
    
    logger.info(f"Original prediction: {original_pred:.4f}")
    
    results_list.append({
        'position': -1,
        'original_token': -1,
        'new_token': -1,
        'prediction': original_pred,
        'label': shortest_label.item(),
        'is_original': True
    })
    
    # Iterate through each position in the shortest sequence
    for pos in range(shortest_length):
        original_token = shortest_seq[pos].item()
        logger.info(f"\nAnalyzing position {pos}/{shortest_length-1}, original token: {original_token}")
        
        # Try replacing with each token in vocabulary
        for new_token in tqdm(range(vocab_size), desc=f"Position {pos}"):
            # Skip if it's the same as original (already have that)
            if new_token == original_token:
                continue
            
            # Create modified sequence
            modified_seq = shortest_seq.clone()
            modified_seq[pos] = new_token
            
            # Get prediction
            model.eval()
            with torch.no_grad():
                input_seq = modified_seq.unsqueeze(0).to(args.device)
                attention_mask = atten_masks[shortest_idx].unsqueeze(0).to(args.device)
                
                inputs = {
                    "input_ids": input_seq,
                    "attention_mask": attention_mask,
                    "extra_features": extra_feat_tensor
                }
                outputs = model(**inputs)
                pred = outputs['logits'].detach().cpu().numpy()[0, 0]
            
            results_list.append({
                'position': pos,
                'original_token': original_token,
                'new_token': new_token,
                'prediction': pred,
                'label': shortest_label.item(),
                'is_original': False,
                'pred_change': pred - original_pred
            })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results_list)
    output_file = os.path.join(eval_output_dir, 'token_shuffle_analysis.csv')
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved token shuffle analysis to {output_file}")
    
    # Create visualization
    visualize_token_shuffle_results(results_df, eval_output_dir, shortest_length)
    
    return results_df


def visualize_token_shuffle_results(results_df, output_dir, seq_length):
    """Create visualizations for token shuffle analysis."""
    
    # Filter out original prediction
    variant_results = results_df[results_df['is_original'] == False].copy()
    
    # 1. Heatmap of prediction changes by position and token
    plt.figure(figsize=(15, 8))
    
    # Create pivot table for heatmap
    pivot_data = variant_results.pivot_table(
        values='pred_change',
        index='new_token',
        columns='position',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, cmap='RdBu_r', center=0, cbar_kws={'label': 'Prediction Change'})
    plt.title('Prediction Change by Token Substitution')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Replacement Token ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_shuffle_heatmap.svg'))
    plt.close()
    
    # 2. Box plot of prediction changes by position
    plt.figure(figsize=(12, 6))
    positions = sorted(variant_results['position'].unique())
    position_data = [variant_results[variant_results['position'] == pos]['pred_change'].values 
                     for pos in positions]
    
    plt.boxplot(position_data, labels=positions)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Position in Sequence')
    plt.ylabel('Prediction Change')
    plt.title('Distribution of Prediction Changes by Position')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_shuffle_boxplot.svg'))
    plt.close()
    
    # 3. Summary statistics by position
    summary_stats = variant_results.groupby('position')['pred_change'].agg([
        ('mean_change', 'mean'),
        ('std_change', 'std'),
        ('max_change', 'max'),
        ('min_change', 'min'),
        ('abs_mean_change', lambda x: np.abs(x).mean())
    ]).reset_index()
    
    summary_stats.to_csv(os.path.join(output_dir, 'token_shuffle_summary_by_position.csv'), index=False)
    
    # 4. Plot mean absolute change by position
    plt.figure(figsize=(10, 6))
    plt.bar(summary_stats['position'], summary_stats['abs_mean_change'])
    plt.xlabel('Position in Sequence')
    plt.ylabel('Mean Absolute Prediction Change')
    plt.title('Importance of Each Position (Mean Absolute Change)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_position_importance.svg'))
    plt.close()
    
    logger.info("Token shuffle visualizations created")


def evaluate(args, model, tokenizer, prefix="", evaluate=True, val=False, extraFeatures=None, scaler=None):
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    labels, seqs, atten_masks, tr_ids = load_data(args, tokenizer, split='test.fasta')
    #train_dataset = mask_tokens(torch.tensor(seqs), labels, tokenizer)
    # Create tr_ids_index for the evaluation dataset
    if extraFeatures is not None:
        # Ensure extraFeatures index is string for matching
        if not pd.api.types.is_string_dtype(extraFeatures.index):
            extraFeatures.index = extraFeatures.index.astype(str)
            
        eval_tr_ids_index_list = []
        ef_index_list_str = extraFeatures.index.tolist()
        for tr_id_str_val in tr_ids: # Use stringified tr_ids from current dev split
            try:
                eval_tr_ids_index_list.append(ef_index_list_str.index(tr_id_str_val))
            except ValueError:
                logger.error(f"Evaluation: Transcript ID '{tr_id_str_val}' from dev split not found in extraFeatures index.")
                # Handle missing IDs during evaluation, e.g., by skipping or using default features
                # For now, raising an error to highlight data inconsistency
                raise ValueError(f"Evaluation: Transcript ID '{tr_id_str_val}' not found in extraFeatures index.")
        tr_ids_index = torch.tensor(eval_tr_ids_index_list, dtype=torch.long)
    else:
        tr_ids_index = torch.zeros_like(labels, dtype=torch.long)
        
    eval_dataset = TensorDataset(seqs, atten_masks, torch.zeros_like(seqs), labels, tr_ids_index)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        if extraFeatures is not None:
            tmp_extraFeatures = extraFeatures.iloc[[int(x) for x in list(batch[4])]].to_numpy()
            tmp_extraFeatures = torch.tensor(tmp_extraFeatures, dtype=torch.float32).to(args.device)
        else:
            tmp_extraFeatures = None

        with torch.no_grad():
            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device), "labels": batch[3].to(args.device), "extra_features": tmp_extraFeatures}
            outputs = model(**inputs)
            tmp_eval_loss = outputs['loss']
            logits = outputs['logits']

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.squeeze(preds)

    results = compute_metrics('sts-b', preds, out_label_ids)
    results['eval_loss'] = eval_loss
    plotPredictions(preds, out_label_ids, results, args.label, eval_output_dir)

    output_eval_file = os.path.join(eval_output_dir, prefix, "test_results.json")
    with open(output_eval_file, "w") as writer:
        json.dump({**{k: (v.item() if isinstance(v, (np.float32, np.int32, np.int64)) else v) for k, v in results.items()}}, writer, indent=4)

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results

def main():
    parser = argparse.ArgumentParser()

    # BASIC
    parser.add_argument("--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
    parser.add_argument("--data_dir", default=None, type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--extraFeatures", default=None, type=str, help="Path the the csv file containing the extra features",)
    parser.add_argument("--mfe", default=None, type=str, help="Path the the csv file containing the MFE features from ViennaRNA",)
    parser.add_argument("--scaler", default=None, type=str, help="Path the the joblib file containing the scaler",)
    parser.add_argument("--label", default='', type=str, help="label for the run",)
    parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Path to pre-trained model or shortcut name selected in the list",)
    parser.add_argument("--task_name", default='rnaprom', type=str, help="Script only prepared for promoter task" )
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--tokenizer_name",default="rna3",type=str, help="Pretrained tokenizer name or path if not the same as model_name",)

    # OBJECTIVE
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument("--do_visualize", action="store_true", help="Whether to calculate attention score.")

    # VALIDATION DURING TRAINING / EVALUATE 
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",)
    parser.add_argument("--do_visualize_during_training", action="store_true", help="Steps to generate an image")
    parser.add_argument("--image_steps", type=int, default=0, help="Steps to generate an image")
    
    # MODEL CONFIGS (only use)

    # TRAINING DETAILS
    parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",)
    parser.add_argument("--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup over warmup_percent*total_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--n_process", default=2, type=int, help="number of processes used for data process",)
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",)
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",)
    parser.add_argument("--neptune", default=False, help="Neptune")
    parser.add_argument("--neptune_tags", type=list, default=["trial"], help="Neptune tags")
    parser.add_argument("--neptune_description", type=str, default="TRIAL minilm fine-tuning", help="Neptune description")
    parser.add_argument("--neptune_token", type=str, default=None, help="Neptune API token")
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project")
    parser.add_argument("--memory_size", type=int, default=None, help="number of memory tokens to use in RMT.",)
    parser.add_argument("--block_size", type=int, default=None, help="Total token input size of base model.",)
    parser.add_argument("--max_n_segments", type=int, default=None, help="Maximun number of segments to include from long input.",)
    


    # OTHER
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)


    args = parser.parse_known_args()[0]

    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['predict'].items():
                parser.set_defaults(**{key: value})
            for key, value in yaml_params['modelParams'].items():
                parser.set_defaults(**{key: value})
            for key, value in yaml_params['RMT'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    # Segons els que he entès, local_rank és per si utilitzes més d'una màquina. Pot ser que utilitzis 1+ gpu però totes a la mateixa màquina
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print("devices", device)
        print("Number of gpus", args.n_gpu)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

        # EVALUATION ON THE TEST SET-----------------------------------------------------------------------------------------------------
    results = {}
    tokenizer_path = args.tokenizer_name if args.tokenizer_name else 'AIRI-Institute/gena-lm-bert-base-fly'
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        # Fallback or re-raise, depending on desired behavior
        logger.info("Falling back to default tokenizer 'AIRI-Institute/gena-lm-bert-base-fly'")
        tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-fly')

    scaler = joblib.load(args.scaler) if os.path.exists(args.scaler) else None  # Initialize scaler

    # 1. Load original extraFeatures from args.extraFeatures CSV
    if args.extraFeatures is not None:
        logger.info(f"Loading original extra features from: {args.extraFeatures}")
        try:
            extraFeatures_df = pd.read_csv(args.extraFeatures, index_col=0)
            # Drop specified columns if they exist
            columns_to_drop = ['Decay Rate', 'Residuals']
            existing_columns_to_drop = [col for col in columns_to_drop if col in extraFeatures_df.columns]
            if existing_columns_to_drop:
                extraFeatures_df.drop(columns=existing_columns_to_drop, inplace=True)
                logger.info(f"Dropped columns: {existing_columns_to_drop} from original extra features.")
        except FileNotFoundError:
            logger.error(f"Original extra features file not found: {args.extraFeatures}")
            extraFeatures_df = None
        except Exception as e:
            logger.error(f"Error loading original extra features from {args.extraFeatures}: {e}")
            extraFeatures_df = None
    else:
        extraFeatures_df = None
        logger.info("No original extra features CSV provided (args.extraFeatures is None).")

    # 2. Load ViennaRNA features (MFE)
    logger.info(f"Attempting to load ViennaRNA features from: {args.mfe}")
    if args.mfe and os.path.exists(args.mfe):
        try:
            vienna_df = pd.read_csv(args.mfe)
            if "id" not in vienna_df.columns:
                logger.warning("ViennaRNA features file found but missing 'id' column. Cannot merge MFE.")
                vienna_df = None
            else:
                vienna_df.set_index("id", inplace=True)
                if 'mfe' in vienna_df.columns:
                    logger.info("Found 'mfe' column in ViennaRNA features.")
                    vienna_df_mfe = vienna_df[['mfe']].copy() # Use .copy() to avoid SettingWithCopyWarning
                    vienna_df_mfe['mfe'] = pd.to_numeric(vienna_df_mfe['mfe'], errors='coerce').fillna(0)
                    vienna_df = vienna_df_mfe # Assign back the processed DataFrame
                else:
                    logger.warning("'mfe' column not found in ViennaRNA features file. Skipping MFE.")
                    vienna_df = None
        except Exception as e:
            logger.error(f"Error loading or processing ViennaRNA features from {args.mfe}: {e}")
            vienna_df = None
    else:
        logger.warning(f"ViennaRNA features file not found at {args.mfe}. Proceeding without MFE.")
        vienna_df = None

    # 3. Merge DataFrames
    if extraFeatures_df is not None and vienna_df is not None:
        logger.info("Merging original extra features with ViennaRNA MFE features.")
        # Ensure indices are of the same type for robust merging
        extraFeatures_df.index = extraFeatures_df.index.astype(str)
        vienna_df.index = vienna_df.index.astype(str)
        extraFeatures_df = extraFeatures_df.merge(vienna_df, left_index=True, right_index=True, how='left')
        if 'mfe' in extraFeatures_df.columns: # MFE column exists due to merge
             extraFeatures_df['mfe'] = extraFeatures_df['mfe'].fillna(0) # Fill NaNs for IDs in extraFeatures_df but not in vienna_df
        logger.info("Merge complete.")
    elif vienna_df is not None and extraFeatures_df is None:
        logger.info("Using only ViennaRNA MFE features as no original extra features were provided.")
        extraFeatures_df = vienna_df.copy() # Use a copy
        extraFeatures_df.index = extraFeatures_df.index.astype(str) # Ensure index is string
    # If extraFeatures_df is not None and vienna_df is None, extraFeatures_df is used as is (index type already handled or assumed consistent).
    # If both are None, extraFeatures_df remains None.

    if extraFeatures_df is not None:
        # Ensure index is string type if it was not already (e.g. if only original extraFeatures_df was used)
        if not pd.api.types.is_string_dtype(extraFeatures_df.index):
            extraFeatures_df.index = extraFeatures_df.index.astype(str)
        logger.info(f"Final extra features DataFrame shape: {extraFeatures_df.shape}")
        logger.info(f"Final extra features columns: {extraFeatures_df.columns.tolist()}")
    else:
        logger.info("No extra features will be used.")

    # 4. Determine num_extra_features for model initialization
    num_extra_features = extraFeatures_df.shape[1] if extraFeatures_df is not None else 0
    
    model = None # Initialize model to None
    if not args.do_visualize: 
        model_path = args.model_name_or_path if args.model_name_or_path else 'AIRI-Institute/gena-lm-bert-base-fly'
        try:
            model = GenaLMWithExtraFeatures.from_pretrained(model_path)
            logger.info(f"Model initialized from {model_path} with num_extra_features: {num_extra_features}")
        except Exception as e:
            logger.error(f"Failed to initialize model from {model_path}: {e}")
            # Decide on fallback or re-raise
            raise
        logger.info("finish loading model")

        if args.local_rank == 0 and torch.distributed.is_initialized(): # Check if distributed is initialized
            torch.distributed.barrier()

        if model: model.to(args.device)
    # else: model remains None if only visualizing.

    model.eval()

    scaledVals = scaler.transform(extraFeatures_df) if extraFeatures_df is not None else extraFeatures_df
    extraFeatures_df = pd.DataFrame(scaledVals, index=extraFeatures_df.index, columns=extraFeatures_df.columns) if extraFeatures_df is not None else None
    
    # Run token shuffle analysis
    logger.info("Starting token shuffle analysis...")
    shuffle_results = token_shuffle_analysis(args, model, tokenizer, extraFeatures=extraFeatures_df, scaler=scaler)
    
    ## Also run standard evaluation for comparison
    #logger.info("Running standard evaluation...")
    #result = evaluate(args, model, tokenizer, extraFeatures=extraFeatures_df, scaler=scaler) # Results saved in file eval_results.txt


if __name__ == "__main__":
    main()
