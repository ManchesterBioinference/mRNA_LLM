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
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
from sklearn.discriminant_analysis import StandardScaler
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

# Ray Tune imports
import ray
import ray.air
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

from data_loaders import load_and_cache_examples_3utr as load_and_cache_examples
from data_loaders import visualize 

from dvclive import Live

from __init__ import glue_compute_metrics as compute_metrics
#/mnt/mr01-home01/m65338lb/worktrees/rnaDecay/gena_lm_extraFeatures/3UTRBERT/functions/src/transformers/data/metrics/__init__.py:

from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
) 

from GenaLMWithExtraFeatures import GenaLMWithExtraFeatures
import pandas as pd # Add pandas import

logger = logging.getLogger(__name__)
live = None #Live('dvclive/TE', cache_images=True)

TOKEN_ID_GROUP = ["bert", "3utrlong", "3utrlongcat", "xlnet", "albert"]

def mask_tokens(inputs, labels, tokenizer, kmer = 4, mask_prob=0.15):
    labels = inputs.clone()
    attention_mask = torch.zeros_like(inputs)

    for i in range(inputs.size(0)):
        input_sequence = inputs[i]
        attention_mask[i] = (~(input_sequence == 0)).int()

        # Create a mask to exclude special tokens and pad tokens
        combined_mask = torch.tensor([0 if x in tokenizer.all_special_ids else 1 for x in input_sequence.tolist()], dtype=torch.bool)

        # Calculate the number of tokens to mask
        num_to_mask = int(mask_prob * (combined_mask).sum().item())
        num_maskToken = round(0.8 * num_to_mask/kmer)
        num_random = round(0.1 * num_to_mask/kmer)

        # Randomly select tokens to mask and create mask
        possibleInd = torch.arange(input_sequence.size(0))[combined_mask][:-(kmer-1)]

        pool = possibleInd[torch.randperm(possibleInd.size(0))].tolist()
        mask_indices = []
        for _ in range(num_maskToken):
            ind = pool.pop()
            mask_indices += list(range(ind,ind+kmer))
            pool = [p for p in pool if p not in range(ind-(kmer-1),ind+kmer)] # have to prevent overlapping masks
        rand_indices = []
        for _ in range(num_random):
            ind = pool.pop()
            rand_indices += list(range(ind,ind+kmer))
            pool = [p for p in pool if p not in range(ind-(kmer-1),ind+kmer)]

        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        input_sequence[mask_indices] = tokenizer.mask_token_id

        # 10% of the time, replace masked input tokens with random word
        if len(rand_indices) > 0:
            random_words = torch.randint(tokenizer.vocab_size-5, (len(rand_indices),), dtype=torch.long)+5 #5 because of the special tokens taking indices 0-4
            input_sequence[rand_indices] = random_words

    return TensorDataset(inputs, attention_mask, torch.zeros_like(inputs), labels)#labels#

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
        am = [1] * len(s) + [0] * (args.max_seq_length - len(s))
        # pad s to max_seq_length
        s = s + [tokenizer.pad_token_id]*(args.max_seq_length-len(s))
        if s not in seqs: # prevent duplicates
            seqs.append(s)
            attention_masks.append(am)
            labels.append(float(record.id))
            tr_ids.append(record.description.split(' ')[1])

    return torch.tensor(labels), torch.tensor(seqs), torch.tensor(attention_masks), tr_ids

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        print('no limit')
        return
    if args.save_total_limit <= 0:
        print('limit <= 0')
        return

    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    logger.info("Total checkpoints: {}".format(len(checkpoints_sorted)))
    logger.info("Saving total limit: {}".format(args.save_total_limit))
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def pearsonr_torch(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den
    return r

def train(args, train_dataset, model, tokenizer, extraFeatures=None, scaler=None):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        # Non-classifier parameters with weight decay (e.g., BERT layers)
        {
            "params": [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay) and not (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
            ],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,  # Slower LR for BERT (e.g., 2e-5)
        },
        # Non-classifier parameters without weight decay (e.g., biases, layer norm)
        {
            "params": [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay) and not (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
            ],
            "weight_decay": 0.0,
            "lr": args.learning_rate,  # Slower LR for BERT
        },
        # Classifier parameters with weight decay
        {
            "params": [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay) and (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
            ],
            "weight_decay": args.classifier_decay,  # From your tested classifier setup
            "lr": args.classifier_lr,  # Faster LR for classifier
        },
        # Classifier parameters without weight decay (e.g., biases)
        {
            "params": [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay) and (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
            ],
            "weight_decay": 0.0,
            "lr": args.classifier_lr,  # Faster LR for classifier
        },
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent * t_total)

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1, args.beta2)
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    
    if args.do_visualize_during_training:
        tata_dataset = load_and_cache_examples(args, args.task_name, tokenizer, viz=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.model_name_or_path):
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss, train_loss = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )

    best_auc = 0
    last_auc = 0
    stop_count = 0
    # Initialize early stopping parameters
    best_val_loss = float('inf')
    best_val_spearmanr = float('-inf')
    patience_counter = 0
    patience_threshold = args.patience  # Number of epochs to wait for improvement

    epoch_counter = 0    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        preds = None
        out_label_ids = None
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if extraFeatures is not None:
                tmp_extraFeatures = extraFeatures.iloc[[int(x) for x in list(batch[4])]].to_numpy()
                tmp_extraFeatures = torch.tensor(tmp_extraFeatures, dtype=torch.float32).to(args.device)
            else:
                tmp_extraFeatures = None

            inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device), "labels": batch[3].to(args.device), "extra_features": tmp_extraFeatures}
            outputs = model(**inputs)
            loss = outputs['loss']
            logits = outputs['logits']
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            #live.log_metric('batch/loss',loss.item())
            #loss += (1-abs(pearsonr_torch(np.squeeze(logits), inputs["labels"])))
            loss.backward()

            tr_loss += loss.item()
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                if args.local_rank in [-1, 0] and args.do_visualize_during_training and global_step % args.image_steps == 0:
                    kmer = int(args.tokenizer_name[-1])
                    attention_scores, _ = visualize(args, model, tokenizer, prefix="", kmer=kmer, pred_dataset=tata_dataset)
                    fig, ax = plt.subplots(figsize=(6,3))
                    sns.set()
                    sns.set(font_scale=1.2)
                    ax = sns.heatmap(attention_scores, cmap='YlGnBu', vmin=0, ax=ax, yticklabels=1500)
                    plt.xticks(np.arange(0,300,25), np.arange(-250,50,25), rotation=45, fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.tight_layout()
                    ax.set_xlabel('Coordinate', fontsize=14)
                    ax.set_ylabel('Sequence', fontsize=14)
                    live.log_image("train/attention", fig)
                    

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    output_dir = os.path.join(args.output_dir, 'checkpoints/',"checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
                    # Save the scaler along with the model
                    if args.extraFeatures is not None:
                        joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
                        
                    logger.info("Saving model checkpoint and scaler to %s", output_dir)
                    
                    _rotate_checkpoints(args, checkpoint_prefix)
                    
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        epoch_counter += 1
        train_loss /= len(train_dataloader)
        live.log_metric('train/loss', train_loss)
        preds = np.squeeze(preds)
        results = compute_metrics('sts-b', preds, out_label_ids)
        for key, value in results.items():
            live.log_metric(f"train/{key}", value)
        plotPredictions(preds, out_label_ids, results)
        
        if args.local_rank in [-1, 0] and args.evaluate_during_training:# and args.logging_steps > 0 and global_step % args.logging_steps == 0:
            logs = {}
            results = evaluate(args, model, tokenizer, evaluate=False, val=True, extraFeatures=extraFeatures)
            val_loss = results['loss']

            for key, value in results.items():
                live.log_metric(f"val/{key}", value)


            for key, value in results.items():
                eval_key = "val_{}".format(key)
                logs[eval_key] = value

            if results['spearmanr'] > best_val_spearmanr:
                best_val_spearmanr = results['spearmanr']
                live.log_metric('global/best_spearmanr', best_val_spearmanr)
                live.log_metric('global/pearson', results['pearson'])
                output_dir = os.path.join(args.output_dir, "best_spearmanr")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                
                # Save the scaler along with the model
                if args.extraFeatures is not None:
                    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
                    
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving best spearmannr model checkpoint and scaler to %s", output_dir)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                live.log_metric('global/best_val_loss', best_val_loss)
                patience_counter = 0
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                checkpoint_dir = os.path.join(args.output_dir,'checkpoints')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                model_to_save = model.module if hasattr(model, "module") else model
                #model_to_save.memory_cell.model.config.save_pretrained(output_dir)
                #torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                # Save the scaler along with the model
                if args.extraFeatures is not None:
                    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving best model checkpoint to %s", output_dir)
            else:
                if epoch_counter >= args.numEpochsBeforeEarlyStopping:
                    patience_counter += 1
                    logger.info("Validation loss did not improve. Patience counter: %d", patience_counter)
        
            if patience_counter >= patience_threshold:
                logger.info("Early stopping triggered. Stopping training.")
                train_iterator.close()
                break

            loss_scalar = (tr_loss - logging_loss) / args.logging_steps
            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar
            logs["loss"] = loss_scalar
            logging_loss = tr_loss

            #print(json.dumps({**logs, **{"step": global_step}}))
            print(json.dumps({**{k: (v.item() if isinstance(v, (np.float32, np.int32, np.int64)) else v) for k, v in logs.items()}, **{"step": global_step}}))

        live.next_step()

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # Append results to CSV
    csv_file_path = "hyperparameterTuningResults.csv"
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, 'a') as f:
        if not file_exists:
            f.write("best_val_spearmanr,num_train_epochs,learning_rate,patience,warmup_percent\n")
        f.write(f"{best_val_spearmanr},{args.num_train_epochs},{args.learning_rate},{args.patience},{args.warmup_percent}\n")
    logger.info(f"Appended results to {csv_file_path}")

    return global_step, tr_loss / global_step

def plotPredictions(preds, out_label_ids, results, stage='train'):
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    
    fig = plt.figure()
    # Create a scatter plot
    plt.scatter(out_label_ids, preds, alpha=0.5)
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Labels')
    plt.xlim(min(out_label_ids.min(), preds.min()), max(out_label_ids.max(), preds.max()))
    plt.ylim(min(out_label_ids.min(), preds.min()), max(out_label_ids.max(), preds.max()))
    
    # Calculate Pearson and Spearman correlation coefficients
    pearson_corr = results.get('pearson', pearsonr(out_label_ids, preds)[0])
    spearman_corr = results.get('spearmanr', spearmanr(out_label_ids, preds)[0])
    
    # Annotate the plot with the correlation coefficients
    plt.annotate(f'Pearson: {pearson_corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.annotate(f'Spearman: {spearman_corr:.2f}', xy=(0.05, 0.90), xycoords='axes fraction')
    
    # Log the image using dvclive
    live.log_image(f'{stage}_predictions_vs_true_labels_{live.step}.png',fig)

def evaluate(args, model, tokenizer, prefix="", evaluate=True, val=False, extraFeatures=None):
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    # Ensure tr_ids from load_data are strings for matching with extraFeatures index
    labels, seqs, atten_masks, tr_ids_raw = load_data(args, tokenizer, split='dev.fasta')
    tr_ids = [str(tid) for tid in tr_ids_raw]


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

        inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device), "labels": batch[3].to(args.device), "extra_features": tmp_extraFeatures}

        with torch.no_grad():
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

    preds = np.squeeze(preds)
    results = compute_metrics('sts-b', preds, out_label_ids)
    results['loss'] = eval_loss / nb_eval_steps
    plotPredictions(preds, out_label_ids, results, stage='val')

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        eval_result = args.data_dir.split("/")[-1] + " "

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            eval_result = eval_result + str(results[key])[:5] + " "
        writer.write(eval_result + "\n")

    return results



def main():
# %%
    global live
    parser = argparse.ArgumentParser()

    # BASIC
    parser.add_argument("--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
    parser.add_argument("--runName", default='', type=str, help="Name of the run",)
    parser.add_argument("--data_dir", default="output/data/decay", type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--extraFeatures", default=None, type=str, help="Path the the csv file containing the extra features",)
    parser.add_argument("--mfe", default=None, type=str, help="Path the the csv file containing the MFE features from ViennaRNA",)
    parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Path to pre-trained model or shortcut name selected in the list",)
    parser.add_argument("--task_name", default='rnaprom', type=str, help="Script only prepared for promoter task" )
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--tokenizer_name",default=None,type=str, help="Pretrained tokenizer name or path if not the same as model_name",)

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
    parser.add_argument("--max_seq_length", default=5000, type=int, help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",)
    parser.add_argument("--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",)
    parser.add_argument("--numEpochsBeforeEarlyStopping", default=10, type=int, help="Number of epochs to wait before tracking validation early stopping.",)
    parser.add_argument("--patience", default=5, type=int, help="Number of epochs to wait before validation early stopping is triggered.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_lr", default=5e-4, type=float, help="The initial classifier learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--classifier_weight_decay", default=0.1, type=float, help="Weight decay for classifier.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermediate layer.")
    parser.add_argument("--projector_dropout", default=0.1, type=float, help="Dropout rate of extra features projector.")
    parser.add_argument("--classifier_dropout_prob", default=0.1, type=float, help="Dropout rate of classification head.")
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
    
    # Ray Tune arguments
    parser.add_argument("--use_ray_tune", action="store_true", help="Use Ray Tune for hyperparameter optimization.")
    parser.add_argument("--ray_tune_samples", type=int, default=20, help="Number of Ray Tune trials to run.")
    parser.add_argument("--ray_tune_max_epochs", type=int, default=10, help="Maximum epochs for Ray Tune ASHA scheduler.")
    parser.add_argument("--ray_tune_initial_points", type=int, default=1, help="Number of initial random points for Ray Tune HyperOpt.")
    parser.add_argument("--ray_tune_grace_period", type=int, default=1, help="Minimum epochs before early stopping in ASHA.")
    parser.add_argument("--ray_tune_reduction_factor", type=int, default=2, help="Reduction factor for ASHA scheduler.")
    parser.add_argument("--ray_tune_cpu_per_trial", type=int, default=2, help="Number of CPUs per Ray Tune trial.")
    parser.add_argument("--ray_tune_gpu_per_trial", type=float, default=1.0, help="Number of GPUs per Ray Tune trial.")
    parser.add_argument("--ray_tune_local_dir", type=str, default="./ray_results", help="Local directory for Ray Tune results.")
    parser.add_argument("--train_final_model", action="store_true", help="Train a final model with the best Ray Tune configuration.")
    

    # OTHER
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)


    args = parser.parse_known_args()[0]

# %%
    # Read parameters from YAML file
    if args.params:
        with open(args.params, 'r') as file:
            yaml_params = yaml.safe_load(file)
            for key, value in yaml_params['fineTuneModel'].items():
                parser.set_defaults(**{key: value})
            for key, value in yaml_params['modelParams'].items():
                parser.set_defaults(**{key: value})
            for key, value in yaml_params['rayTune'].items():
                parser.set_defaults(**{key: value})

    args = parser.parse_args()

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            args.model_name_or_path = args.output_dir
            #raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
        logger.info('CONTINUE FROM: ', args.model_name_or_path)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )


    # Setup CUDA, GPU & distributed training
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

    # Set seed
    set_seed(args)

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


    # Prepare task
    #args.task_name = args.task_name.lower()
    #if args.task_name not in processors:
    #    raise ValueError("Task not found: %s" % (args.task_name))
    #processor = processors[args.task_name]()
    #args.output_mode = 'regression' #output_modes[args.task_name]  # 'classification' or 'regressions'
    #label_list = processor.get_labels()  # [0,1]
    num_labels = 1#len(label_list)


    # LOAD AND INITIALIZE MODELS -----------------------------------------------------------------------------------------------------
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    
    # Moved tokenizer loading and extra features processing outside do_visualize
    # as model and features are needed for training/evaluation regardless of visualization.
    # Ensure tokenizer_name from args is used if available, otherwise default.
    tokenizer_path = args.tokenizer_name if args.tokenizer_name else 'AIRI-Institute/gena-lm-bert-base-fly'
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        # Fallback or re-raise, depending on desired behavior
        logger.info("Falling back to default tokenizer 'AIRI-Institute/gena-lm-bert-base-fly'")
        tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-fly')

    scaler = None  # Initialize scaler

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
    


    # TRAIN  -----------------------------------------------------------------------------------------------------
    if args.do_train:
        if args.use_ray_tune:
            model = None # Initialize model to None
            if not args.do_visualize: 
                model_path = args.model_name_or_path if args.model_name_or_path else 'AIRI-Institute/gena-lm-bert-base-fly'
                try:
                    model = GenaLMWithExtraFeatures(
                        model_path if model_path else 'AIRI-Institute/gena-lm-bert-base-fly', 
                        num_extra_features=num_extra_features,
                        dropout_percent=args.hidden_dropout_prob, 
                        hidden_dropout_prob=args.hidden_dropout_prob, 
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        projector_dropout=args.projector_dropout, 
                        classifier_dropout_prob=args.classifier_dropout_prob
                    )
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

            logger.info("Training/evaluation parameters %s", args)
            # Ray Tune hyperparameter optimization
            logger.info("Starting Ray Tune hyperparameter optimization...")
            
            # Initialize Ray
            ray.init(ignore_reinit_error=True)
            
            # Define the trainable function locally to avoid serialization issues
            def ray_tune_trainable(config, base_args_dict=None, extraFeatures_dict=None, num_extra_features=0):
                """
                Ray Tune trainable function that will be optimized.
                All inputs are now basic Python types for serialization.
                """
                # Import required libraries inside the function to avoid issues with multiprocessing
                import torch
                import logging
                import pandas as pd
                import argparse
                import os
                import random
                from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
                from GenaLMWithExtraFeatures import GenaLMWithExtraFeatures
                from sklearn.preprocessing import StandardScaler
                from torch.utils.data import DataLoader, RandomSampler, TensorDataset
                from data_loaders import load_and_cache_examples_3utr as load_and_cache_examples
                from __init__ import glue_compute_metrics as compute_metrics
                import numpy as np
                from tqdm import tqdm
                from ray import tune
                
                # Create a NEW logger instance inside the function to avoid serialization issues
                try:
                    trial_id = tune.get_trial_id()
                except:
                    trial_id = f"trial_{random.randint(1000, 9999)}"
                
                logger = logging.getLogger(f"ray_tune_{trial_id}")
                logger.setLevel(logging.INFO)
                
                # Remove any existing handlers to avoid conflicts
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                
                # Add a simple console handler (no file handles)
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

                # Reconstruct args from dictionary
                args = argparse.Namespace(**base_args_dict)

                set_seed(args)

                # Reconstruct extraFeatures_df from dictionary
                if extraFeatures_dict is not None:
                    data_dict = extraFeatures_dict['data']
                    index = extraFeatures_dict['index']
                    columns = extraFeatures_dict['columns']
                    
                    # Reconstruct DataFrame from split format
                    extraFeatures_df = pd.DataFrame(
                        data=data_dict['data'], 
                        index=data_dict['index'], 
                        columns=data_dict['columns']
                    )
                    
                    # Ensure index and columns match what we expect
                    extraFeatures_df.index = index
                    extraFeatures_df.columns = columns
                else:
                    extraFeatures_df = None
                
                # Update args with hyperparameters from Ray Tune
                args.learning_rate = config["bert_lr"]
                args.classifier_lr = config["classifier_lr"]
                args.weight_decay = config["bert_weight_decay"]
                args.classifier_weight_decay = config["classifier_weight_decay"]
                args.hidden_dropout_prob = config.get("bert_hidden_dropout", 0.1)
                args.attention_probs_dropout_prob = config.get("bert_atten_dropout", 0.1)
                args.classifier_dropout_prob = config["classifier_dropout"]
                args.projector_dropout = config.get("projector_dropout", 0.1)
                # args.adam_epsilon = config.get("adam_epsilon", 1e-8)
                # args.beta1 = config.get("adam_beta1", 0.9)
                # args.beta2 = config.get("adam_beta2", 0.999)
                # args.num_train_epochs = config.get("num_epochs", 3)
                
                # Create unique output directory for this trial
                try:
                    trial_name = tune.get_trial_id()
                except:
                    trial_name = f"trial_{random.randint(1000, 9999)}"
                args.output_dir = os.path.join(base_args_dict['output_dir'], "ray_tune_trials", trial_name)
                
                # Disable visualization and some logging for faster training
                args.do_visualize_during_training = False
                args.save_steps = -1  # Disable checkpoint saving during tuning
                args.logging_steps = 1000
                args.evaluate_during_training = True
                
                # Set device
                device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
                args.device = device
                args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
                
                # Load tokenizer
                tokenizer_path = args.tokenizer_name if args.tokenizer_name else 'AIRI-Institute/gena-lm-bert-base-fly'
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                except Exception as e:
                    logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
                    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-fly')
                
                # Initialize model
                model_path = args.model_name_or_path if args.model_name_or_path else 'AIRI-Institute/gena-lm-bert-base-fly'
                model = GenaLMWithExtraFeatures(
                    model_path if model_path else 'AIRI-Institute/gena-lm-bert-base-fly', 
                    num_extra_features=num_extra_features,
                    dropout_percent=args.hidden_dropout_prob, 
                    hidden_dropout_prob=args.hidden_dropout_prob, 
                    attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                    projector_dropout=args.projector_dropout, 
                    classifier_dropout_prob=args.classifier_dropout_prob
                )
                model.to(args.device)
                
                # Prepare data - need to redefine load_data here to avoid global scope issues
                def load_data_local(args, tokenizer, test_run=False, split='train.fasta'):
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
                        am = [1] * len(s) + [0] * (args.max_seq_length - len(s))
                        # pad s to max_seq_length
                        s = s + [tokenizer.pad_token_id]*(args.max_seq_length-len(s))
                        if s not in seqs: # prevent duplicates
                            seqs.append(s)
                            attention_masks.append(am)
                            labels.append(float(record.id))
                            tr_ids.append(record.description.split(' ')[1])

                    return torch.tensor(labels), torch.tensor(seqs), torch.tensor(attention_masks), tr_ids
                
                labels, seqs, atten_masks, tr_ids = load_data_local(args, tokenizer)
                tr_ids_str = [str(tid) for tid in tr_ids]
                
                scaler = None
                if extraFeatures_df is not None:
                    # Ensure extraFeatures_df index is string type for matching
                    if not pd.api.types.is_string_dtype(extraFeatures_df.index):
                        extraFeatures_df.index = extraFeatures_df.index.astype(str)

                    # Filter extraFeatures_df to include only rows relevant to the current tr_ids for fitting the scaler
                    extraFeatures_train_subset = extraFeatures_df[extraFeatures_df.index.isin(tr_ids_str)]
                    
                    if not extraFeatures_train_subset.empty:
                        scaler = StandardScaler()
                        scaler.fit(extraFeatures_train_subset)
                        # Transform the entire extraFeatures_df using the fitted scaler
                        scaled_values = scaler.transform(extraFeatures_df)
                        extraFeatures_df = pd.DataFrame(scaled_values, columns=extraFeatures_df.columns, index=extraFeatures_df.index)
                
                # Create tr_ids_index for the TensorDataset
                current_split_ef_indices = []
                if extraFeatures_df is not None:
                    ef_index_list_str = extraFeatures_df.index.tolist()
                    for tr_id_str_val in tr_ids_str:
                        try:
                            current_split_ef_indices.append(ef_index_list_str.index(tr_id_str_val))
                        except ValueError:
                            logger.error(f"Training: Transcript ID '{tr_id_str_val}' from training data not found in extraFeatures_df index.")
                            raise ValueError(f"Transcript ID '{tr_id_str_val}' not found in extraFeatures_df index during training data preparation.")
                    tr_ids_index = torch.tensor(current_split_ef_indices, dtype=torch.long)
                else: 
                    tr_ids_index = torch.zeros_like(labels, dtype=torch.long)
                
                train_dataset = TensorDataset(seqs, atten_masks, torch.zeros_like(seqs), labels, tr_ids_index)
                
                # Local evaluate function to avoid global scope issues
                def evaluate_local(args, model, tokenizer, evaluate=True, val=False, extraFeatures=None):
                    results = {}
                    # Load validation data
                    labels, seqs, atten_masks, tr_ids_raw = load_data_local(args, tokenizer, split='dev.fasta')
                    tr_ids = [str(tid) for tid in tr_ids_raw]

                    # Create tr_ids_index for the evaluation dataset
                    if extraFeatures is not None:
                        # Ensure extraFeatures index is string for matching
                        if not pd.api.types.is_string_dtype(extraFeatures.index):
                            extraFeatures.index = extraFeatures.index.astype(str)
                            
                        eval_tr_ids_index_list = []
                        ef_index_list_str = extraFeatures.index.tolist()
                        for tr_id_str_val in tr_ids:
                            try:
                                eval_tr_ids_index_list.append(ef_index_list_str.index(tr_id_str_val))
                            except ValueError:
                                logger.error(f"Evaluation: Transcript ID '{tr_id_str_val}' from dev split not found in extraFeatures index.")
                                raise ValueError(f"Evaluation: Transcript ID '{tr_id_str_val}' not found in extraFeatures index.")
                        tr_ids_index = torch.tensor(eval_tr_ids_index_list, dtype=torch.long)
                    else:
                        tr_ids_index = torch.zeros_like(labels, dtype=torch.long)
                        
                    eval_dataset = TensorDataset(seqs, atten_masks, torch.zeros_like(seqs), labels, tr_ids_index)

                    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
                    from torch.utils.data import SequentialSampler
                    eval_sampler = SequentialSampler(eval_dataset)
                    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    eval_loss = 0.0
                    nb_eval_steps = 0
                    preds = None
                    out_label_ids = None
                    
                    for batch in eval_dataloader:
                        model.eval()
                        if extraFeatures is not None:
                            tmp_extraFeatures = extraFeatures.iloc[[int(x) for x in list(batch[4])]].to_numpy()
                            tmp_extraFeatures = torch.tensor(tmp_extraFeatures, dtype=torch.float32).to(args.device)
                        else:
                            tmp_extraFeatures = None

                        inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device), "labels": batch[3].to(args.device), "extra_features": tmp_extraFeatures}

                        with torch.no_grad():
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

                    preds = np.squeeze(preds)
                    results = compute_metrics('sts-b', preds, out_label_ids)
                    results['loss'] = eval_loss / nb_eval_steps
                    return results
                
                # Training function
                def train_for_tune(args, train_dataset, model, tokenizer, extraFeatures=None, scaler=None):
                    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
                    from torch.utils.data.distributed import DistributedSampler
                    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
                    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

                    if args.max_steps > 0:
                        t_total = args.max_steps
                        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
                    else:
                        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

                    # Optimizer setup with different learning rates
                    no_decay = ["bias", "LayerNorm.weight"]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in model.named_parameters() 
                                if not any(nd in n for nd in no_decay) and not (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
                            ],
                            "weight_decay": args.weight_decay,
                            "lr": args.learning_rate,
                        },
                        {
                            "params": [
                                p for n, p in model.named_parameters() 
                                if any(nd in n for nd in no_decay) and not (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
                            ],
                            "weight_decay": 0.0,
                            "lr": args.learning_rate,
                        },
                        {
                            "params": [
                                p for n, p in model.named_parameters() 
                                if not any(nd in n for nd in no_decay) and (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
                            ],
                            "weight_decay": args.classifier_weight_decay,
                            "lr": args.classifier_lr,
                        },
                        {
                            "params": [
                                p for n, p in model.named_parameters() 
                                if any(nd in n for nd in no_decay) and (n.startswith("classifier") or n.startswith("extraFeaturesProjector"))
                            ],
                            "weight_decay": 0.0,
                            "lr": args.classifier_lr,
                        },
                    ]

                    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent * t_total)
                    
                    optimizer = AdamW(
                        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1, args.beta2)
                    )
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

                    # Training loop
                    model.zero_grad()
                    best_val_loss = float('inf')
                    best_val_spearmanr = float('-inf')
                    
                    for epoch in range(int(args.num_train_epochs)):
                        model.train()
                        epoch_loss = 0.0
                        num_batches = 0
                        
                        for step, batch in enumerate(train_dataloader):
                            if extraFeatures is not None:
                                tmp_extraFeatures = extraFeatures.iloc[[int(x) for x in list(batch[4])]].to_numpy()
                                tmp_extraFeatures = torch.tensor(tmp_extraFeatures, dtype=torch.float32).to(args.device)
                            else:
                                tmp_extraFeatures = None

                            inputs = {
                                "input_ids": batch[0].to(args.device), 
                                "attention_mask": batch[1].to(args.device), 
                                "labels": batch[3].to(args.device), 
                                "extra_features": tmp_extraFeatures
                            }
                            outputs = model(**inputs)
                            loss = outputs['loss']

                            if args.n_gpu > 1:
                                loss = loss.mean()

                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps

                            loss.backward()
                            epoch_loss += loss.item()
                            num_batches += 1

                            if (step + 1) % args.gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                                optimizer.step()
                                scheduler.step()
                                model.zero_grad()

                        # Evaluate at the end of each epoch
                        val_results = evaluate_local(args, model, tokenizer, evaluate=False, val=True, extraFeatures=extraFeatures)
                        val_loss = val_results['loss']
                        val_spearmanr = val_results['spearmanr']
                        
                        # Update best metrics
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                        if val_spearmanr > best_val_spearmanr:
                            best_val_spearmanr = val_spearmanr
                            patience_counter = 0  # Reset patience counter if we improve
                        else:
                            patience_counter += 1

                        # Report metrics to Ray Tune
                        tune.report({
                            "best_loss": best_val_loss,
                            "best_val_spearmanr": best_val_spearmanr,
                            "loss": val_loss,
                            "val_spearmanr": val_spearmanr,
                            "val_pearson": val_results.get('pearson', 0.0),
                            "train_loss": epoch_loss / num_batches,
                            "epoch": epoch
                        })

                        if patience_counter >= args.patience:
                            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                            break

                    return best_val_loss, best_val_spearmanr

                # Run training
                try:
                    best_loss, best_spearmanr = train_for_tune(args, train_dataset, model, tokenizer, extraFeatures=extraFeatures_df, scaler=scaler)
                    
                    # Final report
                    tune.report({
                        "best_val_loss": best_loss,
                        "best_val_spearmanr": best_spearmanr
                    })
                except Exception as e:
                    logger.error(f"Training failed: {e}")
                    tune.report({
                        "best_val_loss": float('inf'), 
                        "val_spearmanr": float('-inf')
                    })
            
            # Define search space using Ray Tune syntax for modern API
            # Ensure all fixed values are properly converted to avoid placeholder issues
            num_epochs_value = int(float(args.num_train_epochs))  # Ensure it's an integer
            
            search_space = {
                "bert_lr": tune.loguniform(1e-6, 1e-4),  # 1e-6 to 1e-4
                "classifier_lr": tune.loguniform(1e-5, 1e-3),  # 1e-5 to 1e-3
                "bert_weight_decay": tune.uniform(0.0, 0.1),  # 0.0 to 0.1
                "classifier_weight_decay": tune.uniform(0.0, 0.1),  # 0.0 to 0.1
                "bert_hidden_dropout": tune.uniform(0.1, 0.5),  # 0.1 to 0.5
                "bert_atten_dropout": tune.uniform(0.1, 0.5),  # 0.1 to 0.5
                "classifier_dropout": tune.uniform(0.1, 0.5),  # 0.1 to 0.5
                "projector_dropout": tune.uniform(0.1, 0.7),  # 0.1 to 0.7
                # "adam_epsilon": tune.choice([1e-8]),  # Fixed value using tune.choice
                # "adam_beta1": tune.choice([0.9]),  # Fixed value using tune.choice
                # "adam_beta2": tune.choice([0.999]),  # Fixed value using tune.choice
                # "num_epochs": tune.choice([num_epochs_value]),  # Fixed value using tune.choice
            }
            
            # Define ASHA scheduler for early stopping
            scheduler = ASHAScheduler(
                max_t=args.ray_tune_max_epochs,
                grace_period=args.ray_tune_grace_period,
                reduction_factor=args.ray_tune_reduction_factor
            )
            
            # Define HyperOpt search algorithm for Bayesian optimization
            search_alg = HyperOptSearch(
                n_initial_points=args.ray_tune_initial_points,  # Number of random points before Bayesian optimization starts
            )

            # Convert non-serializable objects to serializable forms
            # Only include the specific args we need to avoid serialization issues
            
            # Convert relative paths to absolute paths for Ray Tune workers
            model_path_absolute = args.model_name_or_path
            if model_path_absolute and not model_path_absolute.startswith(('/', 'AIRI-Institute/', 'microsoft/', 'google/', 'facebook/', 'bert-', 'gpt-', 'openai/')):
                # Convert relative path to absolute path
                model_path_absolute = os.path.abspath(model_path_absolute)
                logger.info(f"Converting relative model path '{args.model_name_or_path}' to absolute path '{model_path_absolute}'")
            
            data_dir_absolute = os.path.abspath(args.data_dir) if args.data_dir else args.data_dir
            output_dir_absolute = os.path.abspath(args.output_dir) if args.output_dir else args.output_dir
            
            base_args_dict = {
                'runName': args.runName,
                'data_dir': data_dir_absolute,
                'output_dir': output_dir_absolute,
                'tokenizer_name': args.tokenizer_name,
                'model_name_or_path': model_path_absolute,
                'no_cuda': args.no_cuda,
                'max_seq_length': args.max_seq_length,
                'per_gpu_train_batch_size': args.per_gpu_train_batch_size,
                'per_gpu_eval_batch_size': args.per_gpu_eval_batch_size,
                'patience': args.patience,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'max_grad_norm': args.max_grad_norm,
                'max_steps': args.max_steps,
                'warmup_steps': args.warmup_steps,
                'warmup_percent': args.warmup_percent,
                'seed': args.seed,
                'local_rank': args.local_rank,
                'logging_steps': args.logging_steps,
                'save_steps': args.save_steps,
                'train_batch_size': getattr(args, 'train_batch_size', args.per_gpu_train_batch_size),
                'eval_batch_size': getattr(args, 'eval_batch_size', args.per_gpu_eval_batch_size),
                'device': getattr(args, 'device', None),
                'n_gpu': getattr(args, 'n_gpu', 0),
                # Training parameters that will be overridden by Ray Tune config
                'learning_rate': args.learning_rate,
                'classifier_lr': args.classifier_lr,
                'weight_decay': args.weight_decay,
                'classifier_weight_decay': args.classifier_weight_decay,
                'hidden_dropout_prob': args.hidden_dropout_prob,
                'attention_probs_dropout_prob': args.attention_probs_dropout_prob,
                'classifier_dropout_prob': args.classifier_dropout_prob,
                'projector_dropout': args.projector_dropout,
                'adam_epsilon': args.adam_epsilon,
                'beta1': args.beta1,
                'beta2': args.beta2,
                'num_train_epochs': args.num_train_epochs,
                'do_visualize_during_training': args.do_visualize_during_training,
                'evaluate_during_training': args.evaluate_during_training,
            }
            
            # Convert DataFrame to dict if it exists
            extraFeatures_dict = None
            if extraFeatures_df is not None:
                extraFeatures_dict = {
                    'data': extraFeatures_df.to_dict('split'),  # More robust than default to_dict()
                    'index': extraFeatures_df.index.tolist(),
                    'columns': extraFeatures_df.columns.tolist()
                }
            
            # Create the trainable function with serializable parameters
            trainable_with_data = tune.with_parameters(
                ray_tune_trainable,
                base_args_dict=base_args_dict, 
                extraFeatures_dict=extraFeatures_dict, 
                num_extra_features=num_extra_features
            )
            
            # Wrap with resource specification
            trainable_with_resources = tune.with_resources(
                trainable_with_data,
                resources={
                    "cpu": args.ray_tune_cpu_per_trial,
                    "gpu": args.ray_tune_gpu_per_trial
                }
            )
            
            # Convert relative path to absolute path for Ray Tune storage
            ray_tune_storage_path = os.path.abspath(args.ray_tune_local_dir)
            
            # Create Tuner with modern API
            tuner = tune.Tuner(
                trainable_with_resources,
                tune_config=tune.TuneConfig(
                    metric="best_val_spearmanr",
                    mode="max",
                    num_samples=args.ray_tune_samples,
                    scheduler=scheduler,
                    search_alg=search_alg,
                ),
                param_space=search_space,
                run_config=ray.air.RunConfig(
                    name="trial_results",
                    storage_path=ray_tune_storage_path,
                    verbose=1,
                ),
            )
            
            # Run the hyperparameter search
            results = tuner.fit()
            
            # Get best result from the new API
            best_result = results.get_best_result(metric="best_val_spearmanr", mode="max")
            best_config = best_result.config
            best_metrics = best_result.metrics
            
            logger.info("=" * 50)
            logger.info("RAY TUNE OPTIMIZATION COMPLETE")
            logger.info("=" * 50)
            logger.info(f"Best config: {best_config}")
            logger.info(f"Best validation Spearman correlation: {best_metrics['best_val_spearmanr']:.4f}")
            logger.info(f"Best validation loss: {best_metrics.get('loss', 'N/A')}")
            
            # Save results to CSV
            results_df = results.get_dataframe()
            results_csv_path = os.path.join(args.ray_tune_local_dir, "ray_tune_results.csv")
            results_df.to_csv(results_csv_path, index=False)
            logger.info(f"Ray Tune results saved to: {results_csv_path}")
            
            # Save best config to JSON
            best_config_path = os.path.join(args.ray_tune_local_dir, "best_ray_tune_config.json")
            with open(best_config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            logger.info(f"Best config saved to: {best_config_path}")
            
            # Optionally train a final model with the best configuration
            if hasattr(args, 'train_final_model') and args.train_final_model:
                logger.info("Training final model with best configuration...")
                live = Live(os.path.join('dvclive/TE', args.runName), cache_images=True)

                # Force single GPU to match Ray Tune conditions exactly
                if args.n_gpu > 1:
                    logger.info(f"Forcing single GPU for final model training to match Ray Tune conditions")
                    args.n_gpu = 1
                    torch.cuda.set_device(0)  # Use GPU 0
                    args.device = torch.device("cuda:0")
                    logger.info(f"Final model will use device: {args.device}")

                # Update args with best config
                args.learning_rate = best_config["bert_lr"]
                args.classifier_lr = best_config["classifier_lr"]
                args.weight_decay = best_config["bert_weight_decay"]
                args.classifier_weight_decay = best_config["classifier_weight_decay"]
                args.hidden_dropout_prob = best_config.get("bert_hidden_dropout", 0.1)
                args.attention_probs_dropout_prob = best_config.get("bert_atten_dropout", 0.1)
                args.classifier_dropout_prob = best_config["classifier_dropout"]
                args.projector_dropout = best_config.get("projector_dropout", 0.1)
                
                # Re-initialize model with best config
                model = GenaLMWithExtraFeatures(
                    model_path if model_path else 'AIRI-Institute/gena-lm-bert-base-fly', 
                    num_extra_features=num_extra_features,
                    dropout_percent=args.hidden_dropout_prob, 
                    hidden_dropout_prob=args.hidden_dropout_prob, 
                    attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                    projector_dropout=args.projector_dropout, 
                    classifier_dropout_prob=args.classifier_dropout_prob
                )
                model.to(args.device)
                
                # Prepare data and train final model
                labels, seqs, atten_masks, tr_ids = load_data(args, tokenizer)
                tr_ids_str = [str(tid) for tid in tr_ids]
                
                # Prepare extra features and scaler
                scaler = None
                if extraFeatures_df is not None:
                    extraFeatures_train_subset = extraFeatures_df[extraFeatures_df.index.isin(tr_ids_str)]
                    if not extraFeatures_train_subset.empty:
                        scaler = StandardScaler()
                        scaler.fit(extraFeatures_train_subset)
                        scaled_values = scaler.transform(extraFeatures_df)
                        extraFeatures_df = pd.DataFrame(scaled_values, columns=extraFeatures_df.columns, index=extraFeatures_df.index)
                
                # Create tr_ids_index
                current_split_ef_indices = []
                if extraFeatures_df is not None:
                    ef_index_list_str = extraFeatures_df.index.tolist()
                    for tr_id_str_val in tr_ids_str:
                        try:
                            current_split_ef_indices.append(ef_index_list_str.index(tr_id_str_val))
                        except ValueError:
                            raise ValueError(f"Transcript ID '{tr_id_str_val}' not found in extraFeatures_df index during training data preparation.")
                    tr_ids_index = torch.tensor(current_split_ef_indices, dtype=torch.long)
                else: 
                    tr_ids_index = torch.zeros_like(labels, dtype=torch.long)
                
                train_dataset = TensorDataset(seqs, atten_masks, torch.zeros_like(seqs), labels, tr_ids_index)
                global_step, tr_loss = train(args, train_dataset, model, tokenizer, extraFeatures=extraFeatures_df, scaler=scaler)
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            
            ray.shutdown()
            
        else:
            # Regular training without Ray Tune
            live = Live(os.path.join('dvclive/TE', args.runName), cache_images=True)

            # load best ray tune config if available
            best_config_path = os.path.join(args.ray_tune_local_dir, "best_ray_tune_config.json")
            if os.path.exists(best_config_path):
                with open(best_config_path) as f:
                    best_config = json.load(f)
                # Update args with best config
                args.learning_rate = best_config["bert_lr"]
                args.classifier_lr = best_config["classifier_lr"]
                args.weight_decay = best_config["bert_weight_decay"]
                args.classifier_weight_decay = best_config["classifier_weight_decay"]
                args.hidden_dropout_prob = best_config.get("bert_hidden_dropout", 0.1)
                args.attention_probs_dropout_prob = best_config.get("bert_atten_dropout", 0.1)
                args.classifier_dropout_prob = best_config["classifier_dropout"]
                args.projector_dropout = best_config.get("projector_dropout", 0.1)
                logger.info("Loaded best Ray Tune config: %s", best_config)


            model = None # Initialize model to None
            if not args.do_visualize: 
                model_path = args.model_name_or_path if args.model_name_or_path else 'AIRI-Institute/gena-lm-bert-base-fly'
                try:
                    model = GenaLMWithExtraFeatures(
                        model_path if model_path else 'AIRI-Institute/gena-lm-bert-base-fly', 
                        num_extra_features=num_extra_features,
                        dropout_percent=args.hidden_dropout_prob, 
                        hidden_dropout_prob=args.hidden_dropout_prob, 
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        projector_dropout=args.projector_dropout, 
                        classifier_dropout_prob=args.classifier_dropout_prob
                    )
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

            logger.info("Training/evaluation parameters %s", args)

            # Force single GPU to match Ray Tune conditions exactly
            if args.n_gpu > 1:
                logger.info(f"Forcing single GPU for final model training to match Ray Tune conditions")
                args.n_gpu = 1
                torch.cuda.set_device(0)  # Use GPU 0
                args.device = torch.device("cuda:0")
                logger.info(f"Final model will use device: {args.device}")

            if model is None: 
                raise ValueError("Model not initialized. Cannot proceed with training. Check --do_visualize flag or model loading steps.")


            labels, seqs, atten_masks, tr_ids = load_data(args, tokenizer)
            tr_ids_str = [str(tid) for tid in tr_ids] # Ensure tr_ids from load_data are strings for matching

            if extraFeatures_df is not None:
                # Ensure extraFeatures_df index is string type for matching
                if not pd.api.types.is_string_dtype(extraFeatures_df.index):
                     extraFeatures_df.index = extraFeatures_df.index.astype(str)

                # Filter extraFeatures_df to include only rows relevant to the current tr_ids for fitting the scaler
                extraFeatures_train_subset = extraFeatures_df[extraFeatures_df.index.isin(tr_ids_str)]
                
                if not extraFeatures_train_subset.empty:
                    logger.info(f"Fitting scaler on training subset of extra features (shape: {extraFeatures_train_subset.shape}).")
                    scaler = StandardScaler()
                    scaler.fit(extraFeatures_train_subset)
                    # Transform the entire extraFeatures_df using the fitted scaler
                    logger.info("Applying scaler to the entire extra features DataFrame.")
                    scaled_values = scaler.transform(extraFeatures_df) # transform expects numpy array or df with same columns
                    extraFeatures_df = pd.DataFrame(scaled_values, columns=extraFeatures_df.columns, index=extraFeatures_df.index)
                else:
                    logger.warning("No overlapping transcript IDs found between loaded training data and extra features for scaling. Scaler not fitted. Extra features might not be scaled or used effectively.")
            
            # Create tr_ids_index for the TensorDataset
            current_split_ef_indices = []
            if extraFeatures_df is not None:
                ef_index_list_str = extraFeatures_df.index.tolist() # Already ensured to be string

                for tr_id_str_val in tr_ids_str: # Use stringified tr_ids from current split
                    try:
                        current_split_ef_indices.append(ef_index_list_str.index(tr_id_str_val))
                    except ValueError:
                        logger.error(f"Training: Transcript ID '{tr_id_str_val}' from training data not found in extraFeatures_df index. This sample will be problematic.")
                        raise ValueError(f"Transcript ID '{tr_id_str_val}' not found in extraFeatures_df index during training data preparation.")
                tr_ids_index = torch.tensor(current_split_ef_indices, dtype=torch.long)
            else: 
                tr_ids_index = torch.zeros_like(labels, dtype=torch.long) 
            
            train_dataset = TensorDataset(seqs, atten_masks, torch.zeros_like(seqs), labels, tr_ids_index)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer, extraFeatures=extraFeatures_df, scaler=scaler)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            live.end()
    

            # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
            if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                # Create output directory if needed
                if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                    os.makedirs(args.output_dir)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

                # Save the scaler along with the model
                if scaler is not None: # Changed condition to check if scaler exists
                    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

if __name__ == "__main__":
    main()
