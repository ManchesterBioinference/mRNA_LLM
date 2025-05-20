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


import numpy as np
import torch
from decouple import config
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from data_loaders import load_and_cache_examples_3utr as load_and_cache_examples
from data_loaders import visualize 

from dvclive import Live

from src.transformers import glue_compute_metrics as compute_metrics

from transformers import (
    AutoTokenizer,
) 

from GenaLMWithExtraFeatures import GenaLMWithExtraFeatures
logger = logging.getLogger(__name__)
live = Live('dvclive/decayPredict', cache_images=True)

TOKEN_ID_GROUP = ["bert", "3utrlong", "3utrlongcat", "xlnet", "albert"]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def plotPredictions(preds, out_label_ids, results):
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
    live.log_image('predictions_vs_true_labels_pred.png',fig)

def load_data(args, tokenizer, test_run=False, split='train.fasta'):
    from Bio import SeqIO
    data = list(SeqIO.parse(os.path.join(args.data_dir, split), 'fasta'))
    labels = []
    seqs = []
    tr_ids = []
    for record in data:
        utr5, utr3 = record.seq.split(',')
        utr5 = tokenizer.encode(str(utr5).replace('U',"T"), add_special_tokens=True)
        utr3 = tokenizer.encode(str(utr3).replace('U',"T"), add_special_tokens=True) #, max_length=args.max_seq_length-len(utr5)+1, pad_to_max_length=False, truncation=True)
        s = utr5 + utr3[1:]
        if len(s) < 10 or len(s) > args.max_seq_length:
            continue
        # pad s to max_seq_length
        s = s + [tokenizer.pad_token_id]*(args.max_seq_length-len(s))
        if s not in seqs: # prevent duplicates
            seqs.append(s)
            labels.append(float(record.id))
            tr_ids.append(record.description.split(' ')[1])

    return torch.tensor(labels), torch.tensor(seqs), tr_ids

def evaluate(args, model, tokenizer, prefix="", evaluate=True, val=False):
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    labels, seqs, tr_ids = load_data(args, tokenizer, split='test.fasta')
    #train_dataset = mask_tokens(torch.tensor(seqs), labels, tokenizer)
    if args.extraFeatures is not None:
        extraFeatures = pd.read_csv(args.extraFeatures, index_col=0)
        extraFeatures.drop(['Decay Rate', 'Residuals'], axis=1, inplace=True)
    else:
        extraFeatures = None

    tr_ids_index = torch.tensor([extraFeatures.index.tolist().index(tr_id) for tr_id in tr_ids]) if extraFeatures is not None else torch.zeros_like(seqs)
    eval_dataset = TensorDataset(seqs, torch.ones_like(seqs),torch.zeros_like(seqs),labels, tr_ids_index) #load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate, val=val)

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
    plotPredictions(preds, out_label_ids, results)

    output_eval_file = os.path.join(eval_output_dir, prefix, "test_results.txt")
    with open(output_eval_file, "a") as writer:
        eval_result = args.data_dir.split("/")[-1] + " "

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            eval_result = eval_result + str(results[key])[:5] + " "
        writer.write(eval_result + "\n")

    for key, value in results.items():
        live.log_metric(f"test/{key}", value)

    return results

def main():
    parser = argparse.ArgumentParser()

    # BASIC
    parser.add_argument("--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
    parser.add_argument("--data_dir", default=None, type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--extraFeatures", default=None, type=str, help="Path the the csv file containing the extra features",)
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
    args.model_name_or_path = 'output/ftModel'
    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base-fly')
    checkpoints = [args.model_name_or_path]
    # To evaluate all checkpoints in the folder
    logger.info("Testing the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

        # # Load the model configuration
        # config = config_class.from_pretrained(checkpoint)
        # # Update the configuration for regression
        # config.id2label = None
        # config.label2id = None

        # model = model_class.from_pretrained(checkpoint,config=config)
        # cell = MemoryCell(model, num_mem_tokens=args.memory_size, stage='finetune')
        # model = RecurrentWrapper(cell,segment_size=args.block_size, max_n_segments=args.max_n_segments)
        model = GenaLMWithExtraFeatures.from_pretrained(args.model_name_or_path)
        model.eval()
        model.to(args.device)
        result = evaluate(args, model, tokenizer, prefix=prefix) # Results saved in file eval_results.txt
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)
    print(results)

    return results


if __name__ == "__main__":
    main()
