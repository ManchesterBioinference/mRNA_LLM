import argparse
import yaml
import random
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
#from ferret import SHAPExplainer

import numpy as np
import torch
import pickle
import copy
import os
import pandas as pd
import joblib


from transformers import (
    AutoTokenizer,
) 

from GenaLMWithExtraFeatures import GenaLMWithExtraFeatures
import torch 
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser()

#################################################################
# BASIC
parser.add_argument("--params", default='params.yaml', type=str, help="Path to the YAML file containing parameters.",)
parser.add_argument("--data_dir", default=None, type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
parser.add_argument("--model_name_or_path", default='output/ftModel/best_spearmanr', type=str, help="Path to pre-trained model or shortcut name selected in the list",)
parser.add_argument("--task_name", default='rnaprom', type=str, help="Script only prepared for promoter task" )
parser.add_argument("--output_file", default=None, type=str, help="The output directory where the model predictions and checkpoints will be written.",)
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


# OTHER
parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",)
parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
#################################################################

parser.add_argument("--sequence_file", default="output/data/decay/train.fasta", type=str, help="Path to the TSV file containing sequences")
parser.add_argument("--save_path", default="deleteme", type=str, help="the directory for output")
parser.add_argument("--extraFeatures", default=None, type=str, help="Path the the csv file containing the extra features",)
parser.add_argument("--mfe", default=None, type=str, help="Path the the csv file containing the MFE features from ViennaRNA",)
parser.add_argument("--debug", default=False, type=bool, help="Path the the csv file containing the extra features",)


args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params['predict'].items():
            parser.set_defaults(**{key: value})
        for key, value in yaml_params['modelParams'].items():
            parser.set_defaults(**{key: value})
        for key, value in yaml_params['importanceAnalysis'].items():
            parser.set_defaults(**{key: value})
        for key, value in yaml_params['randomizeSeqsAndExtraFeatures'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()
#args.sequence_file = 'output/data/sanityCheck/originalSeqs.fasta'

# Set seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    args.n_gpu = torch.cuda.device_count()
else: 
    args.n_gpu = 0
print(f"Using {args.n_gpu} GPUs.")
set_seed(args)

model = GenaLMWithExtraFeatures.from_pretrained(args.model_name_or_path) #, num_labels=1, id2label={0: "LABEL_0"})
model.to(device)
if device.type == 'cuda' and args.n_gpu > 1:
    model = torch.nn.DataParallel(model)  # Wrap the model for multi-GPU training
    print("Model wrapped with DataParallel for multi-GPU training.")
model.eval()

# Load the scaler
scaler_path = os.path.join(args.model_name_or_path, 'scaler.joblib')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = None

try:
    t = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if not t.is_fast:
        print("Warning: Loaded tokenizer is not a 'fast' tokenizer. Performance might be suboptimal.")
except Exception:
    print("Failed to load fast tokenizer, trying with use_fast=False.")
    t = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)


if args.extraFeatures is not None:
    print(f"Loading original extra features from: {args.extraFeatures}")
    try:
        extraFeatures_df = pd.read_csv(args.extraFeatures, index_col=0)
        # Drop specified columns if they exist
        columns_to_drop = ['Decay Rate', 'Residuals']
        existing_columns_to_drop = [col for col in columns_to_drop if col in extraFeatures_df.columns]
        if existing_columns_to_drop:
            extraFeatures_df.drop(columns=existing_columns_to_drop, inplace=True)
            print(f"Dropped columns: {existing_columns_to_drop} from original extra features.")
    except FileNotFoundError:
        print(f"Original extra features file not found: {args.extraFeatures}")
        extraFeatures_df = None
    except Exception as e:
        print(f"Error loading original extra features from {args.extraFeatures}: {e}")
        extraFeatures_df = None
else:
    extraFeatures_df = None
    print("No original extra features CSV provided (args.extraFeatures is None).")

# 2. Load ViennaRNA features (MFE)
print(f"Attempting to load ViennaRNA features from: {args.mfe}")
if os.path.exists(args.mfe):
    try:
        vienna_df = pd.read_csv(args.mfe)
        if "id" not in vienna_df.columns:
            print("ViennaRNA features file found but missing 'id' column. Cannot merge MFE.")
            vienna_df = None
        else:
            vienna_df.set_index("id", inplace=True)
            if 'mfe' in vienna_df.columns:
                print("Found 'mfe' column in ViennaRNA features.")
                vienna_df_mfe = vienna_df[['mfe']].copy() # Use .copy() to avoid SettingWithCopyWarning
                vienna_df_mfe['mfe'] = pd.to_numeric(vienna_df_mfe['mfe'], errors='coerce').fillna(0)
                vienna_df = vienna_df_mfe # Assign back the processed DataFrame
            else:
                print("'mfe' column not found in ViennaRNA features file. Skipping MFE.")
                vienna_df = None
    except Exception as e:
        print(f"Error loading or processing ViennaRNA features from {args.mfe}: {e}")
        vienna_df = None
else:
    print(f"ViennaRNA features file not found at {args.mfe}. Proceeding without MFE.")
    vienna_df = None

# 3. Merge DataFrames
if extraFeatures_df is not None and vienna_df is not None:
    print("Merging original extra features with ViennaRNA MFE features.")
    # Ensure indices are of the same type for robust merging
    extraFeatures_df.index = extraFeatures_df.index.astype(str)
    vienna_df.index = vienna_df.index.astype(str)
    extraFeatures_df = extraFeatures_df.merge(vienna_df, left_index=True, right_index=True, how='left')
    if 'mfe' in extraFeatures_df.columns: # MFE column exists due to merge
         extraFeatures_df['mfe'] = extraFeatures_df['mfe'].fillna(0) # Fill NaNs for IDs in extraFeatures_df but not in vienna_df
    print("Merge complete.")
elif vienna_df is not None and extraFeatures_df is None:
    print("Using only ViennaRNA MFE features as no original extra features were provided.")
    extraFeatures_df = vienna_df.copy() # Use a copy
    extraFeatures_df.index = extraFeatures_df.index.astype(str) # Ensure index is string
# If extraFeatures_df is not None and vienna_df is None, extraFeatures_df is used as is (index type already handled or assumed consistent).
# If both are None, extraFeatures_df remains None.

extra_features_lookup_dict = None # Initialize
if extraFeatures_df is not None:
    # Ensure index is string type if it was not already (e.g. if only original extraFeatures_df was used)
    if not pd.api.types.is_string_dtype(extraFeatures_df.index):
        extraFeatures_df.index = extraFeatures_df.index.astype(str)
    
    # Apply scaling if scaler exists
    if scaler is not None:
        scaled_values = scaler.transform(extraFeatures_df)
        extraFeatures_df_processed = pd.DataFrame(scaled_values, columns=extraFeatures_df.columns, index=extraFeatures_df.index)
    else:
        extraFeatures_df_processed = extraFeatures_df.copy() # Use a copy if no scaling

    print(f"Final extra features DataFrame shape: {extraFeatures_df_processed.shape}")
    print(f"Final extra features columns: {extraFeatures_df_processed.columns.tolist()}")
    
    # Convert to dictionary for faster lookups
    extra_features_lookup_dict = {idx: row_values.to_numpy() for idx, row_values in extraFeatures_df_processed.iterrows()}
else:
    print("No extra features will be used.")


# Assuming 'model', 'device', 't' (tokenizer), 'args', 'extraFeatures_df', 'scaler'
# are defined and initialized correctly before this block.
# 'model' should be on 'device' and already wrapped with torch.nn.DataParallel if args.n_gpu > 1.

# Lists to store results
decay = []  # Stores actual decay rates
predDecay = []  # Stores predicted decay rates
seqCount = []  # Stores a counter or ID for each sequence processed

processed_sequence_counter = 0  # Overall counter for sequences from the input file

# Lists to accumulate data for the current batch
current_batch_input_ids = []
current_batch_attention_masks = []
current_batch_extra_features = [] # Will store numpy arrays (if features are used) or None
current_batch_actual_decay_rates = []
current_batch_sequence_counters = [] # Stores the 'processed_sequence_counter' for items in batch

# Determine effective batch size for model input.
# If args.n_gpu > 0, DataParallel handles splitting this total batch across GPUs.
# Each GPU will process args.per_gpu_pred_batch_size.
if args.n_gpu > 0: # Handles single or multiple GPUs
    effective_batch_size = args.per_gpu_pred_batch_size * args.n_gpu
else: # CPU
    effective_batch_size = args.per_gpu_pred_batch_size
print(f"Effective batch size for prediction: {effective_batch_size}")

# Helper function to process a collected batch
def process_filled_batch(ids_list, masks_list, extras_list, actuals_list, counters_list):
    if not ids_list:
        return

    input_ids_tensor = torch.tensor(ids_list, dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor(masks_list, dtype=torch.long).to(device)
    
    extra_features_tensor = None
    if extras_list and extras_list[0] is not None: # If feature data exists for the batch
        # Assumes all items in extras_list are numpy arrays of consistent shape,
        # due to skipping sequences with missing features if features are generally expected.
        try:
            # Convert list of numpy arrays to a single multi-dimensional numpy array, then to tensor
            extra_features_tensor = torch.tensor(np.array(extras_list), dtype=torch.float32).to(device)
        except Exception as e:
            print(f"Error converting extra features to tensor: {e}. This batch might be skipped or processed without features.")
            # Depending on model requirements, you might want to raise error or ensure fallback
            extra_features_tensor = None # Fallback or ensure model can handle this
    
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, extra_features=extra_features_tensor)
        logits = outputs.logits
        # Ensure logits are 1D (batch_size,)
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

    current_predictions = [round(logit.item(), 2) for logit in logits]
    
    # Extend the main result lists
    predDecay.extend(current_predictions)
    decay.extend(actuals_list)
    seqCount.extend(counters_list)

    # Clear the batch accumulation lists for the next batch
    ids_list.clear()
    masks_list.clear()
    extras_list.clear()
    actuals_list.clear()
    counters_list.clear()

# Iterate through sequences from the FASTA file
for seq_record in tqdm(SeqIO.parse(args.sequence_file, 'fasta'), desc="Processing sequences"):
    processed_sequence_counter += 1
    
    # Sequence preprocessing from original code
    sequence_string = str(seq_record.seq).replace('U', "T")
    
    parts = sequence_string.split(',')
    if len(parts) != 2:
        # print(f"Skipping sequence {seq_record.id} (count: {processed_sequence_counter}): Incorrect format (expected UTR5,UTR3).")
        continue
    utr5_str, utr3_str = parts

    tokenized_utr5 = t.encode(utr5_str, add_special_tokens=True, truncation=False)
    tokenized_utr3 = t.encode(utr3_str, add_special_tokens=True, truncation=False)

    if not tokenized_utr3: # Should not happen with add_special_tokens=True
        combined_tokens = tokenized_utr5
    else:
        combined_tokens = tokenized_utr5 + tokenized_utr3[1:] # Remove CLS of UTR3
    
    if len(combined_tokens) > args.max_seq_length:
        # print(f"Skipping sequence {seq_record.id} (count: {processed_sequence_counter}): Combined token length {len(combined_tokens)} exceeds max_seq_length {args.max_seq_length}.")
        continue 

    # Padding
    padding_length = args.max_seq_length - len(combined_tokens)
    padded_tokens = combined_tokens + [t.pad_token_id] * padding_length
    attention_mask = [1] * len(combined_tokens) + [0] * padding_length

    # Extra features processing
    current_sequence_extra_features = None # Default to None
    if extra_features_lookup_dict is not None:
        # Original logic for FASTA header: >decay_rate_val id_for_features other_stuff
        # seq_record.name is 'decay_rate_val'
        # seq_record.description is the full line 'decay_rate_val id_for_features other_stuff'
        desc_parts = seq_record.description.split()
        if len(desc_parts) > 1:
            seq_id_for_features = desc_parts[1] # The ID used for lookup in extraFeatures_df
            current_sequence_extra_features = extra_features_lookup_dict.get(seq_id_for_features)
            if current_sequence_extra_features is None:
                # print(f"Warning: For sequence {seq_record.id} (count: {processed_sequence_counter}), feature ID '{seq_id_for_features}' not found in extra_features_lookup_dict. Skipping this sequence.")
                continue # Skip if features are expected but missing for this ID
        else:
            # print(f"Warning: Could not parse feature ID from description for sequence {seq_record.id} (count: {processed_sequence_counter}): '{seq_record.description}'. Skipping this sequence.")
            continue # Skip if description format is unsuitable for feature ID parsing
    elif args.extraFeatures is not None: # If extraFeatures CSV was specified but lookup dict is None (e.g. empty after processing)
        # This case implies that features were expected but something went wrong or the file was empty.
        # Depending on desired behavior, you might want to skip all sequences or log a more prominent warning.
        # For now, assume if extra_features_lookup_dict is None, we proceed without features if args.extraFeatures was also None.
        # If args.extraFeatures was provided but lookup_dict is None (e.g. empty file), this means no features are available.
        # The original code would skip if extraFeatures_df was not None but ID was missing.
        # If features are mandatory when args.extraFeatures is set, this logic might need adjustment.
        # The current logic: if extra_features_lookup_dict is None, no features are added.
        # If it's not None, but ID is missing, sequence is skipped. This seems consistent.
        pass


    # Actual decay rate from FASTA name/header
    try:
        actual_decay_rate = round(float(seq_record.name), 2)
    except ValueError:
        # print(f"Warning: Could not parse decay rate from seq_record.name '{seq_record.name}' for sequence {seq_record.id} (count: {processed_sequence_counter}). Skipping this sequence.")
        continue # Skip if actual decay rate is not parseable

    # Add processed data to current batch lists
    current_batch_input_ids.append(padded_tokens)
    current_batch_attention_masks.append(attention_mask)
    current_batch_extra_features.append(current_sequence_extra_features)
    current_batch_actual_decay_rates.append(actual_decay_rate)
    current_batch_sequence_counters.append(processed_sequence_counter)

    # If batch is full, process it
    if len(current_batch_input_ids) >= effective_batch_size:
        process_filled_batch(
            current_batch_input_ids, 
            current_batch_attention_masks, 
            current_batch_extra_features, 
            current_batch_actual_decay_rates, 
            current_batch_sequence_counters
        )

# Process any remaining sequences in the last batch (if not empty)
if current_batch_input_ids:
    process_filled_batch(
        current_batch_input_ids, 
        current_batch_attention_masks, 
        current_batch_extra_features, 
        current_batch_actual_decay_rates, 
        current_batch_sequence_counters
    )

# write actual and predicted decay rates as two columns in a csv file
df = pd.DataFrame({'originalPrediction': decay, 'perturbedPrediction': predDecay, 'sequenceCount': seqCount})
df.to_csv(args.save_path, index=False)
