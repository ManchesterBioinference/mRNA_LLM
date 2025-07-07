from myShap import Explainer
from myShap.maskers import Text as TextMasker
from ferret import BaseExplainer
from ferret.explainers.explanation import Explanation
from ferret.explainers.utils import parse_explainer_args
import joblib

class SHAPExplainer(BaseExplainer):
    NAME = "Partition SHAP"

    def compute_feature_importance(self, text, target=None, extra_features=None, ID=None, actual=None, predicted = None, **explainer_args):
        # For regression, target is not needed, but kept for compatibility (ignored)
        init_args, call_args = parse_explainer_args(explainer_args)

        # SHAP silent mode
        init_args["silent"] = init_args.get("silent", True)
        # Default to 'Partition' algorithm
        init_args["algorithm"] = init_args.get("algorithm", "partition")
        # Seed for reproducibility
        init_args["seed"] = init_args.get("seed", 42)

        # Store extra_features for use in prediction function
        self.extra_features = extra_features

        # Define prediction function for regression (raw output, no softmax)
        def func(texts):
            # If we have extra features, make sure to include them in forward pass
            if self.extra_features is not None:
                # Expand extra_features to match batch size if needed
                batch_size = len(texts)
                expanded_features = self.extra_features.unsqueeze(0).expand(batch_size, -1)
                
                # Call forward with extra features directly
                outputs = self.helper._forward(texts, extra_features=expanded_features)
                
                return outputs[1].cpu().numpy()  # Return raw outputs for regression
            else:
                # Call forward without extra features
                outputs = self.helper._forward( texts)
                
                return outputs[1].cpu().numpy()  # Return raw outputs for regression

        # Set up the masker and explainer
        masker = TextMasker(self.tokenizer)
        explainer_partition = Explainer(model=func, masker=masker, **init_args)
        
        # Compute SHAP values
        shap_values = explainer_partition([text], **call_args)
        
        # For regression, shap_values.values shape is [n_samples, n_features] (no class dimension)
        attr = shap_values.values[0]  # Take all feature contributions for the single output

        # Create explanation (target=None or 0 to indicate regression, adjust as needed)
        output = Explanation(ID, actual, predicted, masker._segments_s.tolist(), attr, self.NAME, target=None)
        #output = Explanation(text, masker._segments_s.tolist(), attr, self.NAME, target=None)
        return output
    
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


from transformers import (
    AutoTokenizer,
) 

from GenaLMWithExtraFeatures import GenaLMWithExtraFeatures

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

args = parser.parse_args()

# Set seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    args.n_gpu = 1
else: 
    args.n_gpu = 0
set_seed(args)

model = GenaLMWithExtraFeatures.from_pretrained(args.model_name_or_path) #, num_labels=1, id2label={0: "LABEL_0"})
model.to(device)
model.eval()

# Load the scaler
scaler_path = os.path.join(args.model_name_or_path, 'scaler.joblib')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = None

#seq = 'UCUA CUAC UACU ACUU CUUC UUCG UCGA CGAC GACU ACUG CUGU UGUA GUAA UAAG AAGC AGCC GCCA CCAA CAAA AAAA AAAC AACA ACAG CAGC AGCG GCGU CGUU GUUU UUUU UUUU UUUU UUUU UUUU UUUU UUUU UUUU UUUC UUCU UCUU CUUU UUUU UUUG UUGU UGUU GUUG UUGA UGAG GAGU AGUG GUGC UGCA GCAG CAGC AGCA GCAC CACA ACAA CAAC AACU ACUA CUAU UAUA AUAU UAUG AUGA UGAC GACG ACGA CGAA GAAA AAAU AAUA AUAU UAUG AUGU UGUG GUGC UGCA GCAA CAAA AAAU AAUA AUAU UAUG AUGU UGUU GUUU UUUA UUAU UAUC AUCU UCUA CUAC UACC ACCA CCAA CAAC' #1
#seq = 'GCCA CCAA CAAA AAAA AAAA AAAA AAAA AAAA AAAC AACA ACAA CAAA AAAA AAAA AAAG AAGA AGAA GAAG AAGA AGAG GAGA AGAA GAAA AAAC AACA ACAU CAUC AUCG UCGA CGAA GAAA AAAA AAAU AAUA AUAU UAUA AUAG UAGG AGGG GGGG GGGA GGAA GAAC AACA ACAU CAUG AUGU UGUC GUCU UCUA CUAA UAAA AAAA AAAG AAGA AGAA GAAG AAGC AGCA GCAG CAGA AGAA GAAC AACA ACAG CAGG AGGG GGGC GGCC GCCU CCUA CUAA UAAA AAAG AAGU AGUU GUUC UUCA UCAA CAAA AAAA AAAU AAUG AUGG UGGU GGUG GUGG UGGC GGCU GCUG CUGU UGUG GUGC UGCC GCCU CCUC CUCC' #4
#seq = tokenizer.encode(seq, add_special_tokens=True)

t = AutoTokenizer.from_pretrained(args.model_name_or_path)

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

if extraFeatures_df is not None:
    # Ensure index is string type if it was not already (e.g. if only original extraFeatures_df was used)
    if not pd.api.types.is_string_dtype(extraFeatures_df.index):
        extraFeatures_df.index = extraFeatures_df.index.astype(str)
    scaled_values = scaler.transform(extraFeatures_df) if scaler is not None else extraFeatures_df
    extraFeatures_df = pd.DataFrame(scaled_values, columns=extraFeatures_df.columns, index=extraFeatures_df.index)
    print(f"Final extra features DataFrame shape: {extraFeatures_df.shape}")
    print(f"Final extra features columns: {extraFeatures_df.columns.tolist()}")
else:
    print("No extra features will be used.")

# Run bench.explain for each sequence
explanations_list = []
decay = []
predDecay = []
myShap = SHAPExplainer(model, t)
for seq in tqdm(SeqIO.parse(args.sequence_file, 'fasta')):
    seq.seq = str(seq.seq).replace('U', "T")
    # Check seq length after tokenization
    utr5, utr3 = str(seq.seq).split(',')
    utr5 = t.encode(str(utr5), add_special_tokens=True)
    utr3 = t.encode(str(utr3), add_special_tokens=True) #, max_length=args.max_seq_length-len(utr5)+1, pad_to_max_length=False, truncation=True)
    s = utr5 + utr3[1:]
    if len(s) > args.max_seq_length:
        print(f"Skipping sequence {seq.id} due to length {len(s)}")
        continue
    am = [1] * len(s) + [0] * (args.max_seq_length - len(s))
    s = s + [t.pad_token_id] * (args.max_seq_length - len(s))
    # Get extra features
    seq_id = seq.description.split()[1]
    if extraFeatures_df is not None:
        tmp_extraFeatures = extraFeatures_df.loc[seq_id].to_numpy()
    else:
        tmp_extraFeatures = None

    actualDecayRate = round(float(seq.name),2)
    predictedDecayRate = round(model(input_ids=torch.tensor([s]).to(device), attention_mask=torch.tensor([am]).to(device), extra_features=torch.tensor([tmp_extraFeatures], dtype=torch.float32).to(device)).logits.item(),2)
    decay.append(actualDecayRate)
    predDecay.append(predictedDecayRate)

    if not args.debug:
        # Pass extra_features to the compute_feature_importance method
        e = myShap(str(seq.seq), target=None, extra_features=torch.tensor(tmp_extraFeatures, dtype=torch.float32).to(device), show_progress=True, ID = seq_id, actual = actualDecayRate, predicted = predictedDecayRate)
        new_e = copy.copy(e)
        new_e.scores /= np.linalg.norm(e.scores, ord=1) #L1 normalization axis=-1, 
        explanations_list.append(new_e)

if not args.debug:
    # Save the explanations_list using pickle
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(explanations_list, f)

# plot the decay vs predicted decay and save the fig using dvclive
import matplotlib.pyplot as plt
from dvclive import Live
from scipy.stats import pearsonr, spearmanr

live = Live('dvclive/decayImportanceAnalysis', cache_images=True)

fig = plt.figure()
# Create a scatter plot
plt.scatter(decay, predDecay, alpha=0.5)
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.title('Predictions vs True Labels')
plt.xlim(min(decay + predDecay), max(decay + predDecay))
plt.ylim(min(decay + predDecay), max(decay + predDecay))

# Calculate Pearson and Spearman correlation coefficients
pearson_corr = pearsonr(decay, predDecay)[0]
spearman_corr = spearmanr(decay, predDecay)[0]

# Annotate the plot with the correlation coefficients
plt.annotate(f'Pearson: {pearson_corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
plt.annotate(f'Spearman: {spearman_corr:.2f}', xy=(0.05, 0.90), xycoords='axes fraction')

# Log the image using dvclive
fastaName = os.path.basename(args.sequence_file).split('.')[0]
live.log_image(f'predictions_vs_true_labels_{fastaName}.png',fig)