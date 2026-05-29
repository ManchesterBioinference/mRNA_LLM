import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

import os
import json

class myClassifier(nn.Module):
    def __init__(self, num_extra_features=None, num_labels=1, dropout_percent=0.1, max_length=512, projector_dropout=None, classifier_dropout_prob=None):
        super(myClassifier, self).__init__()
        self.num_extra_features = num_extra_features
        self.num_labels = num_labels
        self.dropout_percent = dropout_percent
        self.device = None
        self.max_length = max_length
        self.projector_dropout = projector_dropout
        if classifier_dropout_prob is None:
            classifier_dropout_prob = dropout_percent
        self.classifier_dropout_prob = classifier_dropout_prob

        if self.projector_dropout is not None and self.num_extra_features > 0:
            self.extraFeaturesProjector = nn.Sequential(
                nn.Linear(self.num_extra_features, self.num_extra_features),
                nn.ReLU(),
                nn.Dropout(self.projector_dropout),
                nn.Linear(self.num_extra_features, self.num_extra_features)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.num_extra_features, 256),
            nn.ReLU(),
            nn.Dropout(self.classifier_dropout_prob),
            nn.Linear(256, self.num_labels)
        )

    def forward(
        self,
        extra_features=None,
        labels=None,
    ):

        if hasattr(self, 'extraFeaturesProjector'):
            extra_features = self.extraFeaturesProjector(extra_features)
        logits = self.classifier(extra_features)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #return {'loss': loss, 'logits': logits} 
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def save_pretrained(self, save_directory):
        """
        Save the model weights and configuration to a directory.
        
        Args:
            save_directory (str): Directory path where the model will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration as a dictionary
        config = {
            "num_labels": self.num_labels,
            "num_extra_features": self.num_extra_features, # Ensure this is saved correctly
            "dropout_percent": self.dropout_percent,
            "projector_dropout": self.projector_dropout,
            "classifier_dropout_prob": self.classifier_dropout_prob,
            "max_length": self.max_length
        }
        
        config = {key: (float(value) if isinstance(value, torch.Tensor) or isinstance(value, float) else value) for key, value in config.items()}
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def from_pretrained(save_directory):
        """
        Load the model weights and configuration from a directory.
        
        Args:
            save_directory (str): Directory path where the model was saved
        
        Returns:
            ORLDModel: Loaded ORLDModel instance
        """
        # Load configuration from config.json
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize the model with the configuration
        model = myClassifier(
            num_extra_features=config["num_extra_features"], # Ensure this is loaded correctly
            num_labels=config["num_labels"],
            dropout_percent=config["dropout_percent"],
            projector_dropout=config.get("projector_dropout"),
            classifier_dropout_prob=config.get("classifier_dropout_prob"),
            max_length=config["max_length"]
        )
        
        # Load model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        
        return model

    def to(self, device):
        """
        Move the model to a specified device (CPU or GPU).
        
        Args:
            device (torch.device): The device to move the model to
        """
        super(myClassifier, self).to(device)
        self.device = device



import argparse
import yaml
import pandas as pd
from Bio import SeqIO
import numpy as np
import psutil

from __init__ import glue_compute_metrics as compute_metrics
from dvclive import Live
import logging
import joblib
from sklearn.discriminant_analysis import StandardScaler

# Ray Tune imports
import ray
import ray.air
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

parser = argparse.ArgumentParser(description="Train a myClassifier model")
parser.add_argument("--params", type=str, required=True, help="Path to the extra features .csv file")
parser.add_argument("--label", type=str, required=True, help="Label the defines the extra features to use")
parser.add_argument("--extraFeatures", type=str, required=True, help="Path to the extra features .csv file")
parser.add_argument("--mfe", type=str, default=None, help="Path to the mfe .csv file")
parser.add_argument("--mastResults", type=str, default=None, help="Path to the mast results .txt file")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model checkpoints")
parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--per_device_batch_size", type=int, default=48, help="Batch size for training")
#parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
#parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
#parser.add_argument("--dropout_percent", type=float, default=0.01, help="Dropout percentage for the model")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--n_gpu", type=int, default=None, help="Number of GPUs available")

# Ray Tune arguments
parser.add_argument("--ray_tune_samples", type=int, default=20, help="Number of Ray Tune trials to run.")
parser.add_argument("--ray_tune_max_epochs", type=int, default=10, help="Maximum epochs for Ray Tune ASHA scheduler.")
parser.add_argument("--ray_tune_grace_period", type=int, default=1, help="Minimum epochs before early stopping in ASHA.")
parser.add_argument("--ray_tune_initial_points", type=int, default=10, help="Minimum epochs before early stopping in ASHA.")
parser.add_argument("--ray_tune_reduction_factor", type=int, default=2, help="Reduction factor for ASHA scheduler.")
parser.add_argument("--ray_tune_cpu_per_trial", type=int, default=2, help="Number of CPUs per Ray Tune trial.")
parser.add_argument("--ray_tune_gpu_per_trial", type=float, default=1.0, help="Number of GPUs per Ray Tune trial.")
parser.add_argument("--ray_tune_local_dir", type=str, default="./ray_results", help="Local directory for Ray Tune results.")
parser.add_argument("--train_final_model", action="store_true", help="Train a final model with the best Ray Tune configuration.")

args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params[args.label+'ClassificationMLP'].items():
            parser.set_defaults(**{key: value})
        for key, value in yaml_params['rayTuneClassificationMLP'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()

# Convert paths to absolute paths to avoid issues with Ray Tune multiprocessing
args.data_dir = os.path.abspath(args.data_dir)
args.output_dir = os.path.abspath(args.output_dir)
args.extraFeatures = os.path.abspath(args.extraFeatures)
args.ray_tune_local_dir = os.path.abspath(args.ray_tune_local_dir)

# Create output directories if they don't exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.ray_tune_local_dir, exist_ok=True)

# Convert additional paths if they exist
if hasattr(args, 'params') and args.params:
    args.params = os.path.abspath(args.params)
if hasattr(args, 'mfe') and args.mfe:
    args.mfe = os.path.abspath(args.mfe)
if hasattr(args, 'mastResults') and args.mastResults:
    args.mastResults = os.path.abspath(args.mastResults)

logger = logging.getLogger(__name__)

def set_seed(args):
    #random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu and args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def plotPredictions(preds, out_label_ids, results, stage='train', live_logger=None):
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    
    fig = plt.figure(figsize=(5,5))
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
    
    # Log the image using dvclive only if live_logger is provided
    if live_logger is not None:
        live_logger.log_image(f'{stage}/predictions_vs_true_labels_{live_logger.step}.png', fig)

    if stage == 'test':
        pd.DataFrame({'True Labels': out_label_ids, 'Predictions': preds}).to_csv(os.path.join(live_logger._dir, f'{stage}_predictions_vs_true_labels.csv'), index=False)
        plt.savefig(os.path.join(live_logger._dir, f'{stage}_predictions_vs_true_labels.svg'), dpi=300)

    plt.close(fig)  # Close figure to prevent memory leaks

def prepMotifCounts(mastResultsPath):
    # Read the file, skipping the first two lines that start with #
    df = pd.read_csv(mastResultsPath, sep='\s+', skiprows=2, skipfooter=1, engine='python')
    df.columns = ['sequence_name', 'strand', 'id', 'alt_id', 'hit_start','hit_end','score','p_value']

    # Create new column 'trID' by splitting sequence_name on '_' and taking index 0
    df['trID'] = df['sequence_name'].str.split('_').str[0]

    # Count alt_ids for each unique trID value
    alt_id_counts_per_trID = df.groupby(['trID','alt_id']).count().reset_index().iloc[:,:3]
    alt_id_counts_per_trID.columns = ['trID', 'alt_id_count', 'counts']

    # Pivot the dataframe to have trID as rows and alt_id_count as columns with counts as values
    alt_id_counts_per_trID_pivoted = alt_id_counts_per_trID.pivot(index='trID', columns='alt_id_count', values='counts').fillna(0).astype(int)
    alt_id_counts_per_trID_pivoted.columns = alt_id_counts_per_trID_pivoted.columns+'_rbp'
    return(alt_id_counts_per_trID_pivoted)

def train_model(config, data_dict=None, args=None):
    """Training function for Ray Tune"""
    set_seed(args)
    # Ensure args paths are absolute (in case they weren't converted properly)
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    if not os.path.isabs(args.ray_tune_local_dir):
        args.ray_tune_local_dir = os.path.abspath(args.ray_tune_local_dir)
    
    # Update args with config from Ray Tune
    args.learning_rate = config["learning_rate"]
    args.weight_decay = config["weight_decay"]
    args.dropout_percent = config["dropout_percent"]
    args.projector_dropout = config["projector_dropout"]
    args.classifier_dropout_prob = config["classifier_dropout_prob"]
    
    # Reconstruct DataFrames from serialized format
    def deserialize_dataframe(serialized_df):
        """Convert serialized dict back to DataFrame"""
        return pd.DataFrame(
            data=serialized_df['data'],
            index=serialized_df['index'],
            columns=serialized_df['columns']
        )
    
    # Reconstruct extraFeatures DataFrames
    extraFeatures = {
        'train': deserialize_dataframe(data_dict['extraFeatures']['train']),
        'dev': deserialize_dataframe(data_dict['extraFeatures']['dev']),
        'test': deserialize_dataframe(data_dict['extraFeatures']['test'])
    }
    
    # Reconstruct labels Series
    labels = {
        'train': pd.Series(data_dict['labels']['train'], index=data_dict['label_indices']['train']),
        'dev': pd.Series(data_dict['labels']['dev'], index=data_dict['label_indices']['dev']),
        'test': pd.Series(data_dict['labels']['test'], index=data_dict['label_indices']['test'])
    }
    
    # Set up model with config
    model = myClassifier(
        num_extra_features=data_dict['num_features'],
        num_labels=1,
        dropout_percent=config["dropout_percent"],
        projector_dropout=config["projector_dropout"],
        classifier_dropout_prob=config["classifier_dropout_prob"],
    )
    
    # Set up optimizer with config
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    
    batch_size = args.per_device_batch_size
    num_batches = len(extraFeatures['train']) // batch_size + (1 if len(extraFeatures['train']) % batch_size > 0 else 0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    best_val_spearmanr = float('-inf')
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.ray_tune_max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(extraFeatures['train']))
            batch_features = torch.tensor(extraFeatures['train'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(labels['train'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features, labels=batch_labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        train_loss /= num_batches
        
        # Validation
        model.eval()
        val_preds = None
        val_labels = None
        val_loss = 0.0
        val_batches = len(extraFeatures['dev']) // batch_size + (1 if len(extraFeatures['dev']) % batch_size > 0 else 0)
        
        with torch.no_grad():
            for batch_idx in range(val_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(extraFeatures['dev']))
                batch_features = torch.tensor(extraFeatures['dev'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(labels['dev'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)
                
                outputs = model(batch_features, labels=batch_labels)
                val_logits = outputs.logits
                loss = outputs.loss
                val_loss += loss.item()
                
                if val_preds is None:
                    val_preds = val_logits.detach().cpu().numpy()
                    val_labels = batch_labels.detach().cpu().numpy()
                else:
                    val_preds = np.append(val_preds, val_logits.detach().cpu().numpy(), axis=0)
                    val_labels = np.append(val_labels, batch_labels.detach().cpu().numpy(), axis=0)
        
        val_loss /= val_batches
        val_preds = np.squeeze(val_preds)
        val_results = compute_metrics('sts-b', val_preds, val_labels)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        # Track best validation spearman correlation
        if val_results['spearmanr'] > best_val_spearmanr:
            best_val_spearmanr = val_results['spearmanr']
        
        # Report metrics to Ray Tune
        tune.report({
            "best_val_loss": best_val_loss,
            "best_val_spearmanr": best_val_spearmanr,
            "loss": val_loss,
            "spearmanr": val_results['spearmanr'],
            "pearson": val_results['pearson'],
        })
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break

def main():
    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()
    else:
        args.n_gpu = 0
    set_seed(args)

    # Configure logging at the start of main()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    
    # Log the absolute paths being used
    logger.info(f"Using absolute paths:")
    logger.info(f"  data_dir: {args.data_dir}")
    logger.info(f"  output_dir: {args.output_dir}")
    logger.info(f"  extraFeatures: {args.extraFeatures}")
    logger.info(f"  ray_tune_local_dir: {args.ray_tune_local_dir}")
    if hasattr(args, 'params') and args.params:
        logger.info(f"  params: {args.params}")
    if hasattr(args, 'mfe') and args.mfe:
        logger.info(f"  mfe: {args.mfe}")
    if hasattr(args, 'mastResults') and args.mastResults:
        logger.info(f"  mastResults: {args.mastResults}")
    
    # get train, validation, and test data directories
    trIDs = {}
    for item in ['train', 'dev', 'test']:
        seqs = SeqIO.parse(os.path.join(args.data_dir, item + '.fasta'), 'fasta')
        # Extract element index 1 from seq.description for each sequence
        trID = [seq.description.split()[1] for seq in seqs]
        trIDs[item] = trID

    # Load extra features
    extraFeaturesDF = pd.read_csv(args.extraFeatures,index_col=0)
    if args.label == 'allExtraFeatures' or args.label == 'extraFeaturesAndMotifCounts':
        # get mfe
        vienna_df = pd.read_csv(args.mfe)
        vienna_df['mfe'] = pd.to_numeric(vienna_df['mfe'], errors='coerce').fillna(0)
        extraFeaturesDF = extraFeaturesDF.merge(vienna_df, left_index=True, right_index=True, how='left')
    if args.label == 'extraFeaturesAndMotifCounts':
        # get motif counts
        motifCountsDF = prepMotifCounts(args.mastResults)
        extraFeaturesDF = extraFeaturesDF.merge(motifCountsDF, left_index=True, right_index=True, how='left')
    extraFeaturesDF = extraFeaturesDF.fillna(0)
    extraFeatures = {item: extraFeaturesDF[extraFeaturesDF.index.isin(trIDs[item])] for item in ['train', 'dev', 'test']}
    labels = {item: extraFeaturesDF[extraFeaturesDF.index.isin(trIDs[item])]['Decay Rate'] for item in ['train', 'dev', 'test']}
    if args.label == 'codonOnly':
        colFilt = [col for col in extraFeaturesDF.columns if len(col) == 3]
    else:
        colFilt = [col for col in extraFeaturesDF.columns if col not in ['Decay Rate','Residuals']]

    extraFeatures = {item: extraFeatures[item][colFilt] for item in ['train', 'dev', 'test']}
    scaler = StandardScaler()
    scaler.fit(extraFeatures['train'])
    extraFeatures = {item: pd.DataFrame(scaler.transform(extraFeatures[item]),columns=extraFeatures[item].columns, index=extraFeatures[item].index) for item in ['train', 'dev', 'test']}

    # Prepare data dictionary for Ray Tune (convert DataFrames to serializable format)
    def serialize_dataframe(df):
        """Convert DataFrame to serializable dict format"""
        return {
            'data': df.values.tolist(),
            'index': df.index.tolist(),
            'columns': df.columns.tolist()
        }
    
    data_dict = {
        'extraFeatures': {
            'train': serialize_dataframe(extraFeatures['train']),
            'dev': serialize_dataframe(extraFeatures['dev']),
            'test': serialize_dataframe(extraFeatures['test'])
        },
        'labels': {
            'train': labels['train'].values.tolist(),
            'dev': labels['dev'].values.tolist(),
            'test': labels['test'].values.tolist()
        },
        'label_indices': {
            'train': labels['train'].index.tolist(),
            'dev': labels['dev'].index.tolist(),  
            'test': labels['test'].index.tolist()
        },
        'num_features': extraFeatures['train'].shape[1]
    }

    # Initialize Ray with explicit CPU count
    import psutil
    
    # Get available CPUs (either from SLURM or system)
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        logger.info(f"Using SLURM_CPUS_PER_TASK: {num_cpus}")
    elif 'SLURM_CPUS_ON_NODE' in os.environ:
        num_cpus = int(os.environ['SLURM_CPUS_ON_NODE'])
        logger.info(f"Using SLURM_CPUS_ON_NODE: {num_cpus}")
    else:
        num_cpus = psutil.cpu_count(logical=True)
        logger.info(f"Using system CPU count: {num_cpus}")
    
    # Calculate max concurrent trials
    max_concurrent_trials = num_cpus // args.ray_tune_cpu_per_trial
    logger.info(f"CPU allocation: {num_cpus} total CPUs, {args.ray_tune_cpu_per_trial} CPU per trial")
    logger.info(f"Max concurrent trials: {max_concurrent_trials}")
    
    # Initialize Ray with explicit resource specification
    ray.init(
        ignore_reinit_error=True,
        num_cpus=num_cpus,
        num_gpus=0,  # Since you're using CPU-only trials
        include_dashboard=False  # Disable dashboard to save resources
    )

    # Define hyperparameter search space
    config = {
        "learning_rate": tune.loguniform(1e-5, 5e-3),
        "weight_decay": tune.uniform(0.001, 0.1),
        "dropout_percent": tune.uniform(0.0, 0.9),
        "projector_dropout": tune.uniform(0.1, 0.9),  # None disables the projector
        "classifier_dropout_prob": tune.uniform(0.1, 0.9),
    }

    # Set up ASHA scheduler
    scheduler = ASHAScheduler(
        max_t=args.ray_tune_max_epochs,
        grace_period=args.ray_tune_grace_period,
        reduction_factor=args.ray_tune_reduction_factor
    )

    # Set up HyperOpt search algorithm
    search_alg = HyperOptSearch(n_initial_points=args.ray_tune_initial_points)

    try:
        tuner = tune.Tuner.restore(os.path.abspath(os.path.join(args.ray_tune_local_dir, args.label)),trainable=train_model)
        logger.info("Restored Ray Tune tuner from previous run.")
        results = tuner.get_results()
        if len(results) != args.ray_tune_samples:
            logger.info("Number of results does not match expected samples. Finishing hyperparameter tuning...")
            results = tuner.fit()
        else:
            results = tuner.get_results()
    except:
        # Run hyperparameter tuning
        logger.info("Starting Ray Tune hyperparameter optimization...")
        tuner = tune.Tuner(
            tune.with_parameters(train_model, data_dict=data_dict, args=args),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=args.ray_tune_samples,
                metric="best_val_loss",
                mode="min",
                max_concurrent_trials=max_concurrent_trials  # Explicitly set max concurrent trials
            ),
            param_space=config,
            run_config=ray.air.RunConfig(
                storage_path=args.ray_tune_local_dir,
                name=args.label,
                stop={"training_iteration": args.ray_tune_max_epochs}
            )
        )

        results = tuner.fit()
    best_result = results.get_best_result("best_val_spearmanr", "max")
    
    logger.info(f"Best hyperparameters found: {best_result.config}")
    logger.info(f"Best validation spearmanr: {best_result.metrics['best_val_spearmanr']}")

    # Save best hyperparameters
    best_config_path = os.path.join(args.output_dir, "best_hyperparameters.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump(best_result.config, f, indent=2)
    logger.info(f"Best hyperparameters saved to {best_config_path}")

    # Train final model with best hyperparameters if requested
    if args.train_final_model:
        logger.info("Training final model with best hyperparameters...")
        
        # Update args with best hyperparameters
        args.learning_rate = best_result.config["learning_rate"]
        args.weight_decay = best_result.config["weight_decay"]
        args.dropout_percent = best_result.config["dropout_percent"]
        args.projector_dropout = best_result.config["projector_dropout"]
        args.classifier_dropout_prob = best_result.config["classifier_dropout_prob"]

        # Initialize final model
        model = myClassifier(
            num_extra_features=extraFeatures['train'].shape[1],
            num_labels=1,
            dropout_percent=args.dropout_percent,
            projector_dropout=args.projector_dropout,
            classifier_dropout_prob=args.classifier_dropout_prob,
        )

        # Set up optimizer with best hyperparameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # Training parameters
        num_epochs = args.num_train_epochs
        batch_size = args.per_device_batch_size
        num_batches = len(extraFeatures['train']) // batch_size + (1 if len(extraFeatures['train']) % batch_size > 0 else 0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Initialize DVCLive for final model training
        live = Live(os.path.join('dvclive/codonOnlyClassificationMLP', args.label + '_final'), cache_images=True)

        # Train the final model
        logger.info("Starting final model training...")
        best_val_loss = float('inf')
        best_val_spearmanr = float('-inf')
        patience_counter = 0
        patience_threshold = args.patience

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss = 0.0
            model.train()
            preds = None
            out_label_ids = None
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(extraFeatures['train']))
                batch_features = torch.tensor(extraFeatures['train'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(labels['train'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)

                optimizer.zero_grad()
                outputs = model(batch_features, labels=batch_labels)
                logits = outputs.logits
                loss = outputs.loss
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = batch_labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, batch_labels.detach().cpu().numpy(), axis=0)

            train_loss /= num_batches
            live.log_metric('train/loss', train_loss)

            preds = np.squeeze(preds)
            results = compute_metrics('sts-b', preds, out_label_ids)
            for key, value in results.items():
                live.log_metric(f"train/{key}", value)
            plotPredictions(preds, out_label_ids, results, live_logger=live)

            # Validation
            logger.info("Starting evaluation...")
            model.eval()
            val_preds = None
            val_labels = None
            val_loss = 0.0
            val_batches = len(extraFeatures['dev']) // batch_size + (1 if len(extraFeatures['dev']) % batch_size > 0 else 0)
            
            with torch.no_grad():
                for batch_idx in range(val_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(extraFeatures['dev']))
                    batch_features = torch.tensor(extraFeatures['dev'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)
                    batch_labels = torch.tensor(labels['dev'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)

                    outputs = model(batch_features, labels=batch_labels)
                    val_logits = outputs.logits
                    loss = outputs.loss
                    val_loss += loss.item()

                    if val_preds is None:
                        val_preds = val_logits.detach().cpu().numpy()
                        val_labels = batch_labels.detach().cpu().numpy()
                    else:
                        val_preds = np.append(val_preds, val_logits.detach().cpu().numpy(), axis=0)
                        val_labels = np.append(val_labels, batch_labels.detach().cpu().numpy(), axis=0)

            val_loss /= val_batches
            live.log_metric('val/loss', val_loss)
            val_preds = np.squeeze(val_preds)
            val_results = compute_metrics('sts-b', val_preds, val_labels)
            for key, value in val_results.items():
                live.log_metric(f"val/{key}", value)
            plotPredictions(val_preds, val_labels, val_results, stage='val', live_logger=live)
            live.next_step()

            # Save best models
            if val_results['spearmanr'] > best_val_spearmanr:
                best_val_spearmanr = val_results['spearmanr']
                live.log_metric('global/best_spearmanr', best_val_spearmanr)
                output_dir = os.path.join(args.output_dir, "best_spearmanr")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving best spearmannr model checkpoint and scaler to %s", output_dir)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                live.log_metric('global/best_val_loss', best_val_loss)
                patience_counter = 0
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving best model checkpoint to %s", output_dir)
            else:
                patience_counter += 1
                logger.info("Validation loss did not improve. Patience counter: %d", patience_counter)
            
            if patience_counter >= patience_threshold:
                logger.info("Early stopping triggered. Stopping training.")
                break

        # Test the final model
        logger.info("Testing final model...")
        model.eval()
        test_preds = None
        test_labels = None
        test_loss = 0.0
        test_batches = len(extraFeatures['test']) // batch_size + (1 if len(extraFeatures['test']) % batch_size > 0 else 0)

        with torch.no_grad():
            for batch_idx in range(test_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(extraFeatures['test']))
                batch_features = torch.tensor(extraFeatures['test'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(labels['test'].iloc[start_idx:end_idx].values, dtype=torch.float32).to(device)

                outputs = model(batch_features, labels=batch_labels)
                test_logits = outputs.logits
                # accumulate loss reported by the model
                if outputs.loss is not None:
                    test_loss += outputs.loss.item()

                if test_preds is None:
                    test_preds = test_logits.detach().cpu().numpy()
                    test_labels = batch_labels.detach().cpu().numpy()
                else:
                    test_preds = np.append(test_preds, test_logits.detach().cpu().numpy(), axis=0)
                    test_labels = np.append(test_labels, batch_labels.detach().cpu().numpy(), axis=0)

        # Average test loss across batches
        if test_batches > 0:
            test_loss /= test_batches

        test_preds = np.squeeze(test_preds)
        test_results = compute_metrics('sts-b', test_preds, test_labels)
        # add loss to test_results and log it
        test_results['loss'] = test_loss
        live.log_metric('test/loss', test_loss)

        for key, value in test_results.items():
            live.log_metric(f"test/{key}", value)
        plotPredictions(test_preds, test_labels, test_results, stage='test', live_logger=live)
        live.end()

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()