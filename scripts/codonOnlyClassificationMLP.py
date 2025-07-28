import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

import os
import json

class myClassifier(nn.Module):
    def __init__(self, num_extra_features=None, num_labels=1, dropout_percent=0.1, max_length=512, hidden_size=256):
        super(myClassifier, self).__init__()
        self.num_extra_features = num_extra_features
        self.num_labels = num_labels
        self.dropout_percent = dropout_percent
        self.device = None
        self.max_length = max_length
        self.hidden_size = hidden_size #256 is the same size as in the original model

        self.dropout = nn.Dropout(self.dropout_percent)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_extra_features, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_percent),
            nn.Linear(self.hidden_size, self.num_labels)
        )

    def forward(
        self,
        extra_features=None,
        labels=None,
    ):

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
            "hidden_size": self.hidden_size,
            "num_labels": self.num_labels,
            "num_extra_features": self.num_extra_features, # Ensure this is saved correctly
            "dropout_percent": self.dropout_percent,
            "max_length": self.max_length
        }
        
        config = {key: (float(value) if isinstance(value, torch.Tensor) or isinstance(value, float) else value) for key, value in config.items()}
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
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
            hidden_size=config["hidden_size"],
            dropout_percent=config["dropout_percent"],
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


from __init__ import glue_compute_metrics as compute_metrics
from dvclive import Live
import logging
import joblib
from sklearn.discriminant_analysis import StandardScaler


parser = argparse.ArgumentParser(description="Train a myClassifier model")
parser.add_argument("--params", type=str, required=True, help="Path to the extra features .csv file")
parser.add_argument("--label", type=str, required=True, help="Label the defines the extra features to use")
parser.add_argument("--extraFeatures", type=str, required=True, help="Path to the extra features .csv file")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model checkpoints")
parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--per_device_batch_size", type=int, default=48, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
parser.add_argument("--dropout_percent", type=float, default=0.01, help="Dropout percentage for the model")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_known_args()[0]

# Read parameters from YAML file
if args.params:
    with open(args.params, 'r') as file:
        yaml_params = yaml.safe_load(file)
        for key, value in yaml_params[args.label+'ClassificationMLP'].items():
            parser.set_defaults(**{key: value})

args = parser.parse_args()

live = Live(os.path.join('dvclive/codonOnlyClassificationMLP',args.label), cache_images=True)
logger = logging.getLogger(__name__)

def set_seed(args):
    #random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
    live.log_image(f'{stage}/predictions_vs_true_labels_{live.step}.png',fig)

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
    

    # get train, validation, and test data directories
    trIDs = {}
    for item in ['train', 'dev', 'test']:
        seqs = SeqIO.index(os.path.join(args.data_dir, item + '.fasta'), 'fasta')
        # Extract element index 1 from seq.description for each sequence
        trID = [seq.description.split()[1] for seq in seqs.values()]
        trIDs[item] = trID


    # Load extra features
    extraFeaturesDF = pd.read_csv(args.extraFeatures,index_col=0)
    extraFeatures = {item: extraFeaturesDF[extraFeaturesDF.index.isin(trIDs[item])] for item in ['train', 'dev', 'test']}
    labels = {item: extraFeaturesDF[extraFeaturesDF.index.isin(trIDs[item])]['Decay Rate'] for item in ['train', 'dev', 'test']}
    if args.label == 'codonOnly':
        colFilt = [col for col in extraFeaturesDF.columns if len(col) == 3]
    elif args.label == 'allExtraFeatures':
        colFilt = [col for col in extraFeaturesDF.columns if col not in ['Decay Rate','Residuals']]
    extraFeatures = {item: extraFeatures[item][colFilt] for item in ['train', 'dev', 'test']}
    scaler = StandardScaler()
    scaler.fit(extraFeatures['train'])
    extraFeatures = {item: pd.DataFrame(scaler.transform(extraFeatures[item]),columns=extraFeatures[item].columns, index=extraFeatures[item].index) for item in ['train', 'dev', 'test']}

    model = myClassifier(
        num_extra_features=extraFeatures['train'].shape[1],
        num_labels=1,  # Assuming regression task
        dropout_percent=args.dropout_percent,
    )

    # set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # set up training loop
    num_epochs = args.num_train_epochs
    batch_size = args.per_device_batch_size
    num_batches = len(extraFeatures['train']) // batch_size + (1 if len(extraFeatures['train']) % batch_size > 0 else 0)

    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    ### Train the model ###
    logger.info("Starting training...")
    best_val_loss = float('inf')
    best_val_spearmanr = float('-inf')
    patience_counter = 0
    patience_threshold = args.patience
    epoch_counter = 0
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = 0.0
        model.train()
        preds = None
        out_label_ids = None
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(extraFeatures['train']))
            batch_features = torch.tensor(extraFeatures['train'].iloc[start_idx:end_idx].values, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
            batch_labels = torch.tensor(labels['train'].iloc[start_idx:end_idx].values, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

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

            #logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item()}")

        epoch_counter += 1
        train_loss /= num_batches
        live.log_metric('train/loss',train_loss)

        preds = np.squeeze(preds)
        results = compute_metrics('sts-b', preds, out_label_ids)
        for key, value in results.items():
            live.log_metric(f"train/{key}", value)
        plotPredictions(preds, out_label_ids, results)

        ### Evaluate on validation set for early stopping ##
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
                batch_features = torch.tensor(extraFeatures['dev'].iloc[start_idx:end_idx].values, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
                batch_labels = torch.tensor(labels['dev'].iloc[start_idx:end_idx].values, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

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
        plotPredictions(val_preds, val_labels, val_results, stage='val')
        live.next_step()

        if val_results['spearmanr'] > best_val_spearmanr:
            best_val_spearmanr = val_results['spearmanr']
            live.log_metric('global/best_spearmanr', best_val_spearmanr)
            output_dir = os.path.join(args.output_dir, "best_spearmanr")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            
            # Save the scaler along with the model
            if args.extraFeatures is not None:
                joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
                
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving best spearmannr model checkpoint and scaler to %s", output_dir)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            output_dir = os.path.join(args.output_dir, "best_checkpoint")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)

            # Save the scaler along with the model
            if args.extraFeatures is not None:
                joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving best model checkpoint to %s", output_dir)
        else:
            patience_counter += 1
            logger.info("Validation loss did not improve. Patience counter: %d", patience_counter)
        
        if patience_counter >= patience_threshold:
            logger.info("Early stopping triggered. Stopping training.")
            break

    ### test the model ###
    model.eval()
    test_preds = None
    test_labels = None
    test_batches = len(extraFeatures['test']) // batch_size + (1 if len(extraFeatures['test']) % batch_size > 0 else 0)

    with torch.no_grad():
        for batch_idx in range(test_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(extraFeatures['test']))
            batch_features = torch.tensor(extraFeatures['test'].iloc[start_idx:end_idx].values, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
            batch_labels = torch.tensor(labels['test'].iloc[start_idx:end_idx].values, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(batch_features, labels=batch_labels)
            test_logits = outputs.logits

            if test_preds is None:
                test_preds = test_logits.detach().cpu().numpy()
                test_labels = batch_labels.detach().cpu().numpy()
            else:
                test_preds = np.append(test_preds, test_logits.detach().cpu().numpy(), axis=0)
                test_labels = np.append(test_labels, batch_labels.detach().cpu().numpy(), axis=0)

    test_preds = np.squeeze(test_preds)
    test_results = compute_metrics('sts-b', test_preds, test_labels)
    for key, value in test_results.items():
        live.log_metric(f"test/{key}", value)
    plotPredictions(test_preds, test_labels, test_results, stage='test')
    live.end()

if __name__ == "__main__":
    main()