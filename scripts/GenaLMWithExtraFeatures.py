import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

import os
import json

class GenaLMWithExtraFeatures(nn.Module):
    def __init__(self, base_model_name, num_extra_features=0, num_labels=1, dropout_percent=0.1, max_length=512, classifier_dropout_prob=0.1):
        super(GenaLMWithExtraFeatures, self).__init__()
        self.model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True).bert
        self.num_extra_features = num_extra_features # This will now include MFE
        self.num_labels = num_labels
        self.dropout_percent = dropout_percent
        self.classifier_dropout_prob = classifier_dropout_prob
        self.device = next(self.parameters()).device
        self.max_length = max_length

        self.dropout = nn.Dropout(self.dropout_percent)
        # self.classifier = nn.Linear(self.model.config.hidden_size + self.num_extra_features, self.num_labels) #regression
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size + self.num_extra_features, 256),
            nn.ReLU(),
            nn.Dropout(self.classifier_dropout_prob),
            nn.Linear(256, self.num_labels)
        )


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        extra_features=None,
        **kwargs
    ):

        outputs = self.model(input_ids, attention_mask=attention_mask,)

        last_hidden_stat = outputs[0]
        pooled_output = last_hidden_stat[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        if extra_features is not None:
            # Check values without affecting gradient computation
            #with torch.no_grad():
            #    # Print mean of extra_features and pooled_output for comparison
            #    print(extra_features.shape)
            #    print(pooled_output.shape)
            #    print(f"Extra features mean: {torch.mean(extra_features, dim=0)}")
            #    print(f"Pooled output mean: {torch.mean(pooled_output, dim=0)}")
            # Ensure extra_features is 2D: [batch_size, num_extra_features]
            if extra_features.ndim == 1:
                extra_features = extra_features.unsqueeze(1)
            pooled_output = torch.cat((pooled_output, extra_features), dim=1)

        logits = self.classifier(pooled_output)


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
            "base_model_name": self.model.config._name_or_path,
            "hidden_size": self.model.config.hidden_size,
            "num_labels": self.num_labels,
            "num_extra_features": self.num_extra_features, # Ensure this is saved correctly
            "dropout_percent": self.dropout_percent,
            "classifier_dropout_prob": self.classifier_dropout_prob,
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
        model = GenaLMWithExtraFeatures(
            base_model_name=config["base_model_name"],
            num_extra_features=config["num_extra_features"], # Ensure this is loaded correctly
            num_labels=config["num_labels"],
            dropout_percent=config["dropout_percent"],
            classifier_dropout_prob=config["classifier_dropout_prob"],
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
        super(GenaLMWithExtraFeatures, self).to(device)
        self.model.to(device)
        self.device = device
