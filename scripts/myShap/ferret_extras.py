### Class from ferret.explainers.explanation.py, I tweaked it to contain the items I wanted
from dataclasses import dataclass
import numpy as np
@dataclass
class Explanation:
    """Generic class to represent an Explanation"""

    id: str
    actual: float
    prediction: float
    tokens: list
    scores: np.array
    explainer: str
    target: int
###
### Class from ferret.model_utils.py, I tweaked it to work with GenaLMWithExtraFeatures
import math
import pdb
from typing import List, Union
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding


class ModelHelper:
    """
    Wrapper class to interface with HuggingFace models
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _tokenize(self, text: str, **tok_kwargs) -> BatchEncoding:
        """
        Base tokenization strategy for a single text.

        Note that we truncate to the maximum length supported by the model.

        :param text str: the string to tokenize
        """
        if ',' in text:
            utr5, utr3 = text.split(',')
            utr5 = self.tokenizer.encode(str(utr5).replace('U',"T"), add_special_tokens=True)
            utr3 = self.tokenizer.encode(str(utr3).replace('U',"T"), add_special_tokens=True) #, max_length=args.max_seq_length-len(utr5)+1, pad_to_max_length=False, truncation=True)
            s = utr5 + utr3[1:]
            am = [1]*len(s)
            am += [0]*(self.model.max_length-len(s))
            # pad s to max_length
            s = s + [self.tokenizer.pad_token_id]*(self.model.max_length-len(s))
            return torch.tensor(s), torch.tensor(am)
        else:
            return self.tokenizer(text, return_tensors="pt", truncation=True, **tok_kwargs)

    def get_input_embeds(self, text: str) -> torch.Tensor:
        """Extract input embeddings

        :param text str: the string to extract embeddings from.
        """
        item = self._tokenize(text)
        item = {k: v.to(self.model.device) for k, v in item.items()}
        embeddings = self._get_input_embeds_from_ids(item["input_ids"][0])
        embeddings = embeddings.unsqueeze(0)
        return embeddings

    def _get_input_embeds_from_ids(self, ids) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def get_tokens(self, text: str, **tok_kwargs) -> List[str]:
        """Extract a list of tokens

        :param text str: the string to extract tokens from.
        """
        item = self._tokenize(text)
        input_len = item["attention_mask"].sum()
        ids = item["input_ids"][0][:input_len]
        return self.tokenizer.convert_ids_to_tokens(ids, **tok_kwargs)

    def _forward_with_input_embeds(
        self,
        input_embeds,
        attention_mask,
        batch_size=8,
        show_progress=False,
        output_hidden_states=False,
    ):
        input_len = input_embeds.shape[0]
        n_batches = math.ceil(input_len / batch_size)
        input_batches = torch.tensor_split(input_embeds, n_batches)
        mask_batches = torch.tensor_split(attention_mask, n_batches)

        if show_progress:
            pbar = tqdm(total=n_batches, desc="Batch", leave=False)

        outputs = list()
        for emb, mask in zip(input_batches, mask_batches):
            out = self.model(
                inputs_embeds=emb,
                attention_mask=mask,
                output_hidden_states=output_hidden_states,
            )
            outputs.append(out)

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        logits = torch.cat([o.logits for o in outputs])
        return outputs, logits

    def _forward(
        self,
        text: Union[str, List[str]],
        batch_size=8,
        show_progress=False,
        use_input_embeddings=False,
        output_hidden_states=True,
        extra_features=None, ################# Logan Brase Addition
        **tok_kwargs
    ):
        if isinstance(text, str):
            text = [text]

        n_batches = math.ceil(len(text) / batch_size)
        batches = np.array_split(text, n_batches)

        outputs = list()
        with torch.no_grad():

            if show_progress:
                pbar = tqdm(total=n_batches, desc="Batch", leave=False)

            for batch in batches:
                item = self._tokenize(batch.tolist(), padding="longest", **tok_kwargs)
                item = {k: v.to(self.model.device) for k, v in item.items()}
                if extra_features is not None:  ################# Logan Brase Addition
                    item['extra_features'] = extra_features.to(self.model.device)  ################# Logan Brase Addition

                if use_input_embeddings:
                    ids = item.pop("input_ids")  # (B,S,d_model)
                    input_embeddings = self._get_input_embeds_from_ids(ids)
                    out = self.model(
                        inputs_embeds=input_embeddings,
                        **item,
                        output_hidden_states=output_hidden_states,
                    )
                else:
                    #print('model device:', self.model.device)
                    out = self.model(**item, output_hidden_states=output_hidden_states)
                outputs.append(out)

                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()

        logits = torch.cat([o.logits for o in outputs])
        return outputs, logits
###

### Class from ferret.explainers.__init__.py, need to include it so my modified ModelHelper is called rather than the original one
"""Explainers API"""

from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    def __init__(self, model, tokenizer):
        self.helper = ModelHelper(model, tokenizer)

    @property
    def device(self):
        return self.helper.model.device

    @property
    def tokenizer(self):
        return self.helper.tokenizer

    def _tokenize(self, text, **tok_kwargs):
        return self.helper._tokenize(text, **tok_kwargs)

    def get_tokens(self, text):
        return self.helper.get_tokens(text)

    def get_input_embeds(self, text):
        return self.helper.get_input_embeds(text)

    @abstractmethod
    def compute_feature_importance(self, text: str, target: int, **explainer_args):
        pass

    def __call__(self, text: str, target: int = 1, **explainer_args):
        return self.compute_feature_importance(text, target, **explainer_args)
    
###