import abc

import torch

from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from src.modeling.basemodel import BaseModel, DistilBertBaseModel

    
class Biencoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.pad = 0

    def forward_embedding(self, input_ids):
        device = input_ids.device
        attn_mask = torch.ones(input_ids.size()).long().to(device).masked_fill(input_ids == 0, 0)
        device = input_ids.device
        embeds = self._get_embedding(input_ids, attention_mask=attn_mask)
        return embeds

    def prepare_candidates(
            self,
            y_input_ids: torch.Tensor,
            candidate_input_ids: torch.Tensor
    ):
        ground_truth_input_ids = y_input_ids.unsqueeze(1)
        combined_input_ids = torch.cat([ground_truth_input_ids, candidate_input_ids], 1)

        return combined_input_ids

    def _get_embedding(self, input_ids, attention_mask):
        embeds = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        embeds = embeds #* BERT_SCALING_FACTOR
        return embeds

    def forward(
            self,
            input_ids: torch.Tensor,
            y_input_ids: torch.Tensor = None,
            candidate_input_ids: torch.Tensor = None,
            candidate_embeds: torch.Tensor = None,
            labels: torch.Tensor = None,
            **kwargs
    ):
        bsz, num_factoids = input_ids.shape[:2]
        device = input_ids.device

        attn_mask = torch.ones(input_ids.size()).long().to(device).masked_fill(input_ids == 0, 0)

        x_embeds = self._get_embedding(input_ids, attn_mask)
        
        if candidate_embeds is not None:
            y_embeds = candidate_embeds
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
        elif candidate_input_ids is not None:
            y_input_ids = candidate_input_ids
            num_cands = y_input_ids.shape[1]
            y_input_ids = y_input_ids.reshape([-1, y_input_ids.shape[-1]])
            y_attn_mask = torch.ones(y_input_ids.size()).long().to(device).masked_fill(y_input_ids == self.pad, 0)
            y_embeds = self._get_embedding(y_input_ids, y_attn_mask)
            y_embeds = y_embeds.reshape([bsz, num_cands, -1])
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
        else:
            y_attn_mask = torch.ones(y_input_ids.size()).long().to(device).masked_fill(y_input_ids == self.pad, 0)
            y_embeds = self._get_embedding(y_input_ids, y_attn_mask)
            scores = torch.matmul(x_embeds, y_embeds.T)
            labels = torch.arange(bsz, device=device)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(scores, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=scores
        )


BERT_SCALING_FACTOR = 0.06

class BiencoderMixin:
    """A Mixin that performs biencoder-like functions for an arbitrary embedding model"""

    def _get_embedding(*args, **kwargs):
        pass

    def forward_embedding(self, input_ids):
        device = input_ids.device
        attn_mask = torch.ones(input_ids.size()).long().to(device).masked_fill(input_ids == 0, 0)
        device = input_ids.device
        embeds = self._get_embedding(input_ids, attention_mask=attn_mask)
        return embeds

    def forward_biencoder(
        self,
        input_ids: torch.Tensor,
        y_input_ids: torch.Tensor = None,
        candidate_input_ids: torch.Tensor = None,
        candidate_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):

        bsz, num_factoids = input_ids.shape[:2]
        device = input_ids.device

        attn_mask = torch.ones(input_ids.size()).long().to(device).masked_fill(input_ids == 0, 0)

        x_embeds = self._get_embedding(input_ids, attn_mask)

        if candidate_embeds is not None:
            y_embeds = candidate_embeds
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
        elif candidate_input_ids is not None:
            y_input_ids = candidate_input_ids
            num_cands = y_input_ids.shape[1]
            y_input_ids = y_input_ids.reshape([-1, y_input_ids.shape[-1]])
            y_attn_mask = torch.ones(y_input_ids.size()).long().to(device).masked_fill(y_input_ids == self.pad, 0)
            y_embeds = self._get_embedding(y_input_ids, y_attn_mask)
            y_embeds = y_embeds.reshape([bsz, num_cands, -1])
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
        else:
            y_attn_mask = torch.ones(y_input_ids.size()).long().to(device).masked_fill(y_input_ids == self.pad, 0)
            y_embeds = self._get_embedding(y_input_ids, y_attn_mask)
            scores = torch.matmul(x_embeds, y_embeds.T)
            labels = torch.arange(bsz, device=device)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(scores, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=scores
        )


class DistilBertBiencoder(DistilBertBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.pad = 0

    def forward_embedding(self, input_ids):
        device = input_ids.device
        attn_mask = torch.ones(input_ids.size()).long().to(device).masked_fill(input_ids == 0, 0)
        device = input_ids.device
        embeds = self._get_embedding(input_ids, attention_mask=attn_mask)
        return embeds

    def _get_embedding(self, input_ids, attention_mask):
        embeds = self.distilbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        embeds = embeds  #* BERT_SCALING_FACTOR
        return embeds

    def forward(
            self,
            input_ids: torch.Tensor,
            y_input_ids: torch.Tensor = None,
            candidate_input_ids: torch.Tensor = None,
            candidate_embeds: torch.Tensor = None,
            labels: torch.Tensor = None,
            **kwargs
    ):

        bsz, num_factoids = input_ids.shape[:2]
        device = input_ids.device

        attn_mask = torch.ones(input_ids.size()).long().to(device).masked_fill(input_ids == 0, 0)

        x_embeds = self._get_embedding(input_ids, attn_mask)

        if candidate_embeds is not None:
            y_embeds = candidate_embeds
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
        elif candidate_input_ids is not None:
            y_input_ids = candidate_input_ids
            num_cands = y_input_ids.shape[1]
            y_input_ids = y_input_ids.reshape([-1, y_input_ids.shape[-1]])
            y_attn_mask = torch.ones(y_input_ids.size()).long().to(device).masked_fill(y_input_ids == self.pad, 0)
            y_embeds = self._get_embedding(y_input_ids, y_attn_mask)
            y_embeds = y_embeds.reshape([bsz, num_cands, -1])
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
        else:
            y_attn_mask = torch.ones(y_input_ids.size()).long().to(device).masked_fill(y_input_ids == self.pad, 0)
            y_embeds = self._get_embedding(y_input_ids, y_attn_mask)
            scores = torch.matmul(x_embeds, y_embeds.T)
            labels = torch.arange(bsz, device=device)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(scores, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=scores
        )


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


