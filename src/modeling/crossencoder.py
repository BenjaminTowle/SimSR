import torch

from torch import nn
from typing import Optional
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class CrossEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.pad = 0

    def get_length(self, input_ids):
        mask = input_ids.eq(self.pad)
        length = (~mask).sum(1, keepdim=True)

        return length, mask

    def truncate(self, input_ids):
        assert len(input_ids.shape) == 2, "input_ids must be a 2d tensor"
        max_length = self.get_length(input_ids)[0].max().item()

        return input_ids[:, :max_length]

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_input_ids: Optional[torch.Tensor] = None,
        y_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        bsz = input_ids.size(0)
        device = input_ids.device
        
        if candidate_input_ids is None:
            num_cands = bsz
            candidate_input_ids = y_input_ids.unsqueeze(0).expand(bsz, -1, -1)
            labels = torch.arange(bsz).to(device)
        else:
            num_cands = candidate_input_ids.size(1)

        candidate_input_ids = candidate_input_ids.reshape([bsz*num_cands, -1])

        input_ids = self.truncate(input_ids)
        input_ids = input_ids.unsqueeze(1).expand(-1, num_cands, -1).reshape([bsz*num_cands, -1])

        token_type_ids = torch.cat([
            torch.zeros(input_ids.size()), torch.ones(candidate_input_ids.size())
        ], dim=-1).to(device).int()

        joint_input_ids = torch.cat([
            input_ids, candidate_input_ids
        ], dim=-1)

        attention_mask = ~joint_input_ids.eq(self.pad)

        outputs = self.bert(
            input_ids=joint_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls = self.cls(outputs.last_hidden_state[:, 0, :])

        scores = cls.reshape([bsz, num_cands])

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(scores, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=scores
        )


class BinaryCrossEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.pad = 0

    def get_length(self, input_ids):
        mask = input_ids.eq(self.pad)
        length = (~mask).sum(1, keepdim=True)

        return length, mask

    def truncate(self, input_ids):
        assert len(input_ids.shape) == 2, "input_ids must be a 2d tensor"
        max_length = self.get_length(input_ids)[0].max().item()

        return input_ids[:, :max_length]

    def forward(
            self,
            input_ids: torch.Tensor,
            candidate_input_ids: Optional[torch.Tensor] = None,
            y_input_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ):
        bsz = input_ids.size(0)
        device = input_ids.device

        if candidate_input_ids is None:
            num_cands = 2
            negatives = torch.cat([
                y_input_ids[1:], y_input_ids[:1]
            ], dim=0)
            candidate_input_ids = torch.stack([
                y_input_ids, negatives
            ], dim=1)
            #candidate_input_ids = y_input_ids.unsqueeze(0).expand(bsz, -1, -1)
            labels = torch.stack([
                torch.ones(bsz).to(device),
                torch.zeros(bsz).to(device)
            ], dim=1).reshape(-1)
        else:
            num_cands = candidate_input_ids.size(1)

        candidate_input_ids = candidate_input_ids.reshape([bsz * num_cands, -1])

        input_ids = self.truncate(input_ids)
        input_ids = input_ids.unsqueeze(1).expand(-1, num_cands, -1).reshape([bsz * num_cands, -1])

        token_type_ids = torch.cat([
            torch.zeros(input_ids.size()), torch.ones(candidate_input_ids.size())
        ], dim=-1).to(device).int()

        joint_input_ids = torch.cat([
            input_ids, candidate_input_ids
        ], dim=-1)

        attention_mask = ~joint_input_ids.eq(self.pad)

        outputs = self.bert(
            input_ids=joint_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls = self.cls(outputs.last_hidden_state[:, 0, :]).squeeze(-1)
        scores = nn.Sigmoid()(cls)

        loss = None
        if self.training:
            loss = nn.BCELoss()(scores, labels)
        else:
            scores = scores.reshape([bsz, num_cands])
            if labels is not None:
                loss = nn.CrossEntropyLoss()(scores, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=scores
        )


