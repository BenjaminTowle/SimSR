import torch

from transformers import (
    BertPreTrainedModel,
    BertModel,
    DistilBertPreTrainedModel,
    DistilBertModel
)
from abc import ABC, abstractmethod


class BaseModelMixin:

    def get_length(self, input_ids):
        mask = input_ids.eq(self.pad)
        length = (~mask).sum(1, keepdim=True)

        return length, mask

    def truncate(self, input_ids):
        assert len(input_ids.shape) == 2, "input_ids must be a 2d tensor"
        max_length = self.get_length(input_ids)[0].max().item()

        return input_ids[:, :max_length]


class BaseModel(BertPreTrainedModel, BaseModelMixin, ABC):

    def __init__(self, config, pad: int = 0):
        super().__init__(config)
        self.bert = BertModel(config)
        self.pad = pad

    @abstractmethod
    def forward_embedding(self, input_ids: torch.Tensor):
        pass


class DistilBertBaseModel(BaseModelMixin, DistilBertPreTrainedModel):
    def __init__(self, config, pad: int = 0):
        print("start: base super")
        super().__init__(config)
        print("end: base super")
        self.distilbert = DistilBertModel(config)
        self.pad = pad

    @abstractmethod
    def forward_embedding(self, input_ids: torch.Tensor):
        pass
