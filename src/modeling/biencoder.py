import abc
import torch

from dataclasses import dataclass
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertTokenizer
from typing import Optional

from src.modeling.basemodel import BaseModel, DistilBertBaseModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@dataclass
class BiencoderOutput(SequenceClassifierOutput):
    x_embeds: Optional[torch.Tensor] = None
    y_embeds: Optional[torch.Tensor] = None

@dataclass
class CVAEOutput:
    x_embeds: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

def get_length(input_ids, pad=0):
    mask = input_ids.eq(pad)
    length = (~mask).sum(1, keepdim=True)

    return length, mask

def truncate(input_ids):
    assert len(input_ids.shape) == 2, "input_ids must be a 2d tensor"
    max_length = get_length(input_ids)[0].max().item()

    return input_ids[:, :max_length]


class VAEHead(nn.Module):

    def __init__(self, z, model_dim) -> None:
        super().__init__()

        self.q_y = nn.Sequential(
            nn.Linear(model_dim * 2, z),
            nn.Tanh()
        )

        self.log_var = nn.Linear(z, z)
        self.mu = nn.Linear(z, z)

        self.proj = nn.Sequential(
            nn.Linear(model_dim + z, z),
            nn.Tanh(),
            nn.Linear(z, model_dim)
        )

    @staticmethod
    def _get_ground_truth_embeds(
        y_embeds,
        labels=None
    ):
        if y_embeds.ndim == 2:
            return y_embeds

        assert labels is not None, "'labels' must be provided when ground truth 'y_embeds' is ambiguous."
        
        return y_embeds.gather(1, labels.unsqueeze(
                -1).unsqueeze(-1).expand(-1, -1, y_embeds.size(-1))
        ).squeeze(1)

    def forward(
        self,
        x_embeds,
        y_embeds,
        labels=None
    ):

        # Compute posterior
        y_embeds = self._get_ground_truth_embeds(y_embeds, labels)
        q_y = self.q_y(torch.cat([x_embeds, y_embeds], dim=-1))
        mu = self.mu(q_y)
        logvar = self.log_var(q_y)
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)
        z = eps * std + mu

        y_hat = self.proj(torch.cat([x_embeds, z], dim=-1))
        loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        return CVAEOutput(
            x_embeds=y_hat,
            logvar=logvar,
            mu=mu,
            loss=loss
        )


class BiencoderMixin:
    """
    A Mixin that performs biencoder-like functions for an arbitrary embedding model
    """

    @abc.abstractmethod
    def get_context_embedding(self, input_ids):
        """
        Child-model implements this function
        """
        pass

    @abc.abstractmethod
    def get_response_embedding(self, input_ids):
        """
        Child-model implements this function
        """
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

        x_embeds = self.get_context_embedding(input_ids)

        if candidate_embeds is not None:
            y_embeds = candidate_embeds
        else:
            if candidate_input_ids is not None:
                y_input_ids = candidate_input_ids
            old_shape = y_input_ids.size() # Either bsz x len or bsz x k x len
            y_embeds = self.get_response_embedding(y_input_ids.reshape([-1, old_shape[-1]]))
            y_embeds = y_embeds.reshape(list(old_shape[:-1]) + [-1])

        return BiencoderOutput(
            x_embeds=x_embeds,
            y_embeds=y_embeds
        )

    @staticmethod
    def asymmetric_loss(X, Y):
        """
        Learn one-directional p(y|x)
        """
        s_x_y = torch.sum(X * Y, dim=1)
        s_x = torch.matmul(X, Y.T)
        lse_s_x = torch.logsumexp(s_x, 1, keepdim=False)
        loss = lse_s_x.add(-s_x_y)
        return loss.mean()

    @staticmethod
    def symmetric_loss(X, Y):
        """
        Learns bi-directional p(y|x) and p(x|y)
        """
        s_x_y = torch.sum(X * Y, dim=1)
        s_x = torch.matmul(X, Y.T)
        s_y = torch.matmul(Y, X.T)
        s_x_s_y = torch.cat((s_x, s_y), dim=1)
        lse_s_x_s_y = torch.logsumexp(s_x_s_y, 1, keepdim=False)
        loss = lse_s_x_s_y.add(-s_x_y)
        return loss.mean()

    def loss_fn(
        self,
        x_embeds,
        y_embeds,
        labels=None
    ):

        scores = None
        if y_embeds.ndim == 2:
            loss_fn = self.symmetric_loss if self.use_symmetric_loss else self.asymmetric_loss
            loss = loss_fn(x_embeds, y_embeds)

        else:
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(scores, labels)
        
        return SequenceClassifierOutput(
            logits=scores,
            loss=loss
        )


class BertBiencoder(BiencoderMixin, BaseModel):
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
        embeds = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return embeds

    def get_context_embedding(self, input_ids):
        return self._get_embedding(input_ids, mode="context")

    def get_response_embedding(self, input_ids):
        return self._get_embedding(input_ids, mode="response")

    def forward(
            self,
            input_ids: torch.Tensor,
            y_input_ids: torch.Tensor = None,
            candidate_input_ids: torch.Tensor = None,
            candidate_embeds: torch.Tensor = None,
            labels: torch.Tensor = None,
    ):

        outputs = self.forward_biencoder(
            input_ids=input_ids,
            y_input_ids=y_input_ids,
            candidate_input_ids=candidate_input_ids,
            candidate_embeds=candidate_embeds,
            labels=labels
        )

        outputs = self.loss_fn(
            x_embeds=outputs.x_embeds,
            y_embeds=outputs.y_embeds,
            labels=labels
        )

        return outputs


class DistilBertBiencoder(BiencoderMixin, DistilBertBaseModel):
    def __init__(self, config, use_symmetric_loss=False):
        print("start: matching super")
        self.use_symmetric_loss = use_symmetric_loss
        super().__init__(config)
        print("end: matching super")

    def forward_embedding(self, input_ids):
        device = input_ids.device
        device = input_ids.device
        embeds = self._get_embedding(input_ids)
        return embeds

    def _get_embedding(self, input_ids, mode="context"):
        input_ids = truncate(input_ids)
        attn_mask = torch.ones(input_ids.size()).long().to(input_ids.device).masked_fill(input_ids == 0, 0)
        embeds = self.distilbert(input_ids, attention_mask=attn_mask).last_hidden_state[:, 0, :]

        return embeds

    def get_context_embedding(self, input_ids):
        return self._get_embedding(input_ids, mode="context")

    def get_response_embedding(self, input_ids):
        return self._get_embedding(input_ids, mode="response")

    def forward(
            self,
            input_ids: torch.Tensor,
            y_input_ids: torch.Tensor = None,
            candidate_input_ids: torch.Tensor = None,
            candidate_embeds: torch.Tensor = None,
            labels: torch.Tensor = None,
    ):

        outputs = self.forward_biencoder(
            input_ids=input_ids,
            y_input_ids=y_input_ids,
            candidate_input_ids=candidate_input_ids,
            candidate_embeds=candidate_embeds,
            labels=labels
        )

        outputs = self.loss_fn(
            x_embeds=outputs.x_embeds,
            y_embeds=outputs.y_embeds,
            labels=labels
        )

        return outputs
        

class DistilBertCVAE(DistilBertBiencoder):
    def __init__(
        self, 
        config, 
        use_symmetric_loss=False, 
        z: int = 512,
        kld_weight: float = 1.0,
        use_kld_annealling: bool = False,
        kld_annealling_steps: int = -1,
        use_message_prior: bool = False
    ):
    
        print("start: mcvae super")
        super().__init__(config, use_symmetric_loss=use_symmetric_loss)
        print("end: mcvae super")

        self.z = z
        self.kld_weight = kld_weight
        self.kld_cur_weight = 0.0 if use_kld_annealling else kld_weight
        self.use_kld_annealling = use_kld_annealling
        self.kld_annealling_steps = kld_annealling_steps
        self.use_message_prior = use_message_prior

        self.cvae = VAEHead(z, config.dim)


    def generate_embedding(self, embeds, num_samples=1):
        embeds = embeds.expand(num_samples, -1)
        z = torch.randn_like(embeds)[:, :self.z]
        embeds = self.cvae.proj(torch.cat([embeds, z], dim=-1))
        return embeds

    def forward(
            self,
            input_ids: torch.Tensor,
            y_input_ids: torch.Tensor = None,
            candidate_input_ids: torch.Tensor = None,
            candidate_embeds: torch.Tensor = None,
            labels: torch.Tensor = None,
    ):

        with torch.no_grad():
            m_outputs = self.forward_biencoder(
                input_ids=input_ids,
                y_input_ids=y_input_ids,
                candidate_input_ids=candidate_input_ids,
                candidate_embeds=candidate_embeds,
                labels=labels
            )

        cvae_outputs = self.cvae(
            x_embeds=m_outputs.x_embeds,
            y_embeds=m_outputs.y_embeds,
            labels=labels
        )

        outputs = self.loss_fn(
            x_embeds=cvae_outputs.x_embeds,
            y_embeds=m_outputs.y_embeds,
            labels=labels
        )
        if self.use_kld_annealling:
            self.kld_cur_weight = min(
                self.kld_cur_weight + 1/self.kld_annealling_steps, self.kld_weight)
        outputs.loss += self.kld_cur_weight * cvae_outputs.loss

        return outputs
