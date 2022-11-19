import torch
import numpy as np

from transformers import BertTokenizer
from typing import Optional, List

from src.agents.baseagent import BaseAgent, BaseAgentOutput
from src.modeling.crossencoder import CrossEncoder

class CrossEncoderRetriever(BaseAgent):

    def __init__(
        self, device="cuda", **kwargs
    ) -> None:

        self.model = CrossEncoder.from_pretrained("crossencoder/checkpoint-49290").to(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = device

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        k=1,
        docs: Optional[List[str]] = None
    ):

        assert docs is not None, "argument 'docs' must be specified for CrossEncoder"

        query_tokens = self.tokenizer([query], return_tensors="pt")
        docs_tokens = self.tokenizer(docs, padding=True, max_length=32, truncation=True, return_tensors="pt")

        scores = self.model(
            input_ids=query_tokens.input_ids.to(self.device),
            candidate_input_ids=docs_tokens.input_ids.to(self.device).unsqueeze(0)
        ).logits

        topk = torch.topk(scores, k=k, dim=-1)
        topk_idxs = topk.indices.cpu().numpy()
        topk_values = topk.values.cpu().numpy()
        chosen_docs = np.array(docs)[topk_idxs].tolist()[0]

        return BaseAgentOutput(
            docs=chosen_docs,
            doc_scores=scores.cpu().numpy()
        )