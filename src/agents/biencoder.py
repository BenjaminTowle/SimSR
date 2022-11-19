import copy
import numpy as np
import torch

from datasets import Dataset
from transformers import BertTokenizer
from typing import Optional, List

from src.agents.baseagent import BaseAgent, BaseAgentOutput
from src.modeling.biencoder import Biencoder
from src.utils import compute_f1_matrix

class BiEncoderFAISSRetriever(BaseAgent):

    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290"
    ) -> None:
        
        self.model = Biencoder.from_pretrained(model_path).to(device)
        self.index = self.build_index(dataset, device=device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = device

    def _build_index(self, samples, device: str = "cuda"):
        """Map fn"""
        samples["embeddings"] = self.model.forward_embedding(torch.tensor(samples["y_input_ids"], device=device)).cpu().numpy()
        return samples
        
    @torch.no_grad()
    def build_index(
        self, 
        dataset: Dataset,
        device: str = "cuda"
    ):

        dataset = dataset.map(lambda x: self._build_index(x, device=device), batched=True, batch_size=8)
        dataset.add_faiss_index(column="embeddings")

        return dataset

    @torch.no_grad()
    def act(
            self,
            query: str,
            k: int,
            docs: Optional[List[str]] = None,
    ):

        query_tokens = self.tokenizer([query], return_tensors="pt")

        query_embed = self.model.forward_embedding(
            query_tokens.input_ids.to(self.device))[0].cpu().numpy()

        _, retrieved_examples = self.index.get_nearest_examples("embeddings", query_embed, k=k)
        docs = retrieved_examples["response"]
       
        return BaseAgentOutput(
            docs=docs,
        )


class BiEncoderRetriever(BaseAgent):

    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290"
    ) -> None:
        
        self.model = Biencoder.from_pretrained(model_path).to(device)
        self.index = self.build_index(dataset, device=device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = device

    def _build_index(self, samples, device: str = "cuda"):
        """Map fn"""
        samples["embeddings"] = self.model.forward_embedding(torch.tensor(samples["y_input_ids"], device=device)).cpu().numpy()
        return samples
        
    @torch.no_grad()
    def build_index(
        self, 
        dataset: Dataset,
        device: str = "cuda"
    ):

        dataset = dataset.map(lambda x: self._build_index(x, device=device), batched=True, batch_size=8)
        candidate_embeds = torch.tensor([dataset["embeddings"]], device=device)
        candidate_input_ids = torch.tensor([dataset["y_input_ids"]], device=device)
        responses = dataset["response"]

        return {
            "candidate_embeds": candidate_embeds, 
            "candidate_input_ids": candidate_input_ids, 
            "responses": responses
        }

    @torch.no_grad()
    def act(
            self,
            query: str,
            k: int,
            docs: Optional[List[str]] = None,
            return_probs = False,
            query_encoder = None
    ):

        query_tokens = self.tokenizer([query], return_tensors="pt")

        bsz = 1
        candidate_embeds = None
        if docs is None:
            candidate_embeds = self.index["candidate_embeds"].expand(bsz, -1, -1)
            candidate_input_ids = self.index["candidate_input_ids"].expand(bsz, -1, -1)
        else:
            candidate_input_ids = self.tokenizer(docs, return_tensors="pt", padding=True, truncation=True)

        if query_encoder is None:
            outputs = self.model(
                input_ids=query_tokens.input_ids.to(self.device),
                candidate_input_ids=candidate_input_ids,
                candidate_embeds=candidate_embeds
            )
            scores = outputs.logits  # m x n
        else:
            ctx_embed = query_encoder.forward_embedding(query_tokens.input_ids.to(self.device))
            ctx_embed = ctx_embed.expand(candidate_input_ids.size(1), -1)
            outputs = query_encoder(
                ctx_embed=ctx_embed,
                doc_embeds=candidate_embeds[0]
            )
            scores = outputs.logits.unsqueeze(0)

        top_k = torch.topk(scores, k=k, dim=-1).indices
        #F.softmax(outputs.logits, dim=-1)

        batch_responses = []
        batch_scores = []
        doc_embeds = []
        for i in range(top_k.size(0)):
            batch_responses.append([self.index["responses"][j] for j in top_k[i]])
            batch_scores.append(np.array([scores[i, j].cpu().numpy().item() for j in top_k[i]]))
            doc_embeds.append(np.array([self.index["candidate_embeds"][0][j].cpu().numpy() for j in top_k[i]]))
        
        return BaseAgentOutput(
            docs=batch_responses[0],
            doc_scores=batch_scores[0],
            doc_embeds=doc_embeds[0]
        )


class BiEncoderModelBasedRetriever(BaseAgent):

    """
    A model-based simulation approach that searches for sets of candidates by evaluating their expected max sim with the ground truth.
    """

    def __init__(
        self, 
        dataset: Dataset,
        device="cuda"
    ) -> None:
        super().__init__(dataset, device)

        self.world_model = BiEncoderRetriever(dataset, device)
        self.policy_model = BiEncoderRetriever(dataset, device, model_path="biencoder/checkpoint-49290")

    
    @torch.no_grad()
    def act(self, query: str, k: int, docs: Optional[List[str]] = None, return_probs = False):

        outputs = self.policy_model.act(query, docs=docs, k=k*4)
        docs = outputs.docs

        world_docs = self.world_model.act(query, k=k*4).docs

        scores = compute_f1_matrix(docs, world_docs)
        idxs = list(range(len(docs)))

        while len(idxs) > k:
            best_score = -1.0
            best_idx = None
            for idx in idxs:
                tmp_idxs = copy.copy(idxs)
                tmp_idxs.remove(idx)

                S = scores[tmp_idxs]
                e_score = np.mean(np.max(S, axis=0), axis=-1).item()

                best_score, best_idx = max([(best_score, best_idx), (e_score, idx)], key=lambda x: x[0])

            idxs.remove(best_idx)

        S = scores[idxs]

        rewards = [0 if i not in idxs else 1 for i in range(len(docs))]
        self.value_model.update(docs, rewards)
        best_answer = [docs[idx] for idx in idxs]

        return BaseAgentOutput(
            docs=best_answer
        )


class BiEncoderSetRetriever(BaseAgent):

    def __init__(self, dataset: Dataset, device="cuda") -> None:
        super().__init__(dataset, device)

        self.policy_model = BiEncoderRetriever(dataset, device)

    def _process_docs(docs):
        docs = [doc.replace("[CLS]", "").replace(" [PAD]", "") for doc in docs]
        docs = [doc.split("[SEP]") for doc in docs]
        new_docs = []
        for doc in docs:
            for d in doc:
                if len(d) > 0:
                    new_docs.append(d)

        return new_docs

    @torch.no_grad()
    def act(self, query: str, k: int, docs: Optional[List[str]] = None, return_probs=False):

        if docs is not None:
            raise Warning("docs is not None, but model expects docs to be tokenized as sets!")

        docs = self.policy_model(query, docs=docs, k=k)

        def _process_doc(docs):
            docs = [doc.replace("[CLS]", "").replace(" [PAD]", "") for doc in docs]
            docs = [doc.split("[SEP]") for doc in docs]
            new_docs = []
            for doc in docs:
                for d in doc:
                    if len(d) > 0:
                        new_docs.append(d)

            return new_docs

        if docs is None:
            outputs = self.policy_model.retrieve(query, k=4)
            docs = outputs.docs

        # Convert to k * 4 docs
        docs = _process_doc(docs)

        return BaseAgentOutput(
            docs=docs,
        )