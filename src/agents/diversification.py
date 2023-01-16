import numpy as np
import torch

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, List
from scipy.stats import zscore

from src.agents.baseagent import BaseAgent, BaseAgentOutput
from src.agents.biencoder import BiEncoderFAISSRetriever


class MMR(BaseAgent):
    """
    Implements Maximum Marginal Relevance algorithm: 
    Carbonell Jaime and Jade Goldstein. 1998. The use of
    MMR, diversity-based reranking for reordering documents and producing summaries. In SIGIR, pages
    335-336.
    """

    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290",
        model_type="distilbert",
        n = 15
    ) -> None:

        self.agent = BiEncoderFAISSRetriever(
            dataset=dataset,
            device=device,
            model_path=model_path,
            model_type=model_type
        )

        self.n = n

    @torch.no_grad()
    def act(
        self,
        query: str,
        k: int,
        docs: Optional[List[str]] = None
    ):

        outputs = self.agent.act(
            query=query,
            k=self.n,
            docs=docs
        )

        docs = outputs.docs
        doc_embeds = np.array(outputs.doc_embeds)
        inter_doc_scores = np.matmul(doc_embeds, doc_embeds.T)
        inter_doc_scores = zscore(inter_doc_scores)
        doc_scores = outputs.doc_scores
        doc_scores = zscore(doc_scores)
        l = 0.5

        chosen_idxs = []
        remaining_idxs = list(range(len(docs)))
        for k_ in range(k):
            if k_ == 0:
                idx = np.argmax(doc_scores)
                chosen_idxs.append(idx)
                remaining_idxs.remove(idx)
                continue

            best_idx, best_mmr = None, -99999.
            for idx in remaining_idxs:
                mmr = (1 - l) * np.argsort(np.max(inter_doc_scores[idx])) - l * np.argsort(doc_scores[idx])
                best_idx, best_mmr = max([(best_idx, best_mmr), (idx, mmr)], key=lambda x: x[1])

            chosen_idxs.append(best_idx)
            remaining_idxs.remove(best_idx)

        chosen_docs = [docs[idx] for idx in chosen_idxs]

        return BaseAgentOutput(
            docs=chosen_docs
        )


class TopicClustering(BaseAgent):
    """
    Uses neural topic model to perform clustering on replies, ensuring no reply set contains two replies from same cluster.
    Topic model is taken from this paper: https://arxiv.org/abs/2209.09824.
    """
    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290",
        model_type="distilbert"
    ) -> None:

        self.device = device

        self.agent = BiEncoderFAISSRetriever(
            dataset=dataset,
            device=device,
            model_path=model_path,
            model_type=model_type
        )

        rpath = f"cardiffnlp/tweet-topic-21-multi"

        self.roberta_tokenizer = AutoTokenizer.from_pretrained(rpath)
        self.roberta = AutoModelForSequenceClassification.from_pretrained(rpath).to(device)

        self.doc2label = {}
    
    
    def _build_index(self, samples):
        """Map fn"""
        inputs = self.roberta_tokenizer(
            samples["responses"], max_length=32, padding="max_length", truncation=True, return_tensors="pt"
        )

        outputs = self.roberta(
            input_ids=inputs.input_ids.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device)
        )

        samples["labels"] = outputs.logits.argmax(-1).cpu().numpy().tolist()
        
        return samples

    @torch.no_grad()
    def act(
        self,
        query: str,
        k: int,
        docs: Optional[List[str]] = None
    ):

        outputs = self.agent.act(
            query=query,
            k=100,
            docs=docs
        )

        docs = [outputs.docs[idx] for idx in reversed(np.argsort(outputs.doc_scores).tolist())]
        
        labels = [None for _ in docs]
        idxs_to_query = []
        docs_to_query = []
        for i, doc in enumerate(docs):
            if doc in self.doc2label:
                labels[i] = self.doc2label[doc]
            else:
                idxs_to_query.append(i)
                docs_to_query.append(doc)

        inputs = self.roberta_tokenizer(
            docs, max_length=32, padding="max_length", truncation=True, return_tensors="pt"
        )

        outputs = self.roberta(
            input_ids=inputs.input_ids.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device)
        )

        labels_to_add = outputs.logits.argmax(-1).cpu().numpy().tolist()
        for idx, label, doc in zip(idxs_to_query, labels_to_add, docs_to_query):
            labels[idx] = label
            self.doc2label[doc] = label

        assert all([l is not None for l in labels])

        chosen_docs = []
        chosen_labels = []
        for doc, label in zip(docs, labels):
            if label not in chosen_labels:
                chosen_docs.append(doc)
                chosen_labels.append(label)
            if len(chosen_docs) == k:
                break

        for doc in chosen_docs:
            if len(chosen_docs) == k:
                break
            chosen_docs.append(doc)

        assert len(chosen_docs) == k

        docs = chosen_docs

        return BaseAgentOutput(docs=docs)
