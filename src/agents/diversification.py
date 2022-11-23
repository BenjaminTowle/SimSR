import numpy as np
import torch
import editdistance

from datasets import Dataset
from typing import Optional, List

from src.agents.baseagent import BaseAgent, BaseAgentOutput
from src.agents.biencoder import BiEncoderFAISSRetriever


class MMR(BaseAgent):

    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290",
        model_type="distilbert"
    ) -> None:

        self.agent = BiEncoderFAISSRetriever(
            dataset=dataset,
            device=device,
            model_path=model_path,
            model_type=model_type
        )

    @torch.no_grad()
    def act(
        self,
        query: str,
        k: int,
        docs: Optional[List[str]] = None
    ):

        outputs = self.agent.act(
            query=query,
            k=k*4,
            docs=docs
        )

        docs = outputs.docs
        doc_embeds = np.array(outputs.doc_embeds)
        inter_doc_scores = np.matmul(doc_embeds, doc_embeds.T)
        doc_scores = outputs.doc_scores
        l = 0.5

        chosen_idxs = []
        remaining_idxs = list(range(len(docs)))
        for k_ in range(k):
            if k_ == 0:
                idx = np.argmax(doc_scores)
                chosen_idxs.append(idx)
                remaining_idxs.remove(idx)
                continue

            best_idx, best_mmr = None, -99.
            for idx in remaining_idxs:
                mmr = l * doc_scores[idx] - (1 - l) * np.max(inter_doc_scores[idx])
                best_idx, best_mmr = max([(best_idx, best_mmr), (idx, mmr)], key=lambda x: x[1])

            chosen_idxs.append(best_idx)
            remaining_idxs.remove(best_idx)

        docs = [docs[idx] for idx in chosen_idxs]

        return BaseAgentOutput(
            docs=docs
        )


class LexicalClustering(BaseAgent):
    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290"
    ) -> None:

        self.agent = BiEncoderFAISSRetriever(
            dataset=dataset,
            device=device,
            model_path=model_path
        )

    @torch.no_grad()
    def act(
        self,
        query: str,
        k: int,
        docs: Optional[List[str]] = None
    ):

        outputs = self.agent.act(
            query=query,
            k=k*4,
            docs=docs
        )


        docs = [outputs.docs[idx] for idx in reversed(np.argsort(outputs.doc_scores).tolist())]

        clusters = {}
        new_docs = []
        # Cluster all words with edit distance of 1
        for doc in docs:
            if new_docs == []:
                new_docs.append(doc)
                continue
        
            ed = max([editdistance.eval(doc.split(), d.split()) for d in new_docs])
            if ed <= 1:
                continue

            new_docs.append(doc)

            if len(new_docs) == k:
                break

        return BaseAgentOutput(docs=new_docs)


