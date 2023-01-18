import abc
import copy
import random
import numpy as np
import torch

from datasets import Dataset
from itertools import combinations
from scipy.special import softmax
from typing import List, Optional

from src.agents.baseagent import BaseAgent, BaseAgentOutput
from src.agents.biencoder import BiEncoderFAISSRetriever
from src.utils import compute_f1_matrix


def _get_clustering(
    clustering: str,
    k=None,
    s=None
):

    if clustering == "ablative":
        return AblativeClustering(k=k)

    elif clustering == "greedy":
        return GreedyClustering(k=k)
    
    elif clustering == "exhaustive":
        return ExhaustiveClustering(k=k)
    
    elif clustering == "samplerank":
        return SampleRankClustering(k=k, s=s)
        
    raise ValueError(clustering)
    


class Clustering(abc.ABC):
    """
    Abstract class for search strategy used.
    """

    def __init__(self, k=3) -> None:
        super().__init__()
        self.k = k

    @abc.abstractmethod
    def run(self, scores) -> List[int]:
        pass


class GreedyClustering(Clustering):

    def run(self, scores):
        idxs = list(range(scores.shape[0]))
        idx = np.argmax(np.sum(scores, axis=-1))
        idxs.remove(idx)
        chosen_idxs = [idx]

        for _ in range(2):
            chosen_idxs, idxs = self._get_idxs(scores, idxs, chosen_idxs)

        return chosen_idxs

    @staticmethod
    def _get_idxs(scores, idxs, chosen_idxs):
        best_score = 0.0
        best_idx = None

        for idx in idxs:
            tmp_idxs = chosen_idxs + [idx]
            S = scores[list(tmp_idxs)]
            e_score = np.mean(np.max(S, axis=0), axis=-1).item()
            best_score, best_idx = max([(best_score, best_idx), (e_score, idx)], key=lambda x: x[0])
        chosen_idxs.append(best_idx)
        idxs.remove(best_idx)

        return chosen_idxs, idxs


class AblativeClustering(Clustering):

    def run(self, scores):
        idxs = list(range(scores.shape[0]))        

        while len(idxs) > self.k:
            best_score = -1.0
            best_idx = None
            for idx in idxs:
                tmp_idxs = copy.copy(idxs)
                tmp_idxs.remove(idx)

                S = scores[tmp_idxs]
                e_score = np.sum(np.max(S, axis=0), axis=-1).item()

                best_score, best_idx = max([(best_score, best_idx), (e_score, idx)], key=lambda x: x[0])

            idxs.remove(best_idx)

        return idxs


class ExhaustiveClustering(Clustering):

    def run(self, scores):
        idxs = list(range(scores.shape[0]))

        best_score = 0.0
        best_idxs = None
        for tmp_idxs in combinations(idxs, self.k):
            S = scores[list(tmp_idxs)]
            e_score = np.mean(np.max(S, axis=0), axis=-1).item()
            best_score, best_idxs = max([(best_score, best_idxs), (e_score, tmp_idxs)], key=lambda x: x[0])
        idxs = best_idxs

        return idxs


class SampleRankClustering(Clustering):

    def __init__(self, k=3, s=300) -> None:
        super().__init__(k)
        self.s = s


    def run(self, scores):
        idxs = list(range(scores.shape[0]))
        best_score = 0
        best_idxs = None

        for _ in range(self.s):
            s_idxs = random.sample(idxs, self.k)
            sample = scores[s_idxs]
            e_score = np.mean(np.max(sample, axis=0), axis=-1).item()

            if e_score > best_score:
                best_score = e_score
                best_idxs = s_idxs
        idxs = best_idxs

        return idxs

        
class BiEncoderModelBasedRetriever(BaseAgent):

    """
    A model-based simulation approach that searches for sets of candidates by evaluating their expected max sim with the ground truth.
    """

    def __init__(
        self, 
        dataset: Dataset,
        policy_model_path,
        model_type: str,
        world_model_path: str = None,
        world_dataset = None,
        device="cuda",
        n = 30,
        s = 30,
        clustering: Optional[str] = None
    ) -> None:
        super().__init__()
        policy_model = BiEncoderFAISSRetriever
        self.policy_model = policy_model(
            dataset=dataset, device=device, model_path=policy_model_path, model_type=model_type
        )

        world_dataset = dataset if world_dataset is None else world_dataset

        self.n = n
        self.s = s
        self.clustering = clustering

        if world_model_path is None:
            self.world_model = self.policy_model
        else:
            self.world_model = BiEncoderFAISSRetriever(dataset=world_dataset, device=device, model_path=world_model_path, model_type=model_type)
        

    @torch.no_grad()
    def act(self, query: str, k: int, docs: Optional[List[str]] = None, return_probs = False):

        if docs is None:
            outputs = self.policy_model.act(query, docs=docs, k=max(self.n, self.s))
            docs = outputs.docs

        world_docs = docs[-self.s:]
        policy_docs = docs[-self.n:]
        
        scores = compute_f1_matrix(policy_docs, world_docs)
        clustering = _get_clustering(clustering=self.clustering, k=k, s=self.s)
        tau = 10.0
        probs = softmax(outputs.doc_scores / tau)
        scores = scores * np.expand_dims(probs, axis=0)
        idxs = clustering.run(scores)
        score = np.mean(np.max(scores[list(idxs)], axis=0), axis=-1).item()

        best_answer = [policy_docs[idx] for idx in idxs]

        return BaseAgentOutput(
            docs=best_answer,
            score=score
        )
