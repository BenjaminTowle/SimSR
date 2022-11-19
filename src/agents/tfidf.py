import numpy as np
import random

from datasets import Dataset
from sklearn.metrics.pairwise import linear_kernel
from typing import Optional, List

from src.agents.baseagent import BaseAgent, BaseAgentOutput
from src.utils import TfidfVectorizer

class TFIDF(BaseAgent):

    def __init__(
        self,
        dataset: Dataset
    ) -> None:
        texts = dataset["response"]
        self.tfidf = TfidfVectorizer().fit(texts)
        self.documents = self.tfidf.transform(texts)
        self.responses = np.array(texts)
        self.special_tokens = ["[CLS]", "[PAD]", "[SEP]"]

    def act(self, query: str, k: int, docs: Optional[List[str]] = None):

        for tok in self.special_tokens:
            query = query.replace(tok, "")

        query = self.tfidf.transform([query])
        docs = self.documents if docs is None else self.tfidf.transform(docs)
        scores = linear_kernel(query, docs)[0]
        top_idxs = np.argpartition(scores, -k)[-k:]

        responses = self.responses[top_idxs].tolist()

        return BaseAgentOutput(docs=responses)

class Random(BaseAgent):

    """
    A random agent that returns a random document.
    """

    def __init__(
        self,
        dataset: Dataset
    ) -> None:
        self.documents = dataset["response"]
        
    def act(self, query: str, k: int, docs: Optional[List[str]] = None):

        docs = self.documents if docs is None else docs
        docs = random.sample(docs, k=k)

        return BaseAgentOutput(docs=docs)
        
