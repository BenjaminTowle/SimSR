import abc
import numpy as np

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BaseAgentOutput:
    docs: List[str]
    score: Optional[float] = None
    doc_embeds: Optional[np.array] = None
    doc_scores: Optional[np.array] = None
    query_embed: Optional[np.array] = None
    contexts: List[str] = None
    targets: List[str] = None


class BaseAgent(abc.ABC):

    """
    Abstract class for instantiating agent.
    An agent is a class which receives a query and set of docs and outputs a set of responses
    """
    @abc.abstractmethod
    def act(
        self,
        query: str,
        k: int,
        docs: Optional[List[str]] = None,
    ) -> BaseAgentOutput:

        """
        query: text string to be queried
        docs: candidate pool to be retrieved from; if not specified will used pre-cached pool.
        k: number of docs to retrieve
        """
        pass
