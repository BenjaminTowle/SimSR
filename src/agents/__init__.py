from typing import Optional

from src.agents.biencoder import BiEncoderRetriever, BiEncoderModelBasedRetriever
from src.agents.tfidf import TFIDF
from src.agents.crossencoder import CrossEncoderRetriever

def get_agent(agent_name: str, k: Optional[int] = 1):
    str2agent = {
        "biencoder": BiEncoderRetriever,
        "biencoder-model-based": BiEncoderModelBasedRetriever,
        "tfidf": TFIDF,
        "crossencoder": CrossEncoderRetriever
    }

    if agent_name in str2agent:
        # TODO pass args to model
        return str2agent[agent_name]()