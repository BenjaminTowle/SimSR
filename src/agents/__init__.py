from datasets import Dataset
from typing import Optional

from src.agents.biencoder import (
    BiEncoderFAISSRetriever,
    MCVAERetriever
)
from src.agents.diversification import (
    MMR,
    TopicClustering
)
from src.agents.modelbased import BiEncoderModelBasedRetriever


def _get_matching_model(args):

    response_set = Dataset.load_from_disk(args.response_set_path)

    return BiEncoderFAISSRetriever(
        response_set,
        args.device,
        model_path=args.model_load_path,
        model_type="distilbert"
    )

def _get_simulation_model(args):

    response_set = Dataset.load_from_disk(args.response_set_path)

    return BiEncoderModelBasedRetriever(
        response_set,
        policy_model_path=args.model_load_path,
        n=args.n,
        s=args.s,
        device=args.device,
        model_type="distilbert",
        clustering=args.clustering
    )

def _get_mcvae_model(args):

    response_set = Dataset.load_from_disk(args.response_set_path)

    return MCVAERetriever(
        response_set,
        device=args.device,
        model_path=args.model_load_path,
        model_type="distilbert",
        n=args.n,
        s=args.s,
        z=args.z,
        use_message_prior=args.use_message_prior
    )

def _get_mmr_model(args):

    response_set = Dataset.load_from_disk(args.response_set_path)

    return MMR(
        response_set,
        device=args.device,
        model_path=args.model_load_path,
        model_type="distilbert",
        n=args.n
    )

def _get_topic_model(args):

    response_set = Dataset.load_from_disk(args.response_set_path)

    return TopicClustering(
        response_set,
        device=args.device,
        model_path=args.model_load_path,
        model_type="distilbert"
    )





def get_agent(args):
    str2agent = {
        "matching": _get_matching_model,
        "simulation": _get_simulation_model,
        "mcvae": _get_mcvae_model,
        "mmr": _get_mmr_model,
        "topic": _get_topic_model
    }

    if args.agent_type in str2agent:
        return str2agent[args.agent_type](args)
    
    raise ValueError(args.agent_type)