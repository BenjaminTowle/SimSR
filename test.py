import torch
import numpy as np
import copy

from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from typing import Optional, List
from tqdm import tqdm
from statistics import mean
from scipy.stats import pearsonr, spearmanr

from src.agents.biencoder import (
    BiEncoderFAISSRetriever, 
    BiEncoderModelBasedRetriever,
    StatefulModelBasedRetrieverForInference,
    StatefulModelBasedRetrieverForTraining
)
    
from src.agents.crossencoder import CrossEncoderRetriever
from src.agents.diversification import MMR, LexicalClustering
from src.args import parse_args
from src.datasets import get_dataset
from src.utils import set_random_seed, load_tokenizer, compute_f1, get_stopwords



class RetrievalPipeline:
    """Stores a sequence of retrieval models"""
    def __init__(self) -> None:
        self.retrievers = []
    
    def add_retriever(self, retriever, k: int):
        self.retrievers.append((retriever, k))

    def act(
        self, 
        query: str,
        **kwargs
    ) -> List[str]:
        
        docs = None
        outputs = None
        for retriever, k in self.retrievers:
            outputs = retriever.act(
                query=query, 
                k=k,
                docs=docs
            )
            docs = outputs.docs

        return outputs



def _process_batch(batch):
    batch["input_ids"] = torch.stack(batch["input_ids"], dim=0).transpose(0, 1).cpu().numpy().tolist()
    return batch

def _get_dataset(args, tokenizer):
    if args.dataset_load_path == "none":
        dataset_dict = get_dataset(args, tokenizer)
        if args.dataset_save_path != "none":
            dataset_dict.save_to_disk(args.dataset_save_path)
        return dataset_dict
    
    dataset_dict = DatasetDict.load_from_disk(args.dataset_load_path)
    return dataset_dict


def main():

    K = 3

    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = load_tokenizer(args)

    dataset_dict = _get_dataset(args, tokenizer)

    r_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["valid"]

    world_dataset = DatasetDict.load_from_disk("../data/personachat-dataset-full")["train"]

    #agent = MMR(
        #r_dataset, model_path="../data/pc-distilbert-biencoder/checkpoint-24645", model_type="distilbert"
    #)

    MODEL_PATH = "../data/pc-distilbert-biencoder/checkpoint-24645"
    MODEL_TYPE = "distilbert"
    

    #biencoder = BiEncoderFAISSRetriever(
    #    r_dataset, model_path=MODEL_PATH, model_type=MODEL_TYPE)

    
    agent = StatefulModelBasedRetrieverForInference(
        dataset=world_dataset, 
        policy_model_path="../data/pc-distilbert-biencoder/checkpoint-24645",
        #world_model_path="../data/pc-distilbert-biencoder/checkpoint-24645",
        #world_dataset=world_dataset,
        model_type="distilbert",
        use_set_retrieval=False,
        n=3,
        state_load_path="agent_state.pkl"
    )

    #crossencoder = CrossEncoderRetriever(
    #    model_path="../data/pc-distilbert-crossencoder/checkpoint-24645"
    #)

    #agent = RetrievalPipeline()
    #agent.add_retriever(biencoder, K)
    #agent.add_retriever(crossencoder, 1)



    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    preds = []
    targets = []

    for batch in tqdm(dataloader):
        batch = _process_batch(batch)
        query = tokenizer["bert"].batch_decode(batch["input_ids"], skip_special_tokens=False)[0]
        outputs = agent.act(query, k=K)

        preds += [outputs.docs]
        targets += batch["responses"]

    with open(f"preds_mb_n10_{str(K)}.txt", "w") as f:
        for t, P in zip(targets, preds):
            f.write(t + "\t" + "|".join(P) + "\n")

    agent.save_state("agent_state.pkl")





if __name__ == "__main__":
    main()
