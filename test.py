import torch
import numpy as np
import copy

from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from typing import Optional, List
from tqdm import tqdm
from statistics import mean
from scipy.stats import pearsonr, spearmanr

from src.agents.biencoder import BiEncoderFAISSRetriever
from src.args import parse_args
from src.utils import set_random_seed, load_tokenizer, compute_f1, get_stopwords



class RetrievalPipeline:
    """Stores a sequence of retrieval models"""
    def __init__(self) -> None:
        self.retrievers = []
    
    def add_retriever(self, retriever, k: int):
        self.retrievers.append((retriever, k))

    def retrieve(
        self, 
        query: str
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

class PRFRetrievalPipeline(RetrievalPipeline):

    def __init__(self) -> None:
        super().__init__()
        self.idx2count = {}

    def _retrieve(self, queries: List[str]):
        docs = None
        for i, (retriever, k) in enumerate(self.retrievers):
            docs = retriever.retrieve(
                query=queries[i],
                k=k,
                docs=docs
            )
        return docs
    
    def retrieve(self, query: str, num_iterations: int = 5) -> List[str]:
        
        queries = [query, query]
        for i in range(num_iterations):
            new_query = self._retrieve(queries)[0]
            if new_query == queries[0]:
                if i not in self.idx2count:
                    self.idx2count[i] = 1
                else:
                    self.idx2count[i] += 1
     
                break
            queries = [new_query, query]
        
        return [new_query]
        

def _eval_fn(batch, stopwords):
    f1 = [max([compute_f1(p, [r], stopwords=None) for p in P]) for P, r in zip(batch["pred"], batch["response"])]
    rare_f1 = [max([compute_f1(p, [r], stopwords=stopwords) for p in P]) for P, r in zip(batch["pred"], batch["response"])]
    persona_f1 = [max([max([compute_f1(p, [k], stopwords=None) for k in K]) for p in P]) for P, K in zip(batch["pred"], batch["persona"])]

    return {
        "f1": f1,
        "rare_f1": rare_f1,
        "persona_f1": persona_f1
    }

def _process_batch(batch):
    batch["input_ids"] = torch.stack(batch["input_ids"], dim=0).transpose(0, 1).cpu().numpy().tolist()
    return batch


def main():

    K = 10

    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = load_tokenizer(args)["bert"]

    dataset = DatasetDict.load_from_disk("../personachat-dataset")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    stopwords = get_stopwords(train_dataset["response"])

    r_dataset = Dataset.load_from_disk("../candidate_pool")

    agent = BiEncoderFAISSRetriever(
        r_dataset, model_path="../biencoder/checkpoint-49290")

    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    preds = []
    all_scores = {}

    for batch in tqdm(dataloader):
        batch = _process_batch(batch)
        query = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)[0]
        outputs = agent.act(query, k=K)

        batch["pred"] = [outputs.docs]
        preds += batch["pred"]

        scores = _eval_fn(batch, stopwords)

        for key, value in scores.items():
            if key not in all_scores:
                all_scores[key] = value
            else:
                all_scores[key] = all_scores[key] + value

    for key, value in all_scores.items():
        print(key, ": ", mean(value))





if __name__ == "__main__":
    main()
