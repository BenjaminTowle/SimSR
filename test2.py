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
from src.agents.diversification import MMR, LexicalClustering
from src.args import parse_args
from src.datasets import get_dataset
from src.utils import set_random_seed, load_tokenizer, compute_f1, get_stopwords
        
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

def _map_fn(
    batch, 
    agent, 
    tokenizer, 
    k,
    max_response_length,

):
    query = tokenizer["bert"].decode(batch["input_ids"], skip_special_tokens=False)
    docs = agent.act(query, k=k).docs
    batch["y_input_ids"] = tokenizer["bert"](
        docs, max_length=max_response_length, padding="max_length", truncation=True, return_tensors="np"
    ).input_ids.reshape(-1).tolist()

    return batch

    

def main():

    K = 3

    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = load_tokenizer(args)

    dataset_dict = _get_dataset(args, tokenizer)

    r_dataset = dataset_dict["train"]

    agent = BiEncoderFAISSRetriever(
        r_dataset, model_path="../data/pc-distilbert-biencoder/checkpoint-24645", model_type="distilbert")

    dataset_dict = dataset_dict.map(lambda x: _map_fn(x, agent, tokenizer, K, max_response_length=args.max_response_length))

    dataset_dict.save_to_disk(f"../data/personachat-set-dataset-{K}")



if __name__ == "__main__":
    main()
