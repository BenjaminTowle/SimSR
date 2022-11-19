from typing import Optional, List

from src.datasets.personachat import get_dataset as get_dataset_personachat

def get_dataset(args, tokenizer):

    dataset_fns = {
        "personachat": get_dataset_personachat
    }

    if args.task in dataset_fns:
        dataset_fn = dataset_fns[args.task]
    else:
        raise ValueError("Task not recognised")

    return dataset_fn(args, tokenizer)