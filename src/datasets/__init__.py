from src.datasets.personachat import get_dataset as get_dataset_personachat
from src.datasets import reddit

def get_dataset(args, tokenizer):

    dataset_fns = {
        "personachat": get_dataset_personachat,
        "reddit": reddit.get_dataset
    }

    if args.task in dataset_fns:
        dataset_fn = dataset_fns[args.task]
    else:
        raise ValueError(f"Task: {args.task} not recognised")

    return dataset_fn(args, tokenizer)