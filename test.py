import torch

from torch.utils.data import DataLoader
from datasets import DatasetDict
from tqdm import tqdm

from src.agents import get_agent
from src.args import parse_args
from src.datasets import get_dataset
from src.utils import set_random_seed, load_tokenizer, compute_f1


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

    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = load_tokenizer(args)

    dataset_dict = _get_dataset(args, tokenizer)

    if args.use_valid:
        eval_dataset = dataset_dict["valid"]
    else:
        eval_dataset = dataset_dict["test"]

    agent = get_agent(args)


    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    preds = []
    pred_scores = []
    actual_scores = []
    targets = []

    for batch in tqdm(dataloader):
        batch = _process_batch(batch)
        query = tokenizer["bert"].batch_decode(batch["input_ids"], skip_special_tokens=False)[0]
        outputs = agent.act(query, k=args.k)

        preds += [outputs.docs]
        pred_scores += [outputs.score]
        actual_scores += [compute_f1(batch["responses"][0], outputs.docs)]
        targets += batch["responses"]


    if args.prediction_save_path == "none":
        exit()

    with open(args.prediction_save_path, "w", encoding="utf-8") as f:
        for t, P in zip(targets, preds):
            P = [p.replace("\n", "").replace("|", "") for p in P]
            f.write(t.replace("\n", "") + "\t" + "|".join(P) + "\n")


if __name__ == "__main__":
    main()
