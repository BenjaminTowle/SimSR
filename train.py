import os

from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl, BertForSequenceClassification
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import set_caching_enabled, DatasetDict, Dataset

from src.datasets import get_dataset
from src.args import parse_args
from src.utils import load_tokenizer, set_random_seed, accuracy_metric
from src.modeling import get_model

os.environ["WANDB_DISABLED"] = "true"
#set_caching_enabled(False)

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

args = parse_args()
set_random_seed(args.seed)

tokenizer = load_tokenizer(args)


def main():

    dataset_dict = get_dataset(args, tokenizer)
    dataset_dict.save_to_disk("personachat-dataset-round")
    dataset_dict = DatasetDict.load_from_disk("personachat-dataset")
    #dataset_dict = dataset_dict.remove_columns("candidate_input_ids")
    #dataset_dict["validation"] = dataset_dict["validation"].remove_columns("labels")

    model = get_model(args)

    if args.output_dir != "none":
        save_strategy = "epoch"
    else:
        save_strategy = "no"
    
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy=save_strategy,
        evaluation_strategy="epoch",
        eval_steps=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        disable_tqdm=True,
        dataloader_drop_last=True
    )

    trainer = Trainer(
        args=training_arguments,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        compute_metrics=accuracy_metric if args.model == "bert" else None,
    )

    trainer.train()


if __name__ == "__main__":
    main()
