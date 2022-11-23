import os

from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from datasets import set_caching_enabled, DatasetDict, Dataset

from src.datasets import get_dataset
from src.args import parse_args
from src.utils import load_tokenizer, set_random_seed, accuracy_metric
from src.modeling import get_model

os.environ["WANDB_DISABLED"] = "true"
set_caching_enabled(False)


args = parse_args()
set_random_seed(args.seed)

tokenizer = load_tokenizer(args)


def _get_dataset(args, tokenizer):
    if args.dataset_load_path == "none":
        dataset_dict = get_dataset(args, tokenizer)
        if args.dataset_save_path != "none":
            dataset_dict.save_to_disk(args.dataset_save_path)
        return dataset_dict
    
    dataset_dict = DatasetDict.load_from_disk(args.dataset_load_path)
    return dataset_dict


def main():

    dataset_dict = _get_dataset(args, tokenizer)
    model = get_model(args)
    model.resize_token_embeddings(len(tokenizer["bert"]) + 1)

    #model.config.gradient_checkpointing = True

    if args.output_dir != "none":
        save_strategy = "epoch"
    else:
        save_strategy = "no"
    
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy=save_strategy,
        evaluation_strategy="epoch",
        eval_steps=1,
        save_total_limit=5,
        learning_rate=5e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=3, #args.num_train_epochs,
        disable_tqdm=True,
        dataloader_drop_last=False,
        #metric_for_best_model="loss",
        #greater_is_better=False,
        #load_best_model_at_end=True
    )

    trainer = Trainer(
        args=training_arguments,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["valid"],
        compute_metrics=accuracy_metric if args.model == "bert" else None,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()


if __name__ == "__main__":
    main()


"""
Results

tinybert [cls] p1 [sep] pk [sep] [cls] u1 [sep] un [sep]
tinybert [cls] p1 [eou] pk [sep] [cls] u1 [eou] un [sep]
tinybert [cls] p1 [eou] pk [sep] [cls] u1 [sep] 55.0 bsz 32
"""