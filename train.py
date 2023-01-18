import os

from transformers import (
    Trainer, 
    TrainingArguments, 
)
from datasets import set_caching_enabled, DatasetDict

from src.datasets import get_dataset
from src.args import parse_args
from src.utils import load_tokenizer, set_random_seed, accuracy_metric
from src.modeling import get_model

os.environ["WANDB_DISABLED"] = "true"
set_caching_enabled(False)


args = parse_args()
set_random_seed(args.seed)

tokenizer = load_tokenizer(args)

def _gpt_format_dataset(
    samples, 
    tokenizer,
    max_response_length
):

    tokens = tokenizer(samples["responses"], max_length=max_response_length, padding="max_length", truncation=True)
    samples["input_ids"] = tokens.input_ids
    samples["labels"] = [[idx if idx != tokenizer.pad_token_id else -100 for idx in idxs] for idxs in tokens.input_ids]
    return samples

def _get_dataset(args, tokenizer):
    if args.dataset_load_path == "none":
        dataset_dict = get_dataset(args, tokenizer)
        if args.dataset_save_path != "none":
            dataset_dict.save_to_disk(args.dataset_save_path)
        return dataset_dict
    
    dataset_dict = DatasetDict.load_from_disk(args.dataset_load_path)

    if args.model_type == "gpt":
        dataset_dict = dataset_dict.map(
            lambda x: _gpt_format_dataset(x, tokenizer["gpt"], args.max_response_length), batched=True, batch_size=100
        )

    return dataset_dict


def main():

    dataset_dict = _get_dataset(args, tokenizer)
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
        save_total_limit=5,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        disable_tqdm=True,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        args=training_arguments,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["valid"],
        compute_metrics=accuracy_metric,
    )

    trainer.train()


if __name__ == "__main__":
    main()
