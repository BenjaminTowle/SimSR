import numpy as np

from datasets import set_caching_enabled, Dataset, DatasetDict

from src.args import parse_args
from src.utils import set_random_seed, load_tokenizer, TFIDF
from src.datasets import get_dataset

set_caching_enabled(False)

def _map_fn(samples, tokenizer):
    samples["y_input_ids"] = tokenizer(samples["response"], padding="max_length", truncation=True, max_length=32).input_ids
    return samples


def main():
    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = load_tokenizer(args)
    #train_dataset = get_dataset(args, tokenizer)["train"]
    train_dataset = DatasetDict.load_from_disk("personachat-dataset-round")["train"]

    responses = train_dataset["response"]
    #input_ids = train_dataset["y_input_ids"]

    # Deduplicate
    unique_responses = []
    #unique_input_ids = []
    for response in responses:
        if response not in unique_responses:
            unique_responses.append(response)
            #unique_input_ids.append(unique_input_ids)
    responses = unique_responses
    #input_ids = unique_input_ids
    

    tfidf = TFIDF(responses)
    e_scores = tfidf(responses, responses)
    #e_scores = np.mean(scores, axis=-1)
    top_idxs = np.argpartition(e_scores, -args.candidate_pool_size)[-args.candidate_pool_size:]

    responses = np.array(responses)[top_idxs].tolist()

    #input_ids = np.array(input_ids)[top_idxs].tolist()

    dataset = Dataset.from_dict({"response": responses})
    dataset = dataset.map(lambda x: _map_fn(x, tokenizer["bert"]), batched=True)
    dataset.save_to_disk("candidate_pool_8192")


if __name__ == "__main__":
    main()