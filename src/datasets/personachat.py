import numpy as np
import os
import re

from datasets import Dataset, DatasetDict


def get_contexts_persona_responses(f, split, max_turns=1, debug=False):
    persona_a = []
    personae_a = []
    persona_b = []
    personae_b = []

    dialog = []
    contexts = []
    responses = []
    persona = []
    candidates = []

    reading_persona = True
    lines = f.readlines()
    for line in lines:
        if "your persona:" in line:
            if reading_persona is False:
                personae_a.append(persona_a)
                personae_b.append(persona_b)
                persona_a = []
                persona_b = []
                dialog = []
                reading_persona = True
            persona_a.append(re.sub(r"\A[0-9]+ (your persona: |partner's persona: )", "", line).replace("\n", ""))
        elif "partner's persona:" in line:
            persona_b.append(re.sub(r"\A[0-9]+ (your persona: |partner's persona: )", "", line))
        else:
            # utterance line is split into speaker A + \t + speaker B + \t\t + candidate_1|candidate_2 etc.
            utts = line.split("\t")
            c = utts[3].replace("\n", "").split("|") if split != "train" else None  # No candidates during training
            context = re.sub(r"\A[0-9]+ ", "", utts[0])  # remove line numbering
            response = utts[1]
            dialog.append(context)
            contexts.append(dialog[-min(max_turns, len(dialog)):]) if max_turns != -1 else contexts.append(dialog[:])
            dialog.append(response)  #  MUST BE ADDED AFTER THE CONTEXT!
            responses.append(response)
            persona.append(persona_a)

            if split != "train":
                # Ground truth contained in last position
                candidates.append(c)
            reading_persona = False

        if not debug:
            continue

        if len(contexts) > 100:
            break


    return contexts, persona, responses, candidates

def load_data(
    data_dir: str, 
    tokenizer,
    split: str = "train",
    max_context_length: int = 64,
    max_response_length: int = 32,
    max_turns: int = 1,
    debug: bool = False
):
    """
    Load dataset from Persona Chat paper: https://arxiv.org/abs/1801.07243
    :return: list of contexts, list of responses, list of personae
    """

    split2file = {
        "train": "train_both_original.txt",
        "valid": "valid_both_original.txt",
        "test": "test_both_original.txt"
    }

    with open(os.path.join(data_dir, split2file[split])) as f:
        contexts, persona, responses, candidates = get_contexts_persona_responses(f, split, max_turns=max_turns, debug=debug)

    # Tokenize
    contexts = [tokenizer.sep_token.join(C) for C in contexts]
    persona = [tokenizer.sep_token.join(P) for P in persona]

    persona_ids = tokenizer(persona).input_ids
    context_ids = tokenizer(contexts, add_special_tokens=False).input_ids

    input_ids = [p + c[-(max_context_length-len(p)-1):] + [tokenizer.sep_token_id] for p, c in zip(persona_ids, context_ids)]
    input_ids = [c + [tokenizer.pad_token_id] * (max_context_length - len(c)) for c in input_ids]
    input_ids = [ids[:max_context_length] for ids in input_ids]

    inputs = {}
    inputs["input_ids"] = input_ids
    inputs["responses"] = responses
    
    if split == "train":
        inputs["y_input_ids"] = tokenizer(responses, padding="max_length", max_length=max_response_length, truncation=True).input_ids
    else:
        inputs["labels"] = [19 for _ in range(len(contexts))]

        candidates = np.array(candidates).reshape(-1).tolist()
        candidate_ids = tokenizer(candidates, padding="max_length", max_length=max_response_length, truncation=True, return_tensors="np").input_ids
        inputs["candidate_input_ids"] = candidate_ids.reshape([len(contexts), -1, max_response_length])

    return Dataset.from_dict(inputs)


def get_dataset(args, tokenizer):
    datasets = {}
    for split in ["test", "train", "valid"]:
        dataset = load_data(
            args.data_dir,
            tokenizer["bert"],
            split=split,
            max_context_length=args.max_context_length,
            max_response_length=args.max_response_length,
            max_turns=args.max_turns,
            debug=args.debug
        )
        datasets[split] = dataset
    dataset_dict = DatasetDict(datasets)
    return dataset_dict
