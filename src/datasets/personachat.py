import numpy as np
import random
import torch
import torch.nn.functional as F

from typing import Optional, List
from transformers import GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from scipy.special import softmax 

from src.models import BertPAROR
from src.utils import compute_f1

def get_tfidf(texts) -> TfidfVectorizer:
    texts = [T[-1] for T in texts]
    return TfidfVectorizer().fit(texts)


def score_persona(
    context: List[str],
    response: List[str],
    persona: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    max_context_length: int,
    max_response_length: int,
    max_persona_length: int,
    policy,
    n_persona,
    device="cuda"
    
):

    context_tokens = tokenizer(context, max_length=max_context_length, truncation=True, padding="max_length")
    response_tokens = tokenizer(response, max_length=max_response_length, truncation=True, padding="max_length")

    persona = np.array(persona).reshape([-1]).tolist()
    persona_tokens = tokenizer(persona, max_length=max_persona_length, truncation=True, padding="max_length", return_tensors="np")

    with torch.no_grad():
        outputs = policy(
            input_ids=torch.tensor(context_tokens.input_ids),
            y_input_ids=torch.tensor(response_tokens.input_ids),
            z_input_ids=torch.tensor(persona_tokens.input_ids).reshape([len(context), -1, persona_tokens.input_ids.shape[-1]])
        )

    scores = []
    for i in range(len(context)):
        scores.append(outputs.persona_scores[i, :, i].cpu().numpy().tolist())
    
    return scores

def process_both(
    samples,
    args,
    tokenizer,
    bert_tokenizer,
    n_persona=5,
    **kwargs
):

    context = [s[-1] for s in samples["history"]]
    samples["response"] = [s[-1] for s in samples["candidates"]]
    tokenizer.padding_side = "left"
    gpt_context = tokenizer(context, padding="max_length", truncation=True, max_length=args.max_context_length)



def process_prior(
    samples,
    args,
    tokenizer,
    **kwargs
):

    samples["response"] = [s[-1] for s in samples["candidates"]]
    eos = tokenizer.eos_token

    response = [eos + s + eos for s in samples["response"]]


    response_tokens = tokenizer(response, max_length=args.max_response_length, truncation=True)

    # Padding
    input_ids = [s for s in response_tokens.input_ids]
    labels = [s for s in response_tokens.input_ids]
    max_length = args.max_response_length

    # Pad left
    attention_mask = [[0] * (max_length - len(s)) + [1] * len(s) for s in input_ids]
    input_ids = [[tokenizer.pad_token_id] * (max_length - len(s)) + s for s in input_ids]
    labels = [[-100] * (max_length - len(s)) + s for s in labels]
        
        #input_ids = [s + [tokenizer.pad_token_id] * (max_length - len(s)) for s in input_ids]
        #labels = [s + [-100] * (max_length - len(s)) for s in labels]

    assert all([len(s) == max_length for s in attention_mask])
    assert all([len(s) == max_length for s in input_ids])
    assert all([len(s) == max_length for s in labels])

    samples["input_ids"] = input_ids
    samples["labels"] = labels
    samples["attention_mask"] = attention_mask

    return samples

    

def process_bandit(
    samples,
    args,
    tokenizer,
    bert_tokenizer,
    split = "train",
    policy = Optional[PreTrainedModel],
    n_persona=5,
    **kwargs
):

    """
    Requires the following outputs:

    WITHOUT PADDING
    gpt_x_input_ids: context
    gpt_z_input_ids: persona

    WITH PADDING
    bert_x_input_ids
    bert_z_input_ids
    bert_y_input_ids
    """
    context = [s[-1] for s in samples["history"]]
    samples["response"] = [s[-1] for s in samples["candidates"]]

    gpt_x_input_ids = tokenizer(context).input_ids




def process_gpt(
    samples,
    args,
    tokenizer,
    bert_tokenizer,
    split = "train",
    policy = Optional[PreTrainedModel],
    n_persona=5,
    **kwargs
):

    """batched map fn"""
    eos = tokenizer.eos_token
    # Max turns = 1 for simplicity now
    context = [s[-1] for s in samples["history"]]
    samples["response"] = [s[-1] for s in samples["candidates"]]


    """
    Select persona with Retrieval model
    """
    if policy is not None:
        persona = pad_sentences(samples["personality"], n_persona, tokenizer.pad_token)
        scores = score_persona(
            context=context,
            response=samples["response"],
            persona=persona,
            tokenizer=bert_tokenizer,
            max_context_length=args.max_context_length,
            max_response_length=args.max_response_length,
            max_persona_length=args.max_persona_length,
            policy=policy,
            n_persona=n_persona
        )

        chosen_persona = []
        max_prob = []
        for i in range(len(context)):
            p = np.random.choice(persona[i], p=softmax(scores[i]))
            #p = max([compute_f1(p, [r]) for p, r in zip(persona, samples[""])])
            max_prob.append(max(softmax(scores[i])))
            chosen_persona.append(p)
        persona = chosen_persona

    else:
        persona = [max([(compute_f1(p, [r]), p) for p in samples["personality"][i]], key=lambda x: x[0])[1] for i, r in enumerate(samples["response"])]
        #persona = [random.choice(s) for s in samples["personality"]]

 
    """
    Structure should be
    context + persona + response
    """

    response = [s + eos for s in samples["response"]]
    context = [c + eos for c in context]
    persona = [p + eos for p in persona]
    
    context_tokens = tokenizer(context)
    persona_tokens = tokenizer(persona)
    response_tokens = tokenizer(response)

    # Manual truncation
    context_ids = [c[-args.max_context_length:] + p[:args.max_persona_length] for c, p in zip(context_tokens.input_ids, persona_tokens.input_ids)]
    response_ids = [r[:args.max_response_length] for r in response_tokens.input_ids]

    # Combine inputs
    input_ids = [s + r for s, r in zip(context_ids, response_ids)]
    labels = [[-100] * len(s) + r for s, r in zip(context_ids, response_ids)]
    max_length = args.max_context_length + args.max_response_length  #max([len(s) for s in input_ids])

    # Pad left
    attention_mask = [[0] * (max_length - len(s)) + [1] * len(s) for s in input_ids]
    input_ids = [[tokenizer.pad_token_id] * (max_length - len(s)) + s for s in input_ids]
    labels = [[-100] * (max_length - len(s)) + s for s in labels]
        
        #input_ids = [s + [tokenizer.pad_token_id] * (max_length - len(s)) for s in input_ids]
        #labels = [s + [-100] * (max_length - len(s)) for s in labels]

    assert all([len(s) == max_length for s in attention_mask])
    assert all([len(s) == max_length for s in input_ids])
    assert all([len(s) == max_length for s in labels])

    samples["input_ids"] = input_ids
    samples["labels"] = labels
    samples["attention_mask"] = attention_mask

    return samples



def pad_sentences(list_of_lists, max_len, pad_token):
    tmp_list = []
    for sublist in list_of_lists:
        sublist = [" " + pad_token + " "] * (max_len - len(sublist)) + sublist
        tmp_list.append(sublist)
    list_of_lists = [item for sublist in tmp_list for item in sublist]

    return tmp_list

def process_bert(
    samples,
    args,
    tokenizer,
    bert_tokenizer,
    split = "train",
    n_persona=5,
    n_turns=1,
    persona_shape="round",
    **kwargs
):

    """batched map fn"""

    #########################################
    # GPT-2 inputs preparation
    #########################################

    context = [s[-1] for s in samples["history"]]
    tokenizer.padding_side = "left"
    eos = tokenizer.eos_token
    persona = pad_sentences(samples["personality"], n_persona, tokenizer.pad_token)
    context_persona = [c + eos + p[i] + eos for c, p in zip(context, persona) for i in range(n_persona)] # flat
    input_tokens = tokenizer(context_persona, max_length=args.max_context_length+args.max_persona_length, truncation=True, padding=True, return_tensors="np")
    input_ids = input_tokens.input_ids.reshape([len(context), n_persona, -1])
    attention_mask = input_tokens.attention_mask.reshape([len(context), n_persona, -1])

    samples["gpt_x_input_ids"] = tokenizer(context).input_ids
    samples["gpt_input_ids"] = input_ids
    samples["gpt_attention_mask"] = attention_mask

    samples["gpt_z_input_ids"] = [tokenizer(P, max_length=args.max_persona_length, truncation=True).input_ids for P in samples["personality"]]
        

    samples["response"] = [s[-1] for s in samples["candidates"]]

    ##########################################
    # BERT inputs preparation
    ##########################################

    # Max turns = 1 for simplicity now
    context = [bert_tokenizer.sep_token.join(s[-min(len(s), n_turns):]) for s in samples["history"]]
    samples["context"] = context
    #context = [s[-1] for s in samples["history"]]
    context_tokens = bert_tokenizer(context, max_length=args.max_context_length, truncation=True, padding="max_length")
    

    if split == "train":
        response = [s[-1] for s in samples["candidates"]]
        response_tokens = bert_tokenizer(response, max_length=args.max_response_length, truncation=True, padding="max_length")
        samples["y_input_ids"] = response_tokens.input_ids
        samples["y_attention_mask"] = response_tokens.attention_mask
    else:
        candidates = np.array(samples["candidates"])
        bsz, label = candidates.shape
        candidates = candidates.reshape([-1]).tolist()
        candidate_tokens = bert_tokenizer(candidates, max_length=args.max_response_length, truncation=True, padding="max_length", return_tensors="np")
        samples["labels"] = [label-1 for _ in range(bsz)]
        samples["candidate_input_ids"] = candidate_tokens.input_ids.reshape([bsz, -1, args.max_response_length]).tolist()
        samples["candidate_attention_mask"] = candidate_tokens.attention_mask.reshape([bsz, -1, args.max_response_length]).tolist()
        
    samples["input_ids"] = context_tokens.input_ids
    samples["attention_mask"] = context_tokens.attention_mask

    padded_persona = pad_sentences(samples["personality"], n_persona, tokenizer.pad_token)
    samples["persona"] = padded_persona

    if persona_shape != "flat":
        persona = pad_sentences(samples["personality"], n_persona, tokenizer.pad_token)
        flat_persona = np.array(persona).reshape([-1]).tolist()
        persona_tokens = bert_tokenizer(flat_persona, max_length=args.max_persona_length, truncation=True, padding="max_length", return_tensors="np")
        
        samples["z_input_ids"] = persona_tokens.input_ids.reshape([-1, n_persona, args.max_persona_length]).tolist()
    else:
        persona = [bert_tokenizer.sep_token.join(p) for p in samples["personality"]]
        persona_tokens = bert_tokenizer(persona, max_length=64, truncation=True, padding="max_length")
        samples["token_type_ids"] = [len(s) * [0] + len(p) * [1] for s, p in zip(samples["input_ids"], persona_tokens.input_ids)]
        samples["input_ids"] = [s + p for s, p in zip(samples["input_ids"], persona_tokens.input_ids)]
        samples["attention_mask"] = [s + p for s, p in zip(samples["attention_mask"], persona_tokens.attention_mask)]

    # Generate persona scores
    with torch.no_grad():
        """
        outputs = kwargs["policy"](
            #input_ids=torch.tensor(batch["gpt_input_ids"]).to(args.device),
            #attention_mask=torch.tensor(batch["gpt_attention_mask"]).to(args.device),
            input_ids=torch.tensor(samples["input_ids"]).to(args.device),
            z_input_ids=torch.tensor(samples["z_input_ids"]).to(args.device),
            y_input_ids=torch.tensor(samples["y_input_ids"]).to(args.device) if split == "train" else None,
            candidate_input_ids=torch.tensor(samples["candidate_input_ids"]).to(args.device) if split != "train" else None
        )
        #probs = F.softmax(outputs.persona_scores[:, :, -1], dim=-1).cpu().numpy().tolist()
        probs = outputs.persona_scores[:, :, -1].cpu().numpy().tolist()
        labels = outputs.persona_scores[:, :, -1].max(-1).indices.cpu().numpy().tolist()
        samples["labels"] = labels
        samples["z_probs"] = probs
        #samples["f1_labels"] = [np.argmax([compute_f1(p, [r]) for p in persona[i]]) for i, r in enumerate(samples["response"])]
        """
        #samples["persona_labels"] = [np.argmax(linear_kernel(kwargs["tfidf"].transform(persona[i]), kwargs["tfidf"].transform([r]))) for i, r in enumerate(samples["response"])]
    samples["persona_scores"] = [np.max(linear_kernel(kwargs["tfidf"].transform(persona[i]), kwargs["tfidf"].transform([r]))) for i, r in enumerate(samples["response"])]
        #samples["context_scores"] = [linear_kernel(kwargs["tfidf"].transform([c]), kwargs["tfidf"].transform([r])).item() for c, r in zip(samples["context"], samples["response"])]
        #samples["z_probs"] = [linear_kernel(kwargs["tfidf"].transform(persona[i]), kwargs["tfidf"].transform([r])) for i, r in enumerate(samples["response"])]

    #samples["paror_chosen_persona"] = [P[i] for P, i in zip(persona, samples["labels"])]
    #samples["f1_chosen_persona"] = [P[i] for P, i in zip(persona, samples["f1_labels"])]

    
    #new_samples = {}
    #keys = list(samples.keys())
    #threshold = np.quantile([max(p) for p in samples["z_probs"]], 0.75)
    #for key in keys:
    #    new_samples[key] = [v for v, p in zip(samples[key], samples["z_probs"]) if max(p) > threshold]
    #samples = new_samples

    #for c, p, f, r in zip(samples["context"], samples["paror_chosen_persona"], samples["f1_chosen_persona"], samples["response"]):
        #print(c, "==>", p, "==>", f, "==>", r)

    return samples


def get_dataset(args, tokenizer):

    dataset_dict = load_dataset("bavard/personachat_truecased")

    tfidf = get_tfidf(dataset_dict["train"]["candidates"])

    if args.policy != "none":
        policy = BertPAROR.from_pretrained(args.policy).to(args.device)
    else:
        policy = None

    #if args.splits != "none":
       # dd = {split: dataset_dict[split] for split in splits}
        #dataset_dict = DatasetDict(dd)
    #else:
    splits = list(dataset_dict.keys())

    if args.debug:
        dd = {split: Dataset.from_dict(dataset_dict[split][:1000]) for split in splits}
        dataset_dict = DatasetDict(dd)

    map_fns = {
        "gpt": process_gpt,
        "bert": process_bert,
        "prior": process_prior
    }

    if tokenizer is not None:

        map_fn = map_fns[args.model]
        dd = {}
        for split in splits:
            dd[split] = dataset_dict[split].map(
                lambda x: map_fn(
                    x, 
                    args=args, 
                    tokenizer=tokenizer["gpt"],
                    bert_tokenizer=tokenizer["bert"], 
                    split=split, 
                    policy=policy,
                    tfidf=tfidf
                ), batched=True, batch_size=100)

        dataset_dict = DatasetDict(dd)
        dataset_dict = dataset_dict.remove_columns(["personality", "candidates", "history", "conv_id", "utterance_idx"])
            
    return dataset_dict

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
    return get_dataset(tokenizer)

if __name__ == "__main__":
    main()
    