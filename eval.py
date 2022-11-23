from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge_score import rouge_scorer

from statistics import mean
from src.utils import normalize_answer, compute_f1

import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

EVAL_FILE = "preds_mb_n10_3.txt"

def bleu(targets, preds, mode=1):
    weights = [1./mode if (i+1) <= mode else 0. for i in range(mode)]
    _bleu = mean(
        [max([sentence_bleu([t], p, weights=weights) for p in P]) for t, P in zip(targets, preds)]
    )
    return _bleu

def meteor(targets, preds):
    _meteor = mean(
        [max([meteor_score([t], p) for p in P]) for t, P in zip(targets, preds)]
    )
    return _meteor

def nist(targets, preds, mode=1):
    _nist = mean(
        [max([sentence_nist([t], p, n=min(len(p), mode)) for p in P]) for t, P in zip(targets, preds)]
    )
    return _nist

def f1(targets, preds):
    f1 = mean(
        [max([compute_f1(p, [r], stopwords=None) for p in P]) for P, r in zip(preds, targets)]
    )
    return f1

def distinct(preds, mode=1):
    d = 0.0
    for pred in preds:
        num_words = sum([len(p) for p in pred])
        ngrams = []
        for p in pred:
            temp = zip(*[p[i:] for i in range(0, mode)])
            ngrams += [' '.join(ngram) for ngram in temp]
        d += len(set(ngrams)) / num_words
    d /= len(preds)

    return d

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3"], use_stemmer=True)


def _blended_rouge(rouge: dict):
    """Mixes together r1, r2, r3 according to https://arxiv.org/pdf/2106.02017.pdf"""
    r1 = rouge["rouge1"].fmeasure / 6
    r2 = rouge["rouge2"].fmeasure / 3
    r3 = rouge["rouge3"].fmeasure / 2
    return r1 + r2 + r3

def rouge(targets, preds):
    avg_score = 0.
    for t, P in zip(targets, preds):
        score = max([_blended_rouge(scorer.score(t, p)) for p in P])
        avg_score += score
    avg_score /= len(preds)

    return avg_score
       

def tokenize(s):
    return s.split()

def process_file(lines):
    targets = []
    preds = []
    for line in lines:
        t, P = line.split("\t")
        targets.append(t)
        preds.append(P.split("|"))
    
    return targets, preds

def main():
    with open(EVAL_FILE) as f:
        lines = f.readlines()
        targets, preds = process_file(lines)

    # Tokenize
    target_tokens = [tokenize(normalize_answer(t)) for t in targets]
    preds_tokens = [[tokenize(normalize_answer(p)) for p in P] for P in preds]

    metrics = {
        "Rouge": rouge(targets, preds),
        "Dist-1": distinct(preds_tokens, 1),
        "Dist-2": distinct(preds_tokens, 2),
        #"Nist-2": nist(target_tokens, preds_tokens, 2),
        #"Nist-4": nist(target_tokens, preds_tokens, 4),
        #"Bleu-2": bleu(target_tokens, preds_tokens, 2),
        #"Bleu-4": bleu(target_tokens, preds_tokens, 4),
        #"Meteor": meteor(target_tokens, preds_tokens),
        #"F1": f1(targets, preds)
    }

    for k, v in metrics.items():
        print(k, ": ", v)


if __name__ == "__main__":
    main()
