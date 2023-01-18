from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge_score import rouge_scorer
from scipy.stats import ttest_ind
from statistics import mean

from src.args import parse_args
from src.utils import normalize_answer, compute_f1

import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

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
        if num_words > 0:
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
    scores = []
    for t, P in zip(targets, preds):
        score = max([_blended_rouge(scorer.score(t, p)) for p in P])
        scores.append(score)

    return scores

def self_rouge(preds):
    # Convert to two lists of targets and preds
    new_preds = []
    new_targets = []
    for P in preds:
        K = len(P)
        if len(P) != 3:
            print(P)
        for k in range(K):
            new_preds.append(P[:k] + P[k+1:])
            new_targets.append(P[k])

    return rouge(new_targets, new_preds)
 

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

def _get_metrics(f):
    lines = f.readlines()
    targets, preds = process_file(lines)

    rge = rouge(targets, preds)
    self_rge = self_rouge(preds) 

    return {"Rouge": rge, "Self-Rouge": self_rge}

def main():
    args = parse_args()
    with open(args.prediction_load_path, encoding="utf-8") as f:
        metrics = _get_metrics(f)

    if args.comparison_load_path == "none":
        for k, v in metrics.items():
            print(k, ": ", mean(v))
        exit()

    with open(args.comparison_load_path, encoding="utf-8") as f:
        comp_metrics = _get_metrics(f)

    for (k, V1), V2 in zip(metrics.items(), comp_metrics.values()):
        pval = ttest_ind(V1, V2)
        print(k, pval)


if __name__ == "__main__":
    main()
