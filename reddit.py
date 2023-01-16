"""
We only want a (pseudo) random subset from the data, so
first figure out how long the dataset is,
then every n / k lines we include.
"""
import math
import random

NUM_SAMPLES = 5000
FILE_PATH = "../data/mrs/mrs/mrs/test/en/test.tsv"
SAVE_PATH = "reddit_test.tsv"

count = 0
with open(FILE_PATH, encoding='utf-8') as f:
    for i, line in enumerate(f):
        count += 1

print(f"{count} total samples")

random.seed(0)
idxs = random.sample(range(count), NUM_SAMPLES)
idxs = sorted(idxs)

interval = math.floor(count / NUM_SAMPLES)
contexts = []
responses = []
with open(FILE_PATH, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == idxs[0]:
            context, response = line.split("\t")
            contexts.append(context)
            responses.append(response)
            idxs.pop(0)
            if len(idxs) == 0:
                break

with open(SAVE_PATH, "w",  encoding='utf-8') as f:
    for c, r in zip(contexts, responses):
        f.write(c + "\t" + r)