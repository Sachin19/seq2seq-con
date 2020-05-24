import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

##
## **Preprocess Options**
##

parser.add_argument('-data', required=True)
parser.add_argument('-output', required=True)
parser.add_argument('-max_ngrams', type=int, default=3)

opt = parser.parse_args()

phrases = defaultdict(int)
with open(opt.data) as f:
    for l in f:
        words = l.strip().split()
        for i in range(len(words)):
            for n in range(opt.max_ngrams):
                if i + n + 1 <= len(words):
                    phrases[" ".join(words[i:i+n+1])] += 1

fwrite = open(opt.output, "w")
phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
for phrase, count in phrases:
    fwrite.write(phrase+"\t"+str(count)+"\n")
fwrite.close()


