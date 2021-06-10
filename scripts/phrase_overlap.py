import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

##
## **Preprocess Options**
##

parser.add_argument('-emb_table', required=True)
parser.add_argument('-phrases', required=True,)

opt = parser.parse_args()

phrases = set()
with open(opt.phrases, encoding='latin1') as f:
    for l in f:
        phrase = l.split("\t", 1)[0]
        # phraseunits = phrase.split("&#32;")
        # phrase = " ".join(phraseunits)
        phrases.add(phrase.replace(" ", "&#32;"))

print(len(phrases))

with open(opt.emb_table, encoding='latin1') as f:
        f.readline()
        i = 0 
        c = 0
        for l in f:
            i += 1
            phrase = l.strip().split(" ", 1)[0]
            if phrase in phrases:
                c += 1

print (c)
print (i)

