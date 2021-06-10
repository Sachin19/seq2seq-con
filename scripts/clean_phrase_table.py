import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

##
## **Preprocess Options**
##

parser.add_argument('-phrases', required=True,)
parser.add_argument('-new_phrases', required=True,)
parser.add_argument('-only_phrase', action="store_true")

opt = parser.parse_args()

phrases = set()
outphrase = open(opt.new_phrases, "w")
with open(opt.phrases, encoding='latin1') as f:
    for l in f:
        src, tgt, scores = l.split("\t")
        tgt = tgt.split()[0]
        score = scores.split()[0]
        if src not in tgt.split():
            if not opt.only_phrase or "&#32;" in src:
                src = " ".join(src.split("&#32;"))
                tgt = " ".join(tgt.split("&#32;"))
                if src != tgt:
                    outphrase.write(src+"\t"+tgt+"\t"+score+"\n")            
            if not opt.only_phrase or "_" in src:
                if src != tgt:
                    outphrase.write(src+"\t"+tgt+"\t"+score+"\n")            

outphrase.close()

