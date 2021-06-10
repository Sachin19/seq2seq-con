import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

##
## **Preprocess Options**
##

parser.add_argument('-phrases', required=True,)
parser.add_argument('-new_phrases', required=True,)

opt = parser.parse_args()

phrases = set()
outphrase = open(opt.new_phrases, "w")
with open(opt.phrases, encoding='latin1') as f:
    for l in f:
        src, tgt = l.split("\t")
        if src in tgt.split():
            outphrase.write(src+"\t"+src+" "+tgt)
        else:
            outphrase.write(src+"\t"+tgt)
outphrase.close()

