import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

##
## **Preprocess Options**
##

parser.add_argument('-dict', required=True,)
parser.add_argument('-output', required=True,)
parser.add_argument('-add_specials', action="store_true")

opt = parser.parse_args()


out = open(opt.output, "w")
if opt.add_specials:
    out.write("<unk>\n")
    out.write("<blank>\n")
    out.write("<s>\n")
    out.write("</s>\n")

with open(opt.dict, encoding="utf-8", errors="surrogateescape") as f:
    for l in f:
        src, tgt = l.split("\t")
        tgt = tgt.split()
        if src in tgt:
            out.write(src+"\n")
        else:
            out.write(tgt[0]+"\n")

out.close()

