import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

##
## **Preprocess Options**
##

parser.add_argument('-data', nargs="+", required=True)
parser.add_argument('-phrases', required=True,)
parser.add_argument('-output', nargs="+", required=True,)

opt = parser.parse_args()

phrases = {}
with open(opt.phrases, encoding='latin1') as f:
    for l in f:
        phrase, score1, score2 = l.split("\t")
        # phraseunits = phrase.split("&#32;")
        # phrase = " ".join(phraseunits)
        phrases[phrase] = float(score1)

print(len(phrases))

for out,dat in zip(opt.output, opt.data):
    print(out, dat)
    fwrite = open(out, "w")
    with open(dat) as f:
        for l in f:
            lout = []
            words = l.strip().split()
            i = 0
            while i < len(words):
                maxscore = -10.
                new_item = words[i]
                increment = 1
                if i+3 <= len(words) and " ".join(words[i:i+3]) in phrases:
                    # lout.append("&#32;".join(words[i:i+3]))
                    score = phrases[" ".join(words[i:i+3])]
                    if score > maxscore:
                        max_score = score
                        new_item = "&#32;".join(words[i:i+3])
                        increment = 3
                
                if i+2 <= len(words) and " ".join(words[i:i+2]) in phrases:
                    score = phrases[" ".join(words[i:i+2])]
                    if score > maxscore:
                        max_score = score
                        new_item = "&#32;".join(words[i:i+2])
                        increment = 2
                
                lout.append(new_item)
                i += increment
            fwrite.write(" ".join(lout)+"\n")
    fwrite.close()


