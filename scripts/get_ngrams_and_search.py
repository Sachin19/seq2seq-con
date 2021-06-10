import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

##
## **Preprocess Options**
##

parser.add_argument('-data', required=True)
parser.add_argument('-outdata', required=True)
parser.add_argument('-output', required=True)
parser.add_argument('-phrases', required=True)
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

phraseset = {}
with open(opt.phrases, encoding='latin1') as f:
    for l in f:
        phrase, score, score2 = l.split("\t")
        # phraseunits = phrase.split("&#32;")
        # phrase = " ".join(phraseunits)
        phraseset[phrase] = float(score)

fwrite = open(opt.output, "w")
phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
c = 0

existing_phrases = {}
for phrase, count in phrases:
    if len(phrase.split()) > 1 and phrase in phraseset:
        c += 1
        existing_phrases[phrase] = phraseset[phrase]
        fwrite.write(phrase+"\t"+str(count)+"\n")
fwrite.close()

fwrite = open(opt.outdata, "w")
with open(opt.data) as f:
    for l in f:
        words = l.strip().split()
        maxscore = -10
        new_words = []
        i = 0
        while i < len(words):
            if i+3 <= len(words) and " ".join(words[i:i+3]) in existing_phrases:
                new_item = "_".join(words[i:i+3])
                new_words.append(new_item)
                i += 3
            elif i+2 <= len(words) and " ".join(words[i:i+2]) in existing_phrases:
                new_item = "_".join(words[i:i+2])
                new_words.append(new_item)
                i+= 2
            else:
                new_words.append(words[i])
                i += 1
        fwrite.write(" ".join(new_words)+"\n")

fwrite.close()
        

print(c)
print(len(phrases))


