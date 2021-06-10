import argparse

import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='add_phrases_to_corpus.py ')

THRESHOLD=0.6
##
## **Preprocess Options**
##

parser.add_argument('-data', required=True)
parser.add_argument('-phrase_table', required=True,)
parser.add_argument('-output', required=True,)

opt = parser.parse_args()

phrases = {}
with open(opt.phrase_table) as f:
    for l in f:
        items = l.split("\t")
        if len(items) > 2:
            source, target, score = items
        else:
            source, target = items
            score = "1.0"
        phrases[source.replace("&#32;"," ")] = (target.split()[0].replace("&#32;", " "), float(score.split()[0]))
print(len(phrases))

fwrite = open(opt.output, "w")
with open(opt.data) as f:
    for l in f:
        #print(l.strip())
        tokens = l.strip().split()
        replaced_tokens = []
        i=0
        while i < len(tokens):
            maxscore=-10
            replacement = ""
            # if i+3 <= len(tokens) and " ".join(tokens[i:i+3]) in phrases:
            #     new_item, score = phrases[" ".join(tokens[i:i+3])]
            #     if new_item != " ".join(tokens[i:i+3]):
            #         print(tokens[i:i+3],new_item,score)
            #         input()
            #     if maxscore < score and score >= THRESHOLD:
            #         maxscore = score
            #         # replacement = new_item.split()
            #         replacement = [" ".join(new_item.split())]
            #         # replacement = [" ".join(tokens[i:i+3])]
            #         increment = 3
            
            if i+2 <= len(tokens) and " ".join(tokens[i:i+2]) in phrases:
                new_item, score = phrases[" ".join(tokens[i:i+2])]
                #if new_item != " ".join(tokens[i:i+2]):
                #    print(tokens[i:i+2],new_item,score)
                #    input()
                if maxscore < score and score >= THRESHOLD:
                    maxscore = score
                    replacement = new_item.split()
                    replacement = ["_".join(new_item.split())]
                    # replacement = ["_".join(tokens[i:i+2])]
                    increment = 2
            
            if tokens[i] in phrases:
                new_item, score = phrases[tokens[i]]
                #if tokens[i] != new_item:
                #    print(tokens[i], new_item, score)
                #    input()
                if maxscore < score and score >= THRESHOLD:
                    replacement = new_item.split()
                    increment = 1
                    maxscore = score
            
            if maxscore < 0:
                replacement = [tokens[i]]
                increment = 1

            i+=increment
            replaced_tokens += replacement
        
        fwrite.write(" ".join(replaced_tokens)+"\n")
fwrite.close()


