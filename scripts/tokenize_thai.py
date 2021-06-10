from pythainlp.tokenize import word_tokenize

import sys

with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
    for l in fin:
        fout.write(" ".join(word_tokenize(l.strip())) + "\n")
