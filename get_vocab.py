import sys
import argparse
import operator
import itertools
import nltk
from collections import defaultdict

parser = argparse.ArgumentParser(
    description='Program to a space separed file and generate its vocabulary',
)
parser.add_argument('-input', type=str, help='A path to a space seperated tokens')
parser.add_argument('-output', type=str, help='Path to output file where the vocab will be written')

args = parser.parse_args()

vocab = defaultdict(int)

with open(args.input) as f:
  for l in f:
    words = l.strip().split()
    for word in words:
      vocab[word] += 1

f.close()
wordbyfreq = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

with open(args.output, "w") as f:
  for word, freq in wordbyfreq:
    f.write(word+"\t"+str(freq)+"\n")

f.close()

