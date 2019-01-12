import pickle
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='get_alignments.py ')

##
## **Preprocess Options**
##

parser.add_argument('-src', required=True, help="Read options from this file")
parser.add_argument('-tgt', required=True, help="Read options from this file")
parser.add_argument('-aligned_file', required=True, help="Read options from this file")
parser.add_argument('-output', required=True, help="Read options from this file")

opt = parser.parse_args()

def main():
  srcFile = open(opt.src)
  tgtFile = open(opt.tgt)
  alignedFile = open(opt.aligned_file)
  alignments = {}
  for line in alignedFile:
    c = line.strip().split()
    # print (c)
    srcLine = srcFile.readline().split()
    tgtLine = tgtFile.readline().split()
    for item in c:
        x,y = item.split("-")
        x = int(x)
        y = int(y)
        if srcLine[x] not in alignments:
          alignments[srcLine[x]] = defaultdict(int)
        alignments[srcLine[x]][tgtLine[y]] += 1

  print ("Built the dictionary, finding best alignments")
  bestAlignments = {}
  len(alignments)
  for srcWord in alignments.keys():
    if len(alignments[srcWord]) == 0:
        continue
    bestTgtWord = max(alignments[srcWord].items(), key=lambda x:x[1])
    bestAlignments[srcWord] = bestTgtWord[0]

  pickle.dump(bestAlignments, open(opt.output,"wb"))

if __name__ == "__main__":
    main()
