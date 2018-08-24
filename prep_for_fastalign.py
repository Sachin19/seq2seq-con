import pickle
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='prepare_alignments.py')

##
## **Preprocess Options**
##

parser.add_argument('-src', required=True, help="Read options from this file")
parser.add_argument('-tgt', required=True, help="Read options from this file")
parser.add_argument('-output', required=True, help="Read options from this file")

opt = parser.parse_args()

def main():
  srcFile = open(opt.src)
  tgtFile = open(opt.tgt)
  mixFile = open(opt.output,"w")

  for srcLine in srcFile:
    tgtLine = tgtFile.readline()
    mixFile.write(srcLine.strip()+" ||| "+tgtLine.strip()+"\n")

  mixFile.close()

if __name__ == "__main__":
    main()


# for i,line in enumerate(f.readlines()):
#    ...:     c = line.strip().split()
#    ...:     for item in c:
#    ...:         x,y = item.split("-")
#    ...:         x = int(x)
#    ...:         y = int(y)
#    ...:         if frlines[i][x] in alignments:
#    ...:             alignments[frlines[i][x]][enlines[i][y]] += 1
#    ...:         else:
#    ...:             alignments[frlines[i][x]] = defaultdict(int)

# for frword in alignments.keys():
#    ...:     if len(alignments[frword]) == 0:
#    ...:         continue
#    ...:     bestenword = max(alignments[frword].items(), key=lambda x:x[1])
#    ...:     best_alignments[frword] = bestenword[0]
