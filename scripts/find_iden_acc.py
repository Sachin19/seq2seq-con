import argparse

parser = argparse.ArgumentParser(description='find_dictionary.py ')

##
## **Preprocess Options**
##
parser.add_argument('-dict', required=True, help="Read options from this file")
parser.add_argument('-common_vocab', required=True, help="Read options from this file")

opt = parser.parse_args()

def main():
    with open(opt.common_vocab) as f:
        common_vocab = set([x.strip() for x in f.readlines()])
    
    match = 0
    acc = 0
    with open(opt.dict) as f:
        for l in f:
            src, tgt = l.strip().split("\t")
            if src == tgt.split()[0]:
                match += 1
                if src in common_vocab:
                    acc += 1

    print("Found", match, "matches")
    print("Accuracy =",float(acc)/len(common_vocab))
      
if __name__ == "__main__":
    main()

