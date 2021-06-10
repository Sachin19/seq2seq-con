import torch
import argparse

import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict


parser = argparse.ArgumentParser(description='get_alignments.py ')

##
## **Preprocess Options**
##

parser.add_argument('-src_vocab', required=True, help="")
parser.add_argument('-tgt_emb', required=True, help="Read options from this file")
parser.add_argument('-tgt_index_save_path', default=None, help="Read options from this file")
parser.add_argument('-output', required=True, help="Read options from this file")

opt = parser.parse_args()

def main():
    src_vocab = torch.load(opt.src_vocab)['tgt'].base_field.vocab
    tgt_words = []
    tgt_index = AnnoyIndex(300, 'angular')
    with open(opt.tgt_emb) as f:
        f.readline()
        i = 0 
        for l in f:
            items = l.strip().split()
            if len(items) == 301:
                tgt_words.append(items[0])
                v = np.array(items[1:], dtype=np.float32)
                tgt_index.add_item(i, v)
            i += 1
    
    print("building the index")
    tgt_index.build(10)

    print("built, now saving the index")
    if opt.tgt_index_save_path is not None:
        tgt_index.save(opt.tgt_index_save_path, prefault=False)
    
    with open(opt.output, "w") as f:
        for i, v in enumerate(src_vocab.vectors):
            neighbours = tgt_index.get_nns_by_vector(v, 2, include_distances=False)
            for n in neighbours:
                f.write(tgt_words[n]+"\n")
    
    print("done")
        
        

  
if __name__ == "__main__":
    main()

