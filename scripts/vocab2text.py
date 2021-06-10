# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import argparse
import numpy as np

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('torchfile', help='the source language embeddings')
    parser.add_argument('output', help='the source language embeddings')
    parser.add_argument('-write_vectors', action='store_true', help='is the torch file a checkpoint and not a vocab file')
    parser.add_argument('-checkpoint', action='store_true', help='is the torch file a checkpoint and not a vocab file')
    args = parser.parse_args()

    # Read input embeddings
    if args.checkpoint:
        vocab = torch.load(args.torchfile, map_location=lambda storage, loc: storage)['vocab']
    else:
        vocab = torch.load(args.torchfile)
    src_vocab = vocab['tgt'].base_field.vocab
    src_words = src_vocab.itos
    x = src_vocab.vectors
    f = open(args.output, "w")
    if args.write_vectors:
        f.write(str(len(x))+" "+str(len(x[0]))+"\n")
    for word, vector in zip(src_words, x):  
        if args.write_vectors:
            f.write(word+" "+" ".join(np.array(vector, dtype='str'))+"\n")
        else:
            f.write(word+"\n")
    f.close()

if __name__ == '__main__':
    main()
