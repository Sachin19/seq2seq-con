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

import embeddings
import torch
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys


BATCH_SIZE = 500


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('output', help='Where to output the dictionary created')
    parser.add_argument('--src_vocab', default=None, help='source words to translate to target')
    parser.add_argument('--delimiter', default="tab", help='tab or space')
    parser.add_argument('--topk', default=1, type=int, help='How many neighbors to output')
    parser.add_argument('--include_all_bigrams', action='store_true', help='How many neighbors to output')
    parser.add_argument('--include_all_trigrams', action='store_true', help='How many neighbors to output')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    if args.delimiter == "tab":
        delim = "\t"
    else:
        delim = " "
    # Read input embeddings
    # src_vocab = torch.load(args.src_vocab)['tgt'].base_field.vocab
    # src_words = src_vocab.itos
    # x = src_vocab.vectors
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    trg_set = set(trg_words)

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
        x = xp.array(x)
        z = xp.array(z)
    
    
    # print(x)
    # print(z)
    # input()
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    iden_translation = set() #source whose translation is the word itself
    # Read dictionary and compute coverage
    if args.src_vocab is not None:
        f = open(args.src_vocab, encoding=args.encoding, errors='surrogateescape')
        vocab = []
        src = []
        for line in f:
            src_word = line.strip().split(delim)[0]
            srcword = src_word.replace(" ", "&#32;")
            # print("before: ", srcword)
            if srcword in src_word2ind:
                # print("after: ", srcword)
                src_ind = src_word2ind[srcword]
                vocab.append(srcword)
                src.append(src_ind)

                if src_word in trg_set:
                    iden_translation.add(srcword)

        if args.include_all_bigrams or args.include_all_trigrams:
            print("yes")
            for word in src_words:
                flag = 0
                ngram = word.split("&#32;")
                if (args.include_all_bigrams and len(ngram) == 2) or\
                    (args.include_all_trigrams and len(ngram) == 3):
                        vocab.append(word)
                        ind = src_word2ind[word]
                        src.append(ind)

                        if word in trg_set:
                            iden_translation.add(word)
        
        src = np.array(src)

    else:
        src = np.arange(len(src_words))
    
    print(len(iden_translation))
    print(len(src))
    

    # Find translations
    translation = collections.defaultdict(list)
    # translation = {}
    if args.retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = torch.Tensor(x[src[i:j]].dot(z.T))
            nnscore, nn = similarities.topk(args.topk, dim=1)
            nn = nn.tolist()
            nnscore = nnscore.tolist()
            # nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = (nn[k], nnscore[k])

    elif args.retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = [k]
    elif args.retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = [nn[k]]
    elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN

            similarities = torch.Tensor(similarities)
            nnscore, nn = similarities.topk(args.topk, dim=1)
            nn = nn.tolist()
            nnscore = nnscore.tolist()

            # nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = (nn[k], nnscore[k])

    # Compute accuracy
    f = open(args.output,"w")
    couldnt = 0
    for i in src:
        jlist = translation[i][0]
        twordscores = translation[i][1]
        twords = []
        for j in jlist:
            twords.append(trg_words[j].replace(" ", "_"))
        
        if src_words[i] in iden_translation and src_words[i] != twords[0]:
            twords = [src_words[i]] + twords[:-1]
            twordscores = [1.] + twordscores[:-1]

        try:
            f.write(src_words[i].replace(" ","_")+"\t"+" ".join(twords)+"\t"+" ".join([str(x) for x in twordscores])+"\n")
        except:
            couldnt+=1
    f.close()
    print("Couldn't do some of them: ",couldnt)
    # accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])
    # print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))


if __name__ == '__main__':
    main()
