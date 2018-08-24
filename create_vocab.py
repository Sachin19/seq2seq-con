import onmt
import numpy as np
import argparse
import torch
import codecs
import json
import sys

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-vocab_size', type=int, default=100000,
                    help="Size of the target vocabulary")

parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")

parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-emb_file',
                    help="Path to an existing target embeddings file")

parser.add_argument('-normalize', action='store_true',
                    help="normalize the target embeddings")

parser.add_argument('-emb_dim', type=int, default=300,
                    help="Embedding Dimension")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

def makeVocabulary(filename, size, target=False, embFile=None):
    print (opt.lower)
    print (opt.normalize)
    special_embeddings=None
    if target:
        special_embeddings = [np.zeros(opt.emb_dim,), np.zeros(opt.emb_dim,), np.zeros(opt.emb_dim,), np.ones(opt.emb_dim,)]
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=opt.lower, special_embeddings=special_embeddings)

    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    if target:
        n=0
        with codecs.open(embFile, "r", "utf-8") as f:
            for l in f:
                items = l.strip().split()
                try:
                    v = np.array(items[1:], dtype=np.float32)
                except Exception as e:
                    print (items)
                    sys.exit(-1)
                vocab.add_embedding(items[0], v, onmt.Constants.UNK_WORD, opt.normalize)
                n+=1

    originalSize = vocab.size()
    vocab, c = vocab.prune(size, target)
    if target:
        vocab.average_unk(onmt.Constants.UNK_WORD, n-c, opt.normalize)
        vocab.convert_embeddings_to_torch()
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab

def initVocabularyWithEmb(name, dataFile, vocabFile, embFile, vocabSize):

    vocab = None
    if embFile is None:
        raise ValueError("Please provide an embedding file for target")

    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize, embFile is not None, embFile)

        vocab = genWordVocab
    return vocab

def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD, onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=opt.lower)
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    #print(type(vocab))
    #with codecs.open(file, 'w') as outfile:
    #    json.dump(dict(vocab), outfile)
    vocab.writeFile(file)

def main():
    dicts = {}
    print('Preparing vocab ....')
    print ("Target Embeddings:",opt.emb_file)
    dicts['tgt'] = initVocabularyWithEmb('target', opt.train_tgt, opt.tgt_vocab, opt.emb_file, opt.vocab_size)

    print('Saving dict to \'' + opt.save_data + '.dict.pt\'...')
    save_data = {'dicts': dicts}
    torch.save(save_data, opt.save_data + '.dict.pt')

if __name__ == "__main__":
    main()
