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

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=100000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=100000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-tgt_emb',
                    help="Path to an existing target embeddings file")
parser.add_argument('-src_emb',
                    help="Path to an existing source embeddings file")

parser.add_argument('-normalize', action='store_true',
                    help="normalize the target embeddings")

parser.add_argument('-remove_unk', action='store_true',
                    help="Remove sentences which contain unks")

parser.add_argument('-seq_length', type=int, default=100,
                    help="Maximum sequence length")

parser.add_argument('-emb_dim', type=int, default=300,
                    help="Output Embedding Dimension")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size, embGiven=False, embFile=None):
    special_embeddings=None
    if embGiven:
        special_embeddings = [np.zeros(opt.emb_dim,), np.zeros(opt.emb_dim,), np.zeros(opt.emb_dim,), np.ones(opt.emb_dim,)]
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=opt.lower, special_embeddings=special_embeddings)

    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    if embGiven:
        n=0
        with codecs.open(embFile, "r", "utf-8") as f:
            for l in f:
                items = l.strip().split()
                if len(items) < 301:
                    continue
                try:
                    v = np.array(items[1:], dtype=np.float32)
                except Exception as e:
                    print (items)
                    continue
                    #sys.exit(-1)
                vocab.add_embedding(items[0], v, onmt.Constants.UNK_WORD, opt.normalize)
                n+=1

    originalSize = vocab.size()
    vocab, c = vocab.prune(size, embGiven)
    if embGiven:
        print (c, size)
        print (len(vocab.idxToLabel), len(vocab.embeddings))
        print (max(vocab.embeddings.keys()))
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

def loadGloveModel(gloveFile, word2idx):
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        if word in word2idx:
            embed = [float(val) for val in splitLine[1:]]
            model[word] = np.array(embed)
    print("Done. ",len(model), " words loaded!")
    return model

def create_mat(idx2word, dim):
    vecs = np.zeros((len(idx2word), dim))

    for i in range(len(idx2word)):
        if idx2word[i] in self.vectors:
            vecs[i] = np.reshape(self.vectors[idx2word[i]], (1, dim))
        else:
            vecs[i] = np.random.rand(1, dim)
    return vecs


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = codecs.open(srcFile, "r", "utf-8")
    tgtF = codecs.open(tgtFile, "r", "utf-8")

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcWords = sline.split()
        tgtWords = tline.split()

        if len(srcWords) <= opt.seq_length and len(tgtWords) <= opt.seq_length:

            srcTensor, sunky = srcDicts.convertToIdx(srcWords,onmt.Constants.UNK_WORD)
            tgtTensor, tunky = tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)
            if (not sunky and not tunky) or not opt.remove_unk:
                src += [srcTensor]
                tgt += [tgtTensor]

                sizes += [len(srcWords)]
            else:
                ignored += 1
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    return src, tgt

def main():
    dicts = {}
    print('Preparing source vocab ....')
    if opt.src_emb:
        dicts['src'] = initVocabularyWithEmb('source', opt.train_src, opt.src_vocab, opt.src_emb,
                                  opt.src_vocab_size)
    else:
        dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size)
    print('Preparing target vocab ....')
    print ("Target Embeddings:",opt.tgt_emb)
    if opt.tgt_emb is not None:
        dicts['tgt'] = initVocabularyWithEmb('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_emb, opt.tgt_vocab_size)
    else:
        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size)


    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                    dicts['src'], dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                }
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
