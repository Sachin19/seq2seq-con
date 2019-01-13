from __future__ import division

import onmt
import torch
import argparse
import math
import codecs
import time

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-tgt_dict',
                    help='Target Embeddings (optional). This is usually for cases when you want to evaluate using a larger embedding table than the one used for training. It should the same format as the target embedding which is part of the training data')
parser.add_argument('-lookup_dict',
                    help='File for dictionary lookup (optional). This is just a python dictionary you can use to look up source word translations when you produce and u<unk>')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-loss', default='cosine',
                    help="""loss function: [l2|cosine|maxmargin|nllvmf]""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size') #recommended beam size for embedding outputs is 1

parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum output sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If lookup_dict
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-use_lm', action="store_true",
                    help='Use a Language Model in Beam search')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-saved_lm', default="",
                    help="""Address of the saved LM""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")

def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal/wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    print(opt)
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt)
    outF = codecs.open(opt.output, 'w', 'utf-8')

    predScoreTotal, predWordsTotal= 0, 0

    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    nsamples = 0.0
    total_time = 0.0
    knntime = 0.0

    for line in addone(codecs.open(opt.src, "r", "utf-8")):

        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        start_time = time.time()
        predBatch, predScore, knntime_ = translator.translate(srcBatch, tgtBatch)
        total_time += (time.time()-start_time)
        knntime += knntime_
        nsamples += len(predBatch)

        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        # if tgtF is not None:
        #     goldScoreTotal += sum(goldScore)
        #     goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0]) + '\n')
            outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    # print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))

                print('')

        srcBatch, tgtBatch = [], []

    # reportScore('PRED', predScoreTotal, count)
    # if tgtF:
    #     reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    samples_per_sec = nsamples/total_time
    print ("Average samples per second: %f, %f, %f" % (nsamples, total_time, samples_per_sec))
    print ("Time per sample %f, KNN Time per sample: %f" % (total_time/nsamples, knntime/nsamples))

if __name__ == "__main__":
    main()
