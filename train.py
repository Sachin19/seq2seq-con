from __future__ import division

import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import sys
import numpy as np
from loss import *

parser = argparse.ArgumentParser(description='train.py')

## Data options
parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from prepare_data.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")
parser.add_argument('-train_anew', action='store_true',
                     help="Load from the train_from model but restart optimizer")
parser.add_argument('-nonlin_gen', action='store_true',
                    help="Make generator (final layer which produces the continuous vector) non linear using a 2 layer MLP")
parser.add_argument('-save_all_epochs', action='store_true',
                    help="Save the model at every epoch (Could be memory consuming")

## Model options
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=1024,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=512,
                    help='Input word embedding sizes')
parser.add_argument('-output_emb_size', type=int, default=300,
                    help='Dimension of the output vector')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-tie_emb', action='store_true',
                    help="Tie input and output embeddings of decoder")
parser.add_argument('-fix_src_emb', action='store_true',
                    help="Initialize and fix the source embeddings")

# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-loss', default='cosine', type=str,
                    help='Loss Function to use: [ce|l2|cosine|maxmargin|nllvmf]')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

#learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

#pretrained word vectors (not really used in this model)
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_output',
                    help="""This will load the output embeddings using which
                    loss will be minimized.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit

def eval(model, loss_fn, target_embeddings, data):
    total_loss = 0
    total_words = 0
    total_other_loss = 0


    model.eval()
    for i in range(len(data)):
        batch = data[i][:-1] # exclude original indices
        outputs = model(batch)
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, other_loss = loss_fn(
                outputs, targets, target_embeddings, model.generator, opt, eval=True)
        total_loss += loss
        total_other_loss += other_loss
        total_words += targets.data.ne(onmt.Constants.PAD).float().sum()

    model.train()
    return total_loss / total_words, total_other_loss / total_words


def trainModel(model, trainData, validData, dataset, target_embeddings, optim):

    print(model)
    sys.stdout.flush()
    model.train()

    # define criterion of each GPU
    if opt.loss == "baseline":
        loss_fn = CrossEntropy
    if opt.loss == "cosine":
        loss_fn = CosineLoss
    elif opt.loss == "l2":
        loss_fn = L2Loss
    elif opt.loss == 'nllvmf':
        loss_fn = NLLvMF
    elif opt.loss == "maxmargin":
        loss_fn = MaxMarginLoss
    else:
        raise ValueError("loss function:%s is not supported"%opt.loss)

    start_time = time.time()

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_other_loss = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_other_loss = 0, 0, 0, 0
        report_samples = 0
        total_samples = 0
        start = time.time()

        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1] # exclude original indices

            model.zero_grad()
            # print batch
            outputs = model(batch)
            targets = batch[1][1:]  # exclude <s> from targets
            loss, gradOutput, other_loss = loss_fn(
                    outputs, targets, target_embeddings, model.generator, opt, False)

            outputs.backward(gradOutput)

            # update the parameters
            optim.step()
            num_words = targets.data.ne(onmt.Constants.PAD).float().sum()
            report_loss += loss
            report_other_loss += other_loss
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            report_samples += targets.size(1)*1.0
            total_samples += targets.size(1)*1.0
            total_loss += loss
            total_other_loss += other_loss
            total_words += num_words

            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; lps: %.5f; mse_lps: %.5f; %3.0f src tok/s; %3.0f tgt tok/s; %3.0f sample/s; %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                      report_loss / report_tgt_words,
                      report_other_loss / report_tgt_words,
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      report_samples/(time.time()-start),
                      time.time()-start_time))

                sys.stdout.flush()
                report_loss = report_tgt_words = report_src_words = report_other_loss = report_samples = 0
                start = time.time()

        print ("Epoch %2d, %6.0f samples, %6.0f s" % (epoch, total_samples, time.time()-start_time))
        return total_loss / total_words, total_other_loss / total_words

    valid_loss, other_loss = eval(model, loss_fn, target_embeddings, validData)
    best_valid_lps = valid_loss
    best_other_loss = other_loss
    print('Validation per step loss: %g' % best_valid_lps)
    print('Validation per step other loss: %g' % (other_loss))

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_lps = train_loss
        print('Train per step loss: %g' % train_lps)
        # print('Train accuracy: %g' % (train_acc*100))

        #  (2) evaluate on the validation set
        valid_loss, other_loss = eval(model, loss_fn, target_embeddings, validData)
        valid_lps = valid_loss
        print('Validation per step loss: %g' % valid_loss)
        print('Validation per step other loss: %g' % (other_loss))

        sys.stdout.flush()
        #  (3) update the learning rate
        # optim.updateLearningRate(valid_loss, epoch)

        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()

        checkpoint = {
        #    'model': model_state_dict,
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }

        torch.save(checkpoint,
                   '%s_latest.pt' % (opt.save_model))
        if opt.save_all_epochs:
          torch.save(checkpoint,'%s_model_%d.pt' % (opt.save_model,epoch))

        if best_valid_lps > valid_lps:
            best_valid_lps = valid_lps
            best_other_loss = other_loss
            print ("Best model found!")
            torch.save(checkpoint,
                   '%s_bestmodel.pt' % opt.save_model)
        elif best_valid_lps == valid_lps: #in case of vMF loss, if the loss is the same, has the cosine loss decreased?
            if best_other_loss > other_loss:
                best_other_loss = other_loss
                print ("Best model found!")
                torch.save(checkpoint, '%s_bestmodel.pt' % opt.save_model)

def main():

    print("Loading data from '%s'" % opt.data)
    dataset = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else None

    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    encoder = onmt.Models.Encoder(opt, dicts['src'], opt.fix_src_emb)
    decoder = onmt.Models.Decoder(opt, dicts['tgt'], opt.tie_emb)

    output_dim = opt.output_emb_size

    if not opt.nonlin_gen:
        generator = nn.Sequential(nn.Linear(opt.rnn_size, output_dim))
    else: #add a non-linear layer before generating the continuous vector
        generator = nn.Sequential(nn.Linear(opt.rnn_size, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))

    #output is just an embedding
    target_embeddings = nn.Embedding(dicts['tgt'].size(), opt.output_emb_size)

    #normalize the embeddings
    norm = dicts['tgt'].embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    target_embeddings.weight.data.copy_(dicts['tgt'].embeddings.div(norm))

    #target embeddings are fixed and not trained
    target_embeddings.weight.requires_grad=False
    # elif opt.loss != "maxmargin": # with max-margin loss, the target embeddings can be fine-tuned as well.
        # target_embeddings.weight.requires_grad=False

    model = onmt.Models.NMTModel(encoder, decoder)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        generator_state_dict = checkpoint['generator']
        encoder_state_dict = [('encoder.'+k,v) for k, v in checkpoint['encoder'].items()]
        decoder_state_dict = [('decoder.'+k,v) for k, v in checkpoint['decoder'].items()]
        model_state_dict = dict(encoder_state_dict+decoder_state_dict)

        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)

        if not opt.train_anew: #load from
            opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
        target_embeddings.cuda()
    else:
        model.cpu()
        generator.cpu()
        target_embeddings.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        if opt.tie_emb:
            decoder.tie_embeddings(target_embeddings)

        if opt.fix_src_emb:
            #fix and normalize the source embeddings
            source_embeddings = nn.Embedding(dicts['src'].size(), opt.output_emb_size)
            norm = dicts['src'].embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            source_embeddings.weight.data.copy_(dicts['src'].embeddings.div(norm))

            #turn this off to initialize embeddings as well as make them trainable
            source_embeddings.weight.requires_grad=False
            if len(opt.gpus) >= 1:
                source_embeddings.cuda()
            else:
                source_embeddings.cpu()
            encoder.fix_embeddings(source_embeddings)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    elif opt.train_anew: #restart optimizer, sometimes useful for training with
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from and not opt.train_anew:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    print('* number of trainable parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, target_embeddings, optim)

if __name__ == "__main__":
    main()
