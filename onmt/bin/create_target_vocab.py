#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_field_vocab_new_tgt,\
                                    _load_vocab

from functools import partial
from multiprocessing import Pool


def maybe_load_vocab(counters, opt):
    tgt_vocab, tgt_vocab_size = _load_vocab(
        opt.tgt_vocab, "tgt", counters,
        opt.tgt_words_min_frequency)
    return tgt_vocab


def build_save_vocab(fields, opt):
    counters = defaultdict(Counter)

    if opt.old_tgt_vocab is not None:
        old_field = torch.load(opt.old_tgt_vocab)['tgt'].base_field

    tgt_vocab = maybe_load_vocab(counters, opt)
    # every corpus has shards, no new one
    vocab_path = opt.save_data + '.vocab.pt'
    emb_file = opt.tgt_emb
    logger.info(emb_file)
    tgt_field = _build_field_vocab_new_tgt(
        fields['tgt'].base_field, counters['tgt'], old_field=old_field, emb_file=emb_file)
    torch.save(tgt_field, vocab_path)


def create_vocab(opt):
    init_logger(opt.log_file)
    logger.info("Building `Fields` object...")
    fields = inputters.get_fields("text", 0, 0)

    logger.info("building the new vocabulary")
    build_save_vocab(fields, opt)

def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.target_vocab_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    create_vocab(opt)


if __name__ == "__main__":
    main()

