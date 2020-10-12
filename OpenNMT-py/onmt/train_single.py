#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import (
    build_dataset_iter,
    load_old_vocab,
    old_style_vocab,
    build_dataset_iter_multiple,
)
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    nontrainable = 0
    for name, param in model.named_parameters():
        if "encoder" in name:
            if param.requires_grad:
                enc += param.nelement()
        else:
            if param.requires_grad:
                dec += param.nelement()
            else:
                nontrainable += param.nelement()
        # logger.info(name)
    return enc + dec, enc, dec, nontrainable


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(
        opt.accum_steps
    ), "Number of accum_count values must match number of accum_steps"
    # Load checkpoint if we resume from a previous training.
    assert (
        not opt.finetune or opt.initialize_with
    ), "Cannot finetune if the no trained model is provided"
    assert (
        opt.pretrain_decoder and opt.initialize_with
    ), "Pretrain needs to be from scratch, not of an already trained model"
    if opt.train_from:
        logger.info("Loading checkpoint from %s" % opt.train_from)
        checkpoint = torch.load(
            opt.train_from, map_location=lambda storage, loc: storage
        )
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info("Loading vocab from checkpoint at %s." % opt.train_from)

        if opt.modify_opts:  # modify some of the following opts with new ones
            model_opt.save_checkpoint_steps = opt.save_checkpoint_steps
            model_opt.train_steps = opt.train_steps
            model_opt.train_only_sec_task = opt.train_only_sec_task
            model_opt.multi_task = opt.multi_task

        if opt.train_only_sec_task:
            vocab = torch.load(opt.data + ".vocab.pt")
        else:
            vocab = checkpoint["vocab"]

    elif opt.initialize_with:
        logger.info("Loading checkpoint from %s" % opt.train_from)
        checkpoint = torch.load(
            opt.train_from, map_location=lambda storage, loc: storage
        )
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])

        model_opt.num_adapters = opt.num_adapters
        model_opt.adapter_dim = opt.adapter_dim
        model_opt.adapt_embeddings = opt.adapt_embeddings
        model_opt.tie_embedding_adapters = opt.tie_embedding_adapters
        model_opt.train_only_adapters = opt.train_only_adapters
        model_opt.new_positional_embeddings = opt.new_positional_embeddings

        new_vocab = torch.load(opt.data + ".vocab.pt")
        vocab = checkpoint["vocab"]
        print(vocab.keys())
        input()
        vocab["tgt"] = new_vocab["tgt"]
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + ".vocab.pt")

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ["src", "tgt"]:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(" * %s vocab size = %d" % (sn, len(sf.vocab)))

    # Build model.
    model, fields = build_model(model_opt, opt, fields, checkpoint)
    print(fields.keys())
    logger.info(model.mtl_generator)
    _check_save_model_path(opt)

    # Build optimizer.
    if opt.train_only_sec_task:
        logger.info(
            "Since train_sec_task is set, the optimizer will have only pos prediction parameters, others will be frozen"
        )
        # freeze other model parameters
        for name, p in model.named_parameters():
            logger.info(name)
            if "mtl_generator" not in name:
                p.requires_grad = False
        opt.reset_optim = True

    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    n_params, enc, dec, nontrainable = _tally_parameters(model)
    logger.info("encoder: %d" % enc)
    logger.info("decoder: %d" % dec)
    logger.info("non-trainable parameters (tgt_out_emb): %d" % nontrainable)
    logger.info("* number of parameters: %d" % n_params)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver
    )

    if batch_queue is None:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt)

    else:
        assert semaphore is not None, "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                yield batch

        train_iter = _train_iter()

    valid_iter = build_dataset_iter("valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info("Starting training on GPU: %s" % opt.gpu_ranks)
    else:
        logger.info("Starting training on CPU, could be very slow")
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps,
    )

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
