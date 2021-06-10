"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
from onmt.modules.ive import logcmk
from onmt.utils.logging import logger


def build_loss_compute(model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
            "order to use --lambda_coverage != 0"

    if opt.loss == "ce":
        if opt.copy_attn:
            criterion = onmt.modules.CopyGeneratorLoss(
                len(tgt_field.vocab), opt.copy_attn_force,
                unk_index=unk_idx, ignore_index=padding_idx
            )
        elif opt.label_smoothing > 0 and train:
            criterion = LabelSmoothingLoss(
                opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
            )
        elif isinstance(model.generator[-1], LogSparsemax):
            criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
        else:
            criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

        # if the loss function operates on vectors of raw logits instead of
        # probabilities, only the first part of the generator needs to be
        # passed to the NMTLossCompute. At the moment, the only supported
        # loss function of this kind is the sparsemax loss.
        use_raw_logits = isinstance(criterion, SparsemaxLoss)
        loss_gen = model.generator[0] if use_raw_logits else model.generator
        if opt.copy_attn:
            compute = onmt.modules.CopyGeneratorLossCompute(
                criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength,
                lambda_coverage=opt.lambda_coverage
            )
        else:
            compute = NMTLossCompute(
                criterion, loss_gen, lambda_coverage=opt.lambda_coverage,
                lambda_align=opt.lambda_align, mtl_generator=model.mtl_generator,
                lambda_mtl=opt.lambda_mtl)
        compute.to(device)
    
    else: #continuous loss functions
        second_target_embeddings = None
        if hasattr(model.decoder, 'new_tgt_out_emb') and opt.use_two_tgt_vocab:
            second_target_embeddings = model.decoder.new_tgt_out_emb

        compute = ContinuousLossCompute(
                generator=model.generator, 
                target_embeddings=model.decoder.tgt_out_emb, 
                second_target_embeddings=model.decoder.new_tgt_out_emb,
                mtl_generator=model.mtl_generator,
                loss_type=opt.loss, 
                ignore_index=padding_idx,
                approximate_vmf=opt.approximate_vmf, 
                lambda_coverage=opt.lambda_coverage,
                lambda_align=opt.lambda_align,
                lambda_vmf=opt.lambda_vmf,
                lambda_mtl=opt.lambda_mtl,
                beta_map=opt.beta_map if opt.emb_map == "orthogonal" else 0.)

    return compute

class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target, sec_loss=None, sec_scores=None, sec_target=None):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()

        num_correct_sec=0.0
        if sec_scores is not None:
            sec_pred = sec_scores.max(1)[1]
            num_correct_sec = sec_pred.eq(sec_target).masked_select(non_padding).sum().item()

        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct, num_correct_sec)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, lambda_align=0.0, mtl_generator=None, lambda_mtl=0.0):
        super(NMTLossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.lambda_align = lambda_align
        self.lambda_mtl = lambda_mtl
        
        self.mtl_generator = mtl_generator
        if self.mtl_generator is not None:
            self.mtl_criterion = nn.NLLLoss(ignore_index=0, reduction='sum')

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
        }

        if self.mtl_generator is not None:
            shard_state.update({
                "sec_target": batch.tgt[range_[0] + 1: range_[1], :, 1],
            })

        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        if self.lambda_align != 0.0:
            # attn_align should be in (batch_size, pad_tgt_size, pad_src_size)
            attn_align = attns.get("align", None)
            # align_idx should be a Tensor in size([N, 3]), N is total number
            # of align src-tgt pair in current batch, each as
            # ['sent_N°_in_batch', 'tgt_id+1', 'src_id'] (check AlignField)
            align_idx = batch.align
            assert attns is not None
            assert attn_align is not None, "lambda_align != 0.0 requires " \
                "alignement attention head"
            assert align_idx is not None, "lambda_align != 0.0 requires " \
                "provide guided alignement"
            pad_tgt_size, batch_size, _ = batch.tgt.size()
            pad_src_size = batch.src[0].size(0)
            align_matrix_size = [batch_size, pad_tgt_size, pad_src_size]
            ref_align = onmt.utils.make_batch_align_matrix(
                align_idx, align_matrix_size, normalize=True)
            # NOTE: tgt-src ref alignement that in range_ of shard
            # (coherent with batch.tgt)
            shard_state.update({
                "align_head": attn_align,
                "ref_align": ref_align[:, range_[0] + 1: range_[1], :]
            })
        return shard_state

    def _compute_loss(self, batch, output, target, sec_target=None, std_attn=None,
                      coverage_attn=None, align_head=None, ref_align=None):

        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        sec_task_loss_for_stats = None
        sec_gtruth_for_stats = None
        sec_scores_for_stats = None
        loss = self.criterion(scores, gtruth)

        if self.mtl_generator is not None and self.lambda_mtl > 0.0:
            sec_scores = self.mtl_generator(bottled_output)
            sec_gtruth = sec_target.view(-1)
            sec_task_loss = self.lambda_mtl * self.mtl_criterion(sec_scores, sec_gtruth)
            loss += sec_task_loss
            
            sec_task_loss_for_stats = sec_task_loss.clone()
            sec_gtruth_for_stats = sec_gtruth.clone() 
            sec_scores_for_stats = sec_scores.clone()

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss
        if self.lambda_align != 0.0:
            if align_head.dtype != loss.dtype:  # Fix FP16
                align_head = align_head.to(loss.dtype)
            if ref_align.dtype != loss.dtype:
                ref_align = ref_align.to(loss.dtype)
            align_loss = self._compute_alignement_loss(
                align_head=align_head, ref_align=ref_align)
            loss += align_loss
        stats = self._stats(loss.clone(), scores, gtruth, sec_task_loss_for_stats, sec_scores_for_stats, sec_gtruth_for_stats)

        return loss, stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss

    def _compute_alignement_loss(self, align_head, ref_align):
        """Compute loss between 2 partial alignment matrix."""
        # align_head contains value in [0, 1) presenting attn prob,
        # 0 was resulted by the context attention src_pad_mask
        # So, the correspand position in ref_align should also be 0
        # Therefore, clip align_head to > 1e-18 should be bias free.
        align_loss = -align_head.clamp(min=1e-18).log().mul(ref_align).sum()
        align_loss *= self.lambda_align
        return align_loss


class ContinuousLossCompute(LossComputeBase):
    """
    Cosine/NLLvMF Loss Computation.
    """

    def __init__(self, generator, target_embeddings, mtl_generator=None, 
                loss_type='nllvmf', ignore_index=0, approximate_vmf=False, normalization="sents", 
                lambda_coverage=0.0, lambda_mtl=0.0, lambda_align=0.0, lambda_vmf=0.2, beta_map=0.001, second_target_embeddings=None):
        super(ContinuousLossCompute, self).__init__(None, generator)
        self.mtl_generator = mtl_generator
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        self.target_embeddings = target_embeddings
        self.second_target_embeddings = second_target_embeddings
        self.approximate_vmf = approximate_vmf
        self.lambda_coverage = lambda_coverage
        self.lambda_align = lambda_align
        self.lambda_vmf = lambda_vmf
        self.lambda_mtl = lambda_mtl
        self.beta_map = beta_map

        if self.mtl_generator is not None:
            self.mtl_criterion = nn.NLLLoss(ignore_index=0, reduction='sum')
    
    @property
    def padding_idx(self):
        return self.ignore_index
    
    def _psd(self, output_emb, target_emb, mask): # power spherical distribution loss
        kappa = output_emb.norm(p=2, dim=-1)
        emb_size = output_emb.size(-1)
        
        output_emb_unitnorm = torch.nn.functional.normalize(output_emb, p=2, dim=-1)
        target_emb_unitnorm = torch.nn.functional.normalize(target_emb, p=2, dim=-1)
        cosine_loss = (1.0 - (output_emb_unitnorm * target_emb_unitnorm).sum(dim=-1)).masked_select(mask).sum()
        
        beta = (emb_size - 1)/2.
        alpha = beta + kappa
        loss = - kappa * torch.log(1. + (output_emb_unitnorm * target_emb_unitnorm).sum(dim=-1)) + (alpha + beta) * np.log(2.) + torch.lgamma(alpha) - torch.lgamma(alpha + beta) 

        loss = loss.masked_select(mask).sum()
        return loss, cosine_loss, None  # scores
   
    def _nllvmf(self, output_emb, target_emb, mask): #unnormalized output_emb
        #approximation of LogC(m, k)
        def logcmkappox(z, d):
            v = d/2 - 1
            arg = torch.sqrt((v + 1) * (v + 1) + z * z)
            return arg - (v - 1) * torch.log(v - 1 + arg)

        kappa = output_emb.norm(p=2, dim=-1)
        emb_size = output_emb.size(-1)

        output_emb_unitnorm = torch.nn.functional.normalize(output_emb, p=2, dim=-1)
        target_emb_unitnorm = torch.nn.functional.normalize(target_emb, p=2, dim=-1)
        cosine_loss = (1.0 - (output_emb_unitnorm * target_emb_unitnorm).sum(dim=-1)).masked_select(mask).sum()
        
        lambda2 = 0.1
        lambda1 = 0.02
        
        if self.approximate_vmf:
            nll_loss = logcmkappox(kappa, emb_size) + torch.log(1 + kappa) * (self.lambda_vmf - (output_emb_unitnorm * target_emb_unitnorm).sum(dim=-1))
        else:
            nll_loss = - logcmk(kappa) + torch.log(1 + kappa) * (self.lambda_vmf - (output_emb_unitnorm * target_emb_unitnorm).sum(dim=-1))

        loss = nll_loss.masked_select(mask).sum()
        return loss, cosine_loss, None  # scores

    def _cosine(self, output_emb, target_emb, mask):
        batch_size = output_emb.size(1)

        target_emb_unitnorm = torch.nn.functional.normalize(target_emb, p=2, dim=-1)
        output_emb_unitnorm = torch.nn.functional.normalize(output_emb, p=2, dim=-1)

        cosine_loss = (1.0 - (output_emb_unitnorm * target_emb_unitnorm).sum(dim=-1)).masked_select(mask).sum()
        return cosine_loss, cosine_loss 
    
    def _l2(self, output_emb, target_emb, mask, normalize=True):            
        target_emb_unitnorm = torch.nn.functional.normalize(target_emb, p=2, dim=-1)
        output_emb_unitnorm = torch.nn.functional.normalize(output_emb, p=2, dim=-1)
        cosine_loss = (1.0 - (output_emb_unitnorm * target_emb_unitnorm).sum(dim=-1)).masked_select(mask).sum()
        if normalize:
            diff = (output_emb_unitnorm - target_emb_unitnorm)
        else:
            diff = (output_emb - target_emb)
        l2_loss = (diff * diff).sum(dim=-1).masked_select(mask).sum()
        return l2_loss, cosine_loss 

    def _make_shard_state(self, batch, output, range_, attns=None):
        shard_state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0]
        }

        if self.mtl_generator is not None:
            shard_state.update({
                "target_other_task": batch.tgt[range_[0] + 1: range_[1], :, 1],
            })

        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        if self.lambda_align != 0.0:
            # attn_align should be in (batch_size, pad_tgt_size, pad_src_size)
            attn_align = attns.get("align", None)
            # align_idx should be a Tensor in size([N, 3]), N is total number
            # of align src-tgt pair in current batch, each as
            # ['sent_N°_in_batch', 'tgt_id+1', 'src_id'] (check AlignField)
            align_idx = batch.align
            assert attns is not None
            assert attn_align is not None, "lambda_align != 0.0 requires " \
                "alignement attention head"
            assert align_idx is not None, "lambda_align != 0.0 requires " \
                "provide guided alignement"
            pad_tgt_size, batch_size, _ = batch.tgt.size()
            pad_src_size = batch.src[0].size(0)
            align_matrix_size = [batch_size, pad_tgt_size, pad_src_size]
            ref_align = onmt.utils.make_batch_align_matrix(
                align_idx, align_matrix_size, normalize=True)
            # NOTE: tgt-src ref alignement that in range_ of shard
            # (coherent with batch.tgt)
            shard_state.update({
                "align_head": attn_align,
                "ref_align": ref_align[:, range_[0] + 1: range_[1], :]
            })
        return shard_state

    def orthogonalize_mapping(self):
        self.target_embeddings[-1].orthogonalize(self.beta_map)
    
    def _get_loss(self, output_emb, target_emb, gtruth):
        if self.loss_type == 'nllvmf':
            loss, cosine_loss, scores = self._nllvmf(output_emb, target_emb, gtruth.ne(self.padding_idx))
        elif self.loss_type == 'cosine':
            loss, cosine_loss = self._cosine(output_emb, target_emb, gtruth.ne(self.padding_idx))
        elif self.loss_type == 'l2':
            loss, cosine_loss = self._l2(output_emb, target_emb, gtruth.ne(self.padding_idx), normalize=False)
        elif self.loss_type == 'psd':
            loss, cosine_loss, scores = self._psd(output_emb, target_emb, gtruth.ne(self.padding_idx))
        return loss, cosine_loss

    def _compute_loss(self, batch, output, target, target_other_task=None, std_attn=None,
                      coverage_attn=None, align_head=None, ref_align=None):

        bottled_output = self._bottle(output)
        output_emb = self.generator(bottled_output)

        gtruth = target.view(-1)
        target_emb = self.target_embeddings(gtruth)
        # reverse_emb = self.target_embeddings[-1].reverse(target_emb)
        # original_emb = self.target_embeddings[0](gtruth)

        # print(target_emb[0], reverse_emb[0], original_emb[0])
        # print(reverse_emb[0] - original_emb[0])
        # # input(target_emb[0])
        # input()
        loss, cosine_loss = self._get_loss(output_emb, target_emb, gtruth)

        if self.second_target_embeddings is not None:
            sec_target_emb = self.second_target_embeddings(gtruth) # the indices of the two vocabularies correspond to (approximate) translations
            sec_loss, sec_cosine_loss = self._get_loss(output_emb, sec_target_emb, gtruth)

            loss = (loss + sec_loss) / 2
            cosine_loss = (cosine_loss + sec_cosine_loss) / 2
        
        # loss = self.criterion(scores, gtruth)

        other_task_loss_for_stats = None
        other_gtruth_for_stats = None
        other_scores_for_stats = None
        if self.mtl_generator is not None:
            other_gtruth = target_other_task.view(-1)
            other_task_logprob = self.mtl_generator(bottled_output)
            other_task_loss = self.lambda_mtl * self.mtl_criterion(other_task_logprob, other_gtruth)
            loss += other_task_loss
            
            other_task_loss_for_stats = other_task_loss.clone()
            other_gtruth_for_stats = other_gtruth.clone() 
            other_scores_for_stats = other_task_logprob.clone()

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss

        if self.lambda_align != 0.0:
            if align_head.dtype != loss.dtype:  # Fix FP16
                align_head = align_head.to(loss.dtype)
            if ref_align.dtype != loss.dtype:
                ref_align = ref_align.to(loss.dtype)
            align_loss = self._compute_alignement_loss(
                align_head=align_head, ref_align=ref_align)
            loss += align_loss

        # stats = self._stats(loss.clone(), cosine_loss.clone(), gtruth)
        stats = self._stats(loss.clone(), cosine_loss.clone(), gtruth, other_task_loss_for_stats, other_scores_for_stats, other_gtruth_for_stats)
        return loss, stats
    
    def _stats(self, loss, cosine_loss, target, other_loss=None, other_scores=None, other_target=None):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        # pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        # num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        # return onmt.utils.Statistics(loss.item(), num_non_padding, cosine_loss.item())

        num_correct_other = 0
        if other_loss is not None:
            pred_other = other_scores.max(1)[1]
            num_correct_other = pred_other.eq(other_target).masked_select(non_padding).sum().item()

        return onmt.utils.Statistics(loss.item(), num_non_padding, cosine_loss.item(), num_correct_other)

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss

    def _compute_alignement_loss(self, align_head, ref_align):
        """Compute loss between 2 partial alignment matrix."""
        # align_head contains value in [0, 1) presenting attn prob,
        # 0 was resulted by the context attention src_pad_mask
        # So, the correspand position in ref_align should also be 0
        # Therefore, clip align_head to > 1e-18 should be bias free.
        align_loss = -align_head.clamp(min=1e-18).log().mul(ref_align).sum()
        align_loss *= self.lambda_align
        return align_loss


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        if len(variables) > 0:
            inputs, grads = zip(*variables)
            torch.autograd.backward(inputs, grads)
