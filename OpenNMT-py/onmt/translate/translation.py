""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
from onmt.inputters.text_dataset import TextMultiField
from onmt.utils.alignment import build_align_pharaoh

from collections import Iterable


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False, phrase_table="", multi_task=False, use_new_target_vocab=False):
        self.data = data
        self.fields = fields
        self._has_text_src = isinstance(
            dict(self.fields)["src"], TextMultiField)
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt
        self.multi_task = multi_task
        self.use_new_target_vocab=use_new_target_vocab

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn, new_vocab=False, length=-1):
        field_dict = dict(self.fields)
        # if new_vocab:
        #     print("new_tgt" in field_dict)
        if new_vocab and "new_tgt" in field_dict:
            tgt_field = field_dict['new_tgt']
            vocab = tgt_field.vocab
        else:
            tgt_field = field_dict["tgt"].base_field
            vocab = tgt_field.vocab
            
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        
        if length > -1:
            tokens = tokens[:length]

        if self.replace_unk and attn is not None and src is not None and len(src_raw) > 0:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    _, max_index = attn[i][:len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens
    
    def _build_sec_target_tokens(self, sec_pred, base_tokens):
        tgt_fields = dict(self.fields)["tgt"].fields
        if len(tgt_fields) <= 1:
            return sec_pred
        
        tgt_pos_vocab = tgt_fields[1][1].vocab
        tokens_all = []
        # print(sec_pred)
        # print(base_tokens)
        # input()
        for i in range(len(base_tokens)): # only predict till the length of actual words
            toks = sec_pred[i]
            tokens = [] 
            # print(toks)
            if toks.dim() > 0:
                for tok in toks:
                    tokens.append(tgt_pos_vocab.itos[tok])
                    # if tokens[-1] == tgt_fields[1][1].eos_token:
                    #     tokens = tokens[:-1]
                    #     break
            else: tokens = tgt_pos_vocab.itos[toks]
            tokens_all.append(tokens)
        return tokens_all

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, new_preds, new_tgt_preds, pred_score, attn, align, gold_score, sec_preds, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["new_predictions"],
                        translation_batch["new_tgt_predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["alignment"],
                        translation_batch["gold_score"],
                        translation_batch['sec_predictions'],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        if not any(align):  # when align is a empty nested list
            align = [None] * batch_size

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None
        tgt = batch.tgt[:, :, 0].index_select(1, perm) \
            if self.has_tgt else None
        
        if self.multi_task:
            sec_tgt = batch.tgt[:, :, 1].index_select(1, perm) \
                if self.has_tgt else None

        translations = []
        for b in range(batch_size):
            if self._has_text_src:
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src[0]
            else:
                src_vocab = None
                src_raw = None
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]

            new_pred_sents = None
            if self.use_new_target_vocab:
                new_pred_sents = [self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    new_preds[b][n], attn[b][n], new_vocab=True)
                    for n in range(self.n_best)]

            gold_sent = None
            new_tgt_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)
                
                new_tgt_sent = None
                if new_tgt_preds is not None:
                    new_tgt_sent = self._build_target_tokens(
                        src[:, b] if src is not None else None,
                        src_vocab, src_raw,
                        new_tgt_preds[b], None, new_vocab=True, length=len(gold_sent))

            gold_sec_sent = None
            sec_pred_sents = None

            if self.multi_task:
                if len(sec_preds[b]) > 0:
                    sec_pred_sents = [self._build_sec_target_tokens(sec_preds[b][n], pred_sents[n]) for n in range(self.n_best)]
                    # sec_pred_sents = self._build_sec_target_tokens(sec_input, pred_sents[0])
                    if sec_tgt is not None:
                        # print(sec_tgt[1:, b].size())
                        # input()
                        gold_sec_sent = self._build_sec_target_tokens(sec_tgt[1:, b].unsqueeze(1), gold_sent)
                    # print(sec_pred_sents)
                    # input()

            # print(pred_sents)
            # print(new_tgt_sent)
            # print(gold_sent)
            # input()
            translation = Translation(
                src[:, b] if src is not None else None,
                src_raw, pred_sents, new_pred_sents, new_tgt_sent, attn[b], pred_score[b],
                gold_sent, gold_score[b], align[b], gold_sec_sent, sec_pred_sents,
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
        word_aligns (List[FloatTensor]): Words Alignment distribution for
            each translation.
    """

    __slots__ = ["src", "src_raw", "pred_sents", "new_pred_sents", "new_tgt_sent", "attns", "pred_scores",
                 "gold_sent", "gold_score", "word_aligns", "gold_sec_sent", "sec_pred_sents"]

    def __init__(self, src, src_raw, pred_sents, new_pred_sents, new_tgt_sent,
                 attn, pred_scores, tgt_sent, gold_score, word_aligns, gold_sec_sent, sec_pred_sents):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.new_pred_sents = new_pred_sents
        self.new_tgt_sent = new_tgt_sent
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        self.word_aligns = word_aligns
        self.sec_pred_sents = sec_pred_sents
        self.gold_sec_sent = gold_sec_sent

    def log(self, sent_number):
        """
        Log translation.
        """

        msg = ['\nSENT {}: {}\n'.format(sent_number, self.src_raw)]
        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))

        if self.new_pred_sents is not None:
            new_best_pred = self.new_pred_sents[0]
            new_pred_sent = ' '.join(new_best_pred)
            msg.append('NEW_PRED {}: {}\n'.format(sent_number, new_pred_sent))

        if self.new_tgt_sent is not None:
            new_tgt_sent = ' '.join(self.new_tgt_sent)
            msg.append('NEW_TGT_PRED {}: {}\n'.format(sent_number, new_tgt_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.word_aligns is not None:
            pred_align = self.word_aligns[0]
            pred_align_pharaoh = build_align_pharaoh(pred_align)
            pred_align_sent = ' '.join(pred_align_pharaoh)
            msg.append("ALIGN: {}\n".format(pred_align_sent))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, " ".join(sent)))

        if self.sec_pred_sents is not None:
            pred_sent = "\n".join([" ".join(x) for x in self.sec_pred_sents])
            msg.append('PRED TAGS {}: {}\n'.format(sent_number, pred_sent))
        
        if self.gold_sec_sent is not None:
            gold_sec_sent = " ".join(["/".join(x) for x in self.gold_sec_sent])
            msg.append('\nGOLD TAGS {}: {}\n'.format(sent_number, gold_sec_sent))

        return "".join(msg)
