""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
from onmt.inputters.text_dataset import TextMultiField
from onmt.utils.alignment import build_align_pharaoh

import re

def isNumeral(s): 
    x = re.findall(r'[0-9]+[\.,][0-9]+',s)
    if len(x) >= 1:
        return True
    return False

class TranslationBuilder2(object):
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
                 has_tgt=False, phrase_table="", multi_task=False):
        self.data = data
        self.fields = fields
        self._has_text_src = isinstance(
            dict(self.fields)["src"], TextMultiField)
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.lookup_dict = None
        if phrase_table.endswith(".dictstr"):
            self.lookup_dict = eval(open(phrase_table).read())
            print("loaded the lookup dict", len(self.lookup_dict))
        else:
            self.lookup_dict = {}
            with open(phrase_table) as f:
                for l in f:
                    k, v = l.strip().split("|||")
                    self.lookup_dict[k] = v
                    
        self.has_tgt = has_tgt
        self.multi_task = multi_task

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field
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
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                _, max_index = attn[i][:len(src_raw)].max(0)
                if tokens[i] == tgt_field.unk_token:
                    
                    _, max_index = attn[i][:len(src_raw)].topk(2, 0)
                    dist = (max_index[0] - i) * 1.0 / len(src) 
                    
                    if src_raw[max_index[0]] in [",", "."] or dist > 0.5 or dist < -0.5:
                        mI = max_index[1].item()
                        # input("Hello")
                    else:
                        mI = max_index[0].item()

                    while mI > 0 and src_raw[mI - 1].endswith("@@"):
                        mI -= 1
                    
                    t = src_raw[mI]
                    if src_raw[mI].endswith("@@"):
                        k = mI + 1
                        while k < len(src_raw) and src_raw[k].endswith("@@"):
                            t += " " + src_raw[k]
                            k += 1
                        if k < len(src_raw):
                            t += " " + src_raw[k]

                    if self.lookup_dict is not None:
                        t = t.replace("@@", "").replace(" ", "")
                        if t in self.lookup_dict:
                            print(t, self.lookup_dict[t])
                            t = self.lookup_dict[t]

                    tokens[i] = t
                                     
                if isNumeral(tokens[i]):
                    # print (tokens[i],)
                    try:
                        mI = max_index[0]
                    except:
                        mI = max_index.item()

                    while mI > 0 and src_raw[mI - 1].endswith("@@"):
                        mI -= 1
                    t = src_raw[mI]

                    if src_raw[mI].endswith("@@"):
                        t = src_raw[mI]
                        k = mI + 1
                        while k < len(src) and src_raw[k].endswith("@@"):
                            t += " " + src_raw[k]
                            k+=1

                        if k < len(src):
                            t += " " + src_raw[k]

                        tokens[i] = t
                    if i + 1 < len(tokens) and tokens[i + 1] in ["a.m.", "p.m."]:
                        tokens[i] = t.replace("."," : ")
                    else:
                        t = t.replace(".","]")
                        t = t.replace(",",".")
                        tokens[i] = t.replace("]", ",")

                if tokens[i] == "%":
                    try:
                        mI = max_index[0]
                    except:
                        mI = max_index.item()
                    if src_raw[mI] == "Prozent":
                        tokens[i] = "percent"
        return tokens
    
    def _build_sec_target_tokens(self, sec_pred, base_tokens):
        tgt_fields = dict(self.fields)["tgt"].fields
        if len(tgt_fields) <= 1:
            return sec_pred
        
        tgt_pos_vocab = tgt_fields[1][1].vocab
        tokens_all = []
        print(sec_pred)
        print(base_tokens)
        input()
        for i in range(len(base_tokens)): # only predict till the length of actual words
            toks = sec_pred[i]
            tokens = [] 
            for tok in toks:
                tokens.append(tgt_pos_vocab.itos[tok])
                # if tokens[-1] == tgt_fields[1][1].eos_token:
                #     tokens = tokens[:-1]
                #     break
            tokens_all.append(tokens)
        return tokens_all

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, align, gold_score, sec_preds, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
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
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)
            gold_sec_sent = None
            sec_pred_sents = None

            if self.multi_task:
                sec_input = sec_preds[b]
                if len(sec_input) > 0:
                    sec_input = sec_input[0]
                    sec_pred_sents = self._build_sec_target_tokens(sec_input, pred_sents[0])
                    if sec_tgt is not None:
                        # print(sec_tgt[1:, b].size())
                        # input()
                        gold_sec_sent = self._build_sec_target_tokens(sec_tgt[1:, b].unsqueeze(1), gold_sent)
                    # print(sec_pred_sents)
                    # input()

            translation = Translation2(
                src[:, b] if src is not None else None,
                src_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b], align[b], gold_sec_sent, sec_pred_sents,
            )
            translations.append(translation)

        return translations


class Translation2(object):
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

    __slots__ = ["src", "src_raw", "pred_sents", "attns", "pred_scores",
                 "gold_sent", "gold_score", "word_aligns", "gold_sec_sent", "sec_pred_sents"]

    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score, word_aligns, gold_sec_sent, sec_pred_sents):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
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
                msg.append("[{:.4f}] {}\n".format(score, sent))

        if self.sec_pred_sents is not None:
            pred_sent = " ".join(["/".join(x) for x in self.sec_pred_sents])
            msg.append('PRED TAGS {}: {}\n'.format(sent_number, pred_sent))
        
        if self.gold_sec_sent is not None:
            gold_sec_sent = " ".join(["/".join(x) for x in self.gold_sec_sent])
            msg.append('\nGOLD TAGS {}: {}\n'.format(sent_number, gold_sec_sent))

        return "".join(msg)
