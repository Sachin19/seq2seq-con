import torch
import codecs

import numpy as np

class Dict(object):
    def __init__(self, data=None, lower=False, special_embeddings=None):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.embeddings = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data, special_embeddings)
                # print self.embeddings.keys()

    def size(self):
        if len(self.embeddings) == 0:
            return len(self.idxToLabel)
        else:
            return len(self.embeddings)

    # Load entries from a file.
    def loadFile(self, filename):
        i = len(self.idxToLabel)
        # print i
        for line in open(filename):
            fields = line.split()
            if len(fields) > 2:
                idx = int(fields[-1])
                label = ' '.join(fields[:-1])
            elif len(fields) == 2:
                label = fields[0]
                idx = int(fields[1])
            else:
                label = fields[0]
                idx = i
                i += 1
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with codecs.open(filename, 'w', "utf-8") as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        # print label, idx
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels, special_embeddings=None):
        for i, label in enumerate(labels):
            self.addSpecial(label)
            if special_embeddings is not None:
                self.add_embedding(label, special_embeddings[i])
    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def add_embedding(self, word, emb, unk=None, normalize=True):
        eps = 1e-6
        if word in self.labelToIdx:
            if normalize:
                emb = emb/(np.linalg.norm(emb)+eps)
            self.embeddings[self.labelToIdx[word]] = emb
        else:
            self.embeddings[self.labelToIdx[unk]] += emb

    def average_unk(self, unk, n, normalize=True):
        self.embeddings[self.labelToIdx[unk]] /= n
        if normalize:
            self.embeddings[self.labelToIdx[unk]] = self.embeddings[self.labelToIdx[unk]]/np.linalg.norm(self.embeddings[self.labelToIdx[unk]])

    def convert_embeddings_to_torch(self):
        embeddings_tensor = np.zeros((self.size(), 200))
        for k, v in self.embeddings.items():
            embeddings_tensor[k] = v
        self.embeddings = torch.Tensor(embeddings_tensor)

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size, target=False):
        # if size >= self.size():
        #     return self, self.size()

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        # print (self.frequencies)
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])
            if target:
                newDict.add_embedding(self.idxToLabel[i], self.embeddings[i])

        c=0
        for i in idx:
            if target:
                if i in self.embeddings:
                    newDict.add(self.idxToLabel[i])
                    newDict.add_embedding(self.idxToLabel[i], self.embeddings[i])
                    c+=1
            else:
                newDict.add(self.idxToLabel[i])
                c+=1
            if c >= size:
                break
        return newDict, c

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]
        unky = False
        if unk in vec:
            unky = True

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec), unky

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []
        for i in idx:
            i
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels
