from collections import Counter
from Util.DataLoader import load_dataset

import numpy as np
import pickle


def create_vocab(path):
    wd_counter, lbl_counter = Counter(), Counter()
    insts = load_dataset(path)
    for inst in insts:
        wd_counter.update(inst.words)
        lbl_counter[inst.label] += 1

    return WordVocab(wd_counter, lbl_counter)


class WordVocab(object):

    def __init__(self, wd_counter, lbl_counter):

        self._min_count = 5
        self._UNK = 0

        self._wd2freq = {wd: count for wd, count in wd_counter.items() if count > self._min_count}

        # 词索引映射
        self._wd2idx = {wd: idx+1 for idx, wd in enumerate(self._wd2freq.keys())}
        self._wd2idx['<unk>'] = self._UNK

        # 索引词映射
        self._idx2wd = {idx: wd for wd, idx in self._wd2idx.items()}

        self._extwd2idx = {}
        self._extidx2wd = {}

        self._lbl2idx = {lbl: idx for idx, lbl in enumerate(lbl_counter.keys())}
        self._idx2lbl = {idx: lbl for lbl, idx in self._lbl2idx}

    # 获取预训练的词向量
    def get_embedding_weight(self, path):
        vec_tabs = {}
        vec_size = 0
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = line.strip().split()
                wd, vector = tokens[0], tokens[1:]
                vec_size = len(vector)
                vec_tabs[wd] = np.asarray(vector, dtype=np.float32)

        self._extwd2idx = {wd: idx+1 for idx, wd in enumerate(vec_tabs)}
        self._extwd2idx['<unk>'] = self._UNK
        self._idx2extword = {idx: wd for wd, idx in self._extwd2idx.items()}

        vocab_size = len(self._extwd2idx)
        embedding_weights = np.zeros((vocab_size, vec_size), dtype=np.float32)
        for i, wd in self._idx2extword.items():
            if i != self._UNK:
                embedding_weights[i] = vec_tabs[wd]
        # embedding_weights[self._UNK] = np.random.uniform(-0.25, 0.25, vec_size)
        embedding_weights[self._UNK] = np.mean(embedding_weights, 0) / np.std(embedding_weights)
        return embedding_weights


    def word2idx(self, wds):
        if isinstance(wds, list):
            return [self._wd2idx.get(wd, self._UNK) for wd in wds]
        else:
            return self._wd2idx.get(wds, self._UNK)

    def idx2word(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2wd.get(idx) for idx in idxs]
        else:
            return self._idx2wd.get(idxs)

    def label2index(self, lbls):
        if isinstance(lbls, list):
            return [self._lbl2idx.get(lbl, -1) for lbl in lbls]
        else:
            return self._lbl2idx.get(lbls, -1)

    def index2label(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2lbl.get(idx) for idx in idxs]
        else:
            return self._idx2lbl.get(idxs)

    def extwd2idx(self, wds):
        if isinstance(wds, list):
            return [self._extwd2idx.get(wd, self._UNK) for wd in wds]
        else:
            return self._extwd2idx.get(wds, self._UNK)

    def idx2extwd(self, idxs):
        if isinstance(idxs, list):
            return [self._extwd2idx.get(idx, '<unk>') for idx in idxs]
        else:
            return self._extwd2idx.get(idxs, '<unk>')

    @property
    def vocab_size(self):
        return len(self._wd2idx)

    @property
    def label_size(self):
        return len(self._lbl2idx)

    def save(self,path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)