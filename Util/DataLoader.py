import numpy as np
import torch


class Instance(object):
    def __init__(self, words, lable):
        self.words = words
        self.label = lable


def load_dataset(path):
    Inst_list = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            lbl, raw_data = line.strip().split('|||')
            words = raw_data.split()
            Inst_list.append(Instance(words, lbl))
    np.random.shuffle(Inst_list)
    return Inst_list

#获取批量数据
def get_batch(dataset, batch_size, shuffle=True):
    #batch数量
    nb_batch = int(np.ceil(len(dataset) / batch_size))

    if shuffle:
        np.random.shuffle(dataset)

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size : (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)
        yield batch_data

#batch数量化，提取数据和标签
def batch_variable(batch_data, wd_vocab, device=torch.device('cuda')):
    #求最长的序列长度
    batch_size = len(batch_data)
    max_seq_len = 0
    for i in range(batch_size):
        if len(batch_data[i].words) > max_seq_len:
            max_seq_len = len(batch_data[i].words)

    #max_seq_len = max([len(inst.words) for inst in batch_data])

    wd_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    lbl_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    mask = torch.zeros(batch_size, max_seq_len).to(device)
    for i, inst in enumerate(batch_data):
        seq_len = len(inst.words)
        wd_idx[i, :seq_len] = torch.tensor(wd_vocab.extwd2idx(inst.words))
        lbl_idx[i] = torch.tensor(wd_vocab.label2index(inst.label))
        mask[i, :seq_len].fill_(1)

    return wd_idx, lbl_idx, mask

