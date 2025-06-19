import json
import os
import torch

def json_load(path, name):
    file = path + "/" + name
    with open(file,'r',encoding='utf-8') as f:
        return json.load(f)
def json_loads(path, name):
    file = path + "/" + name
    data = []
    with open(file,'r') as f:
        for line in f:
            a = json.loads(line)
            data.append(a)
    return data
def gen_ner_labels(ner_list,l, ner2idx):
    labels = torch.FloatTensor(l,l,len(ner2idx)).fill_(0)
    for i in range(0,len(ner_list),3):
        head = ner_list[i]
        tail = ner_list[i+1]
        n = ner2idx[ner_list[i+2]]
        if head >= l or tail >= l:
            continue
        labels[head][tail][n] = 1

    return labels
def gen_rc_labels(rc_list, l, rel2idx):
    labels = torch.FloatTensor(l, l, len(rel2idx)).fill_(0)
    for i in range(0, len(rc_list), 3):
        e1 = rc_list[i]
        e2 = rc_list[i + 1]
        r = rc_list[i + 2]
        if e1 >= l or e2 >= l:
            continue
        labels[e1][e2][rel2idx[r]] = 1

    return labels
def mask_to_tensor(len_list, batch_size):
    token_len = max(len_list)
    tokens = torch.LongTensor(token_len, batch_size).fill_(0)
    for i, s in enumerate(len_list):
        tokens[:s, i] = 1

    return tokens
def sent_to_tensor(len_list, batch_size, sent_list):
    token_len = max(len_list)
    tokens = torch.LongTensor(token_len, batch_size).fill_(0)
    for i, s in enumerate(len_list):
        tokens[:s, i] = torch.tensor(sent_list[i], dtype=torch.long)

    return tokens
def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor
def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked
class collater():
    def __init__(self, ner2idx, rel2idx):
        self.ner2idx = ner2idx
        self.rel2idx = rel2idx

    def __call__(self, data):
        words = [item[0] for item in data]
        ner_labels = [item[1] for item in data]
        rc_labels = [item[2] for item in data]
        bert_len = [item[3] for item in data]
        dist = [item[4] for item in data]

        max_len = max(bert_len)
        batch_size = len(words)
        d = padded_stack(dist)

        ner_labels = [gen_ner_labels(ners, max_len, self.ner2idx) for ners in ner_labels]
        rc_labels = [gen_rc_labels(rcs,max_len, self.rel2idx) for rcs in rc_labels]

        ner_labels = torch.stack(ner_labels, dim=2)
        rc_labels = torch.stack(rc_labels, dim=2)
        mask = mask_to_tensor(bert_len, batch_size)

        return [words,ner_labels,rc_labels,mask,d]
class save_results(object):
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)

        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)

    def save(self, info):
        with open(self.filename, 'a') as out:
            print(info, file=out)
def map_to_idx(text,word2idx):
    ids = [word2idx[t] if t in word2idx else 1 for t in text]
    return ids
def is_overlap(ent1, ent2):
    if ent1[1] < ent2[0] or ent2[1] < ent1[0]:
        return False
    return True