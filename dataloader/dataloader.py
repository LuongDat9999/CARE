import pickle

from tqdm import tqdm

from utils.helper import *
from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
import numpy as np

dis2idx = np.zeros((4000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class dataprocess(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx][0]
        ner_labels = self.data[idx][1]
        rc_labels = self.data[idx][2]
        bert_len = self.data[idx][3]
        _dist_inputs = self.data[idx][4]
        return (words, ner_labels, rc_labels, bert_len, _dist_inputs)

def map_origin_word_to_bert(words, tokenizer):
    bep_dict = {}
    current_idx = 0
    for word_idx, word in enumerate(words):
        bert_word = tokenizer.tokenize(word)
        word_len = len(bert_word)
        bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
        current_idx = current_idx + word_len
    return bep_dict

def ner_label_transform(ner_label, word_to_bert):
    new_ner_labels = []

    for i in range(0, len(ner_label), 3):
        # +1 for [CLS]
        sta = word_to_bert[ner_label[i]][0] + 1
        end = word_to_bert[ner_label[i + 1]][0] + 1
        new_ner_labels += [sta, end, ner_label[i + 2]]

    return new_ner_labels

def rc_label_transform(rc_label, word_to_bert):
    new_rc_labels = []

    for i in range(0, len(rc_label), 3):
        # +1 for [CLS]
        e1 = word_to_bert[rc_label[i]][0] + 1
        e2 = word_to_bert[rc_label[i + 1]][0] + 1
        new_rc_labels += [e1, e2, rc_label[i + 2]]

    return new_rc_labels

def truncate(max_seq_len, words, ner_labels, rc_labels):
    truncated_words = words[:max_seq_len]
    truncated_ner_labels = []
    truncated_rc_labels = []
    for i in range(0, len(ner_labels), 3):
        if ner_labels[i] < max_seq_len and ner_labels[i+1] < max_seq_len:
            truncated_ner_labels += [ner_labels[i], ner_labels[i+1], ner_labels[i+2]]

    for i in range(0, len(rc_labels), 3):
        if rc_labels[i] < max_seq_len and rc_labels[i+1] < max_seq_len:
            truncated_rc_labels += [rc_labels[i], rc_labels[i+1], rc_labels[i+2]]

    return truncated_words, truncated_ner_labels, truncated_rc_labels

def nyt_and_webnlg_preprocess(data, tokenizer):
    processed = []
    for dic in data:
        text = dic['text']
        text = text.split(" ")
        ner_labels = []
        rc_labels = []
        trips = dic['triple_list']
        for trip in trips:
            subj = text.index(trip[0])
            obj = text.index(trip[2])
            rel = trip[1]
            if subj not in ner_labels:
                ner_labels += [subj, subj, "None"]
            if obj not in ner_labels:
                ner_labels += [obj, obj, "None"]

            rc_labels += [subj, obj, rel]

        processed += [(text,ner_labels,rc_labels)]
    res = []
    for x in tqdm(processed):
        words = x[0]
        ner_labels = x[1]
        rc_labels = x[2]

        if len(words) > 128:
            words, ner_labels, rc_labels = truncate(128, words, ner_labels, rc_labels)

        sent_str = ' '.join(words)
        bert_words = tokenizer.tokenize(sent_str)
        bert_len = len(bert_words) + 2
        # bert_len = original sentence + [CLS] and [SEP]

        word_to_bep = map_origin_word_to_bert(words, tokenizer)
        ner_labels = ner_label_transform(ner_labels, word_to_bep)
        rc_labels = rc_label_transform(rc_labels, word_to_bep)

        dist_inputs = torch.zeros((bert_len, bert_len), dtype=torch.long)
        for k in range(bert_len):
            dist_inputs[k, :] += k
            dist_inputs[:, k] -= k

        for i in range(bert_len):
            for j in range(bert_len):
                if dist_inputs[i, j] < 0:
                    dist_inputs[i, j] = dis2idx[-dist_inputs[i, j]] + 9
                else:
                    dist_inputs[i, j] = dis2idx[dist_inputs[i, j]]
        dist_inputs[dist_inputs == 0] = 19
        res.append((words, ner_labels, rc_labels, bert_len, dist_inputs))

    return res

def dataloader(args, ner2idx, rel2idx):
    path = "/Users/luongdat/Desktop/Pycharm/Care/data/WEBNLG"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    train_data = json_load(path, 'train_triples.json')
    test_data = json_load(path, 'test_triples.json')
    dev_data = json_load(path, 'dev_triples.json')

    train_dataset = load_dataset(args, tokenizer, train_data, "train")
    dev_dataset = load_dataset(args, tokenizer, dev_data, "dev")
    test_dataset = load_dataset(args, tokenizer, test_data, "test")

    collate_fn = collater(ner2idx, rel2idx)

    train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, collate_fn=collate_fn)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn)
    dev_batch = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn)

    return train_batch, test_batch, dev_batch

def load_dataset(args, tokenizer, data, split):
    if not os.path.exists('pkl'):
        os.mkdir('pkl')

    file = 'pkl/' + args.data + '_' + split + '.pkl'
    if os.path.exists(file):
        os.remove(file)

    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        data = nyt_and_webnlg_preprocess(data, tokenizer)
        dataset = dataprocess(data)
        with open(file, 'wb') as f:
            pickle.dump(dataset, f)
    return dataset
