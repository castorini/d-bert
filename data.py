from collections import Counter
import glob
import os

from tqdm import tqdm
import torch
import pandas as pd
import torch.utils.data as tud


def load_clean_aol(folder, use_tqdm=True):
    ct_names = glob.glob(os.path.join(folder, 'clean-*.txt'))
    dfs = []
    if use_tqdm: ct_names = tqdm(ct_names)
    for ct_name in ct_names:
        idx = int(ct_name.split('-')[-1].split('.txt')[0])
        df = pd.read_csv(ct_name, sep='\t', quoting=3, error_bad_lines=False, keep_default_na=False).astype(str)
        dfs.append((idx, df))
    dfs = [x[1] for x in sorted(dfs, key=lambda x: x[0])]
    return dfs


class SingleSentenceDataset(tud.Dataset):

    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __getitem__(self, idx):
        return idx, self.sentences[idx]

    def __len__(self):
        return len(self.sentences)

    @classmethod
    def splits(cls, folder, train_file='train.tsv', dev_file='dev.tsv', 
            test_file='test.tsv', column='sentence',
            filter_label=None, label_column='label', **kwargs):
        dfs = [os.path.join(folder, x) for x in (train_file, dev_file, test_file)]
        dfs = [pd.read_csv(df, sep='\t', quoting=3, error_bad_lines=True, keep_default_na=False).astype(str) for df in dfs]
        if filter_label is not None:
            for idx in (0, 1): dfs[idx] = dfs[idx][dfs[idx][label_column] == filter_label]
        sentences_lst = [list(df[column]) for df in dfs]
        return [cls(x) for x in sentences_lst]

    @classmethod
    def pair_splits(cls, folder, train_file='train.tsv', dev_file='dev.tsv', 
            test_file='test.tsv', column1='question1', column2='question2', 
            filter_label='0', label_column='is_duplicate', **kwargs):
        dfs = [os.path.join(folder, x) for x in (train_file, dev_file, test_file)]
        dfs = [pd.read_csv(df, sep='\t', quoting=3, error_bad_lines=True, keep_default_na=False).astype(str) for df in dfs]
        if filter_label is not None:
            for idx in (0, 1): dfs[idx] = dfs[idx][dfs[idx][label_column] == filter_label]
        sentences_lst = [[f' \t '.join((x, y)) for x, y in zip(df[column1], df[column2])] for df in dfs]
        return list(map(cls, sentences_lst))


class Dictionary(object):

    def __init__(self, eos=True, pad=True, sos=True):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        if eos: self.add_word('<eos>')
        if pad: self.add_word('<pad>')
        if sos: self.add_word('<sos>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def sent2idx(self, sentences, **kwargs):
        tokens_lst, tokens_mask = tokenize_batch(sentences, **kwargs)
        tokens_lst = [[self.word2idx[x] for x in sent] for sent in tokens_lst]
        return tokens_lst, tokens_mask


def tokenize_batch(sentences,
                   tokenize_fn=list,
                   eos='<eos>',
                   pad='<pad>',
                   sos=None,
                   max_len=100,
                   pad_to_max=False):
    eos_append = [eos] if eos else []
    sos_prepend = [sos] if sos else []
    tokens_lst = [sos_prepend + tokenize_fn(x) + eos_append for x in sentences]
    tokens_mask = [[1] * len(x) for x in tokens_lst]
    max_len = max_len if pad_to_max else min(max(map(len, tokens_lst)), max_len)
    tokens_lst = [x[:max_len] for x in tokens_lst]
    tokens_mask = [x[:max_len] for x in tokens_mask]
    tokens_lst = [x + ['<pad>'] * (max_len - len(x)) for x in tokens_lst]
    tokens_mask = [x + [0] * (max_len - len(x)) for x in tokens_mask]
    return tokens_lst, tokens_mask


def tokens_reduce(loss, tokens_lst):
    mask = []
    for tokens in tokens_lst:
        mask.append([int(x != '<pad>') for x in tokens])
    mask = torch.Tensor(mask)
    loss = loss * mask
    return loss.sum() / mask.sum()
