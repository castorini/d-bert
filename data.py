from collections import Counter
import glob
import os
import random

from tqdm import tqdm
import pandas as pd
import torch.nn as nn
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


class AOLQueryDataset(tud.Dataset):

    def __init__(self, queries, users):
        super().__init__()
        self.queries = queries
        self.users = users
        self.user_set = set(self.users)
        self._count()

    def _count(self):
        self.user_counter = Counter()
        for q, u in zip(self.queries, self.users):
            self.user_counter[u] += 1

    def count_queries_from(self, user_set):
        return sum(self.user_counter.get(x) for x in user_set)

    def filter(self, user_set):
        new_users = []
        new_queries = []
        for user, query in zip(self.users, self.queries):
            if user in user_set:
                new_users.append(user)
                new_queries.append(query)
        self.users = new_users
        self.queries = new_queries
        self.user_set = set(new_users)
        self._count()

    def __getitem__(self, idx):
        return self.users[idx], self.queries[idx]

    def __len__(self):
        return len(self.queries)

    def sample_split(self, split_prob):
        users = ([], [])
        queries = ([], [])
        for u, q in zip(self.users, self.queries):
            idx = int(random.random() < split_prob)
            users[idx].append(u)
            queries[idx].append(q)
        return [AOLQueryDataset(queries[idx], users[idx]) for idx in range(2)]

    @classmethod
    def splits(cls, folder, val_prob=0.05, test_prob=0.05, use_tqdm=True):
        dfs = load_clean_aol(folder, use_tqdm=use_tqdm)
        df = pd.concat(dfs)
        base_ds = cls(list(df['Query']), list(df['AnonID']))
        training_ds, val_ds  = base_ds.sample_split(val_prob + test_prob)
        dev_ds, test_ds = val_ds.sample_split(test_prob / (val_prob + test_prob))
        return training_ds, dev_ds, test_ds

    @classmethod
    def splits_six(cls,
                   folder,
                   min_test_size=1000000,
                   val_prob=0.02,
                   use_tqdm=True,
                   max_len=128):
        dfs = load_clean_aol(folder, use_tqdm=use_tqdm)
        training_df = pd.concat(dfs[:6])

        training_ds = cls(list(training_df['Query']), list(training_df['AnonID']))
        test_df = pd.concat(dfs[6:])

        test_ds = cls(list(test_df['Query']), list(test_df['AnonID']))
        disjoint_users = list(test_ds.user_set - training_ds.user_set)
        random.shuffle(disjoint_users)
        test_user_set = set()
        for user in disjoint_users:
            test_user_set.add(user)
            n_queries = test_ds.count_queries_from(test_user_set)
            if n_queries > min_test_size:
                break
        test_ds.filter(test_user_set)
        training_ds, dev_ds = training_ds.sample_split(val_prob)
        return training_ds, dev_ds, test_ds


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
