from torchtext.data import Field, TabularDataset, Iterator
from torchtext.vocab import Vectors
import torch

from .base import allennlp_tokenize, basic_tokenize, uniform_unk_init, space_tokenize, \
    bert_tokenize, gpt2_tokenize


_REGISTRY = {}


class RegisteredDataset(TabularDataset):

    def __init_subclass__(cls, name):
        _REGISTRY[name.lower()] = cls


def list_field_mappings(field_tgt, vocab):
    mapping = []
    for word in vocab.stoi:
        if word not in field_tgt.vocab.stoi:
            continue
        mapping.append((vocab.stoi[word], field_tgt.vocab.stoi[word]))
    return mapping


def replace_embeds(embeds_tgt, embeds_src, field_mappings):
    for idx_src, idx_tgt in field_mappings:
        embeds_tgt.weight.data[idx_tgt] = embeds_src.weight.data[idx_src]


class SST2Dataset(RegisteredDataset, name="sst2"):

    N_CLASSES = 2
    TEXT_FIELD = Field(batch_first=True, tokenize=basic_tokenize, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    LOGITS_0 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    LOGITS_1 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev.tsv", test="test.tsv"):
        fields = [("label", cls.LABEL_FIELD), ("sentence", cls.TEXT_FIELD), ("logits_0", cls.LOGITS_0),
            ("logits_1", cls.LOGITS_1)]
        train_ds, dev_ds, test_ds = super(SST2Dataset, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=True)
        del test_ds.fields["logits_0"]
        del test_ds.fields["logits_1"]
        del test_ds.fields["label"]
        return train_ds, dev_ds, test_ds

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev.tsv", test="test.tsv"):
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        sort_within_batch = False
        if sort_within_batch:
            print("Sorting within batch.")
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=sort_within_batch, device=device, sort=False)


class CoLADataset(RegisteredDataset, name="cola"):

    N_CLASSES = 2
    TEXT_FIELD = Field(batch_first=True, tokenize=basic_tokenize, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    LOGITS_0 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    LOGITS_1 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev.tsv", test="test.tsv"):
        fields = [("label", cls.LABEL_FIELD), ("sentence", cls.TEXT_FIELD), ("logits_0", cls.LOGITS_0),
            ("logits_1", cls.LOGITS_1)]
        return super(CoLADataset, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=True)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev.tsv", test="test.tsv"):
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=False, device=device, sort=False)


class STSDataset(RegisteredDataset, name="sts"):
    N_CLASSES = 1
    TEXT_FIELD = Field(batch_first=True, tokenize=basic_tokenize, include_lengths=True)
    SCORE = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence1)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev.tsv", test="test.tsv"):
        fields = [("score", cls.SCORE), ("sentence1", cls.TEXT_FIELD), ("sentence2", cls.TEXT_FIELD)]
        return super(STSDataset, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=True)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev.tsv", test="test.tsv"):
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=False, device=device, sort=False)


class MRPCDataset(RegisteredDataset, name="mrpc"):
    N_CLASSES = 2
    TEXT_FIELD = Field(batch_first=True, tokenize=basic_tokenize, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    LOGITS_0 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    LOGITS_1 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    @staticmethod
    def sort_key(ex):
        return len(ex.question1)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev.tsv", test="test.tsv"):
        fields = [("label", cls.LABEL_FIELD), ("sentence1", cls.TEXT_FIELD), ("sentence2", cls.TEXT_FIELD), 
            ("logits_0", cls.LOGITS_0), ("logits_1", cls.LOGITS_1)]
        return super(MRPCDataset, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=True)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev.tsv", test="test.tsv"):
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=False, device=device, sort=False)


class QQBDataset(RegisteredDataset, name="qqb"):
    N_CLASSES = 2
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=basic_tokenize, include_lengths=True)
    LOGITS_0 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    LOGITS_1 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    @staticmethod
    def sort_key(ex):
        return len(ex.question1)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev.tsv", test="test.tsv"):
        fields = [("is_duplicate", cls.LABEL_FIELD), ("question1", cls.TEXT_FIELD), ("question2", cls.TEXT_FIELD),
            ("logits_0", cls.LOGITS_0), ("logits_1", cls.LOGITS_1)]
        return super(QQBDataset, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=True)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev.tsv", test="test.tsv"):
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=False, device=device, sort=False)


class QNLIDataset(RegisteredDataset, name="qnli"):
    N_CLASSES = 2
    TEXT_FIELD = Field(batch_first=True, tokenize=basic_tokenize, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    LOGITS = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    @staticmethod
    def sort_key(ex):
        return len(ex.question1)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev.tsv", test="test.tsv"):
        fields = [("index", cls.LABEL_FIELD), ("question", cls.TEXT_FIELD), ("sentence", cls.TEXT_FIELD), ("label", cls.LABEL_FIELD),
            ("logits_0", cls.LOGITS), ("logits_1", cls.LOGITS)]
        return super(QNLIDataset, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=True)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev.tsv", test="test.tsv"):
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=False, device=device, sort=False)


class MNLIDataset_MisMatch(RegisteredDataset, name="mnli_mismatch"):
    N_CLASSES = 3
    TEXT_FIELD = Field(batch_first=True, tokenize=basic_tokenize)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    LOGITS_0 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    LOGITS_1 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    LOGITS_2 = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence1)

    @classmethod
    def splits(cls, folder_path, train="train.tsv", dev="dev_mismatched.tsv", test="test_mismatched.tsv"):
        fields = [("gold_label", cls.LABEL_FIELD), ("sentence1", cls.TEXT_FIELD), ("sentence2", cls.TEXT_FIELD), 
            ("logits_0", cls.LOGITS_0), ("logits_1", cls.LOGITS_1), ("logits_2", cls.LOGITS_2)]
        return super(MNLIDataset_MisMatch, cls).splits(folder_path, train=train, validation=dev, test=test, format="tsv", 
            fields=fields, skip_header=True)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, vectors=None,
              unk_init=uniform_unk_init(), device="cuda:0", train="train.tsv", dev="dev_mismatched.tsv", test="test_mismatched.tsv"):
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path, train=train, dev=dev, test=test)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, 
            sort_within_batch=False, device=device, sort=False)


def find_dataset(name):
    return _REGISTRY[name]


def list_datasets():
    return list(_REGISTRY.keys())

