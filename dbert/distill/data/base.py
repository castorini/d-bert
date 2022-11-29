import torch
import numpy as np
import random

from allennlp.data.tokenizers import WordTokenizer
from pytorch_pretrained_bert import tokenization as tok
from pytorch_pretrained_bert import GPT2Tokenizer


BERT_TOKENIZER = None
GPT2_TOKENIZER = None
BERT_PATH = 'bert-large-uncased'
GPT2_PATH = 'gpt2'


def bert_tokenize(text):
    global BERT_TOKENIZER
    if BERT_TOKENIZER is None:
        BERT_TOKENIZER = tok.BertTokenizer.from_pretrained(BERT_PATH)
    return BERT_TOKENIZER.tokenize(text)


def gpt2_tokenize(text):
    global GPT2_TOKENIZER
    if GPT2_TOKENIZER is None:
        GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained(GPT2_PATH)
    return [GPT2_TOKENIZER.decoder[x] for x in GPT2_TOKENIZER.encode(text)]


def space_tokenize(text):
    return text.lower().split(" ")


def allennlp_tokenize(text, **tokenizer_kwargs):
    return [x.text for x in allennlp_full_tokenize(text, **tokenizer_kwargs)]


def allennlp_full_tokenize(text, **tokenizer_kwargs):
    return WordTokenizer(**tokenizer_kwargs).tokenize(text)


def basic_tokenize(text, **tokenizer_kwargs):
    return tok.BasicTokenizer(**tokenizer_kwargs).tokenize(text)


def uniform_unk_init(a=-0.25, b=0.25):
    return lambda tensor: tensor.uniform_(a, b)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)