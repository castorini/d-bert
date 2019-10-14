from collections import Counter
import argparse

from pytorch_transformers import GPT2Tokenizer, TransfoXLTokenizer
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import torch
import torch.utils.data as tud

from .args import add_dict_options, opt, OptionEnum
from .finetune_gpt import gpt_encode, EOS_TOKEN, sample_query
from .finetune_transfoxl import sample_query as transfo_sample_query
from .utils import set_seed, dual_print


class PrefixSampler(object):

    def __init__(self, prefixes, counts):
        self.prefixes = prefixes
        self.counts = counts
        self.sampler = Categorical(probs=torch.Tensor(self.counts))

    def __call__(self):
        return self.prefixes[self.sampler.sample().item()]

    @classmethod
    def from_token_ids(cls, decode, token_ids_lst, use_tqdm=True):
        counter = Counter()
        for token_ids in tqdm(token_ids_lst):
            counter[decode(token_ids[0])] += 1
        prefixes = list(counter.keys())
        counts = list(counter.values())
        return cls(prefixes, counts)


class SampleBatch(object):

    def __init__(self, model, tokenizer, prefix_sampler, max_len=128, max_size=128):
        self.buffer = []
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_size = max_size
        self.prefix_sampler = prefix_sampler

    def try_add_sample(self):
        gpu_count = torch.cuda.device_count()
        while len(self.buffer) < self.max_size or len(self.buffer) % gpu_count != 0:
            self.buffer.append(self.prefix_sampler())

    def simple_sample(self, pair=False, transfo=False):
        text = self.prefix_sampler()
        eos_token = '<eos>' if transfo else EOS_TOKEN
        fin_count = int(pair) + 1
        while True:
            text = transfo_sample_query(self.model, self.tokenizer, text, paired=pair) if transfo else sample_query(self.model, self.tokenizer, text)
            if transfo and text.count(eos_token) >= fin_count:
                return text
            elif not transfo and (eos_token in text and (not pair or '\t' in text)):
                return text
            else:
                text = self.prefix_sampler()

    @staticmethod
    def balance(buffers):
        big_buffer = []
        list(map(lambda x: big_buffer.extend(x.buffer), buffers))
        if len(big_buffer) < len(buffers):
            return
        big_buffer = sorted(big_buffer, key=len)
        buf_idx = 0
        for idx in range(0, len(big_buffer), len(big_buffer) // len(buffers)):
            if idx == 0:
                last_idx = 0
                continue
            buffers[buf_idx].buffer = big_buffer[last_idx:idx]
            last_idx = idx
            buf_idx += 1
        if idx < len(big_buffer) - 1:
            buffers[-1].buffer = big_buffer[idx:]

    def step(self, pair=False):
        if not self.buffer:
            raise ValueError('Buffer empty.')
        try:
            queries, _, _, raw_decode = gpt_encode(self.tokenizer, self.buffer, eos=False, return_raw=True)
        except KeyError:
            self.buffer = [] # hacky fix
            return []

        queries = torch.LongTensor(queries).cuda()
        self.model.eval()
        with torch.no_grad():
            output = self.model(queries)[0]
        preds = output[torch.arange(0, queries.size(0)).long(), torch.tensor(list(map(len, raw_decode))) - 1]
        texts = []
        fin_texts = []
        for pred, toks in zip(preds, raw_decode):
            toks.append(Categorical(logits=pred.view(-1)).sample().item())
            try:
                decode_text = self.tokenizer.decode(toks)
                if len(toks) >= self.max_len or EOS_TOKEN in decode_text:
                    if pair and '\t' not in decode_text: continue
                    fin_texts.append(decode_text)
                    continue
            except KeyError:
                continue
            texts.append(decode_text)
        self.buffer = texts
        return fin_texts


ARGS = [
    OptionEnum.SEED.value.default(1111), # match AWD-LSTM
    opt('--cache-file', type=str, default='aol-cache.pt'),
    opt('--save', type=str, default='prefix_sampler.pt'),
    opt('--gpt2-model', type=str, default='gpt2'),
    opt('--transfo-model', type=str, default='transfo-xl-wt103'),
    opt('--model-type', type=str, choices=['gpt2', 'transfoxl'], default='transfoxl')
]


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)
    sd = torch.load(args.cache_file)

    if args.model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model)
        encode = tokenizer.encode
        decode = lambda x: tokenizer.decoder[x]
    else:
        tokenizer = TransfoXLTokenizer.from_pretrained(args.transfo_model)
        encode = lambda x: [tokenizer.get_idx(x.lower().strip().split()[0])]
        decode = tokenizer.get_sym
    train_ds, _, _ = sd['splits']

    train_loader = tud.DataLoader(train_ds, batch_size=1, shuffle=True)
    token_ids_lst = []
    for batch in train_loader:
        _, sentences = batch
        token_ids_lst.extend(map(encode, sentences))
    sampler = PrefixSampler.from_token_ids(decode, token_ids_lst)
    torch.save(sampler, args.save)


if __name__ == '__main__':
    main()
