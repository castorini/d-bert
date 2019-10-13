from collections import Counter
import argparse
import random
import sys

from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel, TransfoXLTokenizer, TransfoXLLMHeadModel, \
    BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import torch
import torch.nn as nn

from .list import split
from .args import add_dict_options, opt, OptionEnum
from .build_sampler import PrefixSampler, SampleBatch
from .finetune_bert import augment_texts
from .finetune_gpt import gpt_encode, EOS_TOKEN
from .utils import set_seed, dual_print


ARGS = [
    OptionEnum.SEED.value.default(1111), # match AWD-LSTM
    opt('--gpt2-model', type=str, default='gpt2'),
    opt('--transfo-model', type=str, default='transfo-xl-wt103'),
    opt('--bert-model', type=str, default='bert-large-uncased'),
    opt('--transfo', action='store_true'),
    opt('--bert', action='store_true'),
    opt('--resume', type=str),
    opt('--prefix-file', type=str),
    opt('--num-samples', type=int, default=1500000),
    opt('--paired', action='store_true'),
    opt('--num-buffers', type=int, default=2),
    opt('--balance-every', type=int, default=16),
    opt('--simple-sample', action='store_true'),
    opt('--msl', type=int, default=128)
]


def init_sos(model):
    embedding = model.transformer.wte.weight
    sos_tensor = torch.Tensor(1, embedding.data.size(1)).uniform_(-0.1, 0.1).to(embedding.data.device)
    embedding.data = torch.cat((embedding.data, sos_tensor)) # <eos> is <|endoftext|>
    return embedding.data.size(0) - 1


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.prefix_file: prefix_sampler = torch.load(args.prefix_file)
    if args.transfo:
        tokenizer = TransfoXLTokenizer.from_pretrained(args.transfo_model)
        model = TransfoXLLMHeadModel.from_pretrained(args.transfo_model)
    elif args.bert:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        model = BertForMaskedLM.from_pretrained(args.bert_model)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model)
        model = GPT2LMHeadModel.from_pretrained(args.gpt2_model)
        init_sos(model)
    if args.resume: model.load_state_dict(torch.load(args.resume, map_location=lambda s, l: s))
    if not args.simple_sample: model = nn.DataParallel(model)
    model.cuda()

    if args.bert:
        text_batches = list(split(list(sys.stdin), 128))
        for text_batch in tqdm(text_batches, desc='Augmenting'):
            for _ in range(args.num_samples):
                mtext_batch = [' '.join('[MASK]' if (random.random() < 0.2 and '\t' not in x) else x for x in sent.split(' ')) for sent in text_batch]
                print('\n'.join(x.replace('[SEP]', '\t').strip() for x in augment_texts(model, tokenizer, mtext_batch, max_len=args.msl)))
                sys.stdout.flush()
        return

    sample_batches = [SampleBatch(model, tokenizer, prefix_sampler) for _ in range(args.num_buffers)]
    if args.simple_sample:
        for _ in tqdm(range(args.num_samples)):
            print(sample_batches[0].simple_sample(pair=args.paired, transfo=args.transfo))
            sys.stdout.flush()
        return

    n_output = 0
    pbar = tqdm(total=args.num_samples, desc='Generating')
    while n_output < args.num_samples:
        try:
            sample_batch = random.choice(sample_batches)
            sample_batch.try_add_sample()
            fin_texts = sample_batch.step(pair=args.paired)
        except ValueError:
            sample_batch.try_add_sample()
            continue
        for fin_text in fin_texts:
            if n_output >= args.num_samples:
                return
            print(fin_text.replace(EOS_TOKEN, '').replace('<eos>', '\t'))
            sys.stdout.flush()
            pbar.update(1)
            n_output += 1
            if (n_output + 1) % args.balance_every == 0:
                pbar.set_postfix(dict(last_balance=n_output))
                SampleBatch.balance(sample_batches)


if __name__ == '__main__':
    main()
