from collections import Counter
import argparse
import random
import sys

from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLLMHeadModel, BertAdam
from tqdm import tqdm
import torch
import torch.nn as nn

from args import add_dict_options, opt, OptionEnum
from build_gpt_sampler import PrefixSampler, SampleBatch
from utils import set_seed, dual_print


ARGS = [
    OptionEnum.SEED.value.default(1111), # match AWD-LSTM
    opt('--transfo-model', type=str, default='transfo-xl-wt103'),
    opt('--resume', type=str, required=True),
    opt('--prefix-file', type=str, required=True),
    opt('--num-samples', type=int, default=1500000),
    opt('--paired', action='store_true'),
    opt('--num-buffers', type=int, default=2),
    opt('--balance-every', type=int, default=16),
    opt('--simple-sample', action='store_true')
]


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)

    prefix_sampler = torch.load(args.prefix_file)
    tokenizer = TransfoXLTokenizer.from_pretrained(args.transfo_model)
    model = TransfoXLLMHeadModel.from_pretrained(args.transfo_model)
    sos_idx = None
    model.load_state_dict(torch.load(args.resume, map_location=lambda s, l: s))
    if not args.simple_sample: model = nn.DataParallel(model)
    model.cuda()

    sample_batches = [SampleBatch(model, tokenizer, prefix_sampler) for _ in range(args.num_buffers)]
    if args.simple_sample:
        for _ in tqdm(range(args.num_samples)):
            print(sample_batches[0].simple_sample(pair=args.paired))
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
            print(fin_text.replace('<eos>', ''))
            pbar.update(1)
            n_output += 1
            if (n_output + 1) % args.balance_every == 0:
                pbar.set_postfix(dict(last_balance=n_output))
                SampleBatch.balance(sample_batches)


if __name__ == '__main__':
    main()