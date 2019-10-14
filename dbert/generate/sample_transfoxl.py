import argparse

from pytorch_transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from tqdm import tqdm
import torch

from .args import add_dict_options, opt, OptionEnum
from .build_sampler import SampleBatch
from .utils import set_seed, dual_print


ARGS = [
    OptionEnum.SEED.value.default(1111), # match AWD-LSTM
    opt('--transfo-model', type=str, default='transfo-xl-wt103'),
    opt('--resume', type=str, required=True),
    opt('--prefix-file', type=str, required=True),
    opt('--num-samples', type=int, default=1500000),
    opt('--paired', action='store_true')
]


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)

    prefix_sampler = torch.load(args.prefix_file)
    tokenizer = TransfoXLTokenizer.from_pretrained(args.transfo_model)
    model = TransfoXLLMHeadModel.from_pretrained(args.transfo_model)
    model.load_state_dict(torch.load(args.resume, map_location=lambda s, l: s))
    model.cuda()

    sampler = SampleBatch(model, tokenizer, prefix_sampler)
    for _ in tqdm(range(args.num_samples)):
        print(sampler.simple_sample(pair=args.paired))


if __name__ == '__main__':
    main()