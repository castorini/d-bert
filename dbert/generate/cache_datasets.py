import argparse

from tqdm import tqdm
import torch

from .data import SingleSentenceDataset, Dictionary
from .args import add_dict_options, OptionEnum, opt
from .utils import set_seed


ARGS = [
    OptionEnum.DATA_DIR.value.required(True),
    OptionEnum.SEED.value.default(0),
    opt('--output-file', type=str, required=True),
    opt('--no-char-dict', action='store_false', dest='build_char_dict'),
    opt('--resume', type=str),
    opt('--dataset-type', type=str, default='single-sentence', choices=['single-sentence', 'pair-sentence']),
    opt('--column', type=str, default='sentence'),
    opt('--column1', type=str, default='question1'),
    opt('--column2', type=str, default='question2'),
    opt('--filter-label', type=str),
    opt('--label-column', type=str, default='label'),
    opt('--word-level', action='store_true')
]


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.resume:
        save_dict = torch.load(args.resume)
        splits = save_dict['splits']
    else:
        kwargs = dict(filter_label=args.filter_label, label_column=args.label_column, 
            column=args.column, column1=args.column1, column2=args.column2)
        splits_fn = SingleSentenceDataset.splits if args.dataset_type == 'single-sentence' else SingleSentenceDataset.pair_splits
        splits = splits_fn(args.data_dir, **kwargs)
        save_dict = dict(splits=splits)
    print(list(zip(('Training: ', 'Dev: ', 'Test: '), map(len, splits))))

    if args.build_char_dict and not args.resume:
        dictionary = Dictionary()
        for ds in save_dict['splits']:
            for _, query in tqdm(iter(ds), total=len(ds)):
                toks = query.split() if args.word_level else list(query)
                for tok in toks: dictionary.add_word(tok)
        save_dict['dictionary'] = dictionary
    elif args.build_char_dict:
        dictionary = save_dict['dictionary']

    torch.save(save_dict, args.output_file)


if __name__ == '__main__':
    main()
