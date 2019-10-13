from collections import Counter
import argparse
import os
import sys

from tqdm import tqdm
import nltk
import pandas as pd

from .args import opt, add_dict_options


ARGS = [
    opt('--vocab-size', type=int, default=5000),
    opt('--unk-token', type=str, default='<unk>'),
    opt('--files', type=str, nargs='+', required=True),
    opt('--folder', type=str, required=True),
    opt('--columns', type=str, nargs='+', default=['sentence']),
    opt('--new-column', type=str, default='lm')
]


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    os.makedirs(args.folder, exist_ok=True)

    counter = Counter()
    lines_lst = []
    dfs = []
    for file in args.files:
        df = pd.read_csv(file, sep='\t', quoting=3, keep_default_na=False, error_bad_lines=False)
        lines = df[args.columns].values
        lines_lst.append([' \t '.join(x) for x in lines])
        dfs.append(df)
    for lines in lines_lst:
        for line in tqdm(lines, desc='Processing'):
            line = line.strip()
            toks = nltk.word_tokenize(line)
            for tok in toks:
                counter[tok] += 1
    all_counts = list(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    words = [x[0] for x in all_counts[:args.vocab_size - 1]]
    tok_count = sum([x[1] for x in all_counts[:args.vocab_size - 1]])
    oov_count = sum([x[1] for x in all_counts[args.vocab_size - 1:]])
    print(f'OoV: {100 * oov_count / (tok_count + oov_count):.2f}%', file=sys.stderr)
    words.append(args.unk_token)
    words = set(words)

    for lines, name, df in zip(lines_lst, args.files, dfs):
        name = os.path.join(args.folder, name)
        lm_col = []
        for line in tqdm(lines, desc='Outputting'):
            line = line.strip()
            toks = nltk.word_tokenize(line)
            toks = [args.unk_token if tok not in words else tok for tok in toks]
            lm_col.append(' '.join(toks))
        df[args.new_column] = lm_col
        df.to_csv(name, sep='\t', index=False)


if __name__ == '__main__':
    main()
