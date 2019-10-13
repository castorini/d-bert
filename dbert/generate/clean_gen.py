from args import opt, add_dict_options
import argparse
import re

from tqdm import tqdm
import nltk


ARGS = [
    opt('--tokenize', action='store_true'),
    opt('--files', type=str, nargs='+', required=True),
    opt('--paired', action='store_true')
]


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    lines = []
    for file in tqdm(args.files):
        with open(file) as file: lines.extend(file.readlines())

    patt = re.compile(r'^.*[^\s].*\t.*[^\s].*$') if args.paired else re.compile(r'^.*[^\s].*$')
    pbar = tqdm(lines)
    discarded = 0
    for line in pbar:
        line = line.replace('<|endoftext|>', '')
        line = line.replace('<eos>', '\t').strip()
        if not re.match(patt, line) or \
                (args.paired and line.count('\t') != 1) or \
                line.startswith('Better speed can be achieved with'):
            discarded += 1
            pbar.set_postfix(dict(discarded=discarded))
            continue
        if args.tokenize:
            line1, line2 = line.split('\t')
            line = f'{" ".join(nltk.word_tokenize(line1))}\t{" ".join(nltk.word_tokenize(line2))}'
        print(line)
    pbar.close()


if __name__ == '__main__':
    main()
