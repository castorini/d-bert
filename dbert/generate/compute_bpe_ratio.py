import argparse
import sys

from tqdm import tqdm
import sentencepiece as spm

from .args import opt, OptionEnum, add_dict_options


ARGS = [
    OptionEnum.SPM_MODEL.value.required(True)
]


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()

    tot_n = 0
    comp_n = 0
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)

    lines = list(tqdm(sys.stdin))
    pbar = tqdm(lines)
    for idx, line in enumerate(pbar):
        line.strip()
        tot_n += len(line)
        comp_n += len(sp.EncodeAsIds(line))
        if (idx + 1) % (len(lines) // 20) == 0:
            pbar.set_postfix(cp_ratio=f'{tot_n / comp_n:.4f}')
    print(f'{tot_n / comp_n} compression ratio')


if __name__ == '__main__':
    main()