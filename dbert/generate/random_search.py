from random import *
import argparse
import hashlib
import os
import re
import sys

from .args import opt, add_dict_options, OptionEnum


ARGS = [
    OptionEnum.SEED.value,
    opt('--prompt', required=True, type=str),
    opt('--num-iters', type=int, default=1)
]


def gen_prompt(prompt):
    format_args = []
    brace_patt = re.compile(r'\{(.+?)\}')
    m = brace_patt.search(prompt)
    while m:
        _, end_idx = m.span()
        eval_str = m.group(1)
        val = eval(eval_str)
        format_args.append(val)
        m = brace_patt.search(prompt, end_idx)
    idx = 0
    while True:
        old_prompt = prompt
        prompt = brace_patt.sub(f'[{idx}]', prompt, count=1)
        if prompt == old_prompt:
            break
        idx += 1

    backref_patt = re.compile(r'\[(.+?)\]')
    prompt = backref_patt.sub(r'{\1}', prompt)
    prompt = prompt.format(*format_args)
    cmd_hash = hashlib.md5(prompt.encode()).hexdigest()
    prompt = prompt.replace('%cmd_args%', repr(format_args))
    prompt = prompt.replace('%cmd_hash%', cmd_hash)
    return prompt, format_args


def main():
    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()

    seed(args.seed)
    for _ in range(args.num_iters):
        prompt, _ = gen_prompt(args.prompt)
        print(f'Running {prompt}', file=sys.stderr)
        os.system(prompt)


if __name__ == '__main__':
    main()
