import argparse
import random
import sys

from tqdm import tqdm
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-break', default=0, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cache', type=str, required=True)
    parser.add_argument('--n-iter', type=int, default=1)
    args = parser.parse_args()

    sd = torch.load(args.cache)
    training_ds, dev_ds, test_ds = sd['splits']

    random.seed(args.seed)
    for line in tqdm(training_ds.sentences, desc='Writing'):
        print(line)
        for _ in range(args.n_iter):
            roll = random.random()
            if roll < args.sample_break:
                roll /= args.sample_break
                a = int(roll * (len(line) - 1)) + 1
                if a <= 0 or a >= len(line):
                    continue
                print(line[:a])
                print(line[a:])


if __name__ == '__main__':
    main()
