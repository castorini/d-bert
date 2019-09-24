from collections import Counter
import argparse
import datetime
import sys

from tqdm import tqdm

from args import opt, add_dict_options


DOCSTR = \
"""Usage:
for i in `seq 1 9`; do
    cat user-ct-test-collection-0$i.txt | python preprocess.py > clean-0$i.txt
done
cat user-ct-test-collection-10.txt | python preprocess.py > clean-10.txt
"""

ARGS = [
    opt('--action', choices=['background', 'candidate'], default='background'),
    opt('--truncate-length', type=int, default=100),
    opt('--truncate-freq', type=int, default=3)
]

def main():
    parser = argparse.ArgumentParser(epilog=DOCSTR)
    add_dict_options(parser, ARGS)
    args = parser.parse_args()

    if args.action == 'background':
        process_bkgd(args)


def process_bkgd(args):
    lines = list(map(str.strip, sys.stdin))
    query_counter = Counter()
    for line in tqdm(lines, desc='Counting'):
        try:
            anon_id, query, date, _, _ = line.split('\t')
        except:
            anon_id, query, date = line.split('\t')
        try:
            date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        except:
            continue
        if date.month in (3, 4):
            query_counter[query] += 1

    last_query = None
    for line in tqdm(lines, desc='Writing'):
        try:
            anon_id, query, date, b, c = line.split('\t')
            other_data = [date, b, c]
        except:
            anon_id, query, date = line.split('\t')
            other_data = [date]
        if anon_id == 'AnonID':
            print(line)
            continue
        try:
            date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            if date.month not in (3, 4):
                continue
        except:
            pass
        if query_counter[query] < args.truncate_freq or len(query) > args.truncate_length:
            continue
        if query != last_query:
            last_query = query
            print('\t'.join([anon_id, query] + other_data))


if __name__ == '__main__':
    main()