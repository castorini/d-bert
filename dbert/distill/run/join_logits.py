from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits_file", type=str)
    parser.add_argument("--n_logits", type=int, default=2)
    parser.add_argument("--dataset_file", required=True, type=str)
    parser.add_argument("--format", default="tsv", type=str, choices=["tsv", "csv", "txt"])
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--dataset_name", type=str, default="SST-2")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.dataset_file

    sep = "\t" if args.format == "tsv" else ","
    if args.format == "txt":
        sep = "magicmagic12312312358124"
    try:
        ds = pd.read_csv(args.dataset_file, sep=sep, quoting=3, keep_default_na=False, skip_blank_lines=False).astype(str)
    except:
        ds = defaultdict(list)
        with open(args.dataset_file) as f:
            lines = f.readlines()[1:]
        for line in lines:
            a, b = line.split('\t', 1)
            ds['question1'].append(a.strip())
            ds['question2'].append(b.strip())
            ds['is_duplicate'].append(0)
        ds = pd.DataFrame(ds)
    logits = torch.cat(list(map(torch.Tensor, torch.load(args.logits_file)))).squeeze() if args.logits_file else None
    logits_columns = [f"logits_{idx}" for idx in range(args.n_logits)]
    if args.dataset_name == "QQP":
        column_order = ["is_duplicate", "question1", "question2"] + logits_columns
        ds['is_duplicate'] = 0
    elif args.dataset_name == 'qnli':
        column_order = ['index', 'question', 'sentence', 'label'] + logits_columns
        ds['index'] = 0
    elif args.dataset_name == 'cola':
        column_order = ['label', 'sentence'] + logits_columns
    elif args.dataset_name == 'sts':
        logits_columns = ['score']
        column_order = ['sentence1', 'sentence2'] + logits_columns
    elif args.dataset_name == 'mrpc':
        column_order = ['sentence1', 'sentence2'] + logits_columns
    else:
        column_order = ["label", "sentence"] + logits_columns
    for idx in range(args.n_logits):
        if args.dataset_name == 'sts':
            logits_numpy = np.round_(logits.cpu().numpy(), 5)
            ds['score'] = logits_numpy
        else:
            logits_numpy = [0] * len(ds) if logits is None else np.round_(logits[:, idx].cpu().numpy(), 5)
            ds[f"logits_{idx}"] = logits_numpy
    if logits is not None and args.dataset_name != 'sts':
        ds["label"] = logits.max(1)[1].cpu().numpy()
    ds[column_order].to_csv(args.output_file, sep="\t", index=False, quoting=3)


if __name__ == "__main__":
    main()
