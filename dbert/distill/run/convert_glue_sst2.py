import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep="\t")
    df["label"] = df["logits_1"] = df["logits_0"] = [0] * len(df)
    df = df.drop("index", 1)
    column_order = ["label", "sentence", "logits_0", "logits_1"]
    df[column_order].to_csv(sys.stdout, sep="\t", index=False)


if __name__ == "__main__":
    main()