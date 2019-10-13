import argparse
import glob
import os

from tqdm import tqdm
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue_dir", type=str, default="/mnt/nvme/glue")
    args = parser.parse_args()
    test_files = glob.glob(os.path.join(args.glue_dir, "*", "test.tsv"))
    pd.read_csv("")