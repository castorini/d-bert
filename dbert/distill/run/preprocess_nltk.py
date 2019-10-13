import argparse

from tqdm import tqdm
import pandas as pd
import nltk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str)
    parser.add_argument("--text_col", type=str, default="sentence")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pos_tag_only", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_file, sep="\t")
    columns = list(df.columns)
    pos_tags_lst = []
    for text in tqdm(df[args.text_col]):
        word_toks = text.split(" ") if args.pos_tag_only else nltk.word_tokenize(text)
        pos_tags_lst.append(nltk.pos_tag(word_toks))
    df[f"{args.text_col}_pos"] = [" ".join([tag[1] for tag in pos_tags]) for pos_tags in pos_tags_lst]
    if not args.pos_tag_only:
        df[args.text_col] = [" ".join([tag[0] for tag in pos_tags]) for pos_tags in pos_tags_lst]
    if args.output_file is None:
        args.output_file = args.dataset_file
    df.to_csv(args.output_file, index=False, sep="\t")


if __name__ == "__main__":
    main()