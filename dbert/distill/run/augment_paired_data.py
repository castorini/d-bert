from collections import defaultdict
import argparse
import random
import re

from tqdm import tqdm
import pandas as pd
import nltk


def clean_mnli(sentence):
    sentence = str(sentence).replace("(", "").replace(")", "")
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def generate(tags, pos_dict, random_prob, mask_prob, window_prob, window_lengths):
    gen_words = []
    for word, pos_tag in tags:
        roll = random.random()
        if roll < random_prob:
            gen_words.append(random.choice(pos_dict[pos_tag]))
        elif roll < mask_prob + random_prob:
            gen_words.append("[MASK]")
        else:
            gen_words.append(word)
    if random.random() < window_prob:
        window_len = random.choice(window_lengths)
        try:
            idx = random.randrange(len(gen_words) - window_len)
            gen_words = gen_words[idx:idx + window_len]
        except ValueError:
            pass
    gen_sent = " ".join(gen_words)
    return gen_sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", "-f", type=str)
    parser.add_argument("--task", type=str, default="qqp", choices=["qqp", "mnli", "sts", "rte", "mrpc"])
    parser.add_argument("--random_prob", type=float, default=0.1)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--window_prob", default=0, type=float)
    parser.add_argument("--window_lengths", default=[1, 2, 3, 4, 5], nargs="+", type=int)
    parser.add_argument("--single_only_prob", type=float, default=2/3)
    parser.add_argument("--paired_prob", type=float, default=1/3)
    parser.add_argument("--n_iter", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_file, sep="\t", error_bad_lines=False, quoting=3)
    columns = list(df.columns)
    task_cols = dict(qqp=["question1", "question2"], mrpc=["sentence1", "sentence2"],
        mnli=["sentence1", "sentence2"], sts=["sentence1", "sentence2"], rte=["sentence1", "sentence2"])
    cols = task_cols[args.task]
    sents_a, sents_b = list(zip(*df[cols].values.tolist()))

    pos_dict = defaultdict(list)
    dataset = set()

    for sent_a, sent_b in zip(tqdm(sents_a), sents_b):
        all_words = []
        for sent in (sent_a, sent_b):
            try:
                words = nltk.word_tokenize(sent)
            except TypeError:
                continue
            pos_tags = nltk.pos_tag(words)
            for word, pos_tag in pos_tags:
                pos_dict[pos_tag].append(word)
            all_words.append(" ".join(words))
        dataset.add("\t".join(all_words))

    tot = args.single_only_prob + args.paired_prob
    single_prob = args.single_only_prob / tot

    for sent_a, sent_b in zip(tqdm(sents_a), sents_b):
        sent_a = str(sent_a)
        sent_b = str(sent_b)
        try:
            a_tags = nltk.pos_tag(nltk.word_tokenize(sent_a))
            b_tags = nltk.pos_tag(nltk.word_tokenize(sent_b))
        except TypeError:
            continue
        for _ in range(args.n_iter):
            gen_sent1 = " ".join([x[0] for x in a_tags])
            gen_sent2 = " ".join([x[0] for x in b_tags])
            roll = random.random()
            is_double = random.random() > args.single_only_prob
            if roll < 0.5 or is_double:
                gen_sent1 = generate(a_tags, pos_dict, args.random_prob, args.mask_prob, 
                    args.window_prob, args.window_lengths)
            if roll >= 0.5 or is_double:
                gen_sent2 = generate(b_tags, pos_dict, args.random_prob, args.mask_prob,
                    args.window_prob, args.window_lengths)
            if len(gen_sent1.strip()) == 0 or len(gen_sent2.strip()) == 0:
                continue

            ds_sent = f"{gen_sent1}\t{gen_sent2}"
            if ds_sent not in dataset:
                dataset.add(ds_sent)
                print(ds_sent)


if __name__ == "__main__":
    main()
