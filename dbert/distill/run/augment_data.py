from collections import defaultdict
import argparse
import functools
import random
import sys

from tqdm import tqdm
import nltk
import pandas as pd
import torch.nn as nn

import dbert.distill.data as dat
import dbert.distill.model.bert as bt

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=float) #added
    parser.add_argument("--mask_prob", default=0.1, type=float)
    parser.add_argument("--random_prob", default=0.1, type=float)
    parser.add_argument("--window_prob", default=0, type=float)
    parser.add_argument("--bert_gen_prob", default=0.2, type=float)
    parser.add_argument("--window_lengths", default=[1, 2, 3, 4, 5], nargs="+", type=int)
    parser.add_argument("--n_iter", default=20, type=int)
    parser.add_argument("--dataset_file", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", type=str, default="nltk", choices=["allennlp", "nltk"], 
        help="Use NLTK for now.")
    args = parser.parse_args()

    print("args:", args)

    random.seed(args.seed) 
    df = pd.read_csv(args.dataset_file, sep="\t")
    vocab = set()
    dataset = set()
    pos_dict = defaultdict(list)
    for sentence in tqdm(df["sentence"]):
        if args.tokenizer == "nltk":
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
        else:
            toks = dat.allennlp_full_tokenize(sentence)
            words = [tok.text for tok in toks]
            pos_tags = [(tok.text, tok.pos) for tok in toks]
        for word, pos_tag in pos_tags:
            vocab.add(word)
            pos_dict[pos_tag].append(word)
        dataset.add(" ".join(words))

    vocab = list(vocab)
    gen_batch = []
    sorted_sents = sorted(df["sentence"], key=lambda x: len(x.split()), reverse=True)

    for sentence in tqdm(df["sentence"]):
        while len(gen_batch) >= args.batch_size:
            gen_batch = process_gen_batch(bert, gen_batch, dataset)
        if args.tokenizer == "nltk":
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
        else:
            toks = dat.allennlp_full_tokenize(sentence)
            words = [tok.text for tok in toks]
            pos_tags = [(tok.text, tok.pos) for tok in toks]
        for _ in range(args.n_iter):
            use_bert_gen = False
            use_windowing = random.random() < args.window_prob
            mask_prob = args.mask_prob
            gen_words = []
            all_replaced = True
            for idx, (word, pos_tag) in enumerate(zip(words, pos_tags)):
                roll = random.random()
                if roll < mask_prob:
                    word = "[MASK]"
                elif roll < mask_prob + args.random_prob:
                    word = random.choice(pos_dict[pos_tag[1]])
                elif roll < mask_prob + args.random_prob + args.bert_gen_prob:
                    word = "[UNK]"
                    use_bert_gen = True
                else:
                    all_replaced = False
                gen_words.append(word)

            if use_windowing:
                window_len = random.choice(args.window_lengths)
                try:
                    idx = random.randrange(len(gen_words) - window_len)
                    gen_words = gen_words[idx:idx + window_len]
                except ValueError:
                    break
            if args.tokenizer == "nltk":
                gen_sentence = " ".join(gen_words)
            else:
                gen_sentence = reconstruct_allennlp(gen_words, toks)
            if not all_replaced and gen_sentence not in dataset:
                dataset.add(gen_sentence)
                print(gen_sentence)
    

if __name__ == "__main__":
    print("bitch!")
    main()
