import hashlib
import os
import random

from tqdm import tqdm
import nltk
import torch

from .args import read_args


def main():
    config = read_args(default_config="confs/bert.json")
    with open(config.input_file) as f:
        content = f.read()
    sentences = nltk.sent_tokenize(content)
    random.shuffle(sentences)
    print(f"Read {len(sentences)} sentences.")
    vecs = []
    for sent in tqdm(sentences):
        h = hashlib.md5(sent.encode()).hexdigest()
        if config.lookup_word not in sent.lower():
            continue
        path = os.path.join(config.output_folder, h)
        if not os.path.exists(path):
            continue
        try:
            toks, tok_vecs = torch.load(path)
        except:
            print(path)
            continue
        for w, v in zip(toks, tok_vecs.split(1, 1)):
            if w == config.lookup_word:
                vecs.append(v)
    torch.save(vecs, f"{config.lookup_word}-vecs.pt")


if __name__ == "__main__":
    main()