import hashlib
import os
import random

from tqdm import tqdm
import nltk
import torch

import dbert.distill.model as mod
from .args import read_args


def main():
    config = read_args(default_config="confs/bert.json")
    bert = mod.BertWrapper.load(config.bert_model)
    with open(config.input_file) as f:
        content = f.read()
    sentences = nltk.sent_tokenize(content)
    random.shuffle(sentences)
    print(f"Read {len(sentences)} sentences.")
    try:
        os.makedirs(config.output_folder)
    except:
        pass
    for sent in tqdm(sentences):
        h = hashlib.md5(sent.encode()).hexdigest()
        path = os.path.join(config.output_folder, h)
        if os.path.exists(path):
            continue
        try:
            out = bert.extract_vectors(sent)
        except:
            continue
        torch.save(out, path)


if __name__ == "__main__":
    main()