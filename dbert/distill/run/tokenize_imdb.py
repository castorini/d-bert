import argparse
import re
import sys

import nltk


def main():
    punc_error_patt = re.compile(r"([A-z])(\.|\?|\!)([A-z])")
    url_patt = re.compile(r"[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)")
    contiguous_space_patt = re.compile(r"\s+")

    for line in sys.stdin:
        rating, document = line.split("\t")
        rating = rating.index("1")
        if rating not in (3, 4, 5):
            continue
        document = re.sub(url_patt, "", document)
        document = re.sub(punc_error_patt, r"\1\2 \3", document)
        document = re.sub(contiguous_space_patt, " ", document)
        print("\n".join(nltk.sent_tokenize(document)).lower())


if __name__ == "__main__":
    main()