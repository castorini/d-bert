import re


SPACE_PATT = re.compile(r' +')


def space_tokenize(x):
    return SPACE_PATT.split(x.strip())