import random


def split(lst, size):
    for idx in range(0, len(lst), size):
        yield lst[idx:idx + size]