import random


def spm_encode(sp, queries, max_len=128, split_encode=False, sos=False):
    def rand_split_encode(query):
        split_idx = random.randint(0, len(query) - 1)
        a = [] if split_idx == 0 else sp.EncodeAsIds(query[:split_idx])
        b = [] if split_idx == len(query) - 1 else sp.EncodeAsIds(query[split_idx:])
        return a + b
    eos_idx = len(sp)
    encode = rand_split_encode if split_encode else sp.EncodeAsIds
    sos_prepend = [eos_idx + 1] if sos else []
    tokens_lst = [sos_prepend + encode(x) + [eos_idx] for x in queries]
    tokens_mask = [[1] * len(x) for x in tokens_lst]
    old_max_len = max_len
    max_len = min(max(map(len, tokens_lst)), max_len)
    tokens_lst = [x[:max_len] for x in tokens_lst]
    tokens_mask = [x[:max_len] for x in tokens_mask]
    tokens_lst = [x + [0] * (max_len - len(x)) for x in tokens_lst]
    tokens_mask = [x + [0] * (max_len - len(x)) for x in tokens_mask]
    total_in_chars = sum(min(len(q) + len(sos_prepend), old_max_len) for q in queries)
    return tokens_lst, tokens_mask, total_in_chars