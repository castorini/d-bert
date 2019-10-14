from collections import namedtuple
import csv
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_dp_to_single(state_dict):
    mod_key = "module."
    for k, v in list(state_dict.items()):
        if k.startswith(mod_key):
            del state_dict[k]
            state_dict[k[len(mod_key):]] = v
    return state_dict


def convert_single_to_dp(state_dict):
    mod_key = "module."
    for k, v in list(state_dict.items()):
        if k.startswith(mod_key):
            continue
        state_dict[f"{mod_key}{k}"] = v
        del state_dict[k]
    return state_dict
