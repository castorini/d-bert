import random
import sys

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def dual_print(*args, **kwargs):
    try:
        del kwargs['file']
    except:
        pass
    print(*args, file=sys.stderr, **kwargs)
    print(*args, file=sys.stdout, **kwargs)