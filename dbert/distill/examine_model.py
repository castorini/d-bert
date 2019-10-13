import sys

import torch


sd = torch.load(sys.argv[1])
print(sd.get("dev_acc"))
print(sd.get("epoch_idx"))
print(sd.get('dev_pr'))
print(sd.get('dev_sr'))
