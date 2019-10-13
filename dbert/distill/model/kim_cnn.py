import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import fetch_embedding, init_embedding


class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        output_channel = config.output_channel
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        ks = 3

        init_embedding(self, config)
        input_channel = 2 if config.mode == "multichannel" else 1

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, words_dim), padding=(4, 0))
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(ks * output_channel, dataset.N_CLASSES)

    def non_embedding_params(self):
        params = []
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                continue
            params.extend(p for p in m.parameters() if p.dim() == 2)
        return params

    def forward(self, x, **kwargs):
        x = fetch_embedding(self, self.mode, x)
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * ks
        x = torch.cat(x, 1) # (batch, channel_output * ks)
        x = self.dropout(x)
        logits = self.fc1(x) # (batch, target_size)
        return logits