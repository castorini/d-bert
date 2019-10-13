import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import init_embedding, fetch_embedding


class ConvRNNModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        fc_size = config.fc_size
        n_fmaps = config.n_feature_maps
        self.rnn_type = config.rnn_type
        init_embedding(self, config)

        if self.rnn_type.upper() == "LSTM":
            self.bi_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        elif self.rnn_type.upper() == "GRU":
            self.bi_rnn = nn.GRU(self.embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        else:
            raise ValueError("RNN type must be one of LSTM or GRU")
        self.conv = nn.Conv2d(1, n_fmaps, (1, self.hidden_size * 2))
        self.fc1 = nn.Linear(n_fmaps + 2 * self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, config.dataset.N_CLASSES)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = config.mode

    def forward(self, x):
        x = fetch_embedding(self, self.mode, x, squash=True).squeeze(1)
        rnn_seq, rnn_out = self.bi_rnn(x)
        if self.rnn_type.upper() == "LSTM":
            rnn_out = rnn_out[0]
        rnn_out = rnn_out.permute(1, 0, 2)
        x = self.conv(rnn_seq.unsqueeze(1)).squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))
        out = [t.squeeze(1) for t in rnn_out.chunk(2, 1)]
        out.append(x.squeeze(-1))
        x = torch.cat(out, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)