import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import init_embedding, fetch_embedding


class SiameseRNNModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        fc_size = config.fc_size
        self.rnn_type = config.rnn_type
        init_embedding(self, config)

        if self.rnn_type.upper() == "LSTM":
            self.bi_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        elif self.rnn_type.upper() == "GRU":
            self.bi_rnn = nn.GRU(self.embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        else:
            raise ValueError("RNN type must be one of LSTM or GRU")
        self.fc1 = nn.Linear(8 * self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, config.dataset.N_CLASSES)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = config.mode

    def non_embedding_params(self):
        params = []
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                continue
            params.extend(p for p in m.parameters() if p.dim() == 2)
        return params

    def forward(self, sent1_tuple, sent2_tuple):
        sent1, sent1_length = sent1_tuple
        sent2, sent2_length = sent2_tuple
        
        sent1 = fetch_embedding(self, self.mode, sent1, squash=True).squeeze(1)
        sent2 = fetch_embedding(self, self.mode, sent2, squash=True).squeeze(1)
        res = []
        sent = [sent1, sent2]
        sent_length = [sent1_length, sent2_length]
        dynamic = True
        for i in range(2):
            if dynamic:
                orig_sent = sent[i]
                orig_sent_length = sent_length[i]
                sorted_len, sorted_ind = torch.sort(orig_sent_length, descending=True)
                _, sorted_back_ind = torch.sort(sorted_ind)
                sorted_sent = orig_sent[sorted_ind]
                try:
                    packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_len, batch_first=True)
                except RuntimeError as e:
                    sorted_len = torch.eq(sorted_len,0).long() + sorted_len
                    packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_len, batch_first=True)
                rnn_seq, rnn_out = self.bi_rnn(packed)
            
                if self.rnn_type.upper() == "LSTM":
                    rnn_out = rnn_out[0]
            
                rnn_out = rnn_out.permute(1, 0, 2)
                rnn_out = rnn_out[sorted_back_ind]
            else:
                rnn_seq, rnn_out = self.bi_rnn(sent[i])

                if self.rnn_type.upper() == "LSTM":
                    rnn_out = rnn_out[0]

                rnn_out = rnn_out.permute(1, 0, 2)
            
            rnn_out = rnn_out.contiguous().view(rnn_out.size()[0], rnn_out.size()[1] * rnn_out.size()[2])
            res.append(rnn_out)
        s1_enc = res[0]
        s2_enc = res[1]

        res = torch.cat([s1_enc, s2_enc, torch.abs(s1_enc - s2_enc), s1_enc * s2_enc], 1)
        res = F.relu(self.fc1(res))
        logits = self.fc2(res)
        return logits

