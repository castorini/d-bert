import math

import torch
import torch.nn as nn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop, WeightDropout


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        alpha = T(1, input_size).fill_(0)
        beta = T(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        self.alpha = P(alpha)
        self.beta = P(beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).unsqueeze(1).expand_as(x))
        x = x / th.sqrt(th.var(x, 1).unsqueeze(1).expand_as(x) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        do_dropout = self.training and self.dropout > 0.0
        h, c = (torch.zeros(1, x.size(1), self.hidden_size), torch.zeros(1, x.size(1), self.hidden_size)) if hidden is None else hidden
        h = h.view(h.size(1), -1).to(x.device)
        c = c.view(c.size(1), -1).to(x.device)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul((1 - f_t), g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(th.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(th.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)


class LayerNormLSTM(LSTM):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, 
                 dropout_method='pytorch', ln_preact=True, learnable=True):
        super(LayerNormLSTM, self).__init__(input_size=input_size, 
                                            hidden_size=hidden_size, 
                                            bias=bias,
                                            dropout=dropout,
                                            dropout_method=dropout_method)
        if ln_preact:
            self.ln_i2h = LayerNorm(4*hidden_size, learnable=learnable)
            self.ln_h2h = LayerNorm(4*hidden_size, learnable=learnable)
        self.ln_preact = ln_preact
        self.ln_cell = LayerNorm(hidden_size, learnable=learnable)

    def forward(self, x, hidden=None):
        do_dropout = self.training and self.dropout > 0.0
        h, c = (torch.zeros(1, x.size(1), self.hidden_size).to(x.device), 
                torch.zeros(1, x.size(1), self.hidden_size).to(x.device)) if hidden is None else hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        raw_outputs = []
        for inp in x:
            # inp = inp.view(x.size(1), -1)

            # Linear mappings
            i2h = self.i2h(inp)
            h2h = self.h2h(h)
            if self.ln_preact:
                i2h = self.ln_i2h(i2h)
                h2h = self.ln_h2h(h2h)
            preact = i2h + h2h

            # activations
            gates = preact[:, :3 * self.hidden_size].sigmoid()
            g_t = preact[:, 3 * self.hidden_size:].tanh()
            i_t = gates[:, :self.hidden_size] 
            f_t = gates[:, self.hidden_size:2 * self.hidden_size]
            o_t = gates[:, -self.hidden_size:]

            # cell computations
            if do_dropout and self.dropout_method == 'semeniuta':
                g_t = F.dropout(g_t, p=self.dropout, training=self.training)

            c_t = th.mul(c, f_t) + th.mul((1 - f_t), g_t)

            if do_dropout and self.dropout_method == 'moon':
                    c_t.data.set_(th.mul(c_t, self.mask).data)
                    c_t.data *= 1.0/(1.0 - self.dropout)

            c_t = self.ln_cell(c_t)
            h_t = th.mul(o_t, c_t.tanh())

            # Reshape for compatibility
            if do_dropout:
                if self.dropout_method == 'pytorch':
                    F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
                if self.dropout_method == 'gal':
                        h_t.data.set_(th.mul(h_t, self.mask).data)
                        h_t.data *= 1.0/(1.0 - self.dropout)

            raw_outputs.append(h_t)
            # h_t = h_t.view(1, h_t.size(0), -1)
            # c_t = c_t.view(1, c_t.size(0), -1)
        return torch.stack(raw_outputs).to(x.device), h_t


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, 
                 rnn_type, 
                 ntoken, 
                 ninp, 
                 nhid, 
                 nlayers, 
                 dropout=0.5,
                 dropouth=0.5, 
                 dropouti=0.5, 
                 dropoute=0.1, 
                 wdrop=0,
                 tie_weights=True,
                 lstm_type=nn.LSTM,
                 project=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'

        nfinal = nhid if project else ninp
        ndecode = ninp if tie_weights else nhid
        if rnn_type == 'LSTM':
            self.rnns = [lstm_type(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nfinal, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDropout(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDropout(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDropout(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.project = project
        if project:
            self.project_linear = nn.Linear(nfinal, ndecode)
        self.decoder = nn.Linear(ndecode, ntoken)
        self.ntoken = ntoken

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        # output = output.permute(1, 0, 2).contiguous()
        if hasattr(self, 'project_linear'):
            output = self.project_linear(output)
        result = self.decoder(output)
        # result = result.view(output.size(0), output.size(1), self.ntoken)
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result#, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]