import argparse
import time
import math

from tqdm import tqdm
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.utils.data as tud


from utils import unwrap, dual_print
from tokenizers import space_tokenize
import data
import awd_model as model



# from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--test-eval', action='store_true')
parser.add_argument('--no-tied', dest='tied', action='store_false')
parser.add_argument('--lstm-type', type=str, choices=['lstm', 'layernorm'], default='lstm')
parser.add_argument('--project', action='store_true')
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--gpt2-model', type=str, default='gpt2')
parser.add_argument('--vocab-type', type=str, default='char', choices=['char', 'bpe', 'word'])
parser.add_argument('--spm-model', type=str)
parser.add_argument('--cache', type=str, default='aol-cache.pt')
parser.add_argument('--use-sos', action='store_true')
parser.add_argument('--no-split-encode', action='store_false', dest='split_encode')
args = parser.parse_args()

lstm_type = model.LayerNormLSTM if args.lstm_type == 'layernorm' else nn.LSTM

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f, map_location=lambda s, l: s)
        model = unwrap(model)

import os
import hashlib
sd = torch.load(args.cache)
dictionary = sd['dictionary']
train_ds, dev_ds, test_ds = sd['splits']
if args.spm_model:
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)


###############################################################################
# Build the model
###############################################################################

criterion = nn.CrossEntropyLoss(reduction='none')
use_char = args.vocab_type in ('char', 'word')
args.word_level = args.vocab_type == 'word'

ntokens = len(dictionary) if use_char else len(sp) + 1 + int(args.use_sos)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, 
    args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, lstm_type=lstm_type, project=args.project)
###
if args.resume:
    dual_print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in unwrap(model).rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
        model = unwrap(model)
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    dual_print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    if not args.test_eval: model = nn.DataParallel(model, dim=1)
    model.cuda()
#     criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
dual_print('Args:', args)
dual_print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################


def clone_hidden(hidden):
    return [(x.clone().detach(), y.clone().detach()) for x, y in hidden]


from bpe_encode import spm_encode


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    total_n = 0
    for batch in data_source:
        _, queries = batch
        if use_char:
            queries, mask = dictionary.sent2idx(queries, sos='<sos>' if args.use_sos else None, 
                tokenize_fn=space_tokenize if args.word_level else list)
        else:
            queries, mask, nchars = spm_encode(sp, queries, sos=args.use_sos)
        mask = torch.Tensor(mask).cuda().t()
        queries = torch.LongTensor(queries).cuda().t()
        with torch.no_grad():
            output = model(queries[:-1], return_h=False).permute(1, 2, 0)
        targets = queries[1:].t()
        crit = criterion(output, targets)
        mask_tot = mask[1:].sum()
        raw_loss = (crit * mask[1:].t()).sum() / mask_tot
        total_loss += raw_loss.item() * mask_tot
        total_n += mask_tot if use_char else nchars
    return total_loss / total_n


train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)
test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

from torch.distributions.categorical import Categorical


def sample_query(model, prefix_sampler, n=256):
    text = prefix_sampler()
    for _ in range(n):
        try:
            queries, _ = dictionary.sent2idx([text], sos='<sos>' if args.use_sos else None, 
                tokenize_fn=space_tokenize if args.word_level else list)
        except:
            return text
        queries = torch.LongTensor(queries).cuda().t()
        model.eval()
        with torch.no_grad():
            output = model(queries[:-1], return_h=False).permute(1, 2, 0)
            output = output[:, :, -1].contiguous().view(-1)
            cat = Categorical(logits=output)
            next_tok = dictionary.idx2word[cat.sample()]
        text = text + next_tok
    return text


def dbg_query(text, n=256):
    for _ in range(n):
        try:
            queries, _ = dictionary.sent2idx([text], sos='<sos>' if args.use_sos else None,
                tokenize_fn=space_tokenize if args.word_level else list)
        except:
            return text
        queries = torch.LongTensor(queries).cuda().t()
        model.eval()
        with torch.no_grad():
            output = model(queries[:-1], return_h=False).permute(1, 2, 0)
            output = output[:, :, -1].contiguous().view(-1)
            cat = Categorical(logits=output)
            next_tok = dictionary.idx2word[cat.sample()]
        text = text + next_tok
    return text

import sys
if args.test_eval:
    import code
    while True:
        x = input("> ")
        print(dbg_query(x))
    sys.exit(0)


if args.eval_only:
    val_loss = evaluate(dev_loader, args.batch_size)
    dual_print('-' * 89)
    dual_print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
      0, 0, val_loss, math.exp(val_loss), val_loss / math.log(2)))
    dual_print('-' * 89)
    sys.exit(0)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(dictionary) if use_char else len(sp) + 1 + int(args.use_sos)
    batch_idx, i = 0, 0
    total_n = 0
    for batch in train_loader:
        # bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        # optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        # data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        _, queries = batch
        if use_char:
            queries, mask = dictionary.sent2idx(queries, sos='<sos>' if args.use_sos else None,
                tokenize_fn=space_tokenize if args.word_level else list)
        else:
            queries, mask, nchars = spm_encode(sp, queries, split_encode=args.split_encode, sos=args.use_sos)
        mask = torch.Tensor(mask).cuda().t()
        queries = torch.LongTensor(queries).cuda().t()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        # output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        output = model(queries[:-1], return_h=False).permute(1, 2, 0)
        targets = queries[1:].t()
        crit = criterion(output, targets)
        mask_tot = mask[1:].sum().item()
        raw_loss = (crit * mask[1:].t()).sum() / mask_tot
        # raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization # FIX FOR BATCH_FIRST
        # if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        # if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.item() * mask_tot
        total_n += mask_tot if use_char else nchars
        optimizer.param_groups[0]['lr'] = lr2
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / total_n
            elapsed = time.time() - start_time
            dual_print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            total_n = 0
            start_time = time.time()
        ###
        batch_idx += 1
        # i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(dev_loader, args.batch_size)
            dual_print('-' * 89)
            dual_print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            dual_print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                dual_print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(dev_loader, args.batch_size)
            dual_print('-' * 89)
            dual_print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            dual_print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                dual_print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                dual_print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                dual_print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                dual_print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    dual_print('-' * 89)
    dual_print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_loader, args.batch_size)
dual_print('=' * 89)
dual_print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
dual_print('=' * 89)
