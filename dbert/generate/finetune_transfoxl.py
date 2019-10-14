import argparse
import math
import time

from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer, AdamW, WarmupLinearSchedule
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.utils.data as tud

from .args import add_dict_options, opt, OptionEnum
from .utils import set_seed, dual_print


EOS_TOKEN = '<eos>'


ARGS = [
    OptionEnum.LEARNING_RATE.value.default(5e-5),
    OptionEnum.TRAIN_BATCH_SIZE.value.default(48),
    OptionEnum.EVAL_BATCH_SIZE.value.default(48),
    OptionEnum.NUM_TRAIN_EPOCHS.value,
    OptionEnum.SEED.value.default(1111), # match AWD-LSTM
    OptionEnum.WARMUP_PROPORTION.value.default(0.1),
    opt('--weight-decay', type=float, default=1e-3),
    opt('--cache-file', type=str, required=True),
    opt('--log-interval', type=int, default=100),
    opt('--save', type=str, default='gpt2.pt'),
    opt('--resume', type=str),
    opt('--cache-dir', type=str),
    opt('--no-train', action='store_false', dest='do_train'),
    opt('--transfo-model', type=str, default='transfo-xl-wt103'),
    opt('--test-eval', action='store_true'),
    opt('--use-sos', action='store_true'),
    opt('--no-split-encode', action='store_false', dest='split_encode'),
    opt('--no-drop-last', action='store_false', dest='drop_last'),
    opt('--conditioned-model', action='store_true'),
    opt('--reset', action='store_true')
]


def transfo_encode(tokenizer, queries, sos_idx=None, eos=True, max_len=80, 
        split_encode=False, idx2remap=None, return_raw=False, condition_model=False):
    tokens_lst = [tokenizer.tokenize(x.replace('\t', '<eos>'), add_eos=True) for x in queries]
    tokens_lst = [tokenizer.convert_tokens_to_ids(x) for x in tokens_lst]
    eos_append = [0]
    total_words = sum(len(x) - 1 for x in tokens_lst)
    sos_prepend = [sos_idx] if sos_idx is not None else []
    tokens_mask = [[1] * len(x) for x in tokens_lst]
    old_max_len = max_len
    max_len = min(max(map(len, tokens_lst)), max_len)
    tokens_lst = [x[:max_len] for x in tokens_lst]
    raw_decode = tokens_lst
    tokens_mask = [x[:max_len] for x in tokens_mask]
    total_in_chars = 0
    tokens_lst = [x + [0] * (max_len - len(x)) for x in tokens_lst]
    tokens_mask = [x + [0] * (max_len - len(x)) for x in tokens_mask]
    if total_in_chars == 0: total_in_chars = sum(min(len(q) - 1 + len(eos_append) + len(sos_prepend), old_max_len) for q in queries)
    if return_raw:
        return tokens_lst, tokens_mask, total_in_chars, raw_decode
    return tokens_lst, tokens_mask, total_in_chars, total_words


def sample_query(model, tokenizer, text, n=128, paired=False):
    model.cuda()
    encode = lambda x: [tokenizer.get_idx(z) for z in tokenizer.tokenize(x)]
    decoder = tokenizer.get_sym
    encoder = tokenizer.get_idx
    fin_count = int(paired) + 1

    new_toks = [decoder(x) for x in encode(text)]
    full_toks = new_toks.copy()
    past = None
    hist_len = len(new_toks)
    for _ in range(n):
        if hist_len > 1024:
            return ''
        inp_ = [encoder(x) for x in new_toks]
        queries = torch.LongTensor(inp_).unsqueeze(0).cuda()
        model.eval()
        with torch.no_grad():
            output, present = model(queries, mems=past)
            output = output.permute(0, 2, 1)[:, :, -1].contiguous().view(-1)
        try:
            new_tok = Categorical(probs=output.exp()).sample().item()
            full_toks.append(decoder(new_tok))
            if full_toks.count('<eos>') >= fin_count:
                return ' '.join(full_toks)
            hist_len += 1
            new_toks = [decoder(new_tok)]
            past = present
        except KeyError:
            break
    return ' '.join(full_toks)


def init_sos(model):
    embedding = model.transformer.wte.weight
    sos_tensor = torch.Tensor(1, embedding.data.size(1)).uniform_(-0.1, 0.1).to(embedding.data.device)
    embedding.data = torch.cat((embedding.data, sos_tensor)) # <eos> is <|endoftext|>
    return embedding.data.size(0) - 1


def main():
    def evaluate(data_source, split_encode=False):
        model.eval()
        total_loss = 0
        total_words = 0
        total_n = 0
        batch_idx = 0
        for batch in data_source:
            _, queries = batch
            try:
                queries, mask, total_chars, words = transfo_encode(tokenizer, queries, sos_idx, split_encode=split_encode, 
                    condition_model=args.conditioned_model)
            except KeyError:
                continue
            total_words += words
            mask = torch.Tensor(mask).cuda()
            queries = torch.LongTensor(queries).cuda()

            with torch.no_grad():
                output = model(queries[:, :-1])[0].permute(0, 2, 1)
            targets = queries[:, 1:]
            crit = criterion(output, targets)
            mask_tot = mask[:, 1:].sum()
            raw_loss = (crit * mask[:, 1:]).sum() / mask_tot
            loss = raw_loss

            total_loss += raw_loss.item() * mask_tot.item()
            total_n += total_chars
            # print(total_loss / (math.log(2) * total_n))

        cur_loss = total_loss / total_n
        elapsed = time.time() - start_time
        word_ppl = math.exp(total_loss / total_words)
        dual_print('-' * 89)
        dual_print('| end of epoch {:3d} | lr {:05.5f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
            epoch, optimizer.param_groups[0]['lr'],
            elapsed * 1000 / args.log_interval, cur_loss, word_ppl, cur_loss / math.log(2)))
        dual_print('-' * 89)
        return cur_loss / math.log(2)

    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)
    sd = torch.load(args.cache_file)

    tokenizer = TransfoXLTokenizer.from_pretrained(args.transfo_model, cache_dir='transfo-model')
    model = TransfoXLLMHeadModel.from_pretrained(args.transfo_model, cache_dir='transfo-model')
    if args.reset: model.apply(model.init_weights)
    sos_idx = None
    if not args.use_sos: sos_idx = None
    train_ds, dev_ds, test_ds = sd['splits']
    criterion = nn.CrossEntropyLoss(reduction='none')

    train_loader = tud.DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=args.drop_last)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=args.drop_last)
    test_loader = tud.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=args.drop_last)

    no_decay = ['bias']
    params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = args.num_train_epochs * len(train_loader)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
                                     t_total=num_train_optimization_steps)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=lambda s, l: s))
    if args.test_eval:
        while True:
            query = input("> ")
            print(sample_query(model, tokenizer, query))

    model = nn.DataParallel(model).cuda()
    start_time = time.time()
    best_bpc = 1000000

    if not args.do_train:
        evaluate(test_loader, split_encode=False)
        return

    for epoch in range(args.num_train_epochs):
        epoch += 1
        total_loss = 0
        total_words = 0
        total_n = 0
        batch_idx = 0
        for batch in train_loader:
            model.train()
            _, queries = batch
            try:
                queries, mask, total_chars, words = transfo_encode(tokenizer, queries, sos_idx, split_encode=args.split_encode, 
                    condition_model=args.conditioned_model)
            except KeyError:
                dual_print('Skipped batch')
                continue
            total_words += words
            mask = torch.Tensor(mask).cuda()
            queries = torch.LongTensor(queries).cuda()
            optimizer.zero_grad()

            output = model(queries[:, :-1])[0].permute(0, 2, 1)
            targets = queries[:, 1:]
            crit = criterion(output, targets)
            mask_tot = mask[:, 1:].sum()
            raw_loss = (crit * mask[:, 1:]).sum() / mask_tot

            loss = raw_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += raw_loss.item() * mask_tot.item()
            total_n += total_chars
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / total_n
                word_ppl = math.exp(total_loss / total_words)
                total_words = 0
                elapsed = time.time() - start_time
                dual_print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, word_ppl, cur_loss / math.log(2)))
                total_loss = 0
                total_n = 0
                start_time = time.time()
            batch_idx += 1
        bpc = evaluate(dev_loader)
        if bpc < best_bpc:
            best_bpc = bpc
            torch.save(model.module.state_dict(), args.save)
    evaluate(test_loader)


if __name__ == '__main__':
    main()
