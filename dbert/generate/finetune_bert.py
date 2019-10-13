from collections import Counter
import argparse
import functools
import math
import random
import time

from pytorch_transformers import BertTokenizer, BertForMaskedLM, AdamW, WarmupLinearSchedule
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.utils.data as tud

from .args import add_dict_options, opt, OptionEnum
from .utils import set_seed, dual_print
from .data import tokenize_batch


EOS_TOKEN = '[SEP]'


ARGS = [
    OptionEnum.LEARNING_RATE.value.default(5e-5),
    OptionEnum.TRAIN_BATCH_SIZE.value.default(48),
    OptionEnum.EVAL_BATCH_SIZE.value.default(48),
    OptionEnum.NUM_TRAIN_EPOCHS.value,
    OptionEnum.SEED.value.default(1111), # match AWD-LSTM
    OptionEnum.WARMUP_PROPORTION.value.default(0.1),
    opt('--weight-decay', type=float, default=1e-3),
    opt('--cache-file', type=str, default='aol-cache.pt'),
    opt('--log-interval', type=int, default=100),
    opt('--save', type=str, default='bert.pt'),
    opt('--resume', type=str),
    opt('--cache-dir', type=str),
    opt('--no-train', action='store_false', dest='do_train'),
    opt('--bert-model', type=str, default='uncased-bert-large'),
    opt('--test-eval', action='store_true'),
    opt('--use-sos', action='store_true'),
    opt('--no-split-encode', action='store_false', dest='split_encode'),
    opt('--no-drop-last', action='store_false', dest='drop_last'),
    opt('--conditioned-model', action='store_true'),
    opt('--msl', type=int, default=128)
]


def bert_encode(tokenizer, queries, sos_idx=None, eos=True, max_len=128, 
        split_encode=False, idx2remap=None, return_raw=False, condition_model=False, mask_prob=0.15):
    tokens_lst = [tokenizer.tokenize('[CLS] ' + x.replace('\t', EOS_TOKEN) + f' {EOS_TOKEN}') for x in queries]
    targets = []
    masks_lst = []
    targets_lst = []
    segment_ids = []
    for tokens in tokens_lst:
        x = [0] * len(tokens)
        sep_idx = tokens.index(EOS_TOKEN) + 1
        try:
            x[sep_idx:] = 1
        except:
            pass
        segment_ids.append(x)
    ids = list(tokenizer.vocab.values())[1000:]
    new_tokens_lst = []
    for tokens in tokens_lst:
        new_tokens = []
        masks = []
        targets = []
        for tok in tokens:
            use_mask = random.random() < mask_prob and tok != '[SEP]' and tok != '[CLS]'
            masks.append(int(use_mask))
            roll = random.random()
            if use_mask:
                if roll < 0.8: new_tok = tokenizer.vocab['[MASK]']
                elif roll < 0.9 and 0.8 <= roll: new_tok = tokenizer.vocab[tok]
                else: new_tok = random.choice(ids)
                new_tokens.append(new_tok)
                targets.append(tokenizer.vocab[tok])
            else:
                targets.append(-1)
                new_tokens.append(tokenizer.vocab[tok])
        masks_lst.append(masks)
        targets_lst.append(targets)
        new_tokens_lst.append(new_tokens)
    tokens_lst = new_tokens_lst
    input_mask = [[1] * len(x) for x in tokens_lst]
    old_max_len = max_len
    max_len = min(max(map(len, tokens_lst)), max_len)

    tokens_lst = [x[:max_len] for x in tokens_lst]
    targets_lst = [x[:max_len] for x in targets_lst]
    segment_ids = [x[:max_len] for x in segment_ids]
    masks_lst = [x[:max_len] for x in masks_lst]
    input_mask = [x[:max_len] for x in input_mask]

    raw_decode = tokens_lst

    tokens_lst = [x + [0] * (max_len - len(x)) for x in tokens_lst]
    input_mask = [x + [0] * (max_len - len(x)) for x in input_mask]
    targets_lst = [x + [0] * (max_len - len(x)) for x in targets_lst]
    segment_ids = [x + [0] * (max_len - len(x)) for x in segment_ids]
    masks_lst = [x + [0] * (max_len - len(x)) for x in masks_lst]

    return tokens_lst, input_mask, targets_lst, masks_lst, segment_ids


def augment_texts(model, tokenizer, texts, n=128, max_len=128):
    model.cuda()
    model.eval()
    tokens_lst, input_mask, targets_lst, masks_lst, segment_ids = bert_encode(tokenizer, texts, mask_prob=0, max_len=max_len)
    input_mask = torch.LongTensor(input_mask).cuda()
    targets = torch.LongTensor(targets_lst).cuda()
    masks = torch.LongTensor(masks_lst).cuda()
    segment_ids = torch.LongTensor(segment_ids).cuda()
    tokens_lst = torch.LongTensor(tokens_lst).cuda()

    with torch.no_grad():
        preds = model(tokens_lst, token_type_ids=segment_ids, attention_mask=input_mask)
    pred_toks_lst = []
    for tokens, pred in zip(tokens_lst, preds):
        pred_toks = []
        for tok, p in zip(tokens, pred):
            if tok.item() == tokenizer.vocab['[MASK]']:
                pred_toks.append(Categorical(logits=p.view(-1)).sample().item())
            elif tok.item() == tokenizer.vocab['[PAD]'] or tok.item() == tokenizer.vocab['[CLS]']:
                pass
            else:
                pred_toks.append(tok.item())
        pred_toks_lst.append(pred_toks)
    pred_toks = [tokenizer.convert_ids_to_tokens(pred) for pred in pred_toks_lst]
    pred_toks = [' '.join(x).replace(' ##', '').replace('##', '') for x in pred_toks]
    return pred_toks


def init_sos(model):
    embedding = model.transformer.wte.weight
    sos_tensor = torch.Tensor(1, embedding.data.size(1)).uniform_(-0.1, 0.1).to(embedding.data.device)
    embedding.data = torch.cat((embedding.data, sos_tensor)) # <eos> is <|endoftext|>
    return embedding.data.size(0) - 1


def main():
    def evaluate(data_source, split_encode=False):
        model.eval()
        total_loss = 0
        total_n = 0
        batch_idx = 0
        for batch in data_source:
            _, queries = batch
            try:
                tokens_lst, input_mask, targets_lst, masks_lst, segment_ids = bert_encode(tokenizer, queries, sos_idx, split_encode=False, 
                    condition_model=args.conditioned_model, max_len=args.msl)
            except KeyError:
                continue
            input_mask = torch.LongTensor(input_mask).cuda()
            targets = torch.LongTensor(targets_lst).cuda()
            masks = torch.LongTensor(masks_lst).cuda()
            segment_ids = torch.LongTensor(segment_ids).cuda()
            tokens_lst = torch.LongTensor(tokens_lst).cuda()
            queries = tokens_lst

            with torch.no_grad():
                loss = model(queries, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=targets).mean()
            mask_tot = masks.sum()
            total_loss += loss.item() * mask_tot.item()
            total_n += mask_tot.item()

        cur_loss = total_loss / total_n
        elapsed = time.time() - start_time
        dual_print('-' * 89)
        dual_print('| end of epoch {:3d} | lr {:05.5f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
            epoch, optimizer.param_groups[0]['lr'],
            elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
        dual_print('-' * 89)
        return cur_loss

    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)
    sd = torch.load(args.cache_file)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
    model = BertForMaskedLM.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
    sos_idx = None
    train_ds, dev_ds, test_ds = sd['splits']
    criterion = nn.CrossEntropyLoss(reduction='none')

    train_loader = tud.DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=args.drop_last)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=args.drop_last)
    test_loader = tud.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=args.drop_last)

    no_decay = ['bias', 'LayerNorm.weight']
    params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = args.num_train_epochs * len(train_loader)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(args.warmup_proportion * num_train_optimization_steps), t_total=num_train_optimization_steps)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=lambda s, l: s))

    model = nn.DataParallel(model).cuda()
    start_time = time.time()
    best_loss = 1000000

    if not args.do_train:
        evaluate(test_loader, split_encode=False)
        return

    for epoch in range(args.num_train_epochs):
        epoch += 1
        total_loss = 0
        total_n = 0
        batch_idx = 0
        for batch in train_loader:
            model.train()
            _, queries = batch
            try:
                tokens_lst, input_mask, targets_lst, masks_lst, segment_ids = bert_encode(tokenizer, queries, sos_idx, split_encode=False, 
                    condition_model=args.conditioned_model, max_len=args.msl)
            except KeyError:
                continue
            input_mask = torch.LongTensor(input_mask).cuda()
            targets = torch.LongTensor(targets_lst).cuda()
            masks = torch.LongTensor(masks_lst).cuda()
            segment_ids = torch.LongTensor(segment_ids).cuda()
            tokens_lst = torch.LongTensor(tokens_lst).cuda()
            queries = tokens_lst
            optimizer.zero_grad()

            loss = model(queries, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=targets).mean()
            mask_tot = masks.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * mask_tot.item()
            total_n += mask_tot.item()
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
            batch_idx += 1
        loss = evaluate(dev_loader)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.module.state_dict(), args.save)
    evaluate(test_loader)


if __name__ == '__main__':
    main()
