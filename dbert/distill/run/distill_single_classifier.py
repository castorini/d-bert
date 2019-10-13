import os
import sys

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Adadelta
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import read_args
from dbert.distill.data import find_dataset, set_seed, BinaryConfusionMatrix
import dbert.distill.model as mod


def evaluate(model, ds_iter, criterion, export_eval_labels=False):
    ds_iter.init_epoch()
    model.eval()
    n = 0
    loss = 0
    conf_matrix = BinaryConfusionMatrix()
    for batch in ds_iter:
        scores = model(batch.sentence)
        loss += criterion(scores, batch.label).item()
        conf_matrix.ingest(scores, batch.label)
        labels = scores.max(1)[1]
        n += batch.label.size(0)
        if export_eval_labels:
            print("\n".join(list(map(str, labels.cpu().tolist()))))
    loss /= n
    return conf_matrix, loss


def main():
    args = read_args(default_config="confs/kim_cnn_sst2.json")
    set_seed(args.seed)
    try:
        os.makedirs(args.workspace)
    except:
        pass
    torch.cuda.deterministic = True

    bert = mod.BertWrapper.load(args.bert_path, args.bert_weights_path)
    bert_embeds = bert.model.embeddings.word_embeddings
    tokenizer = bert.tokenizer

    dataset_cls = find_dataset(args.dataset_name)
    training_iter, dev_iter, test_iter = dataset_cls.iters(args.dataset_path, bert_embeds, tokenizer,
        batch_size=args.batch_size, train=args.train_file, dev=args.dev_file, test=args.test_file)

    args.dataset = training_iter.dataset
    args.words_num = len(training_iter.dataset.TEXT_FIELD.vocab)

    tgt_metric_dict = dict(sst2="acc", cola="mcc")
    model_dict = dict(bi_rnn=mod.BiRNNModel, kim_cnn=mod.KimCNN)
    tgt_metric_name = tgt_metric_dict.get(args.dataset_name, "acc")

    model = model_dict[args.model](args).to(args.device)
    ckpt_attrs = mod.load_checkpoint(model, args.workspace,
        best=args.load_best_checkpoint) if args.load_last_checkpoint or args.load_best_checkpoint else {}
    offset = ckpt_attrs.get("epoch_idx", -1) + 1
    args.epochs -= offset

    training_pbar = tqdm(total=len(training_iter), position=2)
    training_pbar.set_description("Training")
    dev_pbar = tqdm(total=args.epochs, position=1)
    dev_pbar.set_description("Dev")

    criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.KLDivLoss(reduction="none")
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = Adam(params, lr=args.lr)#, rho=0.95)
    increment_fn = mod.make_checkpoint_incrementer(model, args.workspace, save_last=True,
        best_loss=ckpt_attrs.get("best_dev_loss", 10000))
    non_embedding_params = model.non_embedding_params()

    if args.use_data_parallel:
        model = nn.DataParallel(model)
    if args.eval_test_only:
        test_conf_matrix, _ = evaluate(model, test_iter, criterion, export_eval_labels=args.export_eval_labels)
        print(test_conf_matrix.metrics[tgt_metric_name])
        return
    if args.epochs == 0:
        print("No epochs left from loaded model.", file=sys.stderr)
        return
    for epoch_idx in tqdm(range(args.epochs), position=0):
        training_iter.init_epoch()
        model.train()
        training_pbar.n = 0
        training_pbar.refresh()
        for batch in training_iter:
            training_pbar.update(1)
            optimizer.zero_grad()
            logits = model(batch.sentence)
            kd_logits = torch.stack((batch.logits_0, batch.logits_1), 1)
            focal_weight = 2 * (1 - F.softmax(logits / args.distill_temperature, 1)[torch.arange(0, logits.size(0)).long(), kd_logits.max(1)[1]]).detach()
            # focal_weight = 1
            kd = focal_weight * args.distill_lambda * kd_criterion(F.log_softmax(logits / args.distill_temperature, 1),
                F.softmax(kd_logits / args.distill_temperature, 1)).sum(1)
            loss = args.ce_lambda * criterion(logits, batch.label) + kd.mean()
            loss.backward()
            clip_grad_norm_(non_embedding_params, args.clip_grad)
            optimizer.step()
            conf_matrix = BinaryConfusionMatrix()
            conf_matrix.ingest(logits, batch.label)
            metric = conf_matrix.metrics[tgt_metric_name]
            kwargs = {tgt_metric_name: f"{metric:.2}"}
            training_pbar.set_postfix(**kwargs)

        model.eval()
        conf_matrix, dev_loss = evaluate(model, dev_iter, criterion)
        dev_pbar.update(1)
        dev_metric = conf_matrix.metrics[tgt_metric_name]
        kwargs = {tgt_metric_name: f"{dev_metric:.2}"}
        dev_pbar.set_postfix(**kwargs)
        is_best_dev = increment_fn(-dev_metric, epoch_idx=epoch_idx + offset, **conf_matrix.metrics)

        if is_best_dev:
            kwargs[tgt_metric_name] += " (best)"
            dev_pbar.set_postfix(**kwargs)
            test_conf_matrix, _ = evaluate(model, test_iter, criterion, export_eval_labels=args.export_eval_labels)
            test_metric = test_conf_matrix.metrics[tgt_metric_name]
        print("\n\nDev confusion matrix:", file=sys.stderr)
        print(conf_matrix, file=sys.stderr)
        print(conf_matrix.metrics, file=sys.stderr)
    training_pbar.close()
    dev_pbar.close()
    print(f"Test metric of the best model: {test_metric:.4f}", file=sys.stderr)
    print(test_metric)
    # focal_weight = -0.5 * F.softmax(logits / args.distill_temperature, 1) * F.log_softmax(logits / args.distill_temperature, 1)
    # focal_weight = focal_weight.sum(1).detach()


if __name__ == "__main__":
    main()
