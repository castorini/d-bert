import os
import sys

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adadelta
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import read_args
from dbert.distill.data import find_dataset, set_seed, replace_embeds, list_field_mappings
import dbert.distill.model as mod


def evaluate_score(model, ds_iter, criterion, export_eval_labels=False):
    ds_iter.init_epoch()
    model.eval()
    acc = 0
    n = 0
    loss = 0
    gts = []
    preds = []
    for batch in tqdm(ds_iter):
        scores = model(batch.sentence)
        try:
            gts.extend(batch.score.view(-1).tolist())
            preds.extend(scores.view(-1).tolist())
        except:
            continue
        n += scores.size(0)
        if export_eval_labels:
            print("\n".join(list(map(str, scores.view(-1).cpu().tolist()))))
    if len(gts) == 0:
        return 0, 0
    pr = pearsonr(preds, gts)[0]
    sr = spearmanr(preds, gts)[0]
    return pr, sr


def evaluate(model, ds_iter, criterion, export_eval_labels=False):
    ds_iter.init_epoch()
    model.eval()
    acc = 0
    n = 0
    loss = 0
    for batch in ds_iter:
        scores = model(batch.sentence)
        labels = scores.max(1)[1]
        try:
            loss += criterion(scores, batch.label).item()
            acc += ((labels == batch.label).float().sum()).item()
        except AttributeError: # We're on GLUE
            pass
        n += scores.size(0)
        if export_eval_labels:
            # print("\n".join(list(map(str, labels.cpu().tolist()))))
            label_strs = list(map(str, labels.cpu().tolist()))
            logit_strs = ['\t'.join(map(str, score.cpu().view(-1).tolist())) for score in scores]
            for lbl_str, logit_str in zip(label_strs, logit_strs):
                print('\t'.join((lbl_str, logit_str)))
    acc /= n
    loss /= n
    return acc, loss


def main():
    args = read_args(default_config="confs/kim_cnn_sst2.json")
    set_seed(args.seed)
    try:
        os.makedirs(args.workspace)
    except:
        pass
    torch.cuda.deterministic = True

    dataset_cls = find_dataset(args.dataset_name)
    training_iter, dev_iter, test_iter = dataset_cls.iters(args.dataset_path, args.vectors_file, args.vectors_dir,
        batch_size=args.batch_size, device=args.device, train=args.train_file, dev=args.dev_file, test=args.test_file)

    args.dataset = training_iter.dataset
    args.words_num = len(training_iter.dataset.TEXT_FIELD.vocab)
    model = mod.BiRNNModel(args).to(args.device)
    ckpt_attrs = mod.load_checkpoint(model, args.workspace,
        best=args.load_best_checkpoint) if args.load_last_checkpoint or args.load_best_checkpoint else {}
 
    offset = ckpt_attrs.get("epoch_idx", -1) + 1
    args.epochs -= offset

    training_pbar = tqdm(total=len(training_iter), position=2)
    training_pbar.set_description("Training")
    dev_pbar = tqdm(total=args.epochs, position=1)
    dev_pbar.set_description("Dev")

    criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.MSELoss()
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = Adadelta(params, lr=args.lr, rho=0.95)
    increment_fn = mod.make_checkpoint_incrementer(model, args.workspace, save_last=True,
        best_loss=ckpt_attrs.get("best_dev_loss", 10000))
    non_embedding_params = model.non_embedding_params()
    print(sum(p.numel() for p in list(model.state_dict().values())[2:]))

    if args.use_data_parallel:
        model = nn.DataParallel(model)
    if args.eval_test_only:
        if args.float_score:
            pr, sr = evaluate_score(model, test_iter, criterion, export_eval_labels=args.export_eval_labels)
        else:
            test_acc, _ = evaluate(model, test_iter, criterion, export_eval_labels=args.export_eval_labels)
            print(test_acc)
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
            if args.float_score:
                kd_logits = torch.stack((batch.score,), 1)
            else:
                kd_logits = torch.stack((batch.logits_0, batch.logits_1), 1)
            loss = args.distill_lambda * kd_criterion(logits, kd_logits)
            if not args.float_score: loss += args.ce_lambda * criterion(logits, batch.label)
            loss.backward()
            clip_grad_norm_(non_embedding_params, args.clip_grad)
            optimizer.step()
            if args.float_score:
                training_pbar.set_postfix(loss=f"{loss.item():.4}")
            else:
                acc = ((logits.max(1)[1] == batch.label).float().sum() / batch.label.size(0)).item()
                training_pbar.set_postfix(accuracy=f"{acc:.2}")

        model.eval()
        if args.float_score:
            dev_pr, dev_sr = evaluate_score(model, dev_iter, criterion)
            dev_pbar.update(1)
            dev_pbar.set_postfix(pearsonr=f"{dev_pr:.4}")
            is_best_dev = increment_fn(-dev_pr, dev_sr=dev_sr, dev_pr=dev_pr, epoch_idx=epoch_idx + offset)
            if is_best_dev:
                dev_pbar.set_postfix(pearsonr=f"{dev_pr:.4} (best loss)")
        else:
            dev_acc, dev_loss = evaluate(model, dev_iter, criterion)
            dev_pbar.update(1)
            dev_pbar.set_postfix(accuracy=f"{dev_acc:.4}")
            is_best_dev = increment_fn(-dev_acc, dev_acc=dev_acc, epoch_idx=epoch_idx + offset)

            if is_best_dev:
                dev_pbar.set_postfix(accuracy=f"{dev_acc:.4} (best loss)")
                test_acc, _ = evaluate(model, test_iter, criterion, export_eval_labels=args.export_eval_labels)
    training_pbar.close()
    dev_pbar.close()
    print(f"Test accuracy of the best model: {test_acc:.4f}", file=sys.stderr)
    print(test_acc)


if __name__ == "__main__":
    main()
