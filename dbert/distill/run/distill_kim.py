import os
import sys

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adadelta
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import read_args
from dbert.distill.data import find_dataset, set_seed
import dbert.distill.model as mod


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
        n += batch.sentence.size(0)
        if export_eval_labels:
            print("\n".join(list(map(str, labels.cpu().tolist()))))
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
    model = mod.KimCNN(args).to(args.device)
    ckpt_attrs = mod.load_checkpoint(model, args.workspace,
        best=args.load_best_checkpoint) if args.load_last_checkpoint or args.load_best_checkpoint else {}
    offset = ckpt_attrs.get("epoch_idx", -1) + 1
    args.epochs -= offset

    training_pbar = tqdm(total=len(training_iter), position=2)
    training_pbar.set_description("Training")
    dev_pbar = tqdm(total=args.epochs, position=1)
    dev_pbar.set_description("Dev")

    criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.KLDivLoss(reduction="batchmean")
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = Adadelta(params, lr=args.lr, rho=0.95)
    increment_fn = mod.make_checkpoint_incrementer(model, args.workspace, save_last=True, 
        best_loss=ckpt_attrs.get("best_dev_loss", 10000))
    non_embedding_params = model.non_embedding_params()

    if args.use_data_parallel:
        model = nn.DataParallel(model)
    if args.eval_test_only:
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
            print(batch.sentence[1])
            print(batch.sentence[0])
            training_pbar.update(1)
            optimizer.zero_grad()
            logits = model(batch.sentence)
            loss = args.ce_lambda * criterion(logits, batch.label)
            if args.distill_lambda:
                kd_logits = torch.stack((batch.logits_0, batch.logits_1), 1)
                kd = args.distill_lambda * kd_criterion(F.log_softmax(logits / args.distill_temperature, 1), 
                    F.softmax(kd_logits / args.distill_temperature, 1))
                loss += kd
            # focal_weight = -0.5 * F.softmax(logits / args.distill_temperature, 1) * F.log_softmax(logits / args.distill_temperature, 1)
            # focal_weight = focal_weight.sum(1).detach()
            loss.backward()
            clip_grad_norm_(non_embedding_params, args.clip_grad)
            optimizer.step()
            acc = ((logits.max(1)[1] == batch.label).float().sum() / batch.label.size(0)).item()
            training_pbar.set_postfix(accuracy=f"{acc:.2}")

        model.eval()
        dev_acc, dev_loss = evaluate(model, dev_iter, criterion)
        dev_pbar.update(1)
        dev_pbar.set_postfix(accuracy=f"{dev_acc:.4}")
        is_best_dev = increment_fn(dev_loss, dev_acc=dev_acc, epoch_idx=epoch_idx + offset)

        if is_best_dev:
            dev_pbar.set_postfix(accuracy=f"{dev_acc:.4} (best loss)")
            test_acc, _ = evaluate(model, test_iter, criterion, export_eval_labels=args.export_eval_labels)
    training_pbar.close()
    dev_pbar.close()
    print(f"Test accuracy of the best model: {test_acc:.4f}", file=sys.stderr)
    print(test_acc)


if __name__ == "__main__":
    main()

