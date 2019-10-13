import os
import sys

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adadelta, Adam
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import read_args
from dbert.distill.data import find_dataset, set_seed, list_field_mappings, replace_embeds
import dbert.distill.model as mod


def evaluate(model, ds_iter, criterion, export_eval_labels=False):
    ds_iter.init_epoch()
    model.eval()
    acc = 0
    n = 0
    loss = 0
    for batch in tqdm(ds_iter):
        scores = model(batch.question1, batch.question2)
        loss += criterion(scores, batch.is_duplicate).item()
        labels_prob = F.softmax(scores, dim=1)[:,0]
        labels = scores.max(1)[1]
        labels_prob = labels
        acc += ((labels == batch.is_duplicate).float().sum()).item()
        n += scores.size(0)
        if export_eval_labels:
            print("\n".join(list(map(str, labels_prob.cpu().tolist()))))
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
    model = mod.SiameseRNNModel(args).to(args.device)

    sd = torch.load('sst.pt')['state_dict']
    del sd['static_embed.weight']
    del sd['non_static_embed.weight']
    del sd['fc1.weight']
    del sd['fc1.bias']
    del sd['fc2.weight']
    del sd['fc2.bias']
    model.load_state_dict(sd, strict=False)
    mod.init_embedding(model, args)
    # embs, field_src  = torch.load('embs_tmp.pt')
    # field_mappings = list_field_mappings(dataset_cls.TEXT_FIELD, field_src)
    # replace_embeds(model.non_static_embed, embs, field_mappings)
    model.to(args.device)

    ckpt_attrs = mod.load_checkpoint(model, args.workspace,
        best=args.load_best_checkpoint) if args.load_last_checkpoint or args.load_best_checkpoint else {}
    torch.save((model.non_static_embed, dataset_cls.TEXT_FIELD.vocab), 'qqp-embs.pt')
    return
    offset = ckpt_attrs.get("epoch_idx", -1) + 1
    args.epochs -= offset

    training_pbar = tqdm(total=len(training_iter), position=2)
    training_pbar.set_description("Training")
    dev_pbar = tqdm(total=args.epochs, position=1)
    dev_pbar.set_description("Dev")

    criterion = nn.CrossEntropyLoss()
    kd_criterion = nn.MSELoss()# KLDivLoss(reduction="batchmean")
    filter_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and 'fc' in n]
    params = list(map(lambda x: x[1], filter_params))
    # print([x[0] for x in filter_params])
    optimizer = Adadelta(params, lr=args.lr, rho=0.95)
    #optimizer = Adam(params, lr=args.lr)
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
            training_pbar.update(1)
            optimizer.zero_grad()
            logits = model(batch.question1, batch.question2)
            # kd_logits = torch.stack((batch.logits_0, batch.logits_1), 1)
            #kd = args.distill_lambda * kd_criterion(F.log_softmax(logits / args.distill_temperature, 1),
            #    F.softmax(kd_logits / args.distill_temperature, 1))
            # kd = args.distill_lambda * kd_criterion(logits, kd_logits)
            loss = criterion(logits, batch.is_duplicate)
            loss.backward()
            clip_grad_norm_(non_embedding_params, args.clip_grad)
            optimizer.step()
            acc = ((logits.max(1)[1] == batch.is_duplicate).float().sum() / batch.is_duplicate.size(0)).item()
            training_pbar.set_postfix(accuracy=f"{acc:.2}")

        model.eval()
        dev_acc, dev_loss = evaluate(model, dev_iter, criterion)
        dev_pbar.update(1)
        dev_pbar.set_postfix(accuracy=f"{dev_acc:.4}")
        is_best_dev = increment_fn(dev_loss, dev_acc=dev_acc, epoch_idx=epoch_idx + offset)

        if is_best_dev:
            dev_pbar.set_postfix(accuracy=f"{dev_acc:.4} (best loss)")
            # test_acc, _ = evaluate(model, test_iter, criterion, export_eval_labels=args.export_eval_labels)
    training_pbar.close()
    dev_pbar.close()
    print(f"Test accuracy of the best model: {test_acc:.4f}", file=sys.stderr)
    print(test_acc)


if __name__ == "__main__":
    main()
