# Knowledge Distillation
This repository provides implementations of our original BERT distillation technique, [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136), and our more recent text generation-based method, [Natural Language Generation for Effective Knowledge Distillation](http://ralphtang.com/papers/deeplo2019.pdf).
The latter is more effective than the former, albeit at the cost of computational efficiency, requiring multiple GPUs to fine-tune Transformer-XL or GPT-2 for constructing the transfer set.
Thus, we will henceforth refer to them as `d-lite` and `d-heavy`, respectively.

The codebase admittedly is in a messy state; we plan to continue refactoring it.
If you desire **just the data** from our second paper, you may download that [here](https://nlp.nyc3.digitaloceanspaces.com/distillation-data.zip).

## Transfer Set Construction

Our first task is to construct a transfer set.
The two papers differ for this step only.

### Instructions for `d-lite`
1. Install the dependencies using `pip install -r requirements.txt`.

2. Build the transfer set by running `python -m dbert.distill.run.augment_data --dataset_file (the TSV dataset file) > (output file)` or `python -m dbert.distill.run.augment_paired_data --task (the task) --dataset_file (the TSV dataset file) > (output file)`.

These follow the GLUE datasets' formats.

### Instructions for `d-heavy`
1. Install the dependencies using `pip install -r requirements.txt`.

At the time of the experiments, [`transformers`](https://github.com/huggingface/transformers) was still `pytorch_pretrained_bert`, with no support for GPT-2 345M, so we had to add that manually.
We provide the configuration file in `confs/345m-config.json`.

2. Build a cache dataset using `python -m dbert.generate.cache_datasets --data-dir (directory) --output-file (cache file)`.

The data directory should contain `train.tsv`, `dev.tsv`, and `test.tsv`, as in [GLUE](https://gluebenchmark.com). For sentence-pair datasets, append `--dataset-type pair-sentence`.

3. Fine-tune Transformer-XL using `python -m dbert.generate.finetune_transfoxl --save (checkpoint file) --cache-file (the cache file) --train-batch-size (# that'll fit)`.

You can also use `generate.finetune_gpt` for fine-tuning GPT-2. In our paper, we used a batch size of 48, which might be too much for your system to handle. You can probably reduce it without much change in the final quality.

4. Build a prefix sampler for the transformer-dataset pair with `python -m dbert.generate.build_sampler --cache-file (the cache file) --save (the prefix sampler output file).

For GPT-2, add `--model-type gpt2`.

5. Sample from the Transformer-XL using `python -m dbert.generate.sample_transfoxl --prefix-file (the prefix sampler) > (output file)`.

For sentence-pair sampling, append `--paired`.

## Teacher Fine-tuning

Next, fine-tune the teacher, e.g., large BERT. 

1. Run `python -m dbert.finetune.classifier --config confs/*-ft.json --learning_rate 4e-5 --workspace (output workspace directory)`

See `confs/sst2-ft.json` for an example of the configuration. You need to modify `data_dir` and `--model_file` appropriately, or specify them as command-line options (e.g., `--data_dir`).

2. To export the logits of the transfer set file, run `python -m dbert.finetune.classifier --config confs/*_export.json --no_train --do_test_only --data_dir (the data directory) --export (logits file)`

See `scripts/export_sst.sh` for an example.

## Student Distillation

Finally, we can distill the exported logits into the student model.

1. Join the logits to the original TSV using `python -m dbert.distill.run.join_logits`.

2. Download the word vectors from [here](https://git.uwaterloo.ca/jimmylin/Castor-data).

3. Distill and train a BiLSTM model using `python -m dbert.run.distill_birnn --config confs/birnn_sst2.json`.
