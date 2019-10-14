BASE_DIR=~/data/uncased_L-12_H-768_A-12

python -m pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch --tf_checkpoint_path $BASE_DIR/bert_model.ckpt --bert_config_file $BASE_DIR/bert_config.json --pytorch_dump_path $BASE_DIR/pytorch_model.bin