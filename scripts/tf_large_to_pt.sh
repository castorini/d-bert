BASE_DIR=/mnt/nvme/bert/uncased_L-24_H-1024_A-16

python -m pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch --tf_checkpoint_path $BASE_DIR/bert_model.ckpt --bert_config_file $BASE_DIR/bert_config.json --pytorch_dump_path $BASE_DIR/pytorch_model.bin
