# python -m bertviz.classifier --config confs/rte.json --learning_rate 5e-5 --workspace /home/anonymized/scratch/rte-5e-5-wkspc 
# python -m bertviz.classifier --config confs/rte.json --learning_rate 4e-5 --workspace /home/anonymized/scratch/rte-4e-5-wkspc 
# python -m bertviz.classifier --config confs/rte.json --learning_rate 3e-5 --workspace /home/anonymized/scratch/rte-3e-5-wkspc 
#python -m bertviz.classifier --config confs/mrpc-ft.json --max_seq_length 256 --learning_rate 5e-5 --workspace /home/anonymized/scratch/mrpc-5e-5-wkspc-msl256
#python -m bertviz.classifier --config confs/mrpc-ft.json --max_seq_length 256 --learning_rate 4e-5 --workspace /home/anonymized/scratch/mrpc-4e-5-wkspc-msl256
#python -m bertviz.classifier --config confs/mrpc-ft.json --max_seq_length 256 --learning_rate 3e-5 --workspace /home/anonymized/scratch/mrpc-3e-5-wkspc-msl256
python -m bertviz.classifier --config confs/mrpc-ft.json --max_seq_length 384 --learning_rate 2e-5 --workspace /home/anonymized/scratch/mrpc-2e-5-wkspc-msl384-ep4 --seed 123456 --train_batch_size 32 --num_train_epochs 4

