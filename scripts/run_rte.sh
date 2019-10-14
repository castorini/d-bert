# python -m bertviz.classifier --config confs/rte.json --learning_rate 5e-5 --workspace /home/anonymized/scratch/rte-5e-5-wkspc 
# python -m bertviz.classifier --config confs/rte.json --learning_rate 4e-5 --workspace /home/anonymized/scratch/rte-4e-5-wkspc 
# python -m bertviz.classifier --config confs/rte.json --learning_rate 3e-5 --workspace /home/anonymized/scratch/rte-3e-5-wkspc 
python -m bertviz.classifier --config confs/rte.json --max_seq_length 256 --learning_rate 4e-5 --workspace /home/anonymized/scratch/rte-4e-5-wkspc-msl512 --seed 121345

