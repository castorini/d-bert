# python -m bertviz.classifier -c confs/qqp_export.json --no_train --do_test_only
# python -m bertviz.classifier -c confs/cola_export.json --no_train --do_test_only
# python -m bertviz.classifier -c confs/sst_export.json --no_train --do_test_only --data_dir 800k-export-cbert-sst --export 800k-export-cbert-sst/logits.pt
# python -m bertviz.classifier -c confs/sst_export.json --no_train --do_test_only --data_dir abl-sst2 --export abl-sst2/logits.pt
python -m bertviz.classifier -c confs/sst_export.json --no_train --do_test_only --data_dir sstabl --export sstabl/logits.pt
# python -m bertviz.classifier -c confs/sst_export.json --no_train --do_test_only --data_dir 800k-export-gpt2-345m-sst --export 800k-export-gpt2-345m-sst/logits.pt

