# python -m bertviz.classifier -c confs/qqp_export.json --no_train --do_test_only
# python -m bertviz.classifier -c confs/cola_export.json --no_train --do_test_only --export 800k-export-cbert-cola/logits.pt --data_dir 800k-export-cbert-cola
# python -m bertviz.classifier -c confs/cola_export.json --no_train --do_test_only --export 800k-randmask-cola/logits.pt --data_dir 800k-randmask-cola
# python -m bertviz.classifier -c confs/cola_export.json --data_dir 800k-randmask-cola
python -m bertviz.classifier -c confs/cola_export.json --no_train --do_test_only --data_dir ~/scratch/800k-imdb-cola --export ~/scratch/800k-imdb-cola/logits.pt
# python -m bertviz.classifier -c confs/sst_export.json --no_train --do_test_only

