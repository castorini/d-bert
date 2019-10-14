# python -m bertviz.classifier -c confs/sts_export.json --no_train --do_test_only --export sts-logits-transfo.pt --data_dir 800k-transfo-sts
# python -m bertviz.classifier -c confs/sts_export.json --no_train --do_test_only --export 800k-export-cbert-sts/logits.pt --data_dir 800k-export-cbert-sts
python -m bertviz.classifier -c confs/sts_export.json --no_train --do_test_only --export 800k-export-randmask-sts/logits.pt --data_dir 800k-export-randmask-sts
# python -m bertviz.classifier -c confs/qqp_export.json --no_train --do_test_only
# python -m bertviz.classifier -c confs/cola_export.json --no_train --do_test_only
# python -m bertviz.classifier -c confs/sst_export.json --no_train --do_test_only

