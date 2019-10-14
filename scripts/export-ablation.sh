python -m bertviz.classifier -c confs/qqp_export.json --no_train --do_test_only --export ~/scratch/nw-qqp.pt --data_dir ablation-study/nw-qqp
python -m bertviz.classifier -c confs/qqp_export.json --no_train --do_test_only --export ~/scratch/nwnr-qqp.pt --data_dir ablation-study/nwnr-qqp
python -m bertviz.classifier -c confs/mnli_export.json --no_train --do_test_only --export ~/scratch/nw-mnli.pt --data_dir ablation-study/nw-mnli
python -m bertviz.classifier -c confs/mnli_export.json --no_train --do_test_only --export ~/scratch/nwnr-mnli.pt --data_dir ablation-study/nwnr-mnli

