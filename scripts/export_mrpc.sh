# python -m bertviz.classifier -c confs/sts_export.json --no_train --do_test_only --export sts-logits-transfo.pt --data_dir 800k-transfo-sts
# python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export mrpc-logits-gpt2-345m.pt --data_dir 800k-export
# python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export mrpc-orig-logits.pt --data_dir mrpc-orig-export
# python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export mrpc-transfo-logits.pt --data_dir 800k-export-transfo-mrpc
# python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export mrpc-randmask-logits.pt --data_dir mrpc-randmask-144k-export
python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export 800k-export-mrpc-cbert/logits.pt --data_dir 800k-export-mrpc-cbert
# python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export mrpc-final-randmask-logits-800k.pt --data_dir 800k-randmask-mrpc
# python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export 800k-export-cbert-mrpc/logits.pt --data_dir 800k-export-cbert-mrpc
# python -m bertviz.classifier -c confs/mrpc_export.json --no_train --do_test_only --export mrpc-logits-gpt2.pt --data_dir export-mrpc-gpt2
# python -m bertviz.classifier -c confs/qqp_export.json --no_train --do_test_only
# python -m bertviz.classifier -c confs/cola_export.json --no_train --do_test_only
# python -m bertviz.classifier -c confs/sst_export.json --no_train --do_test_only

