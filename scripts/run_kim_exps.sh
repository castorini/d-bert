# for seed in $(seq $1 $2); do
#     python -m bert_distill.run.distill_kim --distill_lambda 1 --distill_temperature 4 --seed $seed >> kim_cnn_distilled.log
# done

for seed in $(seq $1 $2); do
    python -m bert_distill.run.distill_kim --distill_lambda 0.5 --ce_lambda 0.5 --distill_temperature 6 --seed $seed >> kim_cnn_distilled_high.log
done
