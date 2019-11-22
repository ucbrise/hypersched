#!/bin/bash
set -e

LOGFILE="~/final-time-compare.log"
ulimit -S -n 16384
DIRECTORY=$(dirname "$0")
echo "Directory is " $DIRECTORY
cd $DIRECTORY
cd ../

for model in resnet50; do #
for seed in 1 2 3 4 5; do
for atoms in 8; do
for deadline in 900 1800 3600; do
for maxt in 200; do # 100 50; do


python ~/sosp2019/scripts/evaluate_dynamic_asha.py \
    --num-atoms=$atoms \
    --num-jobs=100 \
    --seed=$seed \
    --sched hyper \
    --result-file=$LOGFILE \
    --max-t=$maxt \
    --global-deadline=$deadline \
    --trainable-id pytorch \
    --model-string $model \
    --data cifar
    # --trainable-id optimus


python ~/sosp2019/scripts/evaluate_dynamic_asha.py \
    --num-atoms=$atoms \
    --num-jobs=100 \
    --seed=$seed \
    --sched asha2 \
    --result-file=$LOGFILE \
    --max-t=$maxt \
    --global-deadline=$deadline \
    --trainable-id pytorch \
    --model-string $model \
    --data cifar
    # --trainable-id optimus

done
done
done
done
done
