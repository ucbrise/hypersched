#!/bin/bash
set -e
set -x

LOGFILE="~/final-imagenet-compare.log"
rm $LOGFILE || true
rm "${LOGFILE}".csv || true
ulimit -S -n 16384

for model in resnet50; do #
for atoms in 16; do
for deadline in 7200; do
for maxt in 500; do # 100 50; do


python ~/sosp2019/scripts/evaluate_dynamic_asha.py \
    --redis-address="localhost:6379" \
    --num-atoms=$atoms \
    --num-jobs=200 \
    --seed=$1 \
    --sched hyper \
    --result-file=$LOGFILE \
    --max-t=$maxt \
    --global-deadline=$deadline \
    --trainable-id pytorch \
    --model-string $model \
    --data imagenet \
    --grid=2
    # --trainable-id optimus


python ~/sosp2019/scripts/evaluate_dynamic_asha.py \
    --redis-address="localhost:6379" \
    --num-atoms=$atoms \
    --num-jobs=200 \
    --seed=$1 \
    --sched asha2 \
    --result-file=$LOGFILE \
    --max-t=$maxt \
    --global-deadline=$deadline \
    --trainable-id pytorch \
    --model-string $model \
    --data imagenet \
    --grid=2
    # --trainable-id optimus

done
done
done
done
done
