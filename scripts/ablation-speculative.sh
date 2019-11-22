#!/bin/bash
DIRECTORY=$(dirname "$0")
echo "Directory is " $DIRECTORY
cd $DIRECTORY
cd ../

# Test Speculative/Retrospective Killing

# This result should show that across different sizes of clusters,
# we

for atoms in 4 8 16; do
for deadline in 10 20 30; do
for seed in {1..5}; do

python ~/sosp2019/scripts/evaluate_hypersched_ablation.py \
    --trainable-id optimus --num-jobs=200 --delay=0.1 \
    --result-file "~/retrokill.log" \
    --no-job-limit --strategy NONE --assume-linear --ignore-overhead \
    --global-deadline=$deadline --num-atoms=$atoms \
    --seed $seed

python ~/sosp2019/scripts/evaluate_hypersched_ablation.py \
    --trainable-id optimus --num-jobs=200 --delay=0.1 \
    --result-file "~/retrokill.log" \
    --no-job-limit --strategy NONE --assume-linear --ignore-overhead \
    --global-deadline=$deadline --num-atoms=$atoms \
    --seed $seed --no-retro
done
done
done
