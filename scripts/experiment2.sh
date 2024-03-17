#!/bin/bash
pruning_ratios=(0.5 0.7 0.9)
CUDA_VISIBLE_DEVICES=0

# iterative pruning with weight reinitialization
for ratio in "${pruning_ratios[@]}"; do
    python3 main.py \
    --model resnetb \
    --depth 20 \
    --priter 4 \
    --epochs 160 \
    --batch-size 64 \
    --sched 'manual' \
    --lr_update_epochs 80 120 \
    --gamma 0.1 \
    --lr 0.1 \
    --seed 1 \
    --pr $ratio \
    --nw 15 \
    --load 'auto' \
    --reinit 'init' \
    --save \
    --prunety 'u'
done

# one-shot pruning with weight reinitialization
for ratio in "${pruning_ratios[@]}"; do
    python3 main.py \
    --model resnetb \
    --depth 20 \
    --priter 1 \
    --epochs 160 \
    --batch-size 64 \
    --sched 'manual' \
    --lr_update_epochs 80 120 \
    --gamma 0.1 \
    --lr 0.1 \
    --seed 1 \
    --pr $ratio \
    --nw 15 \
    --load 'auto' \
    --reinit 'init' \
    --save \
    --prunety 'u'
done


