#!/bin/bash
pruning_ratios=(0.3 0.5)   # overall pruning rations
epochs=(10 30 50 100)      # epochs to save potential EBs
epoch_start=1              # train potential EBs from scratch after pruning

CUDA_VISIBLE_DEVICES=0
# train default model and save at epochs 10, 30, 50, 100
python3 main.py \
--model resnetb \
--depth 20 \
--epochs 160 \
--priter 0 \
--prunety s \
--batch-size 64 \
--pr 0 \
--sched manual \
--lr_update_epochs 80 120 \
--gamma 0.1 \
--lr 0.1 \
--seed 1 \
--nw 15 \
--save \
--epochs-save "${epochs[@]}"

# structure prune the best models saved at epochs 10, 30, 50, 100 and retrain for 160 epochs
for ratio in "${pruning_ratios[@]}"; do
    best_ckpt_file="./checkpoints/resnetb_cifar10_depth=20_batch=64_epoch=160_prty=s/priter=0_epoch=${epochs[0]}_best.pth.tar"    
    for epoch in "${epochs[@]}"; do
        # if best ckpt file for current epoch exists, update the best ckpt file
        ckpt_file="./checkpoints/resnetb_cifar10_depth=20_batch=64_epoch=160_prty=s/priter=0_epoch=${epoch}_best.pth.tar"
        if [ -f $ckpt_file ]; then
            best_ckpt_file=$ckpt_file
        fi
        python3 main.py \
        --model resnetb \
        --depth 20 \
        --epochs 160 \
        --priter 1 \
        --prunety s \
        --batch-size 64 \
        --pr $ratio \
        --sched manual \
        --lr_update_epochs 80 120 \
        --gamma 0.1 \
        --lr 0.1 \
        --seed 1 \
        --nw 15 \
        --load $best_ckpt_file \
        --ann-train $epoch \
        --save
    done
done
