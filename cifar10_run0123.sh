#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export TORCH_DISTRIBUTED_BACKEND=nccl


export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket




cd ~/GroupTopK



for use_error_feedback in "noef" "ef14"; do
    for compressor in "group_topk_no_reshape" "topk_sync" "randk_sync"; do
        for lr in 1e-3; do
            for seed in 1370 1371 1372; do
                for col_rank in 4; do
                    PYTHONPATH=. torchrun  --nproc_per_node=4 --master-port=29501 cifar10/run_cifar10.py \
                        --lr $lr \
                        --use_wandb 1 \
                        --start_compress_iter 1 \
                        --weight_decay 5e-4 \
                        \
                        --compressor $compressor \
                        --use_error_feedback $use_error_feedback \
                        --per_device_train_batch_size 32 \
                        --seed $seed \
                        --num_train_epochs 2 \
                        --compress_ratio 0.2 \
                        --col_rank $col_rank
                done
            done
        done
    done
done

for use_error_feedback in "noef"; do
    for compressor in "none"; do
        for lr in 1e-3; do
            for seed in 1370 1371 1372; do
                for col_rank in 4; do
                    PYTHONPATH=. torchrun  --nproc_per_node=4 --master-port=29501 cifar10/run_cifar10.py \
                        --lr $lr \
                        --use_wandb 1 \
                        --start_compress_iter 1 \
                        --weight_decay 5e-4 \
                        \
                        --compressor $compressor \
                        --use_error_feedback $use_error_feedback \
                        --per_device_train_batch_size 32 \
                        --seed $seed \
                        --num_train_epochs 2 \
                        --compress_ratio 0.2 \
                        --col_rank $col_rank
                done
            done
        done
    done
done


