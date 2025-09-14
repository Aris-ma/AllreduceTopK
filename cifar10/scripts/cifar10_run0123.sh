#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export TORCH_DISTRIBUTED_BACKEND=nccl






cd ~/GroupTopK


for use_error_feedback in "noef"; do
    for compressor in "group_topk_no_reshape"; do
        for optimizer in "adamw"; do
            for seed in 1401; do
                PYTHONPATH=. torchrun  --nproc_per_node=4 --master-port=29501 cifar10/run_cifar10_bits.py \
                    --lr 1e-3 \
                    --use_wandb 1 \
                    --start_compress_iter 1 \
                    --weight_decay 5e-4 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 32 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done


export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket