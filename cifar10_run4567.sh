# 优先resnet50
#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export TORCH_DISTRIBUTED_BACKEND=nccl



cd ~/GroupTopK

for use_error_feedback in "noef" "ef14"; do
    for compressor in "group_topk_no_reshape" "topk_sync" "randk_sync"; do
        for lr in 1e-1; do
            for seed in 1250 1251 1252; do
                PYTHONPATH=. torchrun  --nproc_per_node=4 --master-port=29502 cifar10/run_cifar10.py \
                    --lr $lr \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 32 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4
            done
        done
    done
done



for compressor in "none"; do
    for use_error_feedback in "noef"; do
        for lr in 1e-1; do
            for seed in 1250 1251 1252; do
                PYTHONPATH=. torchrun  --nproc_per_node=4 --master-port=29502 cifar10/run_cifar10.py \
                    --lr $lr \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 32 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4
            done
        done
    done
done



# for use_error_feedback in "noef" "ef14"; do
#     for compressor in "group_topk_no_reshape" "topk_sync" "randk_sync"; do
#         for lr in 1e-1; do
#             for optimizer in "sgd"; do
#                 for seed in 1250 1251 1252; do
#                     PYTHONPATH=. torchrun  --nproc_per_node=4 --master-port=29502 cifar10/run_cifar10.py \
#                         --lr $lr \
#                         --use_wandb 1 \
#                         --start_compress_iter 1000 \
#                         --weight_decay 5e-4 \
#                         \
#                         --compressor $compressor \
#                         --use_error_feedback $use_error_feedback \
#                         --per_device_train_batch_size 32 \
#                         --seed $seed \
#                         --num_train_epochs 200 \
#                         --compress_ratio 0.2 \
#                         --col_rank 4 \
#                         --optimizer $optimizer
#                 done
#             done
#         done
#     done
# done



# for compressor in "none"; do
#     for use_error_feedback in "noef"; do
#         for lr in 1e-1; do
#             for optimizer in "sgd"; do
#                 for seed in 1250 1251 1252; do
#                     PYTHONPATH=. torchrun  --nproc_per_node=4 --master-port=29502 cifar10/run_cifar10.py \
#                         --lr $lr \
#                         --use_wandb 1 \
#                         --start_compress_iter 1000 \
#                         --weight_decay 5e-4 \
#                         \
#                         --compressor $compressor \
#                         --use_error_feedback $use_error_feedback \
#                         --per_device_train_batch_size 32 \
#                         --seed $seed \
#                         --num_train_epochs 200 \
#                         --compress_ratio 0.2 \
#                         --col_rank 4 \
#                         --optimizer $optimizer
#                 done
#             done
#         done
#     done
# done



