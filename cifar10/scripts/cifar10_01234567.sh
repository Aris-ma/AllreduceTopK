#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export TORCH_DISTRIBUTED_BACKEND=nccl






cd ~/GroupTopK


for use_error_feedback in "noef"; do
    for compressor in "none"; do
        for optimizer in "adamw"; do
            for seed in 1410; do
                PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar100.py \
                    --lr 1e-3 \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    --gradient_accumulation_steps 8 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 4 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done



for use_error_feedback in "ef14"; do
    for compressor in "group_topk_no_reshape" "randk_sync"; do
        for optimizer in "adamw"; do
            for seed in 1412; do
                PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar100.py \
                    --lr 1e-3 \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    --gradient_accumulation_steps 8\
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 2 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done




for use_error_feedback in "ef14"; do
    for compressor in "topk_sync"; do
        for optimizer in "adamw"; do
            for seed in 1412; do
                PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar100.py \
                    --lr 1e-3 \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    --gradient_accumulation_steps 4\
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 4 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done

# for use_error_feedback in "noef"; do
#     for compressor in "group_topk_no_reshape"; do
#         for optimizer in "adamw"; do
#             for seed in 1400; do
#                 PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar100_bits.py \
#                     --lr 1e-3 \
#                     --use_wandb 1 \
#                     --start_compress_iter 1 \
#                     --weight_decay 5e-4 \
#                     \
#                     --compressor $compressor \
#                     --use_error_feedback $use_error_feedback \
#                     --per_device_train_batch_size 16 \
#                     --seed $seed \
#                     --num_train_epochs 200 \
#                     --compress_ratio 0.2 \
#                     --col_rank 4 \
#                     --optimizer $optimizer
#             done
#         done
#     done
# done
















for use_error_feedback in "noef" "ef14"; do
    for compressor in "topk_sync"; do
        for optimizer in "adamw"; do
            for seed in 1300; do
                PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar10.py \
                    --lr 1e-3 \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 16 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done


for use_error_feedback in "noef" "ef14"; do
    for compressor in "topk_sync"; do
        for optimizer in "adamw"; do
            for seed in 1300; do
                PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar10_resnet50.py \
                    --lr 1e-3 \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 16 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done

for use_error_feedback in "noef" "ef14"; do
    for compressor in "topk_sync"; do
        for optimizer in "sgd"; do
            for seed in 1300; do
                PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar10.py \
                    --lr 1e-1 \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 16 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done


for use_error_feedback in "noef" "ef14"; do
    for compressor in "topk_sync"; do
        for optimizer in "sgd"; do
            for seed in 1300; do
                PYTHONPATH=. torchrun  --nproc_per_node=8 --master-port=29501 cifar10/run_cifar10_resnet50.py \
                    --lr 1e-1 \
                    --use_wandb 1 \
                    --start_compress_iter 1000 \
                    --weight_decay 5e-4 \
                    \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 16 \
                    --seed $seed \
                    --num_train_epochs 200 \
                    --compress_ratio 0.2 \
                    --col_rank 4 \
                    --optimizer $optimizer
            done
        done
    done
done