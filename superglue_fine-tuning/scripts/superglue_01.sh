#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

export WANDB_API_KEY="0f2bf1daed22671b7865ab947c9cbc917de7f80e"

for TASK_NAME in copa;do
    for compressor in "none";do
        for use_error_feedback in "noef";do
            for seed in 1240;do
                PYTHONPATH=. accelerate launch superglue_fine-tuning/run_superglue_no_trainer.py \
                    --model_name_or_path /data/pretrained_models/roberta-large \
                    --task_name $TASK_NAME \
                    --max_length 512 \
                    --learning_rate 1e-5 \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 4 \
                    --seed $seed \
                    --num_train_epochs 30 \
                    --with_tracking \
                    --report_to wandb \
                    --compress_ratio 0.2 \
                    --r 4
            done
        done
    done
done