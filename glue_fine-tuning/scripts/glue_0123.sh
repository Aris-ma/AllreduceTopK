#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

export WANDB_API_KEY="0f2bf1daed22671b7865ab947c9cbc917de7f80e"

cd GroupTopK

for TASK_NAME in sst2 qnli;do
    for compressor in "group_topk_no_reshape";do
        for use_error_feedback in "ef21";do
            for seed in 1240 1242 1244;do
                PYTHONPATH=. accelerate launch glue_fine-tuning/run_glue_no_trainer_new.py \
                    --model_name_or_path /data/pretrained_models/roberta-base_1 \
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


for TASK_NAME in cola;do
    for compressor in "group_topk_no_reshape";do
        for use_error_feedback in "ef21";do
            for seed in 1240 1242 1244;do
                PYTHONPATH=. accelerate launch glue_fine-tuning/run_glue_no_trainer_new.py \
                    --model_name_or_path /data/pretrained_models/roberta-base_1 \
                    --task_name $TASK_NAME \
                    --max_length 512 \
                    --learning_rate 3e-5 \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --per_device_train_batch_size 8 \
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


for TASK_NAME in mrpc;do
    for compressor in "group_topk_no_reshape";do
        for use_error_feedback in "ef21";do
            for seed in 1240 1242 1244;do
                PYTHONPATH=. accelerate launch glue_fine-tuning/run_glue_no_trainer_new.py \
                    --model_name_or_path /data/pretrained_models/roberta-base_1 \
                    --task_name $TASK_NAME \
                    --max_length 512 \
                    --learning_rate 3e-5 \
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