#!/bin/bash



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890



cd ~/GroupTopK



for TASK_NAME in cola;do
    for compressor in "group_topk_no_reshape";do
        for use_error_feedback in "ef14";do
            for seed in 1513 1514;do
                PYTHONPATH=. accelerate launch glue_1/run_glue_no_trainer_new.py \
                    --model_name_or_path /data/pretrained_models/roberta-base_1 \
                    --task_name $TASK_NAME \
                    --max_length 512 \
                    --learning_rate 3e-5 \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --gradient_accumulation_steps 8 \
                    --per_device_train_batch_size 1 \
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
    for compressor in "randk_sync";do
        for use_error_feedback in "ef14";do
            for seed in 1513 1514;do
                PYTHONPATH=. accelerate launch glue_1/run_glue_no_trainer_new.py \
                    --model_name_or_path /data/pretrained_models/roberta-base_1 \
                    --task_name $TASK_NAME \
                    --max_length 512 \
                    --learning_rate 3e-5 \
                    --compressor $compressor \
                    --use_error_feedback $use_error_feedback \
                    --gradient_accumulation_steps 4 \
                    --per_device_train_batch_size 2 \
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


