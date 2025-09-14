#!/bin/bash

set -xe


# pip install cupy
export SEED=1243
export WANDB_API_KEY="0f2bf1daed22671b7865ab947c9cbc917de7f80e"

# export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=49501
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

export MODEL="llama_130m"
export num_training_steps=10000
export warmup_steps=1000

optimizer=adamw
start_compress_iter=2000
compressor=none
gc=1.0

compress_ratio=0.2

for LEARNING_RATE in 2e-3; do
    for use_error_feedback in ef14; do
        for compressor in "randk_sync"; do 
            # time
            current_time=$(date "+%Y%m%d%H%M%S")
            # tag
            compressor_tag=${compressor}-$use_error_feedback-ratio${compress_ratio}
            output_dir=output/lr$LEARNING_RATE-gc$gc-total_bs${total_batch_size}-seed${SEED}-${compressor_tag}-start_compress${start_compress_iter}-warmup${warmup_steps}-float32
            echo $compressor_tag
            mkdir -p ${output_dir}
            python -m torch.distributed.run --standalone --nproc_per_node=4 c4/run_llama_pretraining_bits.py \
                --model_config c4/configs/$MODEL.json \
                --max_length 256 \
                --dtype float32 \
                --num_training_steps $num_training_steps \
                --warmup_steps $warmup_steps \
                --total_batch_size 256 \
                --batch_size 32 \
                --gradient_accumulation 2 \
                --save_dir c4/results/Adam/$MODEL/lr_$LEARNING_RATE \
                --seed $SEED \
                \
                --optimizer $optimizer \
                --lr $LEARNING_RATE \
                --beta1 0.9 \
                --beta2 0.999 \
                --eps 1e-8 \
                --weight_decay 0.0 \
                \
                --compressor $compressor \
                --start_compress_iter $start_compress_iter \
                --use_error_feedback $use_error_feedback \
                --compress_ratio $compress_ratio \
                \
                --grad_clipping $gc \
                \
                --wandb_project ${MODEL}-bits \
                --output_dir $output_dir \
            2>&1 | tee ${output_dir}/output.log
        done
    done
done

