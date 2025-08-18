#!/bin/bash

set -xe

# 等待指定进程结束的函数
wait_for_process_to_end() {
    local pid=$1
    echo "等待进程 $pid 结束..."
    
    while kill -0 $pid 2>/dev/null; do
        echo "进程 $pid 仍在运行，等待中... $(date)"
        sleep 30  # 每30秒检查一次
    done
    
    echo "进程 $pid 已结束，开始执行训练任务... $(date)"
}

# 等待进程2213078结束
wait_for_process_to_end 2213078

# pip install cupy
export SEED=1234
# export MODEL="llama_130m"
# export num_training_steps=20000
# export warmup_steps=2000

export MODEL="llama_60m"
export num_training_steps=10000
export warmup_steps=1000


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=49501
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

optimizer=adamw
start_compress_iter=1000
compressor=none
gc=1.0

compress_ratio=0.2


# for LEARNING_RATE in 2e-3 4e-3; do
#     for use_error_feedback in ef14; do
#         for compressor in "group_topk_no_reshape"; do 
#             # time
#             current_time=$(date "+%Y%m%d%H%M%S")
#             # tag
#             compressor_tag=${compressor}_new
#             output_dir=output/${current_time}-c4-${optimizer}'('lr$LEARNING_RATE-gc$gc')'-${compressor_tag}-${MODEL}
#             echo $compressor_tag
#             mkdir -p ${output_dir}
#             python -m torch.distributed.run --standalone --nproc_per_node=4 c4/run_llama_pretraining.py \
#                 --model_config c4/configs/$MODEL.json \
#                 --max_length 256 \
#                 --dtype bfloat16 \
#                 --num_training_steps $num_training_steps \
#                 --warmup_steps $warmup_steps \
#                 --eval_every 5000 \
#                 --save_every 10000 \
#                 --total_batch_size 512 \
#                 --batch_size 32 \
#                 --gradient_accumulation 4 \
#                 --save_dir c4/results/Adam/$MODEL/lr_$LEARNING_RATE \
#                 --seed $SEED \
#                 \
#                 --optimizer $optimizer \
#                 --lr $LEARNING_RATE \
#                 --beta1 0.9 \
#                 --beta2 0.999 \
#                 --eps 1e-8 \
#                 --weight_decay 0.0 \
#                 \
#                 --compressor $compressor \
#                 --start_compress_iter $start_compress_iter \
#                 --use_error_feedback $use_error_feedback \
#                 --compress_ratio $compress_ratio \
#                 \
#                 --grad_clipping $gc \
#                 \
#                 --wandb_project SketchTopk \
#                 --output_dir $output_dir \
#             2>&1 | tee ${output_dir}/output.log
#         done
#     done
# done

# export HF_ENDPOINT=https://hf-mirror.com
export MODEL="llama_130m"
export num_training_steps=20000
export warmup_steps=2000


for LEARNING_RATE in 2e-3 4e-3; do
    for use_error_feedback in ef14; do
        for compressor in "topk_sync" "randk_sync"; do 
            # time
            current_time=$(date "+%Y%m%d%H%M%S")
            # tag
            compressor_tag=${compressor}'('$compress_ratio')'-$use_error_feedback-new
            output_dir=output/${current_time}-c4-${optimizer}'('lr$LEARNING_RATE-gc$gc')'-${compressor_tag}-${MODEL}
            echo $compressor_tag
            mkdir -p ${output_dir}
            python -m torch.distributed.run --standalone --nproc_per_node=4 c4/run_llama_pretraining.py \
                --model_config c4/configs/$MODEL.json \
                --max_length 256 \
                --dtype bfloat16 \
                --num_training_steps $num_training_steps \
                --warmup_steps $warmup_steps \
                --eval_every 5000 \
                --save_every 10000 \
                --total_batch_size 512 \
                --batch_size 32 \
                --gradient_accumulation 4 \
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
                --wandb_project SketchTopk \
                --output_dir $output_dir \
            2>&1 | tee ${output_dir}/output.log
        done
    done
done