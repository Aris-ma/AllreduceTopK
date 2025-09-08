#!/bin/bash

set -xe
# pip install cupy
export SEED=1234
# export MODEL="llama_130m"
# export num_training_steps=20000
# export warmup_steps=2000

export MODEL="llama_60m"
export num_training_steps=10000
export warmup_steps=1000

compress_ratio=0.2
r=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=49501
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3


export NCCL_P2P_DISABLE=1    # 关掉 GPU 直连（NVLink/PCIe P2P） :contentReference[oaicite:3]{index=3}
export NCCL_SHM_DISABLE=1    # 关掉 SHM 传输，走网络 :contentReference[oaicite:4]{index=4}
export NCCL_IB_DISABLE=1     # 关掉 InfiniBand/RoCE，走 IP socket :contentReference[oaicite:5]{index=5}
export NCCL_SOCKET_IFNAME=lo # 指定用回环接口（lo）进行通信 :contentReference[oaicite:6]{index=6}


optimizer=adamw
start_compress_iter=10
gc=1.0

for LEARNING_RATE in 1e-3; do
    for use_error_feedback in ef14; do
        for MODEL in llama_60m llama_130m llama_350m llama_1b; do
            for compressor in "none" "topk_sync" "randk_sync"; do
                # time
                current_time=$(date "+%Y%m%d%H%M%S")
                # tag
                compressor_tag=${compressor}
                output_dir=output/${current_time}-c4-${optimizer}'('lr$LEARNING_RATE-gc$gc')'-${compressor_tag}-${MODEL}-prof
                echo $compressor_tag
                mkdir -p ${output_dir}
                python -m torch.distributed.run --standalone --nproc_per_node=4 c4/run_llama_pretraining_prof.py \
                    --model_config c4/configs/$MODEL.json \
                    --max_length 256 \
                    --dtype float32 \
                    --num_training_steps $num_training_steps \
                    --warmup_steps $warmup_steps \
                    --eval_every 500 \
                    --save_every 10000 \
                    --total_batch_size 4 \
                    --batch_size 1 \
                    --gradient_accumulation 1 \
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
                    --col_rank $r \
                    \
                    --grad_clipping $gc \
                    \
                    --wandb_project SketchTopk \
                    --output_dir $output_dir \
                2>&1 | tee ${output_dir}/output.log
            done
        done
    done
done
