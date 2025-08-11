#!/bin/bash

set -xe

export SEED=1234
export MODEL="llama_350m"
export LEARNING_RATE=1e-3

# export compressors=(topk_sync, flex_quant_sync, none)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=49501
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3


# time
current_time=$(date "+%Y%m%d%H%M%S")
# flex quant
quantization_bits=4
# topk
compress_ratio=0.2
sparse_type=tensor
# powerSGD
compress_rank=256

# compressor="flex_quant_sync"
# compressor_tag=${compressor}'(ef14-'${quantization_bits}bits')'
# compressor="topk_sync"
# compressor_tag=${compressor}'(ef14-'${sparse_type}-${compress_ratio}')'
# compressor="none"
# compressor_tag=${compressor}
compressor="powersgd"
compressor_tag=${compressor}'('ef14-rk${compress_rank}')'

output_dir=output/${current_time}-c4-adamw-${compressor_tag}-${MODEL}
echo $compressor_tag
mkdir -p ${output_dir}
python -m torch.distributed.run --standalone --nproc_per_node=4 c4/run_llama_pretraining.py \
    --model_config c4/configs/$MODEL.json \
    --max_length 256 \
    --dtype bfloat16 \
    --num_training_steps 55000 \
    --warmup_steps 5500 \
    --eval_every 1500 \
    --save_every 10000 \
    --total_batch_size 512 \
    --batch_size 32 \
    --gradient_accumulation 4 \
    --save_dir c4/results/Adam/$MODEL/lr_$LEARNING_RATE \
    --seed $SEED \
    \
    --optimizer adamw \
    --lr $LEARNING_RATE \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --weight_decay 0.0 \
    \
    --compressor $compressor \
    --use_error_feedback ef14 \
    --compress_rank $compress_rank \
    --compress_ratio $compress_ratio \
    --sparse_type $sparse_type \
    --bucket_size 256 \
    --quantization_bits $quantization_bits \
    \
    --grad_clipping 1.0 \
    \
    --wandb_project CCBench-C4 \
    --output_dir $output_dir \
2>&1 | tee ${output_dir}/output.log