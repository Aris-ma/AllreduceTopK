# ARC-TopK

Communication remains a central bottleneck in large-scale distributed machine learning, and gradient sparsification has emerged as a promising strategy to alleviate this challenge. 

However, existing gradient compressors face notable limitations: RandK discards structural information and performs poorly in practice, while TopK preserves informative entries but loses the contraction property and requires costly All-Gather operations. 

In this paper, we propose **arctopK**, an All-Reduce-Compatible Top-$K$ compressor that aligns sparsity patterns across nodes using a lightweight sketch of the gradient, enabling index-free All-Reduce while preserving globally significant information. arctopK is provably contractive and, when combined with momentum error feedback (EF21M), achieves linear speedup and sharper convergence rates than the original EF21M under standard assumptions. 

Empirically, arctopK matches the accuracy of TopK while reducing wall-clock training time by up to 60.7\%, offering an efficient and scalable solution that combines the robustness of RandK with the strong performance of TopK.


## Installation

The dependencies requied can be found in the file [requirements.txt](https://github.com/Aris-ma/AllreduceTopK/blob/master/requirements.txt). To install them, please run:

```
pip install -r requirements.txt
```

Our experiment scripts are conducted on python 3.11 with PyTorch 2.7 on NVIDIA RTX 4090

## Reproduce Experiments

### Fine-Tuning RoBERTa on GLUE tasks

Codes of this task are in the file[glue_fine-tuning](https://github.com/Aris-ma/AllreduceTopK/tree/master/glue_fine-tuning),in which [run_glue_no_trainer_new.py](https://github.com/Aris-ma/AllreduceTopK/blob/master/glue_fine-tuning/run_glue_no_trainer_new.py) is the main script for this task.

An example script to run it is shown below:

```
PYTHONPATH=. accelerate launch glue_fine-tuning/run_glue_no_trainer_new.py \
    --model_name_or_path /data/pretrained_models/roberta-base_1 \
    --task_name qnli \
    --max_length 512 \
    --learning_rate 1e-5 \
    --compressor $compressor \
    --use_error_feedback "ef14" \
    --per_device_train_batch_size 4 \
    --seed 1234 \
    --num_train_epochs 30 \
    --with_tracking \
    --report_to wandb \
    --compress_ratio 0.2 \
    --r 4
```

The `compress_ratio` is in the range [0, 1], in our test we use 

For reproductibility purposes, we provide the scripts(加链接) we used to run our experiments. Seeds we often use including 1234, 1236, 1238

## Citation