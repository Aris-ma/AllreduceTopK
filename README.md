# ARC-TopK

Communication remains a central bottleneck in large-scale distributed machine learning, and gradient sparsification has emerged as a promising strategy to alleviate this challenge. 

However, existing gradient compressors face notable limitations: RandK discards structural information and performs poorly in practice, while TopK preserves informative entries but loses the contraction property and requires costly All-Gather operations. 

In this paper, we propose **arctopK**, an All-Reduce-Compatible Top-K compressor that aligns sparsity patterns across nodes using a lightweight sketch of the gradient, enabling index-free All-Reduce while preserving globally significant information. arctopK is provably contractive and, when combined with momentum error feedback (EF21M), achieves linear speedup and sharper convergence rates than the original EF21M under standard assumptions. 

Empirically, arctopK matches the accuracy of TopK while reducing wall-clock training time by up to 60.7\%, offering an efficient and scalable solution that combines the robustness of RandK with the strong performance of TopK.


## Installation

The dependencies are listed in [requirements.txt](https://github.com/Aris-ma/AllreduceTopK/blob/master/requirements.txt). 

You can install them via::

```
pip install -r requirements.txt
```

Our experiments are conducted with python 3.11 with PyTorch 2.7 on NVIDIA RTX 4090 (24 GB) GPUs

## Reproduce Experiments

### Fine-Tuning RoBERTa on GLUE tasks

The code for GLUE experiments is provided in [glue_fine-tuning](https://github.com/Aris-ma/AllreduceTopK/tree/master/glue_fine-tuning).

The main script is [run_glue_no_trainer_new.py](https://github.com/Aris-ma/AllreduceTopK/blob/master/glue_fine-tuning/run_glue_no_trainer_new.py)

An example script is shown below:
```
PYTHONPATH=. accelerate launch glue_fine-tuning/run_glue_no_trainer_new.py \
    --model_name_or_path /data/pretrained_models/roberta-base_1 \
    --task_name qnli \
    --max_length 512 \
    --learning_rate 1e-5 \
    --compressor "group_topk_no_reshape" \
    --use_error_feedback "ef14" \
    --per_device_train_batch_size 4 \
    --seed 1234 \
    --num_train_epochs 30 \
    --with_tracking \
    --report_to wandb \
    --compress_ratio 0.2 \
    --r 4
```

We set `compress_ratio` = 0.2 for all compressors and fix the projection rank at `r` = 4 for ARC-Top-K

Supported error-feedback variants:

* no error feedback(`use_error_feedback`="noef")

* EF21(`use_error_feedback`="ef21")

* EF14(`use_error_feedback`="ef14")

Supported compressors:

* ARC-TopK (`compressor`="group_topk_no_reshape")

* TopK (`compressor`="topk_sync")

* RandK (`compressor`="randk_sync")

* No compression (`compressor`="none")


For reproductibility purposes, we provide the [scripts](https://github.com/Aris-ma/AllreduceTopK/tree/master/glue_fine-tuning/scripts) . 

Seeds we commonly use include 1234, 1236, 1238.

## Citation

```
@misc{arctopk2025,
  title         = {An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning},
  author        = {},
  year          = {2025},
  howpublished  = {\url{https://github.com/Aris-ma/AllreduceTopK}},
}
```