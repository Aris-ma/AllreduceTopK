# PushPull
Implementation for the linear speedup property of PushPull algorithm on neural network.

## Setup

1. Download the MNIST or CIFAR10 dataset and unzip then in the directory `.neural_network_experiments/data`.
2. Change the directory of the datasets in `.neural_network_experiments/datasets/prepare_data.py`.
3. Change the directory of the output files in `.neural_network_experiments/training/training_track_grad_norm.py`.
4. To set up the environment, follow these steps:
```bash
conda create -n pushpull python=3.12
source activate pushpull
pip install -r requirements.txt
```

## Demo

In the `.neural_network_experiments/demo.py`, you can set the parameters like follows, we give a demo on the grid network topology.
```python
n = 4
lr = 3e-2
num_epochs = 100
bs = 128
alpha = 0.9
use_hetero = True
remark = "new"
device = "cuda:0"
root = "./output"
```
you can change the network topology based on the functions in the `.neural_network_experiments/utils/network_utils.py`.

## Synthetic Data Experiments
We conduct the experiments with `cupy`, so it needs GPU. 

In the `./synthetic_experiments/demo_exponential.py`, you can set the parameters like follows, we give a demo on the exponential network topology.
```python
d = 10
L_total = 1440000
n = 16
num_runs = 20
device_id = "cuda:0"
rho = 1e-2
lr = 5e-2
max_it = 3000
bs = 200
```
you can change the network topology based on the functions in the `./synthetic_experiments/network_utils.py`.