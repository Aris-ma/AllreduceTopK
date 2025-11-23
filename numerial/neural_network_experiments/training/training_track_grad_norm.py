import torch
import os
import torch.nn as nn
import pandas as pd
from datasets.prepare_data import get_dataloaders_high_hetero_fixed_batch, get_dataloaders_fixed_batch
from utils.train_utils import get_first_batch
from utils.train_utils import simple_compute_loss_and_accuracy
from training.optimizer_push_pull_grad_norm_track import PushPull_grad_norm_track
from models.cnn import new_ResNet18
from models.fully_connected import SimpleFCN
from tqdm import tqdm
from datetime import datetime

def compute_normalized_avg_gradient_norm(model_list):
    norms = []
    for model in model_list:
        total_norm = 0
        num_params = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2) ** 2
                num_params += param.numel()
        total_norm = (total_norm ** 0.5) / (num_params ** 0.5)
        norms.append(total_norm.item())
    return sum(norms) / len(norms)

def compute_avg_gradient_matrix_norm(model_list):
    num_params = sum(p.numel() for p in model_list[0].parameters() if p.grad is not None)
    num_models = len(model_list)
    all_grads = torch.zeros(num_models, num_params)
    for i, model in enumerate(model_list):
        grads = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
        if grads:
            grad_vector = torch.cat(grads)
            all_grads[i] = grad_vector
    avg_grad = all_grads.mean(dim=0)
    avg_norm = avg_grad.norm(2) / (num_params ** 0.5)
    return avg_norm.item()

def train_track_grad_norm_with_hetero(
    algorithm: str,
    lr: float,  
    A: torch.Tensor,
    B: torch.Tensor,
    dataset_name: str,
    batch_size: int,
    num_epochs: int = 10,
    remark: str = "",
    alpha: float = 0.5,
    root: str = None,
    use_hetero: bool = True,
    device = "cuda:0",
    seed = 42
)-> pd.DataFrame:
    """
    Lower alpha means higher heterogeneity
    """
    device = device
    criterion = nn.CrossEntropyLoss()
    n = A.shape[0]
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)

    if use_hetero:
        print("use heterogeneous data distribution, lower alpha means higher heterogeneity")
        if dataset_name == "CIFAR10":
            model_list = [new_ResNet18().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_high_hetero_fixed_batch(
                n, dataset_name, batch_size, alpha = alpha, seed=seed
            )
            model_class = new_ResNet18
            output_root = "/home/lg/PushPull/output"
            if root is not None:
                output_root = root
                print(f"root: {root}")
        elif dataset_name == "MNIST":
            model_list = [SimpleFCN().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_high_hetero_fixed_batch(
                n, dataset_name, batch_size, alpha = alpha, seed=seed
            )
            model_class = SimpleFCN
            output_root = "/home/lg/PushPull/output"
            if root is not None:
                output_root = root
                print(f"root: {root}")
    else:
        print("use uniform data distribution, alpha is not used")
        if dataset_name == "CIFAR10":
            model_list = [new_ResNet18().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_fixed_batch(
                n, dataset_name, batch_size, repeat=1
            )
            model_class = new_ResNet18
            output_root = "/home/lg/PushPull/output"
            if root is not None:
                output_root = root
                print(f"root: {root}")
        elif dataset_name == "MNIST":
            model_list = [SimpleFCN().to(device) for _ in range(n)]
            trainloader_list, testloader, full_trainloader = get_dataloaders_fixed_batch(
                n, dataset_name, batch_size, repeat=1
            )
            model_class = SimpleFCN
            output_root = "/home/lg/PushPull/output"
            if root is not None:
                output_root = root
                print(f"root: {root}")
    
    torch.backends.cudnn.benchmark = True

    h_data_train, y_data_train = get_first_batch(trainloader_list)
    h_data_train = [
        tensor.to(device, non_blocking=True) for tensor in h_data_train
    ]
    y_data_train = [
        tensor.to(device, non_blocking=True) for tensor in y_data_train
    ]

    def closure():
        total_loss = 0
        for i, model in enumerate(model_list):
            for param in model.parameters():
                param.requires_grad = True
            model.zero_grad()
            output = model(h_data_train[i])
            loss = criterion(output, y_data_train[i])
            loss.backward()
            total_loss += loss.item()
        grad_norm = compute_normalized_avg_gradient_norm(model_list)
        avg_grad_norm = compute_avg_gradient_matrix_norm(model_list)
        return total_loss / len(model_list), grad_norm, avg_grad_norm
    
    if algorithm == "PushPull":
        optimizer = PushPull_grad_norm_track(model_list, lr=lr, A=A, B=B, closure=closure)
        initial_loss, initial_grad_norm, initial_avg_grad_norm = closure()
    else:   
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print("optimizer initialized")
    train_loss_history = []
    test_average_loss_history = []
    test_average_accuracy_history = []

    grad_norm_per_epoch = []

    grad_norm_history = [initial_grad_norm]
    grad_norm_avg_history = [initial_avg_grad_norm]

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in progress_bar:
        train_loss = 0.0

        flag = 0

        for batch_idx, batch in enumerate(zip(*trainloader_list)):
            inputs = [
                data[0].to(device, non_blocking=True) for data in batch
            ]  
            labels = [
                data[1].to(device, non_blocking=True) for data in batch
            ]  
            h_data_train = inputs  
            y_data_train = labels  
            loss, grad_norm, avg_grad_norm = optimizer.step(closure=closure, lr=lr)
            train_loss += loss

            grad_norm_history.append(grad_norm)
            grad_norm_avg_history.append(avg_grad_norm)

            if flag == 0:
                grad_norm_per_epoch.append(avg_grad_norm)
                flag = 1
        
        train_loss = train_loss / len(trainloader_list[0])
        train_loss_history.append(train_loss)
        test_average_loss, test_accuracy = simple_compute_loss_and_accuracy(model_class=model_class, model_list=model_list, testloader=testloader)
        test_average_loss_history.append(test_average_loss)
        test_average_accuracy_history.append(test_accuracy)
        progress_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{train_loss_history[-1]:.4f}",
            test_loss=f"{test_average_loss_history[-1]:.4f}",
            test_accuracy=f"{100 * test_average_accuracy_history[-1]:.4f}%",
        )
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        df = pd.DataFrame({
            "epoch": range(1, epoch + 2),  
            "train_loss(total)": train_loss_history,
            "test_loss(average)": test_average_loss_history,
            "test_accuracy(average)": test_average_accuracy_history,
            "grad_norm_per_epoch": grad_norm_per_epoch,
        })
        csv_filename = remark+f"hetero={use_hetero}, alpha={alpha}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)
        df = pd.DataFrame({
            "grad_norm": grad_norm_history,
            "avg_grad_norm": grad_norm_avg_history,
        })
        csv_filename = remark+f"grad_norm,hetero={use_hetero},s alpha={alpha}, {algorithm}, lr={lr}, n_nodes={n}, batch_size={batch_size}, {today_date}.csv"
        csv_path = os.path.join(output_root, csv_filename)
        df.to_csv(csv_path, index=False)

    return df