import torch
import torch.nn as nn
import itertools
from torch.cuda.amp import autocast
from typing import Tuple


def get_first_batch(trainloader_list: list):
    h_data_train = []
    y_data_train = []

    for trainloader in trainloader_list:
        loader_copy, trainloader = itertools.tee(trainloader, 2)

        first_batch = next(iter(loader_copy))

        h_data_train.append(first_batch[0])
        y_data_train.append(first_batch[1])

    return h_data_train, y_data_train


def compute_normalized_global_gradient_norm(model):
    total_norm = 0.0
    total_elements = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            total_elements += p.grad.numel()
    total_norm = total_norm**0.5
    return total_norm


def compute_loss_and_accuracy(
    model_class, model_list, testloader, full_trainloader, use_amp=False
) -> Tuple[float, float, float, float]:

    criterion = nn.CrossEntropyLoss()

    device = next(model_list[0].parameters()).device

    avg_model = model_class().to(device)
    avg_state_dict = avg_model.state_dict()

    sum_state_dict = {
        key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()
    }

    for model in model_list:
        state_dict = model.state_dict()
        for key in sum_state_dict.keys():
            sum_state_dict[key] += state_dict[key].to(device)

    num_models = len(model_list)
    avg_state_dict = {key: value / num_models for key, value in sum_state_dict.items()}

    avg_model.load_state_dict(avg_state_dict)

    avg_model.eval()
    train_correct = 0
    train_total = 0
    train_total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in full_trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            with autocast(enabled=use_amp):

                outputs = avg_model(inputs)
                loss = criterion(outputs, labels)

            train_total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

    train_average_loss = train_total_loss / len(full_trainloader)
    train_accuracy = train_correct / train_total

    test_correct = 0
    test_total = 0
    test_total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            with autocast(enabled=use_amp):

                outputs = avg_model(inputs)
                loss = criterion(outputs, labels)

            test_total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_average_loss = test_total_loss / len(testloader)
    test_accuracy = test_correct / test_total

    return (
        train_average_loss,
        train_accuracy,
        test_average_loss,
        test_accuracy,
    )


def simple_compute_loss_and_accuracy(
    model_class, model_list, testloader, use_amp=False
) -> Tuple[float, float, float, float]:

    criterion = nn.CrossEntropyLoss()

    device = next(model_list[0].parameters()).device

    avg_model = model_class().to(device)
    avg_state_dict = avg_model.state_dict()

    sum_state_dict = {
        key: torch.zeros_like(param).to(device) for key, param in avg_state_dict.items()
    }

    for model in model_list:
        state_dict = model.state_dict()
        for key in sum_state_dict.keys():
            sum_state_dict[key] += state_dict[key].to(device)

    num_models = len(model_list)
    avg_state_dict = {key: value / num_models for key, value in sum_state_dict.items()}

    avg_model.load_state_dict(avg_state_dict)

    test_correct = 0
    test_total = 0
    test_total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            with autocast(enabled=use_amp):

                outputs = avg_model(inputs)
                loss = criterion(outputs, labels)

            test_total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_average_loss = test_total_loss / len(testloader)
    test_accuracy = test_correct / test_total

    return (
        test_average_loss,
        test_accuracy,
    )
