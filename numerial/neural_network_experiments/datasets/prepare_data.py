import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from typing import Tuple, List
from torch.utils.data import DataLoader, ConcatDataset, Subset

MNIST_transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

MNIST_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

cifar10_transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar10_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def get_dataloaders_fixed_batch(
    n: int, dataset_name: str, batch_size: int, repeat: int = 1
) -> Tuple[
    List[torch.utils.data.DataLoader],
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        trainset = torchvision.datasets.CIFAR10(
            root="/home/lg/PushPull/data/raw/CIFAR10",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.CIFAR10(
            root="/home/lg/PushPull/data/raw/CIFAR10",
            train=False,
            download=False,
            transform=transform_test,
        )
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        trainset = torchvision.datasets.MNIST(
            root="/home/lg/PushPull/data/raw/MNIST",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.MNIST(
            root="/home/lg/PushPull/data/raw/MNIST",
            train=False,
            download=False,
            transform=transform_test,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    original_trainset = trainset

    if repeat > 1:
        trainset = torch.utils.data.ConcatDataset([trainset] * repeat)

    total_train_size = len(trainset)
    subset_sizes = [
        total_train_size // n + (1 if i < total_train_size % n else 0) for i in range(n)
    ]

    subsets = torch.utils.data.random_split(trainset, subset_sizes, generator=generator)

    trainloader_list = [
        torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            generator=generator,
            drop_last=True,
        )
        for subset in subsets
    ]

    full_trainloader = torch.utils.data.DataLoader(
        original_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
        drop_last=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    return trainloader_list, testloader, full_trainloader


def get_dataloaders_high_hetero(
    n: int, dataset_name: str, batch_size: int, repeat: int = 1
) -> Tuple[
    List[torch.utils.data.DataLoader],
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        trainset = torchvision.datasets.CIFAR10(
            root="/home/lg/PushPull/data/raw/CIFAR10",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.CIFAR10(
            root="/home/lg/PushPull/data/raw/CIFAR10",
            train=False,
            download=False,
            transform=transform_test,
        )
        num_classes = 10
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        trainset = torchvision.datasets.MNIST(
            root="/home/lg/PushPull/data/raw/MNIST",
            train=True,
            download=False,
            transform=transform_train,
        )
        testset = torchvision.datasets.MNIST(
            root="/home/lg/PushPull/data/raw/MNIST",
            train=False,
            download=False,
            transform=transform_test,
        )
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    original_trainset = trainset

    if repeat > 1:
        trainset = torch.utils.data.ConcatDataset([trainset] * repeat)

    labels = np.array(trainset.targets)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    subsets = []
    total_size = len(trainset)
    base_size = total_size // n

    alpha = 0.5
    class_dist = np.random.dirichlet([alpha] * n, num_classes)

    for node in range(n):
        node_indices = []
        node_size = base_size + (1 if node < total_size % n else 0)

        target_dist = class_dist[:, node] * node_size

        for cls in range(num_classes):
            num_samples = int(target_dist[cls])
            available_indices = class_indices[cls]

            if len(available_indices) > 0:
                selected = np.random.choice(
                    available_indices,
                    size=min(num_samples, len(available_indices)),
                    replace=False,
                )
                node_indices.extend(selected)

                class_indices[cls] = np.setdiff1d(class_indices[cls], selected)

        subsets.append(torch.utils.data.Subset(trainset, node_indices))

    trainloader_list = [
        torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            generator=generator,
        )
        for subset in subsets
    ]

    full_trainloader = torch.utils.data.DataLoader(
        original_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        generator=generator,
    )

    return trainloader_list, testloader, full_trainloader


def get_dataloaders_high_hetero_fixed_batch(
    n: int,
    dataset_name: str,
    batch_size: int,
    alpha: float = 0.5,
    repeat: int = 1,
    seed: int = 42,
):

    print(f"seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    data_root = "/home/lg/PushPull/data/raw"

    if dataset_name == "CIFAR10":
        transform_train, transform_test = (
            cifar10_transform_train,
            cifar10_transform_test,
        )
        try:
            trainset = torchvision.datasets.CIFAR10(
                root="/home/lg/PushPull/data/raw/CIFAR10",
                train=True,
                download=False,
                transform=transform_train,
            )
            testset = torchvision.datasets.CIFAR10(
                root="/home/lg/PushPull/data/raw/CIFAR10",
                train=False,
                download=False,
                transform=transform_test,
            )
        except Exception as e:
            print(
                f"failed to load CIFAR10 dataset. Please ensure the path {data_root}/CIFAR10 is correct and contains data. Error: {e}"
            )
            raise
        num_classes = 10
    elif dataset_name == "MNIST":
        transform_train, transform_test = MNIST_transform_train, MNIST_transform_test
        try:
            trainset = torchvision.datasets.MNIST(
                root="/home/lg/PushPull/data/raw/MNIST",
                train=True,
                download=False,
                transform=transform_train,
            )
            testset = torchvision.datasets.MNIST(
                root="/home/lg/PushPull/data/raw/MNIST",
                train=False,
                download=False,
                transform=transform_test,
            )
        except Exception as e:
            print(
                f"failed to load MNIST dataset. Please ensure the path {data_root}/MNIST is correct and contains data. Error: {e}"
            )
            raise
        num_classes = 10
    else:
        raise ValueError(f"not support: {dataset_name}")

    original_trainset = trainset

    if repeat > 1:
        original_labels = np.array(original_trainset.targets)
        trainset = ConcatDataset([original_trainset] * repeat)

        labels = np.concatenate([original_labels] * repeat)
    else:

        if hasattr(trainset, "targets"):
            labels = np.array(trainset.targets)
        elif hasattr(trainset, "labels"):
            labels = np.array(trainset.labels)
        else:

            print("error: trainset does not have 'targets' or 'labels' attribute.")
            labels = np.array([sample[1] for sample in trainset])

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    subsets = []
    total_size = len(trainset)
    indices_per_node = [[] for _ in range(n)]

    class_dist = np.random.dirichlet([alpha] * num_classes, n).T

    all_indices_shuffled = np.arange(total_size)

    np.random.shuffle(all_indices_shuffled)

    node_class_samples_target = (class_dist / class_dist.sum(axis=0, keepdims=True)) * (
        total_size / n
    )
    node_class_samples_target = node_class_samples_target.round().astype(int)

    current_total = node_class_samples_target.sum()
    diff = total_size - current_total

    if diff != 0:
        adjustment_indices = np.random.choice(n * num_classes, abs(diff), replace=True)
        adjustments = np.zeros_like(node_class_samples_target.flatten())
        for idx in adjustment_indices:
            adjustments[idx] += np.sign(diff)
        node_class_samples_target = (
            node_class_samples_target.flatten() + adjustments
        ).reshape(node_class_samples_target.shape)
        node_class_samples_target = np.maximum(0, node_class_samples_target)

    final_diff = total_size - node_class_samples_target.sum()
    if final_diff != 0:

        adjust_node, adjust_class = np.random.randint(n), np.random.randint(num_classes)
        node_class_samples_target[adjust_class, adjust_node] += final_diff
        node_class_samples_target[adjust_class, adjust_node] = max(
            0, node_class_samples_target[adjust_class, adjust_node]
        )

    assert (
        node_class_samples_target.sum() == total_size
    ), f"Error: Total samples assigned {node_class_samples_target.sum()} does not match total size {total_size}."

    indices_by_class = [list(idx) for idx in class_indices]

    for idx_list in indices_by_class:
        random.shuffle(idx_list)

    class_pointers = [0] * num_classes
    for node_idx in range(n):
        node_indices = []
        for class_idx in range(num_classes):
            target_count = node_class_samples_target[class_idx, node_idx]
            start = class_pointers[class_idx]
            end = start + target_count

            assigned_indices = indices_by_class[class_idx][start:end]
            node_indices.extend(assigned_indices)

            class_pointers[class_idx] = end

        random.shuffle(node_indices)

        subsets.append(Subset(trainset, node_indices))

    use_persistent_workers = (
        True if (torch.cuda.is_available() or num_workers > 0) else False
    )
    num_workers = 4

    trainloader_list = [
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=use_persistent_workers if num_workers > 0 else False,
            drop_last=True,
            generator=generator,
        )
        for subset in subsets
    ]

    full_trainloader = DataLoader(
        original_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=use_persistent_workers if num_workers > 0 else False,
        drop_last=True,
        generator=generator,
    )

    testloader = DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=use_persistent_workers if num_workers > 0 else False,
    )

    return trainloader_list, testloader, full_trainloader
