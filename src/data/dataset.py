"""Data loading and preprocessing for image classification."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int = 32, augment: bool = True):
    """Get train and test transforms.

    Args:
        image_size: Target image size.
        augment: Whether to apply data augmentation for training.

    Returns:
        Tuple of (train_transform, test_transform).
    """
    # Normalization values for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def get_dataloaders(
    dataset_name: str = "CIFAR10",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 32,
    augment: bool = True,
    data_dir: str = "data/raw",
):
    """Create train and test data loaders.

    Args:
        dataset_name: Name of the torchvision dataset.
        batch_size: Batch size for training and evaluation.
        num_workers: Number of data loading workers.
        image_size: Target image size.
        augment: Whether to apply data augmentation.
        data_dir: Directory to store/load the dataset.

    Returns:
        Tuple of (train_loader, test_loader, class_names).
    """
    train_transform, test_transform = get_transforms(image_size, augment)

    dataset_class = getattr(datasets, dataset_name)

    train_dataset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes

    return train_loader, test_loader, class_names
