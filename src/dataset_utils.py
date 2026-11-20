"""Dataset and preprocessing utilities for ResNet18 image classification."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
)


def build_transforms(train: bool = False) -> transforms.Compose:
    """Return training or evaluation transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _require_dataset_split(split: str) -> Path:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing {split_dir}. Create class folders and collect images first."
        )
    return split_dir


def build_dataloaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train, validation, and test ImageFolder dataloaders."""
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset = datasets.ImageFolder(
        _require_dataset_split("train"), transform=build_transforms(train=True)
    )
    val_dataset = datasets.ImageFolder(
        _require_dataset_split("val"), transform=build_transforms(train=False)
    )
    test_dataset = datasets.ImageFolder(
        _require_dataset_split("test"), transform=build_transforms(train=False)
    )

    if not (train_dataset.classes == val_dataset.classes == test_dataset.classes):
        raise ValueError("Train, validation, and test class folders must match.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_dataset.classes
