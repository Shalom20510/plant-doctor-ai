"""Dataset loading and stratified train/validation splitting helpers."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    """Seed Python and PyTorch for more repeatable experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices_by_class(
    targets: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split each class independently so each class is represented in both sets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_class.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[position] for position in order]
        if len(shuffled) < 2:
            raise ValueError(
                "Every class needs at least 2 images for a train/validation split. "
                "Add more images or remove classes with only one image."
            )
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load ImageFolder data from data_dir/<class_name>/* and split by class.

    The validation set is held out from training and uses deterministic resizing
    rather than random augmentation.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Expected one subfolder per class."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Two ImageFolder instances share the same alphabetical class mapping but use
    # separate transforms. Only indices, not image files, are split.
    train_dataset = datasets.ImageFolder(root=root, transform=train_transform)
    eval_dataset = datasets.ImageFolder(root=root, transform=eval_transform)
    if len(train_dataset.classes) < 2:
        raise ValueError("The dataset must contain at least two class folders.")
    if len(train_dataset) == 0:
        raise ValueError(f"No supported images were found in {root}.")
    if train_dataset.class_to_idx != eval_dataset.class_to_idx:
        raise RuntimeError("Training and validation class mappings do not match.")

    train_indices, val_indices = _split_indices_by_class(
        train_dataset.targets, val_fraction, seed
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_subset, shuffle=True, drop_last=False, **loader_options
    )
    val_loader = DataLoader(
        val_subset, shuffle=False, drop_last=False, **loader_options
    )
    return train_loader, val_loader, list(train_dataset.classes)
