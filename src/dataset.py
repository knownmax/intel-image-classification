"""
dataset.py — Intel Image Classification dataset loader.

Actual directory layout on disk:
    data_dir/
        seg_train/
            seg_train/
                buildings/  forest/  glacier/  mountain/  sea/  street/
        seg_test/
            seg_test/
                buildings/  forest/  glacier/  mountain/  sea/  street/
        seg_pred/
            seg_pred/
                *.jpg   (no class subfolders — unlabelled)

There is no pre-made val split, so 20% of seg_train is held out as val
using a reproducible random split (seeded).
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms


# ImageNet normalisation statistics
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Actual sub-folder names inside data_dir
_TRAIN_SUBDIR = "seg_train/seg_train"
_TEST_SUBDIR  = "seg_test/seg_test"
_PRED_SUBDIR  = "seg_pred/seg_pred"


def _build_transforms(split: str) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline.

    Args:
        split: One of 'train', 'val', 'test', or 'pred'.

    Returns:
        Composed transforms.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])


class _TransformSubset(Dataset):
    """Wrap a Subset with its own transform, independent of the base dataset."""

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def _split_train_val(
    data_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Load seg_train and split into train / val subsets.

    The base dataset uses no-op ToTensor so that _TransformSubset can apply
    split-specific augmentation independently.

    Args:
        data_dir:     Root data directory.
        val_fraction: Fraction of training data reserved for validation.
        seed:         Random seed for reproducible split.

    Returns:
        (train_dataset, val_dataset) with their respective transforms applied.
    """
    base = datasets.ImageFolder(
        root=str(data_dir / _TRAIN_SUBDIR),
        transform=None,   # PIL images passed through; each split applies its own transform
    )

    n = len(base)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(n), [n_train, n_val], generator=generator
    )

    train_ds = _TransformSubset(
        Subset(base, train_indices.indices), _build_transforms("train")
    )
    val_ds = _TransformSubset(
        Subset(base, val_indices.indices), _build_transforms("val")
    )
    return train_ds, val_ds, base.classes


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders matching the actual disk layout.

    - train + val : split from seg_train/seg_train/  (80/20 by default)
    - test        : seg_test/seg_test/

    Args:
        data_dir:     Root directory (contains seg_train/, seg_test/, seg_pred/).
        batch_size:   Mini-batch size for all splits.
        num_workers:  Number of DataLoader worker processes.
        val_fraction: Fraction of train data used for validation.
        seed:         Seed for the train/val split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = Path(data_dir)
    pin_memory = torch.cuda.is_available()

    train_dataset, val_dataset, classes = _split_train_val(root, val_fraction, seed)

    test_dataset = datasets.ImageFolder(
        root=str(root / _TEST_SUBDIR),
        transform=_build_transforms("test"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[dataset] train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}")
    print(f"[dataset] classes: {classes}")

    return train_loader, val_loader, test_loader


def get_pred_loader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, list]:
    """
    Build a DataLoader for the unlabelled seg_pred split.

    Returns:
        (pred_loader, list_of_image_paths)
    """
    root = Path(data_dir) / _PRED_SUBDIR
    pred_dataset = datasets.ImageFolder(
        root=str(root.parent),   # parent so ImageFolder sees seg_pred/ as a pseudo-class
        transform=_build_transforms("pred"),
    )
    image_paths = [s[0] for s in pred_dataset.samples]

    loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[dataset] pred={len(pred_dataset)} unlabelled images")
    return loader, image_paths
