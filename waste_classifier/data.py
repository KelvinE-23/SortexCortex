from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TransformSubset(Dataset):
    """Subset wrapper that lets train and validation use different transforms."""

    def __init__(self, dataset: datasets.ImageFolder, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        dataset_index = self.indices[item]
        return self.dataset[dataset_index]


def _validate_dataset_root(data_dir: str | Path) -> Path:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_path}")
    if not any(path.is_dir() for path in data_path.iterdir()):
        raise ValueError(
            "Dataset folder should contain one subfolder per class, "
            "for example data/cardboard, data/glass, and so on."
        )
    return data_path


def get_transforms(image_size: int = 224):
    """Return beginner-friendly train and validation transforms."""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def create_stratified_split(
    targets: Sequence[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Create a simple per-class split so each class appears in train and val."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    indices_by_class: Dict[int, List[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[target].append(index)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_indices in indices_by_class.values():
        rng.shuffle(class_indices)

        if len(class_indices) == 1:
            train_indices.extend(class_indices)
            continue

        val_count = max(1, int(len(class_indices) * val_ratio))
        if val_count >= len(class_indices):
            val_count = len(class_indices) - 1

        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_datasets(
    data_dir: str | Path,
    image_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Create train and validation datasets from one ImageFolder root."""

    data_path = _validate_dataset_root(data_dir)
    train_transform, val_transform = get_transforms(image_size=image_size)

    base_dataset = datasets.ImageFolder(root=data_path)
    class_names = base_dataset.classes

    train_indices, val_indices = create_stratified_split(
        targets=base_dataset.targets,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_path, transform=val_transform)

    train_subset = TransformSubset(train_dataset, train_indices)
    val_subset = TransformSubset(val_dataset, val_indices)

    split_info = {
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "num_classes": len(class_names),
    }

    return train_subset, val_subset, class_names, split_info


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
):
    """Create train and validation dataloaders."""

    train_dataset, val_dataset, class_names, split_info = build_datasets(
        data_dir=data_dir,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, class_names, split_info
