from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_accuracy = correct_predictions / max(total_samples, 1)
    return epoch_loss, epoch_accuracy


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = {
        "loss": running_loss / max(total_samples, 1),
        "accuracy": correct_predictions / max(total_samples, 1),
    }
    return metrics, all_targets, all_predictions


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    class_names: List[str],
    best_val_accuracy: float,
    image_size: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "best_val_accuracy": best_val_accuracy,
        "image_size": image_size,
    }
    torch.save(checkpoint, path)


def save_training_history(path: str | Path, history: Dict[str, List[float]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def plot_confusion_matrix(
    targets: List[int],
    predictions: List[int],
    class_names: List[str],
    output_path: str | Path,
) -> None:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Validation Confusion Matrix",
    )

    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            axis.text(col, row, format(matrix[row, col], "d"), ha="center", va="center", color=color)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

