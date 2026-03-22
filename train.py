from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from waste_classifier.data import build_dataloaders
from waste_classifier.model import build_resnet18_model
from waste_classifier.utils import (
    ensure_dir,
    evaluate_model,
    get_device,
    plot_confusion_matrix,
    save_checkpoint,
    save_training_history,
    set_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 waste classifier.")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root with one folder per class.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Folder to save checkpoints and plots.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize images to this square size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the pretrained ResNet backbone and train only the classifier head.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train ResNet18 from scratch instead of using ImageNet pretrained weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    plots_dir = ensure_dir(output_dir / "plots")

    train_loader, val_loader, class_names, split_info = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print(f"Classes: {class_names}")
    print(
        f"Train samples: {split_info['train_size']} | "
        f"Validation samples: {split_info['val_size']}"
    )

    device = get_device()
    print(f"Using device: {device}")

    model = build_resnet18_model(
        num_classes=len(class_names),
        freeze_backbone=args.freeze_backbone,
        use_pretrained=not args.no_pretrained,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1.0
    best_checkpoint_path = checkpoints_dir / "best_model.pth"
    last_checkpoint_path = checkpoints_dir / "last_model.pth"

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | train acc: {train_accuracy:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | val acc: {val_metrics['accuracy']:.4f}"
        )

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            class_names=class_names,
            best_val_accuracy=best_val_accuracy,
            image_size=args.image_size,
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                class_names=class_names,
                best_val_accuracy=best_val_accuracy,
                image_size=args.image_size,
            )
            print(f"Saved new best checkpoint to {best_checkpoint_path}")

    history_path = output_dir / "training_history.json"
    save_training_history(history_path, history)

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics, targets, predictions = evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )

    confusion_matrix_path = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
        output_path=confusion_matrix_path,
    )

    print(f"Best validation accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Confusion matrix image: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
