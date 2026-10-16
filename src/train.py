"""Train an EfficientNet-B0 plant-disease classifier with transfer learning.

Expected dataset structure:
    data/plantvillage/color/
        Apple___Apple_scab/
            image1.jpg
        Apple___healthy/
            image2.jpg
        ...

Example:
    python -m src.train --data_dir data/plantvillage/color --epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torchvision import models

from src.data import create_dataloaders, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Plant Doctor AI.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/plantvillage/color"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_v1.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Do not download/use ImageNet pretrained weights (usually slower to converge).",
    )
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """Run a training or evaluation epoch and return loss, accuracy, macro-F1."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * targets.size(0)
        predictions = logits.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    if not all_targets:
        raise RuntimeError("The data loader returned no images.")
    average_loss = total_loss / len(all_targets)
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    return average_loss, float(accuracy), float(macro_f1)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")

    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        f"Classes: {len(class_names)} | "
        f"Training images: {len(train_loader.dataset)} | "
        f"Validation images: {len(val_loader.dataset)}"
    )

    weights = None
    if not args.no_pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_macro_f1 = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f}, macro-F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}, macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "architecture": "efficientnet_b0",
                    "image_size": 224,
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                    "best_epoch": epoch,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "seed": args.seed,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training history saved to {history_path}")
    print(
        "Training complete. Validation metrics are estimates on the held-out split; "
        "they do not guarantee performance on real-world field photos."
    )


if __name__ == "__main__":
    main()
