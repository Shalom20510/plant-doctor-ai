"""Evaluate a trained Plant Doctor AI checkpoint on a held-out dataset split.

Example:
    python -m src.eval --data_dir data/plantvillage/color --checkpoint artifacts/model_v1.pt

The script creates a deterministic per-class validation split using the same
split settings as src.train.py. For a rigorous final report, evaluate against a
separate test set that was never used for model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.data import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, _split_indices_by_class
from src.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained plant-disease model.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/plantvillage/color"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/model_v1.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")

    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {args.data_dir}. "
            "Expected one subfolder per class."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train a model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_classes, model_device = load_model(args.checkpoint, device)

    dataset_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=dataset_transform)
    if dataset.classes != checkpoint_classes:
        raise ValueError(
            "Dataset class names/order do not match the checkpoint. "
            "The evaluator will not silently remap labels because that can "
            "produce misleading metrics. Ensure the training and evaluation "
            "datasets use the same class folders and class order."
        )

    _, validation_indices = _split_indices_by_class(
        dataset.targets, args.val_fraction, args.seed
    )
    validation_dataset = Subset(dataset, validation_indices)
    loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_targets: list[int] = []
    all_predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(model_device, non_blocking=True))
            predictions = logits.argmax(dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_targets.extend(targets.tolist())

    if not all_targets:
        raise RuntimeError("No validation images were available for evaluation.")

    labels = list(range(len(checkpoint_classes)))
    report = classification_report(
        all_targets,
        all_predictions,
        labels=labels,
        target_names=checkpoint_classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(all_targets, all_predictions, labels=labels).tolist()
    results = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.data_dir),
        "device": str(model_device),
        "validation_images": len(all_targets),
        "num_classes": len(checkpoint_classes),
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(all_targets, all_predictions)),
        "macro_f1": float(
            f1_score(all_targets, all_predictions, average="macro", zero_division=0)
        ),
        "class_names": checkpoint_classes,
        "classification_report": report,
        "confusion_matrix": {
            "labels": checkpoint_classes,
            "matrix": matrix,
        },
        "note": (
            "This is a validation split, not an independent external test set. "
            "Results may not reflect performance on field photographs."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "evaluation_report.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Validation images: {len(all_targets)}")
    print(f"Classes: {len(checkpoint_classes)}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print("\nPer-class metrics:")
    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=labels,
            target_names=checkpoint_classes,
            zero_division=0,
        )
    )
    print(f"Confusion matrix and full report saved to: {output_path}")
    print(results["note"])


if __name__ == "__main__":
    main()
