"""Robustness evaluation on original and synthetically degraded leaf images.

Run:
    py -m src.robustness --data_dir data/plantvillage/color --checkpoint artifacts/models/model_v1.pt

Synthetic clutter is an edge-overlay stress test, not a realistic background replacement.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torchvision import datasets, transforms

from src.data import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare classifier performance under image degradation.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/plantvillage/color"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/model_v1.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/robustness_report.json"))
    parser.add_argument("--max_images", type=int, default=0, help="Optional quick-test cap; 0 evaluates all images.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def synthetic_clutter(image: Image.Image, seed: int) -> Image.Image:
    """Overlay repeatable colored shapes near image edges as a clutter proxy."""
    image = image.convert("RGB").copy()
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    rng = random.Random(seed)
    band_x, band_y = max(1, width // 5), max(1, height // 5)
    colors = [(40, 100, 30, 115), (150, 90, 30, 115), (30, 80, 130, 105)]
    for _ in range(12):
        side = rng.randrange(4)
        if side in (0, 1):
            x = rng.randint(0, max(0, width - 1))
            y = rng.randint(0, max(0, band_y - 1)) if side == 0 else rng.randint(max(0, height - band_y), max(0, height - 1))
        else:
            x = rng.randint(0, max(0, band_x - 1)) if side == 2 else rng.randint(max(0, width - band_x), max(0, width - 1))
            y = rng.randint(0, max(0, height - 1))
        radius = rng.randint(max(2, min(width, height) // 40), max(3, min(width, height) // 12))
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=rng.choice(colors))
    return image


def variants(image: Image.Image, seed: int) -> dict[str, Image.Image]:
    image = image.convert("RGB")
    return {
        "original": image,
        "brightness_dim_0.55": ImageEnhance.Brightness(image).enhance(0.55),
        "brightness_bright_1.45": ImageEnhance.Brightness(image).enhance(1.45),
        "rotation_15_degrees": image.rotate(15, resample=Image.Resampling.BILINEAR, fillcolor=(124, 124, 124)),
        "rotation_30_degrees": image.rotate(30, resample=Image.Resampling.BILINEAR, fillcolor=(124, 124, 124)),
        "gaussian_blur_1.5": image.filter(ImageFilter.GaussianBlur(radius=1.5)),
        "synthetic_edge_clutter": synthetic_clutter(image, seed),
    }


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}. Expected one subfolder per class.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.max_images < 0:
        raise ValueError("--max_images must be 0 or a positive integer.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, model_device = load_model(args.checkpoint, device)
    dataset = datasets.ImageFolder(args.data_dir)
    if dataset.classes != class_names:
        raise ValueError("Dataset class names/order do not match checkpoint class order. Align labels before evaluating.")
    # Select a deterministic, approximately class-balanced subset for quick runs.
    # Taking the first N ImageFolder samples would overrepresent the first class.
    by_class: dict[int, list[int]] = {}
    for sample_index, target in enumerate(dataset.targets):
        by_class.setdefault(int(target), []).append(sample_index)

    rng = random.Random(args.seed)
    for class_indices in by_class.values():
        rng.shuffle(class_indices)

    if args.max_images:
        if args.max_images < len(by_class):
            print(
                f"WARNING: --max_images={args.max_images} is smaller than the "
                f"{len(by_class)} classes; some classes cannot be represented."
            )
        indices = []
        class_ids = sorted(by_class)
        # Round-robin selection balances classes as far as the cap allows.
        while len(indices) < min(args.max_images, len(dataset)):
            added_this_round = False
            for class_id in class_ids:
                if by_class[class_id]:
                    indices.append(by_class[class_id].pop())
                    added_this_round = True
                    if len(indices) >= args.max_images:
                        break
            if not added_this_round:
                break
    else:
        indices = list(range(len(dataset)))

    if not indices:
        raise RuntimeError("No images found in dataset.")

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    totals: dict[str, dict[str, list[int]]] = {}
    skipped = 0
    model.eval()
    for index in indices:
        path, target = dataset.samples[index]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            print(f"Skipping unreadable image: {path} ({exc})")
            skipped += 1
            continue
        for condition, variant in variants(image, args.seed + index).items():
            tensor = preprocess(variant).unsqueeze(0).to(model_device)
            with torch.inference_mode():
                prediction = int(model(tensor).argmax(dim=1).item())
            bucket = totals.setdefault(condition, {"targets": [], "predictions": []})
            bucket["targets"].append(int(target))
            bucket["predictions"].append(prediction)

    results: dict = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.data_dir),
        "device": str(model_device), "num_classes": len(class_names),
        "images_requested": len(indices), "unreadable_images_skipped": skipped,
        "sampling": (
            "deterministic approximately class-balanced subset"
            if args.max_images else "all dataset images"
        ),
        "seed": args.seed, "conditions": {},
        "limitations": [
            "Synthetic transformations are stress tests, not a substitute for real field photographs.",
            "Synthetic edge clutter overlays shapes; it does not realistically replace the background.",
            "Use an independent held-out test set for final claims; results depend on dataset representativeness.",
            "Softmax confidence is not a calibrated probability of correctness."
        ],
    }
    original_accuracy = None
    for condition, bucket in totals.items():
        y_true, y_pred = bucket["targets"], bucket["predictions"]
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if condition == "original":
            original_accuracy = accuracy
        results["conditions"][condition] = {
            "images": len(y_true), "accuracy": accuracy, "macro_f1": macro_f1,
            "accuracy_delta_vs_original": None,
        }
    if original_accuracy is not None:
        for metrics in results["conditions"].values():
            metrics["accuracy_delta_vs_original"] = round(metrics["accuracy"] - original_accuracy, 6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images evaluated per condition: {len(indices) - skipped}")
    print("\nCondition                         Accuracy    Macro-F1    Δ accuracy")
    print("-" * 72)
    for name, metrics in results["conditions"].items():
        delta = metrics["accuracy_delta_vs_original"]
        delta_text = "—" if delta is None else f"{delta:+.4f}"
        print(f"{name:32} {metrics['accuracy']:.4f}      {metrics['macro_f1']:.4f}      {delta_text}")
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
