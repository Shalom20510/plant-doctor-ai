"""Image preprocessing and inference helpers for Plant Doctor AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.model import load_model


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized batch tensor for EfficientNet-B0."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    rgb_image = image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(rgb_image).unsqueeze(0)


def load_image(image_source: str | Path | Image.Image) -> Image.Image:
    """Open an image path or copy a supplied PIL image into RGB mode."""
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    path = Path(image_source)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{path.suffix}'. Use JPG, JPEG, or PNG."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read a valid image from: {path}") from exc


def _format_label(label: str) -> tuple[str, str]:
    """Return a readable crop name and class/disease label."""
    crop, separator, condition = label.partition("___")
    if not separator:
        return crop.replace("_", " "), " ".join(label.replace("_", " ").split())
    return crop.replace("_", " "), " ".join(condition.replace("_", " ").split())


def predict_image(
    image_source: str | Path | Image.Image,
    checkpoint_path: str | Path = "artifacts/model_v1.pt",
    top_k: int = 3,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Predict plant-disease classes for one image using a trained checkpoint.

    Returns the top prediction, confidence, and top-k alternatives. Confidence is
    the model's softmax score, not a guarantee that the diagnosis is correct.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    model, class_names, selected_device = load_model(checkpoint_path, device)
    image = load_image(image_source)
    batch = preprocess_image(image).to(selected_device)

    with torch.inference_mode():
        logits = model(batch)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("Model returned an unexpected output shape.")
        probabilities = torch.softmax(logits[0], dim=0)
        count = min(top_k, len(class_names))
        scores, indices = torch.topk(probabilities, k=count)

    alternatives: list[dict[str, Any]] = []
    for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
        label = class_names[index]
        crop, condition = _format_label(label)
        alternatives.append(
            {
                "class_name": label,
                "crop": crop,
                "condition": condition,
                "confidence": float(score),
            }
        )

    best = alternatives[0]
    return {
        "class_name": best["class_name"],
        "crop": best["crop"],
        "condition": best["condition"],
        "confidence": best["confidence"],
        "top_predictions": alternatives,
        "device": str(selected_device),
        "note": (
            "This is an AI estimate, not a confirmed diagnosis. Check the result "
            "against visible symptoms and seek agricultural advice before treatment."
        ),
    }
