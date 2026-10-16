"""Image preprocessing and inference helpers for Plant Doctor AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import json
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.model import load_model as load_checkpoint_model
from src.input_validation import normalize_crop_name


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_device() -> torch.device:
    """Return the preferred inference device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_mapping(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load the saved class-to-index mapping and its inverse."""
    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("Class mapping must be a non-empty JSON object.")
    normalized = {str(name): int(index) for name, index in class_to_idx.items()}
    if sorted(normalized.values()) != list(range(len(normalized))):
        raise ValueError("Class mapping indices must be contiguous from zero.")
    return normalized, {index: name for name, index in normalized.items()}


def get_inference_transform(img_size: int = IMAGE_SIZE):
    """Build the same deterministic preprocessing transform used at inference."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(
    weights_path: str | Path,
    num_classes: int | None = None,
    device: str | torch.device | None = None,
):
    """Backward-compatible app wrapper around the canonical checkpoint loader."""
    model, class_names, selected_device = load_checkpoint_model(weights_path, device)
    if num_classes is not None and len(class_names) != num_classes:
        raise ValueError(
            f"Checkpoint has {len(class_names)} classes, but the class map has {num_classes}."
        )
    return model


def predict_topk(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    idx_to_class: dict[int, str],
    device: str | torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k (label, softmax score) pairs for a preprocessed image tensor."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor.to(device))
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("Model returned an unexpected output shape.")
        probabilities = torch.softmax(logits[0], dim=0)
        count = min(top_k, probabilities.numel())
        scores, indices = torch.topk(probabilities, k=count)
    predictions = []
    for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
        if index not in idx_to_class:
            raise ValueError(f"Predicted class index {index} is missing from the class map.")
        predictions.append((idx_to_class[index], float(score)))
    return predictions


def predict_topk_for_crop(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    idx_to_class: dict[int, str],
    device: str | torch.device,
    crop_name: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Predict among one user-confirmed crop's classes only.

    Returned scores are softmax scores conditional on the selected crop's
    classes. They are not comparable to global 38-class scores and they do not
    independently verify that the uploaded plant really is that crop.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    selected_crop = normalize_crop_name(crop_name)
    matching_indices = []
    for index, label in idx_to_class.items():
        raw_crop = label.partition("___")[0]
        if normalize_crop_name(raw_crop) == selected_crop:
            matching_indices.append((int(index), label))

    if not matching_indices:
        raise ValueError(f"No model classes are available for crop '{crop_name}'.")

    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor.to(device))
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("Model returned an unexpected output shape.")
        indices = [index for index, _ in matching_indices]
        if max(indices) >= logits.shape[1] or min(indices) < 0:
            raise ValueError("The crop class mapping does not match model output indices.")
        probabilities = torch.softmax(logits[0, indices], dim=0)
        count = min(top_k, probabilities.numel())
        scores, local_indices = torch.topk(probabilities, k=count)

    predictions = []
    for score, local_index in zip(scores.cpu().tolist(), local_indices.cpu().tolist()):
        class_index = indices[local_index]
        predictions.append((idx_to_class[class_index], float(score)))
    return predictions



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


def load_image(image_source: str | Path | Image.Image | BinaryIO) -> Image.Image:
    """Load a path, PIL image, or seekable uploaded file and return RGB pixels."""
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    # Streamlit UploadedFile and ordinary file objects are file-like, not paths.
    if hasattr(image_source, "read"):
        try:
            if hasattr(image_source, "seek"):
                image_source.seek(0)
            with Image.open(image_source) as image:
                image.load()
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Could not read a valid uploaded image.") from exc

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
    image_source: str | Path | Image.Image | BinaryIO,
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

    model, class_names, selected_device = load_checkpoint_model(checkpoint_path, device)
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
