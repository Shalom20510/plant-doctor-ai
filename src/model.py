"""EfficientNet-B0 architecture and checkpoint loading for Plant Doctor AI.

This module deliberately does not download weights automatically. A disease
classifier must use weights trained for the configured plant-disease classes;
ImageNet-only weights are not a valid substitute for those weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


# Standard PlantVillage 38-class label set, in ImageFolder alphabetical order.
# If a checkpoint stores its own class_names/classes list, that list takes priority.
PLANTVILLAGE_CLASSES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def build_model(num_classes: int = len(PLANTVILLAGE_CLASSES)) -> nn.Module:
    """Create EfficientNet-B0 with a classifier head for the requested classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    network = models.efficientnet_b0(weights=None)
    in_features = network.classifier[1].in_features
    network.classifier[1] = nn.Linear(in_features, num_classes)
    return network


def _extract_checkpoint(checkpoint: Any) -> tuple[dict[str, torch.Tensor], list[str] | None]:
    """Normalize common checkpoint formats into a state dict and optional labels."""
    class_names: list[str] | None = None
    state: Any = checkpoint

    if isinstance(checkpoint, dict):
        candidate_names = checkpoint.get("class_names", checkpoint.get("classes"))
        if isinstance(candidate_names, (list, tuple)) and all(
            isinstance(name, str) for name in candidate_names
        ):
            class_names = list(candidate_names)

        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict) or not state:
        raise ValueError(
            "Unsupported checkpoint format. Expected a model state_dict or a "
            "checkpoint containing 'state_dict'/'model_state_dict'."
        )

    # Remove the common DataParallel prefix without changing parameter names otherwise.
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }
    if not cleaned:
        raise ValueError("The checkpoint does not contain usable tensor weights.")

    return cleaned, class_names


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, list[str], torch.device]:
    """Load a trained plant-disease model and return (model, labels, device).

    Supported checkpoints may be a raw state_dict or a dictionary containing
    'state_dict'/'model_state_dict' and optional 'class_names'/'classes'.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {path}. "
            "Train or obtain a compatible plant-disease checkpoint before prediction."
        )

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    try:
        checkpoint = torch.load(path, map_location=selected_device, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept weights_only.
        checkpoint = torch.load(path, map_location=selected_device)

    state_dict, checkpoint_classes = _extract_checkpoint(checkpoint)
    classes = checkpoint_classes or PLANTVILLAGE_CLASSES

    # Infer output size from the checkpoint when its classifier head is present.
    classifier_weight = state_dict.get("classifier.1.weight")
    num_classes = (
        int(classifier_weight.shape[0])
        if classifier_weight is not None and classifier_weight.ndim == 2
        else len(classes)
    )
    if len(classes) != num_classes:
        raise ValueError(
            f"Checkpoint outputs {num_classes} classes but {len(classes)} labels "
            "are configured. Supply the exact class order used during training."
        )

    network = build_model(num_classes=num_classes)
    try:
        network.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint weights do not match the expected EfficientNet-B0 "
            "architecture. Use a checkpoint trained with this model definition."
        ) from exc

    network.to(selected_device)
    network.eval()
    return network, classes, selected_device
