"""Focused integration tests for the Plant Doctor AI baseline."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src import model as model_module
from src import predict as predict_module
from src.data import _split_indices_by_class


def _png_bytes(color: tuple[int, int, int] = (30, 150, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_accepts_file_like_upload() -> None:
    uploaded = io.BytesIO(_png_bytes())
    image = predict_module.load_image(uploaded)
    assert image.mode == "RGB"
    assert image.size == (32, 24)


def test_load_image_accepts_pil_image() -> None:
    image = Image.new("RGBA", (10, 12), color=(1, 2, 3, 100))
    loaded = predict_module.load_image(image)
    assert loaded.mode == "RGB"
    assert loaded.size == (10, 12)


def test_load_image_rejects_unsupported_path_extension(tmp_path: Path) -> None:
    unsupported = tmp_path / "leaf.gif"
    unsupported.write_bytes(b"not a real image")
    with pytest.raises(ValueError, match="Unsupported image extension"):
        predict_module.load_image(unsupported)


def test_predict_image_accepts_file_like_input(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedModel(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            assert tuple(batch.shape) == (1, 3, 224, 224)
            return torch.tensor([[0.0, 3.0]])

    monkeypatch.setattr(
        predict_module,
        "load_checkpoint_model",
        lambda checkpoint_path, device: (
            FixedModel(),
            ["Apple___healthy", "Tomato___healthy"],
            torch.device("cpu"),
        ),
    )

    result = predict_module.predict_image(io.BytesIO(_png_bytes()), top_k=2)

    assert result["class_name"] == "Tomato___healthy"
    assert result["crop"] == "Tomato"
    assert len(result["top_predictions"]) == 2
    assert 0.0 <= result["confidence"] <= 1.0


class _TinyCheckpointModel(nn.Module):
    """Small stand-in preserving the classifier key expected by load_model."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Linear(3, 4)
        self.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(4, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def test_load_model_reads_checkpoint_and_saved_class_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    classes = ["Class_A", "Class_B"]
    source_model = _TinyCheckpointModel(num_classes=len(classes))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"state_dict": source_model.state_dict(), "class_names": classes},
        checkpoint,
    )
    monkeypatch.setattr(model_module, "build_model", _TinyCheckpointModel)

    loaded_model, loaded_classes, device = model_module.load_model(checkpoint, "cpu")

    assert loaded_classes == classes
    assert device == torch.device("cpu")
    assert not loaded_model.training
    assert set(loaded_model.state_dict()) == set(source_model.state_dict())


def test_training_and_evaluation_share_deterministic_class_split() -> None:
    # ImageFolder targets are ordered by class and then by filename.
    targets = [0] * 5 + [1] * 6 + [2] * 4
    train_a, validation_a = _split_indices_by_class(targets, 0.2, 42)
    train_b, validation_b = _split_indices_by_class(targets, 0.2, 42)

    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)
    assert sorted(train_a + validation_a) == list(range(len(targets)))
    assert {targets[index] for index in train_a} == {0, 1, 2}
    assert {targets[index] for index in validation_a} == {0, 1, 2}


def test_checkpoint_class_order_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    source_model = _TinyCheckpointModel(num_classes=2)
    torch.save(
        {
            "state_dict": source_model.state_dict(),
            "class_names": ["Class_A"],
        },
        checkpoint,
    )
    monkeypatch.setattr(model_module, "build_model", _TinyCheckpointModel)

    with pytest.raises(ValueError, match="outputs 2 classes but 1 labels"):
        model_module.load_model(checkpoint, "cpu")


def test_crop_conditioned_prediction_excludes_other_plant_classes() -> None:
    class FixedLogitModel(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            # Bell Pepper wins globally, while Tomato healthy is best among
            # Tomato-only classes. The crop-conditioned helper must not return
            # a Bell Pepper label when the user explicitly selected Tomato.
            return torch.tensor([[12.0, 2.0, 4.0, 1.0]])

    idx_to_class = {
        0: "Pepper,_bell___healthy",
        1: "Tomato___Bacterial_spot",
        2: "Tomato___healthy",
        3: "Apple___healthy",
    }
    result = predict_module.predict_topk_for_crop(
        model=FixedLogitModel(),
        image_tensor=torch.zeros((1, 3, 224, 224)),
        idx_to_class=idx_to_class,
        device="cpu",
        crop_name="Tomato",
        top_k=3,
    )

    assert result[0][0] == "Tomato___healthy"
    assert all(label.startswith("Tomato___") for label, _ in result)
    assert 0.0 <= result[0][1] <= 1.0


def test_crop_conditioned_prediction_rejects_unknown_crop() -> None:
    class FixedLogitModel(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="No model classes are available"):
        predict_module.predict_topk_for_crop(
            model=FixedLogitModel(),
            image_tensor=torch.zeros((1, 3, 224, 224)),
            idx_to_class={0: "Apple___healthy", 1: "Tomato___healthy"},
            device="cpu",
            crop_name="Mango",
        )
