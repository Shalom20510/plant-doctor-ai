"""Heuristic input-scope decisions for the Plant Doctor demo.

These rules interpret CLIP prompt scores; they are not calibrated probabilities.
"""
from __future__ import annotations

SUPPORTED_CROPS = (
    "apple", "blueberry", "cherry", "corn", "grape", "orange", "peach",
    "bell pepper", "potato", "raspberry", "soybean", "squash", "strawberry",
    "tomato",
)

SUPPORTED_PROMPTS = tuple(
    f"a close-up photograph of a {crop} leaf showing its surface and possible spots"
    for crop in SUPPORTED_CROPS
)
UNSUPPORTED_LEAF_PROMPT = (
    "a close-up photograph of a leaf from an unsupported plant species"
)
AMBIGUOUS_PROMPT = "an unclear, blurry, or ambiguous photograph"
NON_LEAF_PROMPTS = (
    "a photograph of a person",
    "a photograph of a building",
    "a photograph of a vehicle",
    "a photograph of food",
    "a photograph of an animal",
    "a photograph of an unrelated object",
)

SCOPE_THRESHOLD = 0.28


def decide_input_scope(
    best_description: str,
    supported_score: float,
    best_score: float,
    threshold: float = SCOPE_THRESHOLD,
) -> dict[str, str | bool | float]:
    """Decide whether to continue to disease classification.

    supported_score is the sum of CLIP softmax scores for supported crop prompts.
    It is a relative heuristic, not a calibrated probability.
    """
    if not 0.0 <= supported_score <= 1.0:
        raise ValueError("supported_score must be between 0 and 1")
    if not 0.0 <= best_score <= 1.0:
        raise ValueError("best_score must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if best_description in SUPPORTED_PROMPTS and supported_score >= threshold:
        return {
            "accepted": True,
            "status": "supported_candidate",
            "reason": "The image most closely matches a crop category represented by this model.",
            "supported_score": supported_score,
            "best_score": best_score,
        }
    if best_description == UNSUPPORTED_LEAF_PROMPT:
        return {
            "accepted": False,
            "status": "unsupported_plant",
            "reason": "This appears closer to a plant category outside the model's supported crop classes. Disease prediction was not run.",
            "supported_score": supported_score,
            "best_score": best_score,
        }
    if best_description in NON_LEAF_PROMPTS:
        return {
            "accepted": False,
            "status": "non_leaf",
            "reason": "This does not look like a supported crop-leaf photo. Upload a clear close-up of one leaf. Disease prediction was not run.",
            "supported_score": supported_score,
            "best_score": best_score,
        }
    if best_description == AMBIGUOUS_PROMPT or supported_score < threshold:
        return {
            "accepted": False,
            "status": "inconclusive",
            "reason": "The image is too ambiguous for the scope check. Try a sharper, well-lit close-up of one leaf. Disease prediction was not run.",
            "supported_score": supported_score,
            "best_score": best_score,
        }
    return {
        "accepted": False,
        "status": "inconclusive",
        "reason": "The input could not be confidently matched to a supported crop category. Disease prediction was not run.",
        "supported_score": supported_score,
        "best_score": best_score,
    }
