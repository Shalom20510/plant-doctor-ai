"""Heuristic input-scope and crop-label helpers for Plant Doctor AI.

CLIP prompt scores are relative similarities, not calibrated probabilities.
"""
from __future__ import annotations

import math

SUPPORTED_CROP_NAMES = (
    "apple", "blueberry", "cherry", "corn", "grape", "orange", "peach",
    "bell pepper", "potato", "raspberry", "soybean", "squash", "strawberry",
    "tomato",
)

# One canonical prompt per model crop is retained for readable results and
# compatibility with the simple scope decision function.
SUPPORTED_PROMPTS = tuple(
    f"a close-up photograph of a {crop} leaf showing its surface and possible spots"
    for crop in SUPPORTED_CROP_NAMES
)

# Diverse wording reduces dependence on any single CLIP sentence template.
# A crop-specific morphology prompt is included as one vote, not treated as
# ground truth. These descriptions are heuristics and need labeled image tests.
CROP_MORPHOLOGY_PROMPTS = {
    "apple": "a broad oval apple-tree leaf with a finely serrated edge",
    "blueberry": "a small oval blueberry-bush leaf with a fine toothed edge",
    "cherry": "an oval cherry-tree leaf with a pointed tip and serrated edges",
    "corn": "a long narrow blade-like leaf of a corn maize plant",
    "grape": "a broad lobed grapevine leaf with radiating palmate veins",
    "orange": "a smooth oval citrus leaf from an orange tree",
    "peach": "a long narrow lance-shaped pointed leaf of a peach tree",
    "bell pepper": "a smooth broad pointed oval leaf of a bell pepper plant",
    "potato": "a compound potato-plant leaf with several oval leaflets",
    "raspberry": "a compound raspberry leaf with toothed oval leaflets",
    "soybean": "a soybean leaf with three broad oval leaflets",
    "squash": "a large broad lobed leaf from a squash plant",
    "strawberry": "a strawberry leaf with three toothed leaflets",
    "tomato": "a compound tomato leaf with multiple irregular deeply serrated leaflets",
}

_GENERIC_CROP_TEMPLATES = (
    "a close-up photograph of a {crop} leaf",
    "a photograph of the leaf of a {crop} plant",
    "one {crop} leaf filling the frame, showing its edges and veins",
    "a {crop} leaf being checked for plant disease",
)

CROP_PROMPT_VARIANTS = {
    crop: tuple(
        template.format(crop=crop)
        for template in _GENERIC_CROP_TEMPLATES
    ) + (CROP_MORPHOLOGY_PROMPTS[crop],)
    for crop in SUPPORTED_CROP_NAMES
}

UNSUPPORTED_LEAF_PROMPT = (
    "a close-up photograph of a leaf from an unsupported plant species"
)
UNSUPPORTED_LEAF_PROMPTS = (
    UNSUPPORTED_LEAF_PROMPT,
    "a leaf from a plant species not included in the supported crop list",
    "a close-up of an unknown garden plant leaf",
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


def aggregate_scope_logits(
    crop_prompt_scores: dict[str, list[float]],
    unsupported_prompt_scores: list[float],
    non_leaf_prompt_scores: dict[str, float],
) -> tuple[str, float, float]:
    """Aggregate raw CLIP logits across templates and return (label, best score, supported score).

    Scores are raw image-text logits, not calibrated probabilities. Crop templates
    are averaged within each crop so each crop gets equal weight irrespective of
    the number of prompt variants.
    """
    if set(crop_prompt_scores) != set(SUPPORTED_CROP_NAMES):
        raise ValueError("crop_prompt_scores must contain exactly the supported crop names")
    if any(not values for values in crop_prompt_scores.values()):
        raise ValueError("Every supported crop must have at least one prompt score")
    if not unsupported_prompt_scores or not non_leaf_prompt_scores:
        raise ValueError("At least one unsupported-leaf and one non-leaf score are required")

    all_scores = [
        score for values in crop_prompt_scores.values() for score in values
    ] + list(unsupported_prompt_scores) + list(non_leaf_prompt_scores.values())
    if any(not math.isfinite(float(score)) for score in all_scores):
        raise ValueError("Prompt scores must all be finite numbers")

    mean_crop_logits = {
        crop: sum(values) / len(values)
        for crop, values in crop_prompt_scores.items()
    }
    best_crop = max(mean_crop_logits, key=mean_crop_logits.get)
    best_supported_logit = mean_crop_logits[best_crop]
    unsupported_logit = sum(unsupported_prompt_scores) / len(unsupported_prompt_scores)
    best_non_leaf_prompt, best_non_leaf_logit = max(
        non_leaf_prompt_scores.items(), key=lambda item: item[1]
    )

    group_logits = [best_supported_logit, unsupported_logit, best_non_leaf_logit]
    max_logit = max(group_logits)
    exp_scores = [math.exp(value - max_logit) for value in group_logits]
    denominator = sum(exp_scores)
    group_scores = [value / denominator for value in exp_scores]
    winner = max(range(len(group_scores)), key=group_scores.__getitem__)

    if winner == 0:
        description = SUPPORTED_PROMPTS[SUPPORTED_CROP_NAMES.index(best_crop)]
    elif winner == 1:
        description = UNSUPPORTED_LEAF_PROMPT
    else:
        description = best_non_leaf_prompt

    return description, group_scores[winner], group_scores[0]


def normalize_crop_name(value: str) -> str:
    """Normalize crop aliases used by CLIP prompts and checkpoint class labels."""
    normalized = " ".join(
        value.lower().replace("_", " ").replace(",", " ").replace("(", " ").replace(")", " ").split()
    )
    aliases = {
        "pepper bell": "bell pepper",
        "corn maize": "corn",
        "cherry including sour": "cherry",
    }
    return aliases.get(normalized, normalized)


def crop_name_from_prompt(prompt: str) -> str | None:
    """Return the canonical crop name associated with a known CLIP prompt."""
    for crop, prompts in CROP_PROMPT_VARIANTS.items():
        if prompt in prompts:
            return crop
    for crop, canonical in zip(SUPPORTED_CROP_NAMES, SUPPORTED_PROMPTS):
        if prompt == canonical:
            return crop
    return None


def decide_input_scope(
    best_description: str,
    supported_score: float,
    best_score: float,
    threshold: float = SCOPE_THRESHOLD,
) -> dict[str, str | bool | float]:
    """Decide whether to continue to disease classification.

    supported_score and best_score are relative CLIP group softmax scores,
    not calibrated probabilities of plant identity or diagnosis.
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
