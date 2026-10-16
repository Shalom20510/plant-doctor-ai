"""Dataset-free tests for Plant Doctor input-scope rejection rules."""
import pytest

from src.input_validation import (
    AMBIGUOUS_PROMPT,
    NON_LEAF_PROMPTS,
    SUPPORTED_PROMPTS,
    UNSUPPORTED_LEAF_PROMPT,
    decide_input_scope,
)


def test_supported_crop_candidate_is_accepted():
    result = decide_input_scope(SUPPORTED_PROMPTS[0], 0.62, 0.31)
    assert result["accepted"] is True
    assert result["status"] == "supported_candidate"


def test_non_leaf_image_is_rejected():
    result = decide_input_scope(NON_LEAF_PROMPTS[0], 0.03, 0.02)
    assert result["accepted"] is False
    assert result["status"] == "non_leaf"
    assert "Disease prediction was not run" in result["reason"]


def test_unsupported_plant_category_is_rejected():
    result = decide_input_scope(UNSUPPORTED_LEAF_PROMPT, 0.12, 0.20)
    assert result["accepted"] is False
    assert result["status"] == "unsupported_plant"


def test_ambiguous_or_low_supported_score_is_rejected():
    result = decide_input_scope(AMBIGUOUS_PROMPT, 0.18, 0.16)
    assert result["accepted"] is False
    assert result["status"] == "inconclusive"


def test_supported_prompt_with_low_scope_score_is_rejected():
    result = decide_input_scope(SUPPORTED_PROMPTS[0], 0.12, 0.05)
    assert result["accepted"] is False
    assert result["status"] == "inconclusive"


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError):
        decide_input_scope(SUPPORTED_PROMPTS[0], score, 0.2)

from src.input_validation import (
    CROP_PROMPT_VARIANTS,
    SUPPORTED_CROP_NAMES,
    aggregate_scope_logits,
    crop_name_from_prompt,
    normalize_crop_name,
)


def test_every_supported_crop_has_multiple_prompt_variants():
    assert set(CROP_PROMPT_VARIANTS) == set(SUPPORTED_CROP_NAMES)
    assert all(len(prompts) >= 5 for prompts in CROP_PROMPT_VARIANTS.values())


def test_crop_specific_morphology_prompts_are_distinct():
    assert "serrated leaflets" in CROP_PROMPT_VARIANTS["tomato"][-1]
    assert "three broad oval leaflets" in CROP_PROMPT_VARIANTS["soybean"][-1]
    assert CROP_PROMPT_VARIANTS["tomato"][-1] != CROP_PROMPT_VARIANTS["soybean"][-1]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Bell Pepper", "bell pepper"),
        ("Pepper bell", "bell pepper"),
        ("Corn (maize)", "corn"),
        ("Cherry (including sour)", "cherry"),
        ("Tomato", "tomato"),
    ],
)
def test_crop_name_aliases_normalize_consistently(label, expected):
    assert normalize_crop_name(label) == expected


def test_canonical_prompt_resolves_to_crop_name():
    from src.input_validation import SUPPORTED_PROMPTS
    assert crop_name_from_prompt(SUPPORTED_PROMPTS[-1]) == "tomato"

def _crop_scores(default=0.0):
    return {
        crop: [default] * len(CROP_PROMPT_VARIANTS[crop])
        for crop in SUPPORTED_CROP_NAMES
    }


def test_prompt_ensemble_averages_scores_within_each_crop():
    scores = _crop_scores()
    scores["tomato"] = [8.0, 8.0, 8.0, 8.0, 8.0]
    # One misleading large soybean prompt score should not outweigh its lower
    # average across the full template ensemble.
    scores["soybean"] = [20.0, 2.0, 2.0, 2.0, 2.0]
    description, best_score, supported_score = aggregate_scope_logits(
        scores,
        unsupported_prompt_scores=[4.0, 4.0, 4.0],
        non_leaf_prompt_scores={prompt: 0.0 for prompt in NON_LEAF_PROMPTS},
    )
    assert description == SUPPORTED_PROMPTS[-1]
    assert best_score == pytest.approx(supported_score)
    assert supported_score > 0.90


def test_prompt_ensemble_rejects_a_strong_non_leaf_match():
    description, _, supported_score = aggregate_scope_logits(
        _crop_scores(),
        unsupported_prompt_scores=[0.0, 0.0, 0.0],
        non_leaf_prompt_scores={
            prompt: (7.0 if prompt == NON_LEAF_PROMPTS[0] else 0.0)
            for prompt in NON_LEAF_PROMPTS
        },
    )
    assert description == NON_LEAF_PROMPTS[0]
    assert supported_score < 0.01


def test_prompt_ensemble_rejects_a_strong_unsupported_leaf_match():
    description, _, supported_score = aggregate_scope_logits(
        _crop_scores(),
        unsupported_prompt_scores=[7.0, 7.0, 7.0],
        non_leaf_prompt_scores={prompt: 0.0 for prompt in NON_LEAF_PROMPTS},
    )
    assert description == UNSUPPORTED_LEAF_PROMPT
    assert supported_score < 0.01


def test_prompt_ensemble_rejects_missing_crop_prompt_group():
    scores = _crop_scores()
    scores.pop("tomato")
    with pytest.raises(ValueError, match="exactly the supported crop names"):
        aggregate_scope_logits(
            scores,
            unsupported_prompt_scores=[0.0],
            non_leaf_prompt_scores={NON_LEAF_PROMPTS[0]: 0.0},
        )



def test_prompt_ensemble_rejects_a_strong_ambiguous_match():
    description, _, supported_score = aggregate_scope_logits(
        _crop_scores(),
        unsupported_prompt_scores=[0.0, 0.0, 0.0],
        non_leaf_prompt_scores={prompt: 0.0 for prompt in NON_LEAF_PROMPTS},
        ambiguous_prompt_score=7.0,
    )
    assert description == AMBIGUOUS_PROMPT
    assert supported_score < 0.01
