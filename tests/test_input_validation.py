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
