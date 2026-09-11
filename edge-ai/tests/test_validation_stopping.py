"""
Unit tests for the adaptive validation stopping criteria module.
"""
import os
import sys

# Allow direct import of the module without triggering GUI dependencies
_module_path = os.path.join(
    os.path.dirname(__file__), "..", "gui", "application_validation"
)
sys.path.insert(0, _module_path)

import pytest

from validation_stopping import (
    DEFAULT_CONFUSION_SMOOTHING_PRIOR,
    DEFAULT_DIAG_CREDIBLE_INTERVAL_TOLERANCE,
    DEFAULT_MIN_CLASS_EXAMPLES,
    DEFAULT_OFFDIAG_CREDIBLE_INTERVAL_TOLERANCE,
    ValidationConfig,
    build_confusion_error_model,
    check_stopping_criteria,
)


def _make_rows(n_correct: int, n_total: int, label: str = "copepod") -> list:
    rows = []
    for i in range(n_total):
        final = label if i < n_correct else "other"
        rows.append({"original_label": label, "final_label": final})
    return rows


def test_build_confusion_error_model_returns_counts_and_probabilities():
    rows = [
        {"original_label": "copepod", "final_label": "copepod"},
        {"original_label": "copepod", "final_label": "copepod"},
        {"original_label": "copepod", "final_label": "detritus"},
        {"original_label": "detritus", "final_label": "copepod"},
        {"original_label": "detritus", "final_label": "detritus"},
    ]

    model = build_confusion_error_model(
        rows,
        labels=["copepod", "detritus"],
        smoothing_prior_alpha=DEFAULT_CONFUSION_SMOOTHING_PRIOR,
    )

    assert model["class_order"] == ["copepod", "detritus"]
    assert model["confusion_counts"][0][0] == 2
    assert model["transition_probabilities"][0][0] == pytest.approx(2 / 3)
    assert model["dirichlet_posterior_parameters"][0][0] == pytest.approx(2.75)


def test_uncertain_rows_are_excluded_from_confusion_error_model():
    rows = [
        {"original_label": "copepod", "final_label": "copepod"},
        {"original_label": "copepod", "final_label": "uncertain"},
        {"original_label": "detritus", "final_label": "detritus"},
    ]

    model = build_confusion_error_model(
        rows,
        labels=["copepod", "detritus"],
        smoothing_prior_alpha=DEFAULT_CONFUSION_SMOOTHING_PRIOR,
    )

    assert model["class_order"] == ["copepod", "detritus"]
    assert model["confusion_counts"][0][0] == 1
    assert model["confusion_counts"][0][1] == 0
    assert model["class_example_counts"] == [1, 1]


def test_stopping_rule_requires_min_examples_and_narrow_intervals():
    rows = [
        *[{"original_label": "copepod", "final_label": "copepod"} for _ in range(25)],
        *[{"original_label": "copepod", "final_label": "detritus"} for _ in range(5)],
        *[{"original_label": "detritus", "final_label": "detritus"} for _ in range(25)],
        *[{"original_label": "detritus", "final_label": "copepod"} for _ in range(5)],
    ]
    config = ValidationConfig(
        min_class_examples=20,
        diag_credible_interval_tolerance=0.2,
        offdiag_credible_interval_tolerance=0.2,
    )

    result = check_stopping_criteria(rows, config=config)

    assert result["class_count_criterion_met"] is True
    assert result["credible_interval_criterion_met"] is True
    assert result["all_criteria_met"] is True


def test_stopping_rule_fails_when_intervals_are_wide():
    rows = [
        *[{"original_label": "copepod", "final_label": "copepod"} for _ in range(3)],
        *[{"original_label": "copepod", "final_label": "detritus"} for _ in range(1)],
        *[{"original_label": "detritus", "final_label": "detritus"} for _ in range(3)],
        *[{"original_label": "detritus", "final_label": "copepod"} for _ in range(1)],
    ]
    config = ValidationConfig(
        min_class_examples=4,
        diag_credible_interval_tolerance=0.03,
        offdiag_credible_interval_tolerance=0.03,
    )

    result = check_stopping_criteria(rows, config=config)

    assert result["all_criteria_met"] is False
    assert result["credible_interval_criterion_met"] is False


def test_validation_config_defaults():
    cfg = ValidationConfig()
    assert cfg.min_class_examples == DEFAULT_MIN_CLASS_EXAMPLES
    assert cfg.diag_credible_interval_tolerance == DEFAULT_DIAG_CREDIBLE_INTERVAL_TOLERANCE
    assert cfg.offdiag_credible_interval_tolerance == DEFAULT_OFFDIAG_CREDIBLE_INTERVAL_TOLERANCE
    assert cfg.smoothing_prior_alpha == DEFAULT_CONFUSION_SMOOTHING_PRIOR
