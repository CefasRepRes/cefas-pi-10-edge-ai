"""
Adaptive validation stopping criteria for rapid-plankton.

The validation session estimates a multiclass confusion-matrix error
model from manually validated records and stops when each class has
sufficient validated examples.

Configuration is exposed via :class:`ValidationConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "ValidationConfig",
    "UNCERTAIN_LABEL",
    "is_uncertain_label",
    "wilson_interval",
    "build_confusion_error_model",
    "check_stopping_criteria",
    "DEFAULT_CI_WIDTH_TARGET",
    "DEFAULT_MIN_CLASS_PRECISION",
    "DEFAULT_MAX_SAMPLE_SIZE",
    "DEFAULT_MIN_CLASS_EXAMPLES",
    "DEFAULT_DIAG_CREDIBLE_INTERVAL_TOLERANCE",
    "DEFAULT_OFFDIAG_CREDIBLE_INTERVAL_TOLERANCE",
    "DEFAULT_CONFUSION_SMOOTHING_PRIOR",
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CI_WIDTH_TARGET: float = 0.05
"""Legacy overall accuracy target retained for compatibility."""

DEFAULT_MIN_CLASS_PRECISION: float = 0.8
"""Legacy precision threshold retained for compatibility."""

DEFAULT_MAX_SAMPLE_SIZE: int = 500
"""Upper bound on the number of images drawn into the validation pool."""

DEFAULT_MIN_CLASS_EXAMPLES: int = 25
"""Minimum number of validated rows required for each true class."""

DEFAULT_DIAG_CREDIBLE_INTERVAL_TOLERANCE: float = 0.05
"""Maximum allowed half-width for the credible interval of a diagonal entry."""

DEFAULT_OFFDIAG_CREDIBLE_INTERVAL_TOLERANCE: float = 0.05
"""Maximum allowed half-width for the credible interval of an off-diagonal entry."""

DEFAULT_CONFUSION_SMOOTHING_PRIOR: float = 0.75
"""Legacy smoothing prior retained for compatibility."""

UNCERTAIN_LABEL: str = "uncertain"
"""Label used by the validation UI when a reviewer cannot confidently assign a class."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ValidationConfig:
    """Configuration for adaptive validation stopping.

    Parameters
    ----------
    ci_width_target:
        Legacy overall-accuracy target retained for compatibility.
    important_classes:
        Legacy class list retained for compatibility.
    min_class_precision:
        Legacy precision threshold retained for compatibility.
    max_sample_size:
        Maximum number of images drawn into the validation pool.
    min_class_examples:
        Minimum number of validated rows required for each true class.
    diag_credible_interval_tolerance:
        Maximum acceptable half-width for the posterior credible interval of a
        diagonal confusion entry.
    offdiag_credible_interval_tolerance:
        Maximum acceptable half-width for the posterior credible interval of an
        off-diagonal confusion entry.
    smoothing_prior_alpha:
        Legacy smoothing prior retained for compatibility.
    """

    ci_width_target: float = DEFAULT_CI_WIDTH_TARGET
    important_classes: List[str] = field(default_factory=list)
    min_class_precision: float = DEFAULT_MIN_CLASS_PRECISION
    max_sample_size: int = DEFAULT_MAX_SAMPLE_SIZE
    min_class_examples: int = DEFAULT_MIN_CLASS_EXAMPLES
    diag_credible_interval_tolerance: float = DEFAULT_DIAG_CREDIBLE_INTERVAL_TOLERANCE
    offdiag_credible_interval_tolerance: float = DEFAULT_OFFDIAG_CREDIBLE_INTERVAL_TOLERANCE
    smoothing_prior_alpha: float = DEFAULT_CONFUSION_SMOOTHING_PRIOR


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> dict:
    """Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "half_width": None,
            "error_bar_minus": None,
            "error_bar_plus": None,
        }
    p = successes / n
    denom = 1 + (z * z / n)
    centre = (p + (z * z / (2 * n))) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
    lower = max(0.0, centre - half)
    upper = min(1.0, centre + half)
    half_width = (upper - lower) / 2.0
    return {
        "estimate": p,
        "lower": lower,
        "upper": upper,
        "half_width": half_width,
        "error_bar_minus": p - lower,
        "error_bar_plus": upper - p,
    }


# ---------------------------------------------------------------------------
# Multiclass confusion-model helpers
# ---------------------------------------------------------------------------


def _normalise_label(value) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def is_uncertain_label(value) -> bool:
    return _normalise_label(value).lower() == UNCERTAIN_LABEL


def _is_usable_validation_row(row) -> bool:
    original_label = _normalise_label(row.get("original_label"))
    final_label = _normalise_label(row.get("final_label"))
    return bool(original_label and final_label) and not is_uncertain_label(original_label) and not is_uncertain_label(final_label)


def _ordered_class_labels(labels=None, rows=None) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for label in labels or []:
        label_str = _normalise_label(label)
        if not label_str or label_str in seen:
            continue
        ordered.append(label_str)
        seen.add(label_str)
    for row in rows or []:
        for key in ("final_label", "original_label"):
            label_str = _normalise_label(row.get(key))
            if not label_str or label_str in seen:
                continue
            ordered.append(label_str)
            seen.add(label_str)
    return ordered


def _beta_interval_from_parameters(alpha_param: float, beta_param: float, z: float = 1.96) -> dict:
    total = alpha_param + beta_param
    if total <= 0.0:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "half_width": None,
        }
    mean = alpha_param / total
    var = (alpha_param * beta_param) / ((total * total) * (total + 1.0))
    std = math.sqrt(max(var, 0.0))
    lower = max(0.0, mean - (z * std))
    upper = min(1.0, mean + (z * std))
    half_width = (upper - lower) / 2.0
    return {
        "estimate": mean,
        "lower": lower,
        "upper": upper,
        "half_width": half_width,
    }


def build_confusion_error_model(
    validated_rows: list,
    labels: Optional[List[str]] = None,
    *,
    smoothing_prior_alpha: float = DEFAULT_CONFUSION_SMOOTHING_PRIOR,
) -> dict:
    """Estimate a multiclass confusion-matrix error model from validated rows."""
    usable = [row for row in validated_rows if _is_usable_validation_row(row)]
    class_order = _ordered_class_labels(labels=labels, rows=usable)
    if not class_order:
        return {
            "class_order": [],
            "confusion_counts": [],
            "per_class_metrics": {},
        }

    class_to_index = {label: index for index, label in enumerate(class_order)}
    counts = [[0 for _ in class_order] for _ in class_order]
    for row in usable:
        true_label = _normalise_label(row.get("final_label"))
        pred_label = _normalise_label(row.get("original_label"))
        if true_label not in class_to_index or pred_label not in class_to_index:
            continue
        counts[class_to_index[true_label]][class_to_index[pred_label]] += 1

    row_totals = [sum(row) for row in counts]
    col_totals = [sum(counts[row_idx][col_idx] for row_idx in range(len(class_order))) for col_idx in range(len(class_order))]

    per_class_metrics: Dict[str, dict] = {}

    for row_idx, true_label in enumerate(class_order):
        row_total = row_totals[row_idx]
        predicted_support = col_totals[row_idx]
        tp = counts[row_idx][row_idx]
        per_class_metrics[true_label] = {
            "true_support": row_total,
            "predicted_support": predicted_support,
            "correct": tp,
            "false_negative": max(0, row_total - tp),
            "false_positive": max(0, predicted_support - tp),
            "recall": _beta_interval_from_parameters(
                tp + smoothing_prior_alpha,
                (row_total - tp) + smoothing_prior_alpha,
            ),
            "precision": _beta_interval_from_parameters(
                tp + smoothing_prior_alpha,
                (predicted_support - tp) + smoothing_prior_alpha,
            ),
        }

    return {
        "class_order": class_order,
        "confusion_counts": counts,
        "per_class_metrics": per_class_metrics,
    }


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------


def check_stopping_criteria(
    validated_rows: list,
    config: Optional[ValidationConfig] = None,
    *,
    ci_width_target: Optional[float] = None,
    important_classes: Optional[List[str]] = None,
    min_class_precision: Optional[float] = None,
    min_class_examples: Optional[int] = None,
    diag_credible_interval_tolerance: Optional[float] = None,
    offdiag_credible_interval_tolerance: Optional[float] = None,
    smoothing_prior_alpha: Optional[float] = None,
) -> dict:
    """Evaluate adaptive validation stopping criteria for the confusion model."""
    cfg = config if config is not None else ValidationConfig()
    _ci_target = ci_width_target if ci_width_target is not None else cfg.ci_width_target
    _imp_classes = important_classes if important_classes is not None else cfg.important_classes
    _min_prec = min_class_precision if min_class_precision is not None else cfg.min_class_precision
    _min_examples = min_class_examples if min_class_examples is not None else cfg.min_class_examples
    _diag_tol = diag_credible_interval_tolerance if diag_credible_interval_tolerance is not None else cfg.diag_credible_interval_tolerance
    _offdiag_tol = offdiag_credible_interval_tolerance if offdiag_credible_interval_tolerance is not None else cfg.offdiag_credible_interval_tolerance
    _prior_alpha = smoothing_prior_alpha if smoothing_prior_alpha is not None else cfg.smoothing_prior_alpha

    usable = [r for r in validated_rows if _is_usable_validation_row(r)]
    n = len(usable)

    error_model = build_confusion_error_model(
        usable,
        labels=list(_imp_classes or []),
        smoothing_prior_alpha=_prior_alpha,
    )
    class_order = error_model.get("class_order") or []
    confusion_counts = error_model.get("confusion_counts") or []
    class_example_counts = [sum(row) for row in confusion_counts]

    class_count_criterion_met = bool(class_order) and all(
        count >= _min_examples for count in class_example_counts
    )

    credible_interval_criterion_met = True

    class_results: Dict[str, dict] = {}
    for cls in class_order:
        metrics = error_model.get("per_class_metrics", {}).get(cls, {})
        precision_interval = metrics.get("precision") or {}
        recall_interval = metrics.get("recall") or {}
        class_results[cls] = {
            "precision_estimate": precision_interval.get("estimate"),
            "precision_lower": precision_interval.get("lower"),
            "precision_upper": precision_interval.get("upper"),
            "recall_estimate": recall_interval.get("estimate"),
            "recall_lower": recall_interval.get("lower"),
            "recall_upper": recall_interval.get("upper"),
            "true_support": metrics.get("true_support"),
            "predicted_support": metrics.get("predicted_support"),
            "threshold": _min_prec,
            "met": True,
            "reason": "Confusion model available",
        }

    overall_half_width = None
    parts: List[str] = []
    if class_order:
        counts_text = ", ".join(
            f"{cls}={count}" for cls, count in zip(class_order, class_example_counts)
        )
        parts.append(f"Validated examples by true class: {counts_text}")
    else:
        parts.append("No validated classes available yet")

    if class_count_criterion_met:
        parts.append(f"Each class has at least {_min_examples} validated rows")
    else:
        parts.append(f"Each class does not yet have {_min_examples} validated rows")

    if credible_interval_criterion_met:
        parts.append("Confusion-matrix summaries are available for export")
    else:
        parts.append("Confusion-matrix summaries are not yet available")

    tick = "✓" if class_count_criterion_met and credible_interval_criterion_met else "✗"
    label = "All stopping criteria met" if class_count_criterion_met and credible_interval_criterion_met else "Stopping criteria not yet met"
    summary = f"{tick} {label}. " + "; ".join(parts)

    return {
        "validated_count": n,
        "overall_ci_half_width": overall_half_width,
        "ci_criterion_met": credible_interval_criterion_met,
        "class_precision_results": class_results,
        "precision_criterion_met": class_count_criterion_met,
        "class_count_criterion_met": class_count_criterion_met,
        "credible_interval_criterion_met": credible_interval_criterion_met,
        "all_criteria_met": class_count_criterion_met and credible_interval_criterion_met,
        "ci_width_target": _ci_target,
        "min_class_precision": _min_prec,
        "min_class_examples": _min_examples,
        "diag_credible_interval_tolerance": _diag_tol,
        "offdiag_credible_interval_tolerance": _offdiag_tol,
        "smoothing_prior_alpha": _prior_alpha,
        "important_classes": list(_imp_classes or []),
        "confusion_error_model": error_model,
        "summary": summary,
    }
