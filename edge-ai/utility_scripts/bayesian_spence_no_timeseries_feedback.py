#!/usr/bin/env python3
"""Compatibility wrapper for the cut-feedback Bayesian Spence implementation."""

from bayesian_spence import (  # noqa: F401,F403
    CONFUSION_PRIOR_CONCENTRATION,
    LOG,
    NCHAINS,
    NUMPYRO_MCMC_WARMUP,
    build_validation_alpha,
    build_uncertainty_dataframe as _build_uncertainty_dataframe,
    derive_dirichlet_posterior_parameters,
    extract_confusion_counts,
    extract_validation_error_model,
    load_validation_uncertainty_dataframe as _load_validation_uncertainty_dataframe,
    section4_model,
)


def build_uncertainty_dataframe(*args, **kwargs):
    kwargs.setdefault("timeseries_feedback", False)
    return _build_uncertainty_dataframe(*args, **kwargs)


def load_validation_uncertainty_dataframe(*args, **kwargs):
    kwargs.setdefault("timeseries_feedback", False)
    return _load_validation_uncertainty_dataframe(*args, **kwargs)
