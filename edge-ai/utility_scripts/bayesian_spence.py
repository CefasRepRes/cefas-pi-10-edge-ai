#!/usr/bin/env python3
"""
Unified Bayesian Spence model with a switch for confusion-matrix feedback.

For the original code used in the publication see https://github.com/CefasRepRes/AvoidingConfusion
The original model allows the timeseries to redefine the confusion matrix priors used for your classification uncertainty. In doing so it seemed to prevent the chains from converging, placing wildly inflated counts in minority classes, when given a small validation sample and a proportionately large timeseries. The switch for confusion-matrix feedback (TIMESERIES_FEEDBACK) lets you turn this off meaning the model's understanding of inter-class confusion is fixed to the confusion matrix as is passed in.

"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml_prediction_results_utilities import load_json_payload

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive
except Exception:
    jax = jnp = numpyro = dist = MCMC = NUTS = Predictive = None

LOG = logging.getLogger(__name__)

# The published Section 4 analysis used 20 chains, 5,000 warm-up iterations,
# and 5,000 retained iterations per chain. ``mc_samples`` controls retained draws
# per chain here so callers may deliberately request a smaller development run.
NUMPYRO_MCMC_WARMUP = 50
NCHAINS = 8

# Single-file model switch.
# True  = published Spence joint-feedback model.
# False = validation-only cut-feedback model.
TIMESERIES_FEEDBACK = False

# Validation-only symmetric Dirichlet prior concentration per confusion cell.
# Small positive mass prevents invalid zero Dirichlet parameters.
CONFUSION_PRIOR_CONCENTRATION = 0.01


def extract_validation_error_model(payload: Any) -> Optional[Dict[str, Any]]:
    """Return the classifier-error block exported by the validation workflow."""
    if not isinstance(payload, dict):
        return None
    for key in ("validation_error_model", "classifier_error_model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def extract_confusion_counts(
    error_model: Dict[str, Any], class_order: List[str]
) -> np.ndarray:
    """Return the raw true-by-predicted confusion-count matrix."""
    matrix = error_model.get("confusion_counts")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(
            "Section 4 alignment requires raw 'confusion_counts'; "
            "Dirichlet posterior parameters alone are insufficient."
        )

    counts = np.asarray(matrix)
    n = len(class_order)
    if counts.shape != (n, n):
        raise ValueError("Confusion matrix shape does not match class_order")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Confusion counts must be finite and non-negative")
    rounded = np.rint(counts).astype(np.int32)
    if not np.allclose(counts, rounded):
        raise ValueError("Confusion counts must be integers")
    if np.any(rounded.sum(axis=1) == 0):
        raise ValueError(
            "Every true class needs at least one validation observation for "
            "the Section 4 confusion-matrix likelihood."
        )
    return rounded


def build_validation_alpha(
    confusion_counts: np.ndarray,
    prior_concentration: float = CONFUSION_PRIOR_CONCENTRATION,
) -> np.ndarray:
    """Validation-only posterior parameters for classifier matrix P."""
    if not np.isfinite(prior_concentration) or prior_concentration <= 0:
        raise ValueError("Confusion prior concentration must be finite and > 0")
    alpha = np.asarray(confusion_counts, dtype=np.float64) + prior_concentration
    if np.any(alpha <= 0) or not np.all(np.isfinite(alpha)):
        raise ValueError("Validation Dirichlet parameters must be positive and finite")
    return alpha


def derive_dirichlet_posterior_parameters(
    error_model: Dict[str, Any], class_order: List[str]
) -> List[List[float]]:
    """Compatibility helper for plotting P based on validation data alone."""
    counts = extract_confusion_counts(error_model, class_order)
    return (counts.astype(float) + 1.0).tolist()


def section4_model(
    observed_matrix,
    total_vector,
    validation_matrix,
    n_classes,
    *,
    validation_alpha=None,
    sample_validation_uncertainty: bool = False,
    sample_latent_true_counts: bool = False,
    timeseries_feedback: bool = TIMESERIES_FEEDBACK,
):
    """NumPyro transcription of the Section 4 model with an optional feedback switch."""
    if timeseries_feedback:
        p = numpyro.sample(
            "confusion_matrix",
            dist.Dirichlet(jnp.ones(n_classes, dtype=jnp.float64)).expand((n_classes,)),
        )
        validation_totals = jnp.sum(validation_matrix, axis=1)
        numpyro.sample(
            "validation_confusion_counts",
            dist.Multinomial(total_count=validation_totals, probs=p),
            obs=validation_matrix,
        )
    else:
        fixed_p = jnp.asarray(validation_matrix, dtype=jnp.float64)
        if fixed_p.shape != (n_classes, n_classes):
            raise ValueError("Fixed confusion matrix shape does not match n_classes")
        if sample_validation_uncertainty:
            if validation_alpha is None:
                raise ValueError("validation_alpha is required for uncertainty replay")
            alpha = jnp.asarray(validation_alpha, dtype=jnp.float64)
            if alpha.shape != (n_classes, n_classes):
                raise ValueError("Validation alpha shape does not match n_classes")
            p = numpyro.sample(
                "validation_confusion_matrix_draw",
                dist.Dirichlet(alpha),
            )
        else:
            p = fixed_p
        numpyro.deterministic("confusion_matrix", p)

    alpha_b = numpyro.sample(
        "ecological_mean",
        dist.Dirichlet(jnp.ones(n_classes, dtype=jnp.float64)),
    )
    b_par = numpyro.sample("ecological_concentration", dist.HalfNormal(1000.0))
    ecological_alpha = numpyro.deterministic("ecological_alpha", b_par * alpha_b)
    q = numpyro.sample(
        "prevalence",
        dist.Dirichlet(ecological_alpha).expand((observed_matrix.shape[0],)),
    )

    log_mu = numpyro.sample("log_mu", dist.Normal(0.0, 1000.0))
    mu = numpyro.deterministic("mu", jnp.exp(log_mu))
    k = numpyro.sample("k", dist.HalfNormal(1000.0))
    numpyro.sample(
        "observed_totals",
        dist.NegativeBinomial2(mean=mu, concentration=k),
        obs=total_vector,
    )

    reported_prob = q @ p
    numpyro.sample(
        "observed_counts",
        dist.Multinomial(total_count=total_vector, probs=reported_prob),
        obs=observed_matrix,
    )

    if sample_latent_true_counts:
        numerator = q[:, :, None] * p[None, :, :]
        denominator = reported_prob[:, None, :]
        safe_denominator = jnp.where(denominator > 0, denominator, 1.0)
        allocation = jnp.where(
            denominator > 0, numerator / safe_denominator, 1.0 / n_classes
        )
        numpyro.deterministic(
            "expected_true_counts",
            jnp.einsum("lij,lj->li", allocation, observed_matrix),
        )
        allocation_by_predicted = jnp.moveaxis(allocation, 2, 1)
        latent_by_predicted = numpyro.sample(
            "latent_true_counts_by_predicted",
            dist.Multinomial(
                total_count=observed_matrix, probs=allocation_by_predicted
            ),
        )
        numpyro.deterministic(
            "latent_true_counts", jnp.sum(latent_by_predicted, axis=1)
        )


def _simple(df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    """Fallback used only when the NumPyro/JAX stack is unavailable."""
    rows = []
    for _, source in df.iterrows():
        row = {key: source[key] for key in ("timestamp", "run_name", "blob_name")}
        for class_name in classes:
            value = float(source.get(class_name, 0))
            row.update(
                {
                    f"{class_name}_corrected_mean": value,
                    f"{class_name}_corrected_median": value,
                    f"{class_name}_corrected_lower": max(0.0, value - 0.5),
                    f"{class_name}_corrected_upper": value + 0.5,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def build_uncertainty_dataframe(
    plot_df: pd.DataFrame,
    validation_json: Any,
    *,
    mc_samples: int = 5000,
    mc_seed: int = 42,
    diagnostics_dir: Optional[Path] = None,
    diagnostic_prior_draws: int = 500,
    timeseries_feedback: bool = TIMESERIES_FEEDBACK,
) -> pd.DataFrame:
    """Fit the Bayesian Spence model and summarise corrected latent counts."""
    error_model = extract_validation_error_model(validation_json)
    if error_model is None:
        raise ValueError("Validation JSON has no classifier error model")

    classes = [str(x) for x in error_model.get("class_order", []) if str(x)]
    if not classes:
        raise ValueError("Validation error model has no class_order")
    confusion_counts = extract_confusion_counts(error_model, classes)
    n = len(classes)

    if timeseries_feedback:
        validation_matrix = confusion_counts
        validation_alpha = None
        fixed_confusion_matrix = None
    else:
        validation_alpha = build_validation_alpha(confusion_counts)
        fixed_confusion_matrix = validation_alpha / validation_alpha.sum(
            axis=1, keepdims=True
        )
        validation_matrix = fixed_confusion_matrix

    data_columns = {
        column
        for column in plot_df.columns
        if column not in {"timestamp", "run_name", "blob_name"}
    }
    unknown = sorted(data_columns.difference(classes))
    if unknown:
        raise ValueError(
            "Observed classes are absent from the validation model: "
            + ", ".join(unknown)
        )

    if any(x is None for x in (jax, jnp, numpyro, dist, MCMC, NUTS)):
        LOG.warning(
            "JAX/NumPyro is unavailable; returning uncorrected fallback intervals"
        )
        return _simple(plot_df, classes)

    observed = plot_df.reindex(columns=classes, fill_value=0).to_numpy(float)
    observed_counts = np.rint(observed).astype(np.int32)
    if not np.allclose(observed, observed_counts) or np.any(observed_counts < 0):
        raise ValueError("Observed counts must be non-negative integers")
    totals = observed_counts.sum(axis=1).astype(np.int32)
    if np.any(totals == 0):
        raise ValueError(
            "Section 4's multinomial likelihood requires a positive total at "
            "every retained location/timestamp"
        )

    numpyro.set_host_device_count(NCHAINS)
    kernel = NUTS(
        functools.partial(
            section4_model,
            n_classes=n,
            timeseries_feedback=timeseries_feedback,
        ),
        target_accept_prob=0.95,
        max_tree_depth=12,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=NUMPYRO_MCMC_WARMUP,
        num_samples=max(1, int(mc_samples)),
        num_chains=NCHAINS,
        chain_method="parallel",
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(mc_seed),
        observed_matrix=jnp.asarray(observed_counts),
        total_vector=jnp.asarray(totals),
        validation_matrix=jnp.asarray(validation_matrix),
        validation_alpha=jnp.asarray(validation_alpha) if validation_alpha is not None else None,
    )

    posterior_by_chain = {
        k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    posterior = mcmc.get_samples(group_by_chain=False)
    prior_samples = None
    if diagnostics_dir is not None:
        if timeseries_feedback:
            prior_samples = Predictive(
                functools.partial(
                    section4_model,
                    n_classes=n,
                    timeseries_feedback=timeseries_feedback,
                ),
                num_samples=max(20, int(diagnostic_prior_draws)),
                return_sites=["confusion_matrix", "ecological_mean", "ecological_concentration", "mu", "k"],
            )(
                jax.random.PRNGKey(mc_seed + 10000),
                observed_matrix=jnp.asarray(observed_counts),
                total_vector=jnp.asarray(totals),
                validation_matrix=jnp.asarray(confusion_counts),
            )
        else:
            prior_samples = Predictive(
                functools.partial(
                    section4_model,
                    n_classes=n,
                    timeseries_feedback=timeseries_feedback,
                ),
                num_samples=max(20, int(diagnostic_prior_draws)),
                return_sites=["confusion_matrix", "ecological_mean", "ecological_concentration", "mu", "k"],
            )(
                jax.random.PRNGKey(mc_seed + 10000),
                observed_matrix=jnp.asarray(observed_counts),
                total_vector=jnp.asarray(totals),
                validation_matrix=jnp.asarray(fixed_confusion_matrix),
                validation_alpha=jnp.asarray(validation_alpha),
            )

    posterior = mcmc.get_samples(group_by_chain=False)
    n_replay = max(1, min(50 if timeseries_feedback else 250, int(mc_samples)))
    posterior_size = len(next(iter(posterior.values())))
    if posterior_size > n_replay:
        step = max(1, posterior_size // n_replay)
        idx = np.arange(0, posterior_size, step)[:n_replay]
        posterior = {k: np.asarray(v)[idx] for k, v in posterior.items()}
    else:
        posterior = {k: np.asarray(v) for k, v in posterior.items()}

    if timeseries_feedback:
        generated_quantities = Predictive(
            functools.partial(
                section4_model,
                n_classes=n,
                timeseries_feedback=True,
                sample_latent_true_counts=True,
            ),
            posterior_samples=posterior,
            return_sites=["latent_true_counts", "expected_true_counts"],
        )(
            jax.random.PRNGKey(mc_seed + 1),
            observed_matrix=jnp.asarray(observed_counts),
            total_vector=jnp.asarray(totals),
            validation_matrix=jnp.asarray(confusion_counts),
        )
        p = np.asarray(posterior["confusion_matrix"])
        condition_number = float(np.mean([np.linalg.cond(draw.T) for draw in p]))
    else:
        generated_quantities = Predictive(
            functools.partial(
                section4_model,
                n_classes=n,
                timeseries_feedback=False,
                sample_validation_uncertainty=True,
                sample_latent_true_counts=True,
            ),
            posterior_samples=posterior,
            return_sites=[
                "validation_confusion_matrix_draw",
                "latent_true_counts",
                "expected_true_counts",
            ],
        )(
            jax.random.PRNGKey(mc_seed + 1),
            observed_matrix=jnp.asarray(observed_counts),
            total_vector=jnp.asarray(totals),
            validation_matrix=jnp.asarray(fixed_confusion_matrix),
            validation_alpha=jnp.asarray(validation_alpha),
        )
        replayed_p = np.asarray(generated_quantities["validation_confusion_matrix_draw"])
        p_diag = replayed_p[:, np.arange(n), np.arange(n)]
        LOG.warning(
            "Validation-only P replay: shape=%s, "
            "Noctiluca diagonal min=%.6f mean=%.6f max=%.6f sd=%.6f",
            replayed_p.shape,
            float(p_diag[:, 0].min()),
            float(p_diag[:, 0].mean()),
            float(p_diag[:, 0].max()),
            float(p_diag[:, 0].std()),
        )
        condition_number = float(
            np.mean([np.linalg.cond(draw.T) for draw in replayed_p])
        )

    latent_true_counts = np.asarray(generated_quantities["latent_true_counts"])
    expected_true_counts = np.asarray(generated_quantities["expected_true_counts"])

    if timeseries_feedback:
        corrected_lower = np.percentile(expected_true_counts, 5, axis=0)
        corrected_upper = np.percentile(expected_true_counts, 95, axis=0)
        latent_true_count_lower = np.percentile(latent_true_counts, 5, axis=0)
        latent_true_count_upper = np.percentile(latent_true_counts, 95, axis=0)
        corrected_median = np.median(expected_true_counts, axis=0)
    else:
        corrected_lower = np.percentile(latent_true_counts, 1, axis=0)
        corrected_upper = np.percentile(latent_true_counts, 99, axis=0)
        corrected_median = np.median(expected_true_counts, axis=0)
        latent_true_count_lower = np.percentile(latent_true_counts, 1, axis=0)
        latent_true_count_upper = np.percentile(latent_true_counts, 99, axis=0)

    totals_f = totals.astype(float)
    if not np.array_equal(
        latent_true_counts.sum(axis=2),
        np.broadcast_to(totals, latent_true_counts.shape[:2]),
    ):
        raise RuntimeError("Latent true counts did not preserve observed totals")
    if not np.allclose(
        expected_true_counts.sum(axis=2), totals_f[None, :], rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("Expected true counts did not preserve observed totals")

    expected_composition = expected_true_counts / totals_f[None, :, None]
    corrected_mean = expected_true_counts.mean(axis=0)
    latent_true_count_median = np.median(latent_true_counts, axis=0)

    ecological_alpha = np.asarray(posterior["ecological_alpha"])
    ecological_mean = np.asarray(posterior["ecological_mean"])
    ecological_concentration = np.asarray(posterior["ecological_concentration"])
    mu = np.asarray(posterior["mu"])
    k = np.asarray(posterior["k"])

    if diagnostics_dir is not None and prior_samples is not None:
        try:
            from bayesian_diagnostics import write_bayesian_diagnostics

            write_bayesian_diagnostics(
                output_dir=Path(diagnostics_dir),
                mcmc=mcmc,
                prior_samples={k: np.asarray(v) for k, v in prior_samples.items()},
                posterior_by_chain=posterior_by_chain,
                observed_counts=observed_counts,
                validation_counts=confusion_counts,
                classes=classes,
                expected_true_counts=expected_true_counts,
                labels=[str(x) for x in plot_df["timestamp"]],
                seed=mc_seed,
            )
        except Exception as exc:
            LOG.exception("Could not write Bayesian diagnostics: %s", exc)

    rows = []
    for location, (_, source) in enumerate(plot_df.iterrows()):
        row = {
            "timestamp": source["timestamp"],
            "run_name": source["run_name"],
            "blob_name": source["blob_name"],
            "observed_total": float(totals[location]),
            "mean_sampled_condition_number": condition_number,
            "total_process_mu_mean": float(mu.mean()),
            "total_process_mu_lower": float(np.percentile(mu, 5)),
            "total_process_mu_upper": float(np.percentile(mu, 95)),
            "total_process_k_mean": float(k.mean()),
            "total_process_k_lower": float(np.percentile(k, 5)),
            "total_process_k_upper": float(np.percentile(k, 95)),
            "ecological_concentration_mean": float(ecological_concentration.mean()),
            "ecological_concentration_lower": float(
                np.percentile(ecological_concentration, 5)
            ),
            "ecological_concentration_upper": float(
                np.percentile(ecological_concentration, 95)
            ),
        }
        for i, class_name in enumerate(classes):
            row.update(
                {
                    f"{class_name}_shared_alpha_mean": float(
                        ecological_alpha[:, i].mean()
                    ),
                    f"{class_name}_shared_alpha_lower": float(
                        np.percentile(ecological_alpha[:, i], 5)
                    ),
                    f"{class_name}_shared_alpha_upper": float(
                        np.percentile(ecological_alpha[:, i], 95)
                    ),
                    f"{class_name}_ecological_mean": float(
                        ecological_mean[:, i].mean()
                    ),
                    f"{class_name}_corrected_mean": float(
                        corrected_mean[location, i]
                    ),
                    f"{class_name}_corrected_median": float(
                        corrected_median[location, i]
                    ),
                    f"{class_name}_corrected_lower": float(
                        corrected_lower[location, i]
                    ),
                    f"{class_name}_corrected_upper": float(
                        corrected_upper[location, i]
                    ),
                    f"{class_name}_boundary_frequency": float(
                        np.mean(latent_true_counts[:, location, i] == 0)
                    ),
                    f"{class_name}_expected_composition": float(
                        expected_composition[:, location, i].mean()
                    ),
                    f"{class_name}_composition_median": float(
                        np.median(expected_composition[:, location, i])
                    ),
                    f"{class_name}_composition_lower": float(
                        np.percentile(expected_composition[:, location, i], 5)
                    ),
                    f"{class_name}_composition_upper": float(
                        np.percentile(expected_composition[:, location, i], 95)
                    ),
                    f"{class_name}_expected_true_count": float(
                        corrected_mean[location, i]
                    ),
                    f"{class_name}_latent_true_count_median": float(
                        latent_true_count_median[location, i]
                    ),
                    f"{class_name}_latent_true_count_lower": float(
                        latent_true_count_lower[location, i]
                    ),
                    f"{class_name}_latent_true_count_upper": float(
                        latent_true_count_upper[location, i]
                    ),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows, index=plot_df.index)


def load_validation_uncertainty_dataframe(
    plot_df,
    validation_source,
    *,
    mc_samples,
    mc_seed,
    blob_service_client=None,
    diagnostics_dir: Optional[Path] = None,
    diagnostic_prior_draws: int = 500,
    timeseries_feedback: bool = TIMESERIES_FEEDBACK,
):
    """Load validation JSON, fit the Section 4 model, and return summaries."""
    if not validation_source:
        return None
    try:
        payload = load_json_payload(
            validation_source, blob_service_client=blob_service_client
        )
        return build_uncertainty_dataframe(
            plot_df,
            payload,
            mc_samples=mc_samples,
            mc_seed=mc_seed,
            diagnostics_dir=diagnostics_dir,
            diagnostic_prior_draws=diagnostic_prior_draws,
            timeseries_feedback=timeseries_feedback,
        )
    except Exception as exc:
        LOG.exception("Skipping validation uncertainty: %s", exc)
        return None
