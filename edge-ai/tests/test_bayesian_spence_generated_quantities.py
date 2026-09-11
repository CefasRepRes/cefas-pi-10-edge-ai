"""End-to-end check that latent true counts are produced as a genuine
posterior "generated quantities" step and that all ``*_corrected_*``
summaries come from expected counts only.

Unlike ``test_bayesian_spence.py``, this test intentionally exercises the
real JAX/NumPyro stack (both are pinned in ``edge-ai/requirements.txt``)
rather than lightweight stand-ins, so it loads ``bayesian_spence`` under a
distinct module name to avoid colliding with the stubbed import there.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ``test_bayesian_spence.py`` installs lightweight fake modules under the
# names "jax", "jax.numpy", "numpyro", "numpyro.distributions" and
# "numpyro.infer" in ``sys.modules`` so that importing ``bayesian_spence``
# does not require the real (heavy) JAX/NumPyro stack. If that test module
# ran earlier in this test session those fakes would still be cached in
# ``sys.modules`` here, and would shadow the real packages this module
# needs. Evict them first so the imports below resolve to the genuine
# installed packages.
for _name in (
    "jax",
    "jax.numpy",
    "numpyro",
    "numpyro.distributions",
    "numpyro.infer",
):
    sys.modules.pop(_name, None)

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")

_UTILITY_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "utility_scripts"
sys.path.insert(0, str(_UTILITY_SCRIPTS_DIR))
_MODULE_PATH = _UTILITY_SCRIPTS_DIR / "bayesian_spence.py"
_SPEC = importlib.util.spec_from_file_location(
    "bayesian_spence_generated_quantities_test", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load test module from {_MODULE_PATH}")
MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def uncertainty_df():
    plot_df = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "run_name": "run-a",
                "blob_name": "a.json",
                "fish": 8,
                "detritus": 2,
            },
            {
                "timestamp": "2024-01-02",
                "run_name": "run-b",
                "blob_name": "b.json",
                "fish": 3,
                "detritus": 5,
            },
        ]
    )
    validation_payload = {
        "classifier_error_model": {
            "class_order": ["fish", "detritus"],
            "confusion_counts": [[8, 2], [1, 9]],
        }
    }
    return MODULE.build_uncertainty_dataframe(
        plot_df, validation_payload, mc_samples=50, mc_seed=7
    )


def test_corrected_summaries_are_internally_consistent(uncertainty_df):
    # mean/median/lower/upper must all come from the same (expected-count)
    # posterior quantity, so lower <= median <= upper always holds and the
    # mean is non-negative, for every class.
    for class_name in ("fish", "detritus"):
        lower = uncertainty_df[f"{class_name}_corrected_lower"]
        median = uncertainty_df[f"{class_name}_corrected_median"]
        upper = uncertainty_df[f"{class_name}_corrected_upper"]
        mean = uncertainty_df[f"{class_name}_corrected_mean"]

        assert (lower <= median).all()
        assert (median <= upper).all()
        assert (mean >= 0).all()
        # expected_true_count is documented as an alias of corrected_mean.
        assert np.allclose(
            uncertainty_df[f"{class_name}_expected_true_count"], mean
        )


def test_latent_true_counts_are_a_distinct_sampled_quantity(uncertainty_df):
    # latent_true_count_* summarise the genuinely-sampled discrete latent
    # counts, which is a different posterior quantity from the
    # expected-count-based corrected_* columns, so both must be present
    # and internally ordered.
    for class_name in ("fish", "detritus"):
        lower = uncertainty_df[f"{class_name}_latent_true_count_lower"]
        median = uncertainty_df[f"{class_name}_latent_true_count_median"]
        upper = uncertainty_df[f"{class_name}_latent_true_count_upper"]

        assert (lower <= median).all()
        assert (median <= upper).all()
        assert f"{class_name}_boundary_frequency" in uncertainty_df.columns
        assert (uncertainty_df[f"{class_name}_boundary_frequency"] >= 0).all()
        assert (uncertainty_df[f"{class_name}_boundary_frequency"] <= 1).all()


def test_section4_model_generated_quantities_preserve_observed_totals():
    """Replaying posterior draws through ``section4_model`` with
    ``sample_latent_true_counts=True`` must reproduce each location's
    observed total exactly for the sampled latent counts, and up to
    floating-point tolerance for the expected counts."""
    import functools

    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, Predictive

    n_classes = 2
    observed_counts = np.array([[8, 2], [3, 5]], dtype=np.int32)
    totals = observed_counts.sum(axis=1)
    validation_matrix = np.array([[8, 2], [1, 9]], dtype=np.int32)

    kernel = NUTS(functools.partial(MODULE.section4_model, n_classes=n_classes))
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(0),
        observed_matrix=jnp.asarray(observed_counts),
        total_vector=jnp.asarray(totals),
        validation_matrix=jnp.asarray(validation_matrix),
    )
    posterior = mcmc.get_samples()

    generated = Predictive(
        functools.partial(
            MODULE.section4_model,
            n_classes=n_classes,
            sample_latent_true_counts=True,
        ),
        posterior_samples=posterior,
        return_sites=["latent_true_counts", "expected_true_counts"],
    )(
        jax.random.PRNGKey(1),
        observed_matrix=jnp.asarray(observed_counts),
        total_vector=jnp.asarray(totals),
        validation_matrix=jnp.asarray(validation_matrix),
    )

    latent = np.asarray(generated["latent_true_counts"])
    expected = np.asarray(generated["expected_true_counts"])

    assert latent.shape == (10, 2, n_classes)
    assert np.array_equal(
        latent.sum(axis=2), np.broadcast_to(totals, latent.shape[:2])
    )
    assert np.allclose(expected.sum(axis=2), totals[None, :])
