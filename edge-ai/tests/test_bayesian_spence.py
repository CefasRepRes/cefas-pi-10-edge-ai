import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas  # noqa: F401  (real dependency; ensures it is importable)
import pytest

jax_stub = types.ModuleType("jax")
jax_stub.config = types.SimpleNamespace(update=lambda *args, **kwargs: None)
jax_stub.random = types.SimpleNamespace(PRNGKey=lambda seed: seed)
sys.modules.setdefault("jax", jax_stub)

jax_numpy_stub = types.ModuleType("jax.numpy")
jax_numpy_stub.float64 = "float64"
jax_numpy_stub.int32 = "int32"
jax_numpy_stub.asarray = lambda value, dtype=None: value
jax_numpy_stub.ones = lambda size, dtype=None: [1.0] * size
jax_numpy_stub.exp = lambda value: value
jax_numpy_stub.log = lambda value: value
jax_numpy_stub.sum = lambda value, axis=None: value
jax_numpy_stub.where = lambda condition, x, y: x
jax_numpy_stub.moveaxis = lambda value, source, destination: value
jax_numpy_stub.einsum = lambda subscripts, *operands: operands[-1]
sys.modules.setdefault("jax.numpy", jax_numpy_stub)


numpyro_stub = types.ModuleType("numpyro")
numpyro_stub.sample = lambda *args, **kwargs: None
numpyro_stub.deterministic = lambda *args, **kwargs: None
numpyro_stub.set_host_device_count = lambda *args, **kwargs: None
sys.modules.setdefault("numpyro", numpyro_stub)

numpyro_distributions_stub = types.ModuleType("numpyro.distributions")
numpyro_distributions_stub.Dirichlet = lambda *args, **kwargs: None
numpyro_distributions_stub.Normal = lambda *args, **kwargs: None
numpyro_distributions_stub.HalfNormal = lambda *args, **kwargs: None
numpyro_distributions_stub.NegativeBinomial2 = lambda *args, **kwargs: None
numpyro_distributions_stub.Multinomial = lambda *args, **kwargs: None
sys.modules.setdefault("numpyro.distributions", numpyro_distributions_stub)

numpyro_infer_stub = types.ModuleType("numpyro.infer")
numpyro_infer_stub.MCMC = object
numpyro_infer_stub.NUTS = object
numpyro_infer_stub.Predictive = object
sys.modules.setdefault("numpyro.infer", numpyro_infer_stub)

_UTILITY_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "utility_scripts"
sys.path.insert(0, str(_UTILITY_SCRIPTS_DIR))

MODULE_PATH = _UTILITY_SCRIPTS_DIR / "bayesian_spence.py"
SPEC = importlib.util.spec_from_file_location("bayesian_spence", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load test module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extract_confusion_counts_returns_raw_true_by_predicted_matrix():
    counts = MODULE.extract_confusion_counts(
        {"confusion_counts": [[4, 1], [1, 4]]}, ["fish", "detritus"]
    )
    assert counts.tolist() == [[4, 1], [1, 4]]


def test_extract_confusion_counts_rejects_missing_confusion_counts():
    with pytest.raises(ValueError, match="raw 'confusion_counts'"):
        MODULE.extract_confusion_counts({}, ["fish", "detritus"])


def test_extract_confusion_counts_rejects_mismatched_shape():
    with pytest.raises(ValueError, match="shape"):
        MODULE.extract_confusion_counts(
            {"confusion_counts": [[4, 1]]}, ["fish", "detritus"]
        )


def test_extract_confusion_counts_rejects_empty_true_class_row():
    with pytest.raises(ValueError, match="at least one validation observation"):
        MODULE.extract_confusion_counts(
            {"confusion_counts": [[0, 0], [1, 4]]}, ["fish", "detritus"]
        )


def test_derive_dirichlet_posterior_parameters_from_confusion_counts():
    params = MODULE.derive_dirichlet_posterior_parameters(
        {"confusion_counts": [[4, 1], [1, 4]]}, ["fish", "detritus"]
    )

    # Section 4's published prior is P_i ~ Dirichlet(1), so the validation-only
    # posterior parameters are simply C + 1.
    assert params == [[5.0, 2.0], [2.0, 5.0]]


class _Value:
    """Minimal stand-in for a traced JAX array: supports just enough
    arithmetic (``*``, ``@``, indexing and comparison) for the model
    function to run to completion without performing any real numeric
    computation."""

    def __mul__(self, other):
        return _Value()

    __rmul__ = __mul__

    def __matmul__(self, other):
        return _Value()

    def __truediv__(self, other):
        return _Value()

    def __getitem__(self, item):
        return _Value()

    def __gt__(self, other):
        return _Value()

    def sum(self, axis=None):
        return _Value()


def _make_recording_distribution():
    class _Distribution:
        calls = []

        def __init__(self, *args, **kwargs):
            type(self).calls.append({"args": args, "kwargs": kwargs})

        def expand(self, *args, **kwargs):
            return self

    return _Distribution


def test_section4_model_uses_published_section4_stan_priors(monkeypatch):
    """The Section 4 Stan model (``plankton_model.stan``) specifies:

        P[m] ~ dirichlet(rep_vector(1.0, M))
        alpha_b: implicit uniform simplex prior, i.e. Dirichlet(1)
        log_mu ~ normal(0, 1000)
        k_nb ~ normal(0, 1000)   (constrained <lower=0>, i.e. half-normal)
        b_par ~ normal(0, 1000)  (constrained <lower=0>, i.e. half-normal)

    This drives ``bayesian_spence.section4_model`` directly (with recording
    stand-ins for the NumPyro distribution constructors) and asserts on the
    actual runtime arguments passed to ``dist.Dirichlet``/``Normal``/
    ``HalfNormal``, rather than on the module's source text.
    """
    dirichlet_cls = _make_recording_distribution()
    normal_cls = _make_recording_distribution()
    half_normal_cls = _make_recording_distribution()
    other_cls = _make_recording_distribution()

    monkeypatch.setattr(MODULE.dist, "Dirichlet", dirichlet_cls)
    monkeypatch.setattr(MODULE.dist, "Normal", normal_cls)
    monkeypatch.setattr(MODULE.dist, "HalfNormal", half_normal_cls)
    monkeypatch.setattr(MODULE.dist, "Multinomial", other_cls)
    monkeypatch.setattr(MODULE.dist, "NegativeBinomial2", other_cls)
    monkeypatch.setattr(
        MODULE.numpyro,
        "sample",
        lambda name, distribution, obs=None: obs if obs is not None else _Value(),
    )
    monkeypatch.setattr(
        MODULE.numpyro, "deterministic", lambda name, value: value
    )

    n_classes = 3
    observed_matrix = np.zeros((5, n_classes))
    total_vector = np.zeros(5)
    validation_matrix = np.zeros((n_classes, n_classes))

    MODULE.section4_model(observed_matrix, total_vector, validation_matrix, n_classes)

    dirichlet_concentrations = [call["args"][0] for call in dirichlet_cls.calls]
    # confusion_matrix and ecological_mean both use Dirichlet(1); the third
    # Dirichlet call (prevalence) uses the derived ecological_alpha, which is
    # a _Value() stand-in rather than a literal concentration vector.
    assert dirichlet_concentrations[0] == [1.0] * n_classes  # confusion_matrix
    assert dirichlet_concentrations[1] == [1.0] * n_classes  # ecological_mean

    half_normal_scales = [call["args"][0] for call in half_normal_cls.calls]
    assert half_normal_scales == [1000.0, 1000.0]  # ecological_concentration, k

    normal_args = [call["args"] for call in normal_cls.calls]
    assert normal_args == [(0.0, 1000.0)]  # log_mu


def test_section4_model_only_samples_latent_true_counts_when_requested(monkeypatch):
    """Latent true counts are a "generated quantities" style addition: they
    must only be produced when ``sample_latent_true_counts=True`` (the mode
    used for ``Predictive`` replay after fitting), and never as a side
    effect of the default call used for NUTS fitting.
    """
    sample_calls = []
    deterministic_calls = []

    monkeypatch.setattr(MODULE.dist, "Dirichlet", _make_recording_distribution())
    monkeypatch.setattr(MODULE.dist, "Normal", _make_recording_distribution())
    monkeypatch.setattr(MODULE.dist, "HalfNormal", _make_recording_distribution())
    monkeypatch.setattr(MODULE.dist, "Multinomial", _make_recording_distribution())
    monkeypatch.setattr(MODULE.dist, "NegativeBinomial2", _make_recording_distribution())

    def _record_sample(name, distribution, obs=None):
        sample_calls.append(name)
        return obs if obs is not None else _Value()

    def _record_deterministic(name, value):
        deterministic_calls.append(name)
        return value

    monkeypatch.setattr(MODULE.numpyro, "sample", _record_sample)
    monkeypatch.setattr(MODULE.numpyro, "deterministic", _record_deterministic)

    n_classes = 3
    observed_matrix = np.zeros((5, n_classes))
    total_vector = np.zeros(5)
    validation_matrix = np.zeros((n_classes, n_classes))

    MODULE.section4_model(observed_matrix, total_vector, validation_matrix, n_classes)

    assert "latent_true_counts_by_predicted" not in sample_calls
    assert "expected_true_counts" not in deterministic_calls
    assert "latent_true_counts" not in deterministic_calls

    sample_calls.clear()
    deterministic_calls.clear()

    MODULE.section4_model(
        observed_matrix,
        total_vector,
        validation_matrix,
        n_classes,
        sample_latent_true_counts=True,
    )

    # The discrete latent allocation is sampled, and both it and its
    # conditional expectation are exposed as explicit named sites, exactly
    # as they would be replayed through numpyro.infer.Predictive.
    assert "latent_true_counts_by_predicted" in sample_calls
    assert "expected_true_counts" in deterministic_calls
    assert "latent_true_counts" in deterministic_calls

