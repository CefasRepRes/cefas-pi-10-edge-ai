import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "utility_scripts" / "plot_class_counts_timeseries_streaming_quiet.py"
SPEC = importlib.util.spec_from_file_location("plot_class_counts_timeseries_streaming_quiet", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_uncertainty_dataframe_uses_nonnegative_solver_and_percentiles():
    plot_df = pd.DataFrame(
        [
            {"timestamp": "2024-01-01", "run_name": "run-a", "blob_name": "a.json", "fish": 4, "detritus": 1},
            {"timestamp": "2024-01-02", "run_name": "run-b", "blob_name": "b.json", "fish": 2, "detritus": 0},
        ]
    )
    validation_payload = {
        "classifier_error_model": {
            "class_order": ["fish", "detritus"],
            "dirichlet_posterior_parameters": [
                [4.0, 1.0],
                [1.0, 4.0],
            ],
        }
    }

    uncertainty_df = MODULE.build_uncertainty_dataframe(
        plot_df,
        validation_payload,
        mc_samples=50,
        mc_seed=7,
    )

    assert "fish_corrected_median" in uncertainty_df.columns
    assert "detritus_corrected_median" in uncertainty_df.columns
    assert uncertainty_df.loc[0, "fish_corrected_lower"] <= uncertainty_df.loc[0, "fish_corrected_median"]
    assert uncertainty_df.loc[0, "fish_corrected_median"] <= uncertainty_df.loc[0, "fish_corrected_upper"]
    assert (uncertainty_df["fish_corrected_median"] >= 0).all()
    assert (uncertainty_df["detritus_corrected_median"] >= 0).all()


class FakeBlobDownloader:
    def __init__(self, payload):
        self._payload = payload

    def readall(self):
        return self._payload.encode("utf-8")


class FakeBlobClient:
    def __init__(self, payload):
        self._payload = payload

    def download_blob(self):
        return FakeBlobDownloader(self._payload)


class FakeBlobContainerClient:
    def __init__(self, payload):
        self._payload = payload

    def download_blob(self, _blob_name):
        return FakeBlobDownloader(self._payload)

    def list_blobs(self, name_starts_with=""):
        if isinstance(self._payload, dict):
            return [type("Blob", (), {"name": name})() for name in self._payload if name.startswith(name_starts_with)]
        return []

    def get_blob_client(self, name):
        if isinstance(self._payload, dict):
            return FakeBlobClient(self._payload[name])
        return FakeBlobClient(self._payload)


class FakeBlobServiceClient:
    def __init__(self, payload):
        self._payload = payload

    def get_container_client(self, _container_name):
        return FakeBlobContainerClient(self._payload)


def test_parse_args_defaults_to_blob_prefix_scan():
    cfg = MODULE.parse_args([])
    assert cfg.inference_json == MODULE.DEFAULT_INFERENCE_SOURCE


def test_load_inference_rows_from_source_scans_summary_blobs():
    payload = {
        "runs/run-a/per_tar/2024-10-01/1700/summary.json": '{"class_counts": {"fish": 2, "detritus": 1}}',
        "runs/run-a/per_tar/2024-10-01/1701/summary.json": '{"class_counts": {"fish": 4, "detritus": 0}}',
        "runs/run-b/per_tar/2024-10-01/1700/summary.json": '{"class_counts": {"fish": 1, "detritus": 2}}',
        "runs/run-b/per_tar/2024-10-01/1702/ignored.json": '{}',
    }

    rows = MODULE.load_inference_rows_from_source(
        "https://example.blob.core.windows.net/ml-prediction-results/runs/",
        blob_service_client=FakeBlobServiceClient(payload),
    )

    assert len(rows) == 3
    assert {row["run_name"] for row in rows} == {"run-a", "run-b"}
    assert rows[0]["timestamp"].year == 2024
    assert rows[0]["timestamp"].month == 10
    assert rows[0]["timestamp"].day == 1
    assert rows[0]["timestamp"].hour == 17
    assert rows[0]["timestamp"].minute == 0
    assert rows[0]["fish"] == 2
    assert rows[2]["detritus"] == 2


def test_load_json_payload_downloads_authenticated_azure_blob(monkeypatch):
    payload = MODULE.load_json_payload(
        "https://example.blob.core.windows.net/container/validation.json",
        blob_service_client=FakeBlobServiceClient('{"ok": true}'),
    )

    assert payload == {"ok": True}


def test_load_validation_uncertainty_dataframe_returns_none_when_validation_json_cannot_be_loaded(monkeypatch):
    plot_df = pd.DataFrame(
        [{"timestamp": "2024-01-01", "run_name": "run-a", "blob_name": "a.json", "fish": 4}]
    )

    def fail_load_json(_source, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(MODULE, "load_json_payload", fail_load_json)

    uncertainty_df = MODULE.load_validation_uncertainty_dataframe(
        plot_df,
        "https://example.invalid/validation.json",
        mc_samples=10,
        mc_seed=7,
    )

    assert uncertainty_df is None


def test_load_validation_uncertainty_dataframe_returns_none_when_building_uncertainty_fails(monkeypatch):
    plot_df = pd.DataFrame(
        [{"timestamp": "2024-01-01", "run_name": "run-a", "blob_name": "a.json", "fish": 4}]
    )

    monkeypatch.setattr(MODULE, "load_json_payload", lambda _source, **_kwargs: {"classifier_error_model": {}})
    monkeypatch.setattr(MODULE, "build_uncertainty_dataframe", lambda _plot_df, _payload, **_kwargs: (_ for _ in ()).throw(ValueError("boom")))

    uncertainty_df = MODULE.load_validation_uncertainty_dataframe(
        plot_df,
        "https://example.invalid/validation.json",
        mc_samples=10,
        mc_seed=7,
    )

    assert uncertainty_df is None
