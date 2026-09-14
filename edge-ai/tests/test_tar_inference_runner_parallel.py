"""Tests for parallel TAR processing in tar_inference_runner_by_day.py."""
import importlib
import os
import sys
import threading
import types
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the edge-ai directory and its sub-packages are importable without
# pulling in heavy optional dependencies (torch, pyvips, azure, PySide6).
# We stub out the problematic top-level imports before loading the module.
# ---------------------------------------------------------------------------

EDGE_AI_DIR = Path(__file__).resolve().parent.parent

def _ensure_stub(name, attrs=None):
    """Register a lightweight stub module under *name* if not already present."""
    if name in sys.modules:
        return sys.modules[name]
    parts = name.split(".")
    parent_name = ".".join(parts[:-1])
    if parent_name:
        parent = _ensure_stub(parent_name)
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    if parent_name:
        setattr(sys.modules[parent_name], parts[-1], mod)
    return mod


def _stub_dependencies():
    """Stub every import that the runner needs but that isn't installed in CI."""
    # torch
    torch_mod = _ensure_stub("torch")
    torch_mod.no_grad = lambda: MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None)
    torch_mod.device = lambda x: x
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_name=lambda i: "",
    )
    torch_mod.__version__ = "0.0"
    torch_mod.version = types.SimpleNamespace(cuda="0.0")
    torch_mod.load = MagicMock(return_value={})
    torch_mod.stack = MagicMock()
    torch_mod.nn = types.SimpleNamespace(Module=object)

    # pyvips
    pyvips_mod = _ensure_stub("pyvips")
    pyvips_mod.Error = Exception

    # azure stubs
    _ensure_stub("azure")
    _ensure_stub("azure.core")
    _ensure_stub("azure.core.exceptions", {"ResourceNotFoundError": Exception})
    _ensure_stub("azure.identity", {"DefaultAzureCredential": MagicMock})
    _ensure_stub("azure.storage")
    _ensure_stub("azure.storage.blob", {
        "BlobServiceClient": MagicMock,
        "ContentSettings": MagicMock,
    })

    # PySide6 stubs (workers.py imports it; we don't load workers directly but
    # the runner imports from gui.application_validation sub-modules)
    _ensure_stub("PySide6")
    _ensure_stub("PySide6.QtCore", {"QThread": object, "Signal": lambda *a: None})

    # gps / prediction_records stubs
    _ensure_stub("gps")
    _ensure_stub("prediction_records", {
        "build_prediction_record": MagicMock(return_value={}),
        "extract_gps_payload": MagicMock(return_value=None),
    })

    # gui sub-package stubs
    gui_pkg = _ensure_stub("gui")
    gui_pkg.__path__ = []
    av_pkg = _ensure_stub("gui.application_validation")
    av_pkg.__path__ = []

    _ensure_stub("gui.application_validation.constants", {
        "TAR_PROGRESS_EVERY_IMAGES": 1000,
        "MAX_BLOBS_SAFETY": 100_000,
    })
    _ensure_stub("gui.application_validation.csv_utils", {
        "parse_sample_ratio": lambda x: float(x),
        "DEFAULT_TAR_SAMPLE_RATIO": 1.0,
    })
    _ensure_stub("gui.application_validation.model_utils", {
        "build_inference_transform": MagicMock(return_value=lambda b: b),
        "build_model_from_artifact": MagicMock(return_value=MagicMock()),
        "idx_to_label_fn": MagicMock(return_value=lambda i: str(i)),
    })
    _ensure_stub("gui.application_validation.tar_streaming", {
        "iter_tar_image_bytes_from_blob": MagicMock(return_value=iter([])),
        "per_tar_paths": lambda run_dir, tar_name: (
            str(Path(run_dir) / "per_tar" / tar_name),
            str(Path(run_dir) / "per_tar" / tar_name / "predictions.json"),
            str(Path(run_dir) / "per_tar" / tar_name / "summary.json"),
        ),
        "safe_relative_stem_from_tar_name": lambda t: Path(t).stem,
    })
    _ensure_stub("gui.application_validation.azure_utils", {
        "get_blob_service_client": MagicMock(),
        "utc_now_iso": lambda: "2000-01-01T00:00:00Z",
        "extract_runid_from_model_blob": MagicMock(return_value=None),
    })
    _ensure_stub("gui.application_validation.blob_listing", {
        "quick_count_blobs": MagicMock(return_value=0),
    })
    _ensure_stub("gui.application_validation.workers", {})


_stub_dependencies()

# Insert edge-ai on path so the module resolves its local imports
if str(EDGE_AI_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_AI_DIR))

import tar_inference_runner_by_day as runner  # noqa: E402


# ---------------------------------------------------------------------------
# Tests for _default_parallel_stream_count
# ---------------------------------------------------------------------------

class TestDefaultParallelStreamCount:
    def test_mobilenet_v3_small_returns_16(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenet_v3_small"}) == 16

    def test_mobilenet_alias_returns_16(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenet"}) == 16

    def test_mobilenetv3_alias_returns_16(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenetv3"}) == 16

    def test_resnet_returns_2(self):
        assert runner._default_parallel_stream_count({"arch": "resnet18"}) == 2

    def test_mobilenet_v2_returns_2(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenet_v2"}) == 2

    def test_mobilenetv2_alias_returns_2(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenetv2"}) == 2

    def test_missing_arch_defaults_to_resnet18(self):
        assert runner._default_parallel_stream_count({}) == 2

    def test_none_artifact_defaults_to_resnet18(self):
        assert runner._default_parallel_stream_count(None) == 2

    def test_hyphen_normalised(self):
        # arch names with hyphens should be normalised to underscores
        assert runner._default_parallel_stream_count({"arch": "mobilenet-v3-small"}) == 16


# ---------------------------------------------------------------------------
# Tests for parallel TAR dispatch in main()
# ---------------------------------------------------------------------------

def _make_fake_infer_on_tar(captured_calls, call_lock):
    """Return a fake infer_on_tar_archive that records calls and returns a valid result dict."""
    def _fake(*, container, tar_name, prefix, sample_ratio, model, tfm,
              idx_to_label, device, model_blob, model_runid, run_dir,
              source_blob_service_client, output_blob_service_client,
              output_container, output_prefix, output_slug, log_cb):
        with call_lock:
            captured_calls.append(tar_name)
        return {
            "container": container,
            "tar_name": tar_name,
            "prefix": prefix,
            "sample_ratio": sample_ratio,
            "tar_class_counts": Counter({"plankton": 1}),
            "error_message": None,
            "skipped_existing": False,
        }
    return _fake


class TestParallelDispatch:
    """Verify that the ThreadPoolExecutor path is exercised with the right concurrency."""

    def _run_main_with_tars(self, tar_names, parallel_streams_arg=0, tmp_path=None):
        """Patch main()'s internals and invoke it; return (selected_tars, captured_calls)."""
        captured_calls = []
        call_lock = threading.Lock()

        fake_artifact = {"arch": "resnet18", "model_state_dict": {"_dummy": None}}

        fake_bsc = MagicMock()
        fake_container_client = MagicMock()
        fake_container_client.list_blobs.return_value = [
            types.SimpleNamespace(name=n) for n in tar_names
        ]
        # model bytes download must return bytes (BytesIO wraps it)
        fake_downloader = MagicMock()
        fake_downloader.readall.return_value = b"fake-model-bytes"
        fake_blob_client = MagicMock()
        fake_blob_client.download_blob.return_value = fake_downloader
        fake_bsc.get_container_client.return_value = fake_container_client
        fake_container_client.get_blob_client.return_value = fake_blob_client

        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model
        fake_model.to.return_value = fake_model
        fake_model.load_state_dict = MagicMock()

        # Build a minimal set of args
        args_ns = types.SimpleNamespace(
            account_url="https://fake.blob.core.windows.net",
            models_container="trainedmodels",
            model_blob="model2000-01-01T00-00-00Z.pt",
            target_container="test-container",
            target_prefix="",
            day="2000-01-01",
            run_key="test-run",
            sample_ratio="1.0",
            output_container="",
            output_prefix="runs",
            settings_blob="",
            resume_existing="false",
            allow_empty_day="true",
            no_resume=False,
            parallel_streams=parallel_streams_arg,
        )

        work_dir = tmp_path or Path("/tmp/runner_test")
        work_dir.mkdir(parents=True, exist_ok=True)

        patches = [
            patch.object(runner, "parse_args", return_value=args_ns),
            patch.object(runner, "build_blob_service_client", return_value=fake_bsc),
            patch.object(runner, "build_model_from_artifact", return_value=fake_model),
            patch.object(runner, "build_inference_transform", return_value=lambda b: b),
            patch.object(runner, "idx_to_label_fn", return_value=lambda i: str(i)),
            patch.object(runner, "infer_on_tar_archive", side_effect=_make_fake_infer_on_tar(captured_calls, call_lock)),
            patch("torch.load", return_value=fake_artifact),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.device_count", return_value=0),
            patch.object(Path, "cwd", return_value=work_dir),
        ]

        import contextlib
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            runner.main()

        return captured_calls

    def test_all_tars_processed(self, tmp_path):
        tar_names = [f"2000-01-01/archive_{i:03d}.tar" for i in range(5)]
        calls = self._run_main_with_tars(tar_names, tmp_path=tmp_path)
        assert sorted(calls) == sorted(tar_names)

    def test_parallel_streams_override(self, tmp_path):
        """When --parallel-streams is set explicitly, that value should be used."""
        tar_names = [f"2000-01-01/archive_{i:03d}.tar" for i in range(4)]
        calls = self._run_main_with_tars(tar_names, parallel_streams_arg=4, tmp_path=tmp_path)
        assert len(calls) == 4

    def test_single_tar(self, tmp_path):
        tar_names = ["2000-01-01/only.tar"]
        calls = self._run_main_with_tars(tar_names, tmp_path=tmp_path)
        assert calls == tar_names

    def test_no_tars_writes_empty_day_summary(self, tmp_path):
        """When no TARs are found and allow_empty_day is true, main() should not raise."""
        calls = self._run_main_with_tars([], tmp_path=tmp_path)
        assert calls == []
