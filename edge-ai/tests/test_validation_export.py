import io
import json
import os
import sys
import tarfile

import pytest

# Allow direct import of the module without triggering GUI dependencies
_module_path = os.path.join(
    os.path.dirname(__file__), "..", "gui", "application_validation"
)
sys.path.insert(0, _module_path)

from validation_export import (
    build_validation_dataset_json,
    build_validation_session_folder_name,
    export_validation_session_to_training_libs,
    prepare_validation_summary_for_export,
)


def _make_tar_blob(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class _FakeDownloader:
    def __init__(self, payload: bytes):
        self._payload = payload

    def chunks(self):
        yield self._payload


class _FakeBlobClient:
    def __init__(self, container, name: str):
        self._container = container
        self._name = name

    def download_blob(self, max_concurrency=1):
        return _FakeDownloader(self._container.objects[self._name])


class _FakeContainerClient:
    def __init__(self):
        self.objects = {}

    def list_blobs(self, name_starts_with=""):
        return [
            {"name": name}
            for name in sorted(self.objects)
            if name.startswith(name_starts_with)
        ]

    def upload_blob(self, name, data, overwrite=False):
        if not overwrite and name in self.objects:
            raise RuntimeError(f"Blob already exists: {name}")
        if hasattr(data, "read"):
            data = data.read()
        self.objects[name] = data

    def get_blob_client(self, name):
        return _FakeBlobClient(self, name)


class _FakeBlobServiceClient:
    def __init__(self):
        self.containers = {}

    def get_container_client(self, name):
        return self.containers.setdefault(name, _FakeContainerClient())


@pytest.fixture
def fixed_now():
    return __import__("datetime").datetime(2026, 7, 15, 9, 36, 30, tzinfo=__import__("datetime").timezone.utc)


def test_folder_name_uses_required_timestamp_format(fixed_now):
    assert build_validation_session_folder_name(fixed_now) == "validationsession_202607150936"


def test_prepare_validation_summary_for_export_adds_compatibility_fields():
    summary = {
        "classifier_error_model": {
            "class_order": ["copepod", "detritus"],
            "dirichlet_posterior_parameters": [[2.0, 0.5], [0.5, 2.0]],
        },
        "images": [
            {
                "container": "source-container",
                "source_tar_blob": "sample.tar",
                "tar_member": "RawImages/image1.tif",
                "original_label": "copepod",
                "final_label": "detritus",
                "validated": True,
            }
        ],
    }

    prepared = prepare_validation_summary_for_export(
        summary,
        labels=["copepod", "detritus"],
        metadata={"session_name": "demo"},
    )

    assert prepared["validation_error_model"]["class_order"] == ["copepod", "detritus"]
    assert prepared["classifier_error_model"]["class_order"] == ["copepod", "detritus"]
    assert prepared["validation_session"]["session_name"] == "demo"
    assert prepared["images"][0]["predicted_label"] == "copepod"
    assert prepared["images"][0]["prediction"] == "copepod"
    assert prepared["images"][0]["true_label"] == "detritus"
    assert prepared["images"][0]["label"] == "detritus"
    assert prepared["labels"] == ["copepod", "detritus"]
    assert "dirichlet_posterior_parameters" not in prepared["validation_error_model"]
    assert "posterior_credible_intervals" not in prepared["validation_error_model"]


def test_build_validation_dataset_json_populates_required_fields(fixed_now):
    rows = [
        {
            "container": "source-container",
            "source_tar_blob": "sample.tar",
            "tar_member": "RawImages/image1.tif",
            "original_label": "copepod",
            "final_label": "copepod",
            "model_blob": "model2026-07-15T09-00-00Z.pt",
            "validated": True,
        }
    ]

    dataset = build_validation_dataset_json(
        rows=rows,
        labels=["copepod", "detritus"],
        account_url="https://example.blob.core.windows.net",
        export_folder="validationsession_202607150936",
        generated_utc="2026-07-15T09:36:30Z",
        config=type(
            "Cfg",
            (),
            {
                "important_classes": ["copepod"],
                "min_class_precision": 0.8,
                "ci_width_target": 0.05,
                "max_sample_size": 500,
                "min_class_examples": 25,
                "diag_credible_interval_tolerance": 0.1,
                "offdiag_credible_interval_tolerance": 0.2,
                "smoothing_prior_alpha": 0.75,
            },
        )(),
    )

    assert dataset["schema_version"] == "1.0"
    assert dataset["dataset_identity"]["dataset_id"] == "validationsession_202607150936"
    assert dataset["labelling_session"]["labelling_for_validation_or_improvement"] == "validation"
    assert dataset["ml_assistance"]["used"] is True
    assert dataset["class_options"]["classes"] == ["copepod", "detritus"]
    assert dataset["versioning_and_lineage"]["dataset_version"] == "1.0"
    assert dataset["versioning_and_lineage"]["change_log"][0]["date_utc"] == "2026-07-15T09:36:30Z"
    assert dataset["validation_error_model"]["class_order"] == ["copepod", "detritus"]
    assert dataset["validation_error_model"]["transition_probabilities"][0][0] == pytest.approx(1.0)
    assert "dirichlet_posterior_parameters" not in dataset["validation_error_model"]
    assert "posterior_credible_intervals" not in dataset["validation_error_model"]
    assert "training-libs/validationsession_202607150936" in dataset["generalnotes"]


def test_export_skips_uncertain_rows_when_building_dataset_and_training_libs_export(fixed_now):
    bsc = _FakeBlobServiceClient()
    source = bsc.get_container_client("source-container")
    source.upload_blob(
        "sample.tar",
        _make_tar_blob(
            {
                "RawImages/image1.tif": b"image-one",
                "RawImages/image2.tif": b"image-two",
            }
        ),
        overwrite=False,
    )

    rows = [
        {
            "container": "source-container",
            "source_tar_blob": "sample.tar",
            "tar_member": "RawImages/image1.tif",
            "original_label": "copepod",
            "final_label": "copepod",
            "model_blob": "model2026-07-15T09-00-00Z.pt",
            "validated": True,
        },
        {
            "container": "source-container",
            "source_tar_blob": "sample.tar",
            "tar_member": "RawImages/image2.tif",
            "original_label": "detritus",
            "final_label": "uncertain",
            "model_blob": "model2026-07-15T09-00-00Z.pt",
            "validated": True,
        },
    ]
    validation_summary = {
        "generated_utc": "2026-07-15T09:36:30Z",
        "validated_image_count": 2,
        "images": rows,
    }

    result = export_validation_session_to_training_libs(
        blob_service_client=bsc,
        account_url="https://example.blob.core.windows.net",
        rows=rows,
        validation_summary=validation_summary,
        labels=["copepod", "detritus"],
        now=fixed_now,
    )

    uploaded = bsc.get_container_client("training-libs").objects
    assert f"{result['export_folder']}/copepod/sample__RawImages_image1.tif" in uploaded
    assert f"{result['export_folder']}/uncertain/sample__RawImages_image2.tif" not in uploaded

    dataset = json.loads(uploaded[f"{result['export_folder']}/dataset.json"].decode("utf-8"))
    assert dataset["validation_error_model"]["class_order"] == ["copepod", "detritus"]

    validation_json = json.loads(uploaded[f"{result['export_folder']}/validation.json"].decode("utf-8"))
    assert len(validation_json["images"]) == 2
    assert any(image["final_label"] == "uncertain" for image in validation_json["images"])
    assert validation_json["validation_error_model"]["class_order"] == ["copepod", "detritus"]
    assert validation_json["images"][0]["predicted_label"] == "copepod"
    assert validation_json["images"][0]["true_label"] == "copepod"


def test_export_uploads_metadata_and_images_into_training_libs_with_collision_suffix(fixed_now):
    bsc = _FakeBlobServiceClient()
    training_libs = bsc.get_container_client("training-libs")
    training_libs.upload_blob("validationsession_202607150936/validation.json", b"{}", overwrite=False)

    source = bsc.get_container_client("source-container")
    source.upload_blob(
        "sample.tar",
        _make_tar_blob(
            {
                "RawImages/image1.tif": b"image-one",
                "RawImages/image2.tif": b"image-two",
            }
        ),
        overwrite=False,
    )

    rows = [
        {
            "container": "source-container",
            "source_tar_blob": "sample.tar",
            "tar_member": "RawImages/image1.tif",
            "original_label": "copepod",
            "final_label": "copepod",
            "model_blob": "model2026-07-15T09-00-00Z.pt",
            "validated": True,
        },
        {
            "container": "source-container",
            "source_tar_blob": "sample.tar",
            "tar_member": "RawImages/image2.tif",
            "original_label": "detritus",
            "final_label": "detritus",
            "model_blob": "model2026-07-15T09-00-00Z.pt",
            "validated": True,
        },
    ]
    validation_summary = {
        "generated_utc": "2026-07-15T09:36:30Z",
        "validated_image_count": 2,
        "images": rows,
    }

    result = export_validation_session_to_training_libs(
        blob_service_client=bsc,
        account_url="https://example.blob.core.windows.net",
        rows=rows,
        validation_summary=validation_summary,
        labels=["copepod", "detritus"],
        now=fixed_now,
    )

    assert result["export_folder"] == "validationsession_202607150936_1"

    uploaded = training_libs.objects
    assert f"{result['export_folder']}/validation.json" in uploaded
    assert f"{result['export_folder']}/dataset.json" in uploaded
    assert f"{result['export_folder']}/copepod/sample__RawImages_image1.tif" in uploaded
    assert f"{result['export_folder']}/detritus/sample__RawImages_image2.tif" in uploaded

    dataset = json.loads(uploaded[f"{result['export_folder']}/dataset.json"].decode("utf-8"))
    assert dataset["dataset_identity"]["dataset_id"] == result["export_folder"]
    assert dataset["validation_error_model"]["class_order"] == ["copepod", "detritus"]
