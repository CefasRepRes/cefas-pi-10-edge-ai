import json
import os
import sys

_module_path = os.path.join(os.path.dirname(__file__), "..", "gui", "application_validation")
sys.path.insert(0, _module_path)

from prediction_sources import load_predictions_from_blob_container, load_predictions_from_source


class _FakeDownloader:
    def __init__(self, payload):
        self._payload = payload

    def readall(self):
        return self._payload


class _FakeBlobClient:
    def __init__(self, payload):
        self._payload = payload

    def download_blob(self):
        return _FakeDownloader(self._payload)


class _FakeContainerClient:
    def __init__(self, blobs):
        self._blobs = blobs

    def list_blobs(self, name_starts_with=""):
        return [
            type("Blob", (), {"name": name})()
            for name in self._blobs
            if name.startswith(name_starts_with)
        ]

    def get_blob_client(self, name):
        return _FakeBlobClient(self._blobs[name])


class _FakeBlobServiceClient:
    def __init__(self, container_blobs):
        self._container_blobs = container_blobs

    def get_container_client(self, container):
        return _FakeContainerClient(self._container_blobs.get(container, {}))


def test_load_predictions_from_blob_container_reads_prediction_blobs():
    bsc = _FakeBlobServiceClient({
        "predictions": {
            "runs/run1/per_tar/2025-06-01/2350/predictions.json": json.dumps([
                {
                    "container": "source",
                    "source_tar_blob": "sample.tar",
                    "tar_member": "RawImages/alpha.tif",
                    "predicted_label": "fish_larvae",
                }
            ]).encode("utf-8"),
            "runs/run1/per_tar/2025-06-01/2351/predictions.json": json.dumps([
                {
                    "container": "source",
                    "source_tar_blob": "sample2.tar",
                    "tar_member": "RawImages/beta.tif",
                    "predicted_label": "copepod",
                }
            ]).encode("utf-8"),
            "runs/run1/per_tar/2025-06-01/2351/summary.json": b"{}",
        }
    })

    predictions = load_predictions_from_blob_container(
        bsc,
        "predictions",
        prefix="runs/run1/per_tar",
        log_cb=lambda _: None,
    )

    assert len(predictions) == 2
    assert predictions[0]["source_tar_blob"] == "sample.tar"
    assert predictions[1]["tar_member"].endswith("beta.tif")


def test_load_predictions_from_source_accepts_blob_urls():
    bsc = _FakeBlobServiceClient({
        "predictions": {
            "runs/run1/per_tar/2025-06-01/2350/predictions.json": json.dumps([
                {
                    "container": "source",
                    "source_tar_blob": "sample.tar",
                    "tar_member": "RawImages/alpha.tif",
                    "predicted_label": "fish_larvae",
                }
            ]).encode("utf-8")
        }
    })

    predictions = load_predictions_from_source(
        "https://example.blob.core.windows.net/predictions/runs/run1/per_tar",
        account_url="https://example.blob.core.windows.net",
        blob_service_client=bsc,
        log_cb=lambda _: None,
    )

    assert len(predictions) == 1
    assert predictions[0]["predicted_label"] == "fish_larvae"
