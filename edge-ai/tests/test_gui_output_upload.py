import sys
from pathlib import Path

import pytest

EDGE_AI_DIR = Path(__file__).resolve().parent.parent
if str(EDGE_AI_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_AI_DIR))

from gui.application_validation.output_upload import (  # noqa: E402
    DEFAULT_OUTPUT_CONTAINER,
    DEFAULT_OUTPUT_PREFIX,
    build_gui_output_slug,
    build_output_blob_name,
    upload_json_blob,
)


def test_build_gui_output_slug_uses_guilocalinference_prefix():
    slug = build_gui_output_slug("2026-08-20T13-26-16Z")
    assert slug == "guilocalinference-2026-08-20T13-26-16Z"


def test_build_output_blob_name_uses_prefix_and_relative_path():
    blob_name = build_output_blob_name("runs", "guilocalinference-2026-08-20T13-26-16Z", "per_tar/2024-10-01/1700/predictions.json")
    assert blob_name == "runs/guilocalinference-2026-08-20T13-26-16Z/per_tar/2024-10-01/1700/predictions.json"


def test_build_output_blob_name_defaults_to_prediction_results_prefix():
    blob_name = build_output_blob_name("", "guilocalinference-2026-08-20T13-26-16Z", "run_summary.json")
    assert blob_name == f"{DEFAULT_OUTPUT_PREFIX}/guilocalinference-2026-08-20T13-26-16Z/run_summary.json"


def test_build_output_blob_name_falls_back_to_generated_slug_when_empty():
    blob_name = build_output_blob_name("", "", "run_summary.json")
    assert blob_name.startswith(f"{DEFAULT_OUTPUT_PREFIX}/guilocalinference-")
    assert blob_name.endswith("/run_summary.json")


def test_upload_json_blob_uploads_to_requested_container(tmp_path):
    class FakeContainerClient:
        def __init__(self):
            self.calls = []

        def upload_blob(self, name, fh, overwrite=True, **kwargs):
            self.calls.append({"name": name, "overwrite": overwrite, "kwargs": kwargs, "data": fh.read()})

    class FakeBlobServiceClient:
        def __init__(self, container_client):
            self.container_client = container_client

        def get_container_client(self, container_name):
            assert container_name == DEFAULT_OUTPUT_CONTAINER
            return self.container_client

    container_client = FakeContainerClient()
    blob_service_client = FakeBlobServiceClient(container_client)
    local_path = tmp_path / "summary.json"
    local_path.write_text('{"hello": "world"}', encoding="utf-8")

    upload_json_blob(blob_service_client, DEFAULT_OUTPUT_CONTAINER, "runs/foo/summary.json", local_path)

    assert len(container_client.calls) == 1
    call = container_client.calls[0]
    assert call["name"] == "runs/foo/summary.json"
    assert call["overwrite"] is True
    assert call["data"] == b'{"hello": "world"}'
