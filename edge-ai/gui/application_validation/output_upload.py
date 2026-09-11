import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:  # pragma: no cover - supports environments without azure-storage-blob
    BlobServiceClient = None  # type: ignore[assignment]
    ContentSettings = None  # type: ignore[assignment]

DEFAULT_OUTPUT_CONTAINER = "ml-prediction-results"
DEFAULT_OUTPUT_PREFIX = "runs"
GUI_OUTPUT_SLUG_PREFIX = "guilocalinference"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H-%M-%SZ")


def sanitize_slug(value: Optional[str], *, default: str = "run") -> str:
    candidate = (value or default).strip()
    candidate = candidate.replace(":", "-")
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("_")
    return candidate or default


def build_gui_output_slug(stamp: Optional[str] = None) -> str:
    stamp = (stamp or utc_now_iso()).strip()
    return f"{GUI_OUTPUT_SLUG_PREFIX}-{sanitize_slug(stamp, default='run')}"


def build_output_blob_name(output_prefix: Optional[str], output_slug: Optional[str], relative_path: Path | str) -> str:
    prefix = (output_prefix or DEFAULT_OUTPUT_PREFIX).strip("/")
    slug_value = output_slug or build_gui_output_slug()
    slug = sanitize_slug(slug_value)
    relative = str(relative_path).replace("\\", "/")
    return "/".join([prefix, slug, relative.lstrip("/")])


def upload_json_blob(blob_service_client, container_name: str, blob_name: str, local_path: Path | str):
    """Upload a local JSON file to the given Azure blob container.

    The helper accepts any blob-service client object exposing
    ``get_container_client`` and ``upload_blob``. It always requires a working
    client, and only applies Azure blob content settings when
    ``azure.storage.blob.ContentSettings`` is available.
    """
    container_client = blob_service_client.get_container_client(container_name)
    local_path = Path(local_path)
    with local_path.open("rb") as fh:
        kwargs = {"overwrite": True}
        if ContentSettings is not None:
            kwargs["content_settings"] = ContentSettings(content_type="application/json")
        container_client.upload_blob(blob_name, fh, **kwargs)
