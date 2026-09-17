import json
import os
import re
import sys

try:
    from .tar_streaming import iter_per_tar_prediction_paths
except ImportError:  # pragma: no cover - allows direct test imports without package context
    sys.path.insert(0, os.path.dirname(__file__))
    from tar_streaming import iter_per_tar_prediction_paths


def _parse_blob_url(url: str):
    try:
        from .azure_utils import parse_blob_url
    except ImportError:  # pragma: no cover - allows direct test imports without package context
        url = (url or "").strip()
        if not url:
            return None
        url_no_q = url.split("?", 1)[0]
        match = re.match(r"^(https?://[^/]+)(?:/([^/]+)(?:/(.*))?)?$", url_no_q)
        if not match:
            return None
        account_url = match.group(1)
        container = match.group(2)
        path = (match.group(3) or "").lstrip("/")
        return account_url, container, path
    return parse_blob_url(url)


def _get_blob_service_client(account_url: str, log_cb=None):
    try:
        from .azure_utils import get_blob_service_client
    except ImportError:  # pragma: no cover - allows direct test imports without package context
        raise RuntimeError("Azure Blob SDK dependencies are not available")
    return get_blob_service_client(account_url, log_cb=log_cb)


def _prediction_records_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("predictions", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return None


def _decode_prediction_payload(raw):
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, str):
        return raw
    raise RuntimeError(f"Unsupported prediction payload type: {type(raw).__name__}")


def _load_prediction_file(path: str, log_cb=None):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = _prediction_records_from_payload(payload)
    if records is None:
        raise RuntimeError(f"Unexpected predictions payload in {path}")
    return records


def load_predictions_from_local_dir(per_tar_dir: str, log_cb=None):
    if not per_tar_dir or not os.path.isdir(per_tar_dir):
        return []

    all_predictions = []
    for pred_path in iter_per_tar_prediction_paths(per_tar_dir):
        try:
            all_predictions.extend(_load_prediction_file(pred_path, log_cb=log_cb))
        except Exception as exc:
            if log_cb:
                log_cb(f"Failed to read {pred_path}: {exc}")
    return all_predictions


def load_predictions_from_blob_container(
    blob_service_client,
    container: str,
    prefix: str = "",
    log_cb=None,
):
    if not blob_service_client:
        raise RuntimeError("Missing blob service client")
    if not container:
        raise RuntimeError("Missing blob container")

    prefix = (prefix or "").strip().lstrip("/")
    container_client = blob_service_client.get_container_client(container)

    names = []
    if prefix.lower().endswith("predictions.json"):
        names = [prefix]
    else:
        for blob in container_client.list_blobs(name_starts_with=prefix):
            try:
                name = getattr(blob, "name", None)
                if name is None and isinstance(blob, dict):
                    name = blob.get("name")
                if not name:
                    continue
                if name.lower().endswith("predictions.json"):
                    names.append(name)
            except Exception:
                continue

    all_predictions = []
    for blob_name in sorted(set(names)):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            payload = json.loads(_decode_prediction_payload(blob_client.download_blob().readall()))
        except Exception as exc:
            if log_cb:
                log_cb(f"Failed to read blob {container}/{blob_name}: {exc}")
            continue
        records = _prediction_records_from_payload(payload)
        if records is None:
            if log_cb:
                log_cb(f"Unexpected predictions payload in blob {container}/{blob_name}")
            continue
        all_predictions.extend(records)
    return all_predictions


def _looks_like_local_path(value: str) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip()
    if not value:
        return False
    if value.startswith(("~", "/", "\\")):
        return True
    if value.startswith("./") or value.startswith(".\\"):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    return False


def load_predictions_from_source(source, *, account_url=None, blob_service_client=None, log_cb=None):
    if source is None:
        return []

    if isinstance(source, (list, tuple)):
        return list(source)

    if os.path.isdir(source):
        return load_predictions_from_local_dir(source, log_cb=log_cb)

    if os.path.isfile(source):
        return _load_prediction_file(source, log_cb=log_cb)

    if _looks_like_local_path(source):
        raise RuntimeError(f"Could not find local predictions path: {source}")

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        parsed = _parse_blob_url(source)
        if not parsed:
            raise RuntimeError(f"Could not parse predictions blob URL: {source}")
        parsed_account_url, container, path = parsed
        if blob_service_client is None:
            resolved_account_url = (account_url or parsed_account_url or "").strip().rstrip("/")
            if not resolved_account_url:
                raise RuntimeError("Missing Azure Blob Storage account URL")
            blob_service_client = _get_blob_service_client(resolved_account_url, log_cb=log_cb)
        return load_predictions_from_blob_container(blob_service_client, container, prefix=path, log_cb=log_cb)

    if isinstance(source, str):
        source = source.strip()
        if not source:
            return []
        if "/" in source:
            container, prefix = source.split("/", 1)
            container = container.strip()
            prefix = prefix.strip()
        else:
            container = source.strip()
            prefix = ""
        if container:
            if blob_service_client is None:
                resolved_account_url = (account_url or "").strip().rstrip("/")
                if not resolved_account_url:
                    raise RuntimeError("Missing Azure Blob Storage account URL")
                blob_service_client = _get_blob_service_client(resolved_account_url, log_cb=log_cb)
            return load_predictions_from_blob_container(blob_service_client, container, prefix=prefix, log_cb=log_cb)

    raise RuntimeError(f"Could not resolve predictions source: {source}")
