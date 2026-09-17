import csv
import os
import re
import sys
from datetime import datetime

try:
    from .tar_streaming import safe_relative_stem_from_tar_name
except ImportError:  # pragma: no cover - allows direct test imports without package context
    sys.path.insert(0, os.path.dirname(__file__))
    from tar_streaming import safe_relative_stem_from_tar_name


DEFAULT_TAR_SAMPLE_RATIO = 1.0


def _parse_blob_url(url: str):
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


def _build_resume_summary_blob_name(previous_job_url: str, tar_blob_name: str) -> str | None:
    parsed = _parse_blob_url(previous_job_url)
    if not parsed:
        return None
    _, container, path = parsed
    if not container:
        return None
    resume_prefix = (path or "").strip().strip("/")
    if resume_prefix.endswith("/summary.json"):
        if "/per_tar/" in resume_prefix:
            resume_prefix = resume_prefix.split("/per_tar/", 1)[0]
        else:
            resume_prefix = resume_prefix.rsplit("/", 1)[0]
    if resume_prefix:
        resume_prefix = f"{resume_prefix}/"
    relative_stem = safe_relative_stem_from_tar_name(tar_blob_name)
    return f"{resume_prefix}per_tar/{relative_stem}/summary.json"


def _resume_summary_exists(bsc, previous_job_url: str, tar_blob_name: str) -> bool:
    if not previous_job_url or not tar_blob_name:
        return False
    parsed = _parse_blob_url(previous_job_url)
    if not parsed:
        return False
    _, container, _ = parsed
    if not container:
        return False
    blob_name = _build_resume_summary_blob_name(previous_job_url, tar_blob_name)
    if not blob_name:
        return False

    try:
        container_client = bsc.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        if hasattr(blob_client, "exists"):
            return bool(blob_client.exists())
        return any(blob.name == blob_name for blob in container_client.list_blobs(name_starts_with=blob_name))
    except Exception:
        return False


def parse_sample_ratio(
    value,
    default: float = DEFAULT_TAR_SAMPLE_RATIO,
    *,
    allow_whole_number_percent: bool = False,
) -> float:
    """Parse sample ratio values from CSV/UI, accepting fractions and ``%`` suffixed values."""
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        ratio = float(value)
    else:
        text = str(value).strip()
        if not text:
            return float(default)
        if text.endswith("%"):
            ratio = float(text[:-1].strip()) / 100.0
        else:
            ratio = float(text)
            if allow_whole_number_percent and ratio > 1.0 and ratio <= 100.0:
                ratio /= 100.0

    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"sample ratio must be between 0 and 1 inclusive, got {value!r}")
    return ratio


def load_tar_selection_csv(csv_path: str):
    """Read CSV and return selected TAR rows as (container, tar_blob, prefix, sample_ratio)."""
    selected = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("process", "").strip().upper() == "Y":
                has_sample_ratio = "sample_ratio" in row
                has_sample_percent = "sample_percent" in row
                sample_value = (
                    row.get("sample_ratio")
                    if has_sample_ratio
                    else row.get("sample_percent")
                    if has_sample_percent
                    else None
                )
                try:
                    sample_ratio = parse_sample_ratio(
                        sample_value,
                        allow_whole_number_percent=has_sample_percent and not has_sample_ratio,
                    )
                except ValueError as e:
                    tar_name = row.get("tar_blob") or "(unknown TAR)"
                    raise RuntimeError(f"Invalid sample ratio for {tar_name}: {e}") from e
                selected.append((row["container"], row["tar_blob"], row.get("prefix") or "", sample_ratio))

    if not selected:
        raise RuntimeError("No TARs marked for processing in CSV")

    return selected


def extract_datetime_from_tar_name(name: str):
    """Attempt to extract YYYYMMDDHHMMSS from tar names. Returns ISO string or None."""
    m = re.search(r"(\d{8})(\d{6})", name)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_target_specs(targets=None, *, container: str | None = None, prefix: str = ""):
    if targets is None:
        if container:
            return [{"container": container, "prefix": prefix or ""}]
        return []

    normalized = []
    seen = set()
    for entry in targets:
        if isinstance(entry, dict):
            target_container = entry.get("container")
            target_prefix = entry.get("prefix") or entry.get("selected_prefix") or ""
        elif isinstance(entry, (list, tuple)):
            if len(entry) >= 2:
                target_container, target_prefix = entry[0], entry[1]
            elif len(entry) == 1:
                target_container, target_prefix = entry[0], ""
            else:
                raise RuntimeError(f"Unexpected container target spec: {entry!r}")
        else:
            target_container = entry
            target_prefix = ""

        target_container = str(target_container).strip() if target_container is not None else ""
        target_prefix = str(target_prefix or "")
        if not target_container:
            raise RuntimeError(f"Container target missing container name: {entry!r}")
        key = (target_container, target_prefix)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"container": target_container, "prefix": target_prefix})

    return normalized


def generate_tar_selection_csv(
    *,
    bsc,
    container: str | None = None,
    prefix: str = "",
    csv_path: str,
    preselect_ratio: float = 0.10,
    sample_ratio: float = DEFAULT_TAR_SAMPLE_RATIO,
    targets=None,
    resume_previous_job_url: str | None = None,
    log_cb=None,
):
    """Enumerate TAR blobs and emit editable CSV with Y/N process flag."""
    target_specs = _normalize_target_specs(targets, container=container, prefix=prefix)
    if not target_specs:
        raise RuntimeError("No container targets provided")

    sample_ratio = parse_sample_ratio(sample_ratio)

    rows_written = 0
    preselected_total = 0
    target_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tar_blob", "container", "prefix", "datetime_utc", "process", "sample_ratio"],
        )
        writer.writeheader()

        for target in target_specs:
            target_container = target["container"]
            target_prefix = target["prefix"]
            cc = bsc.get_container_client(target_container)
            tar_names = [
                b.name for b in cc.list_blobs(name_starts_with=target_prefix)
                if b.name.lower().endswith((".tar", ".tar.gz", ".tgz"))
            ]

            if not tar_names:
                if len(target_specs) == 1:
                    raise RuntimeError("No TAR blobs found")
                if log_cb:
                    log_cb(f"Skipping target {target_container}/{target_prefix}: no TAR blobs found")
                continue

            tar_rows = []
            for name in tar_names:
                dt = extract_datetime_from_tar_name(name)
                dt_key = dt or "9999-12-31T23:59:59Z"
                tar_rows.append((dt_key, name))

            tar_rows.sort(key=lambda x: x[0])
            N = len(tar_rows)
            k = max(1, int(round(N * preselect_ratio)))

            if k >= N:
                preselected = {name for _, name in tar_rows}
            else:
                step = (N - 1) / max(k - 1, 1)
                selected_indices = {round(i * step) for i in range(k)}
                preselected = {tar_rows[i][1] for i in selected_indices}

            target_count += 1
            for name in sorted(tar_names):
                resume_skipped = False
                if resume_previous_job_url:
                    try:
                        resume_skipped = _resume_summary_exists(bsc, resume_previous_job_url, name)
                    except Exception as exc:
                        if log_cb:
                            log_cb(f"WARNING: failed to verify resume output for {name}: {exc}")
                        resume_skipped = False
                process_value = "N"
                if not resume_skipped:
                    process_value = "Y" if name in preselected else "N"
                if resume_skipped and log_cb:
                    log_cb(
                        f"Skipping {target_container}/{name} because a resume summary was found at "
                        f"{_build_resume_summary_blob_name(resume_previous_job_url, name)}"
                    )
                if process_value == "Y":
                    preselected_total += 1
                writer.writerow({
                    "tar_blob": name,
                    "container": target_container,
                    "prefix": target_prefix,
                    "datetime_utc": extract_datetime_from_tar_name(name),
                    "process": process_value,
                    "sample_ratio": sample_ratio,
                })
                rows_written += 1

    if rows_written == 0:
        raise RuntimeError("No TAR blobs found")

    if log_cb:
        log_cb(
            f"Wrote TAR selection CSV: {csv_path} "
            f"({preselected_total}/{rows_written} TARs preselected across {target_count} target(s), "
            f"within-TAR sample {sample_ratio:.2%})"
        )
