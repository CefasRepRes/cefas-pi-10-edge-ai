#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
os.environ.pop("VIPS_INFO", None)
os.environ.pop("VIPS_TRACE", None)
os.environ.pop("G_MESSAGES_DEBUG", None)
import pyvips
from azure.core.exceptions import ResourceNotFoundError

EDGE_AI_DIR = Path(__file__).resolve().parent
if str(EDGE_AI_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_AI_DIR))
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from gui.application_validation.constants import TAR_PROGRESS_EVERY_IMAGES
from gui.application_validation.csv_utils import parse_sample_ratio
from gui.application_validation.model_utils import build_inference_transform, build_model_from_artifact, idx_to_label_fn
from gui.application_validation.tar_streaming import iter_tar_image_bytes_from_blob, per_tar_paths, safe_relative_stem_from_tar_name
import gps
from prediction_records import build_prediction_record, extract_gps_payload

def is_self_hosted_github_runner() -> bool:
    github_actions = os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"
    runner_environment = os.environ.get("RUNNER_ENVIRONMENT", "").strip().lower()
    return github_actions and runner_environment == "self-hosted"
    
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def sanitize_slug(value: Optional[str], *, default: str = "run") -> str:
    candidate = (value or default).strip()
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    return candidate or default

def parse_bool(value: Optional[str], *, default: bool = True) -> bool:
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default

def day_tokens(day: str):
    day = (day or "").strip()
    if not day:
        return []
    return [day, day.replace("-", ""), day.replace("-", "_")]

def tar_matches_day(tar_name: str, day: str) -> bool:
    tokens = day_tokens(day)
    if not tokens:
        return True
    name = Path(tar_name).name
    parts = Path(tar_name).parts
    return any(token in name or token in tar_name or token in parts for token in tokens)

def _default_parallel_stream_count(artifact) -> int:
    """Return the default number of concurrent TAR streams for the given model artifact.

    Mirrors the logic in gui/application_validation/workers.py so the runner
    uses the same concurrency heuristic as the GUI inference tab.
    """
    arch = str(artifact.get("arch", "resnet18") if artifact is not None else "resnet18").strip().lower().replace("-", "_")
    aliases = {
        "mobilenet": "mobilenet_v3_small",
        "mobilenetv2": "mobilenet_v2",
        "mobilenet_v2": "mobilenet_v2",
        "mobilenetv3": "mobilenet_v3_small",
        "mobilenetv3_small": "mobilenet_v3_small",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "mobilenetv3_large": "mobilenet_v3_large",
        "mobilenet_v3_large": "mobilenet_v3_large",
    }
    arch = aliases.get(arch, arch)
    return 12 if arch == "mobilenet_v3_small" else 2


def build_blob_service_client(account_url: str) -> BlobServiceClient:
    account_url = (account_url or "").strip().rstrip("/")
    if not account_url:
        raise RuntimeError("Missing Azure Blob Storage account URL")
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    return BlobServiceClient(account_url=account_url, credential=credential)

def list_tar_blobs(container_client, prefix: str):
    names = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        name = getattr(blob, "name", None) or (blob.get("name") if isinstance(blob, dict) else None)
        if not name:
            continue
        lowered = name.lower()
        if lowered.endswith((".tar", ".tar.gz", ".tgz")):
            names.append(name)
    names.sort()
    return names

def candidate_day_prefixes(target_prefix: str, target_day: str):
    target_prefix = (target_prefix or "").strip("/")
    target_day = (target_day or "").strip().strip("/")
    prefixes = []
    if target_day:
        if target_prefix:
            prefixes.append(f"{target_prefix}/{target_day}")
            prefixes.append(f"{target_prefix}/{target_day}/")
        else:
            prefixes.append(target_day)
            prefixes.append(f"{target_day}/")
    if target_prefix:
        prefixes.append(target_prefix)
    else:
        prefixes.append("")
    seen = set()
    unique = []
    for prefix in prefixes:
        if prefix not in seen:
            seen.add(prefix)
            unique.append(prefix)
    return unique

def list_tar_blobs_for_day(container_client, target_prefix: str, target_day: str):
    if not target_day:
        return list_tar_blobs(container_client, target_prefix or "")
    first_error = None
    for prefix in candidate_day_prefixes(target_prefix, target_day):
        try:
            names = list_tar_blobs(container_client, prefix)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
        names = [name for name in names if tar_matches_day(name, target_day)]
        if names:
            logging.info("Found %s TAR archive(s) for %s using blob prefix '%s'", len(names), target_day, prefix)
            return names
        logging.info("No TAR archives for %s using blob prefix '%s'", target_day, prefix)
    if first_error is not None:
        raise first_error
    return []

def upload_json_blob(blob_service_client: BlobServiceClient, container_name: str, blob_name: str, local_path: Path):
    container_client = blob_service_client.get_container_client(container_name)
    with local_path.open("rb") as fh:
        container_client.upload_blob(
            blob_name,
            fh,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )

def blob_exists(container_client, blob_name: str) -> bool:
    try:
        return bool(container_client.get_blob_client(blob_name).exists())
    except ResourceNotFoundError:
        return False

def build_output_blob_name(output_prefix: str, output_slug: str, relative_path: Path | str) -> str:
    prefix = (output_prefix or "runs").strip("/")
    slug = sanitize_slug(output_slug or "run")
    relative = str(relative_path).replace("\\", "/")
    return "/".join([prefix, slug, relative.lstrip("/")])

def expected_per_tar_output_blobs(run_dir: Path, tar_name: str, output_prefix: str, output_slug: str):
    _, pred_path_str, summ_path_str = per_tar_paths(str(run_dir), tar_name)
    pred_path = Path(pred_path_str).resolve()
    summ_path = Path(summ_path_str).resolve()
    relative_pred_path = pred_path.relative_to(run_dir) if pred_path.is_relative_to(run_dir) else Path(os.path.relpath(pred_path, str(run_dir)))
    relative_summ_path = summ_path.relative_to(run_dir) if summ_path.is_relative_to(run_dir) else Path(os.path.relpath(summ_path, str(run_dir)))
    return (
        build_output_blob_name(output_prefix, output_slug, relative_pred_path),
        build_output_blob_name(output_prefix, output_slug, relative_summ_path),
    )

def resolve_blob_service_client(
    client: Optional[BlobServiceClient],
    *,
    account_url: Optional[str] = None,
    required: bool = True,
) -> Optional[BlobServiceClient]:
    if client is not None:
        return client
    if not required:
        return None
    resolved_account_url = (account_url or os.environ.get("AZURE_STORAGE_ACCOUNT_URL", "")).strip().rstrip("/")
    if not resolved_account_url:
        raise RuntimeError("Missing Azure Blob Storage account URL")
    return build_blob_service_client(resolved_account_url)

def infer_on_tar_archive(
    *,
    container: str,
    tar_name: str,
    prefix: str,
    sample_ratio: float,
    model: torch.nn.Module,
    tfm,
    idx_to_label,
    device: torch.device,
    model_blob: str,
    model_runid: Optional[str],
    run_dir: Path,
    source_blob_service_client: Optional[BlobServiceClient],
    output_blob_service_client: Optional[BlobServiceClient],
    output_container: str,
    output_prefix: str,
    output_slug: str,
    log_cb,
):
    run_dir = Path(run_dir).resolve()
    out_dir, pred_path_str, summ_path_str = per_tar_paths(str(run_dir), tar_name)
    relative_out_dir = os.path.relpath(out_dir, str(run_dir))
    pred_path = Path(pred_path_str).resolve()
    summ_path = Path(summ_path_str).resolve()
    log_cb(f"[{container}/{tar_name}] Processing TAR (sample {sample_ratio:.2%})")
    tar_predictions = []
    tar_class_counts = Counter()
    batch_size = 256

    imgs = []
    valid_members = []
    readable_count = 0
    inferred_count = 0
    tar_gps_payload = None
    gps_extracted = False
    gps_attached = False
    import time
    
    THROUGHPUT_LOG_SECS = 60
    
    tar_start_ts = time.monotonic()
    last_progress_ts = tar_start_ts
    last_progress_count = 0
    sampling_method = "full_tar_stream_all_images"
    sampling_bias_note = None
    max_stream_bytes = None
    error_message = None
    if sample_ratio <= 0.0:
        sampling_method = "none_requested_zero_ratio"
        log_cb("    Sampling ratio is 0.00%; skipping image streaming/inference for this TAR")
    elif 0.0 < sample_ratio < 1.0:
        sampling_method = "sequential_tar_prefix_bytes"
        sampling_bias_note = "bias_toward_earlier_tar_members"
    def run_streamed_batch(batch_imgs, batch_members):
        nonlocal inferred_count, last_progress_ts, last_progress_count, gps_attached
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device, non_blocking=True)
        try:
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                confs, pred_idxs = torch.max(probs, dim=1)
            for member_name, conf, pred_idx in zip(batch_members, confs, pred_idxs):
                label = idx_to_label(int(pred_idx.item()))
                conf = float(conf.item())
                tar_class_counts[label] += 1
                gps_payload = tar_gps_payload if not gps_attached else None
                tar_predictions.append(
                    build_prediction_record(
                        container=container,
                        tar_name=tar_name,
                        member_name=member_name,
                        predicted_label=label,
                        confidence=conf,
                        image_bytes=None,
                        gps_module=None,
                        gps_data=gps_payload,
                    )
                )
                gps_attached = True
            inferred_count += len(batch_members)

            now = time.monotonic()
            
            if last_progress_count == 0:
                log_cb(f"    Started inference on {tar_name} (batch_size={batch_size})")
                last_progress_ts = now
                last_progress_count = inferred_count
            elif now - last_progress_ts >= THROUGHPUT_LOG_SECS:
                interval_images = inferred_count - last_progress_count
                interval_seconds = now - last_progress_ts
                ips = interval_images / interval_seconds if interval_seconds > 0 else 0.0
                elapsed = now - tar_start_ts
                overall_ips = inferred_count / elapsed if elapsed > 0 else 0.0
                log_cb(f"    {tar_name} | {inferred_count:,} images processed | {ips:,.0f} img/s recent | {overall_ips:,.0f} img/s overall")
                last_progress_ts = now
                last_progress_count = inferred_count

        
        except Exception as exc:
            log_cb(f"ERROR during batched inference: {exc}")
    resolved_source_blob_service_client = None
    resolved_output_blob_service_client = None
    try:
        resolved_source_blob_service_client = resolve_blob_service_client(source_blob_service_client)
        if output_container:
            resolved_output_blob_service_client = resolve_blob_service_client(output_blob_service_client, required=False)
        container_client = resolved_source_blob_service_client.get_container_client(container)
        blob_client = container_client.get_blob_client(tar_name)
        if sample_ratio > 0.0 and sample_ratio < 1.0:
            try:
                blob_size = int(getattr(blob_client.get_blob_properties(), "size", 0))
            except Exception as exc:
                blob_size = 0
                log_cb(f"    WARNING: could not read blob size for streaming cap ({type(exc).__name__}): {exc}")
            if blob_size > 0:
                max_stream_bytes = max(1, int(blob_size * sample_ratio))
                log_cb(f"    Sampling strategy: sequential TAR prefix stream at {sample_ratio:.2%} of blob bytes (~{max_stream_bytes:,}/{blob_size:,})")
            else:
                log_cb("    Sampling strategy fallback: blob size unavailable, streaming full TAR")
        for member_name, image_bytes in iter_tar_image_bytes_from_blob(
            blob_client,
            tar_name,
            log_cb=log_cb,
            max_bytes=max_stream_bytes,
        ):
            if not gps_extracted:
                tar_gps_payload = extract_gps_payload(image_bytes, gps)
                gps_extracted = True
            try:
                img = tfm(image_bytes)
                readable_count += 1
                imgs.append(img)
                valid_members.append(member_name)
            except pyvips.Error as exc:
                log_cb(f"Skipping unreadable image in TAR with pyvips: {member_name} - {exc}")
                continue
            except Exception as exc:
                log_cb(f"Skipping image in TAR ({type(exc).__name__}): {member_name} - {exc}")
                continue
            if len(imgs) >= batch_size:
                run_streamed_batch(imgs, valid_members)
                imgs = []
                valid_members = []
        run_streamed_batch(imgs, valid_members)
        if readable_count == 0:
            log_cb(f"WARNING: no readable images found in {tar_name}")
        else:
            elapsed = time.monotonic() - tar_start_ts
            overall_ips = inferred_count / elapsed if elapsed > 0 else 0.0
            log_cb(f"    Finished TAR: {tar_name} | {inferred_count:,}/{readable_count:,} readable images inferred | {overall_ips:,.0f} img/s overall")
    except Exception as exc:
        error_message = str(exc)
        log_cb(f"ERROR processing TAR {container}/{tar_name}: {exc}")
    pred_path.write_text(json.dumps(tar_predictions, indent=2), encoding="utf-8")
    summ_path.write_text(
        json.dumps(
            {
                "generated_utc": utc_now_iso(),
                "container": container,
                "source_tar_blob": tar_name,
                "selected_prefix": prefix,
                "per_tar_output_dir": relative_out_dir,
                "model_blob": model_blob,
                "model_runid": model_runid,
                "image_count": len(tar_predictions),
                "sample_ratio_requested": sample_ratio,
                "sample_percent_requested": sample_ratio * 100.0,
                "readable_image_count": readable_count,
                "sampled_image_count": inferred_count,
                "sampling_method": sampling_method,
                "sampling_bias_note": sampling_bias_note,
                "stream_max_bytes": max_stream_bytes,
                "class_counts": dict(tar_class_counts),
                "temperature": 1.0,
                "decoder": "pyvips",
                "within_tar_progress_every_images": TAR_PROGRESS_EVERY_IMAGES,
                "throughput_log_secs": THROUGHPUT_LOG_SECS,
                "error_message": error_message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if resolved_output_blob_service_client is not None:
        relative_pred_path = pred_path.relative_to(run_dir) if pred_path.is_relative_to(run_dir) else Path(os.path.relpath(pred_path, str(run_dir)))
        relative_summ_path = summ_path.relative_to(run_dir) if summ_path.is_relative_to(run_dir) else Path(os.path.relpath(summ_path, str(run_dir)))
        pred_blob_name = build_output_blob_name(output_prefix, output_slug, relative_pred_path)
        summ_blob_name = build_output_blob_name(output_prefix, output_slug, relative_summ_path)
        upload_json_blob(
            resolved_output_blob_service_client,
            output_container,
            pred_blob_name,
            pred_path,
        )
        upload_json_blob(
            resolved_output_blob_service_client,
            output_container,
            summ_blob_name,
            summ_path,
        )
        log_cb(f"Uploaded per-TAR outputs to {output_container}/{pred_blob_name} and {output_container}/{summ_blob_name}")
    return {
        "container": container,
        "tar_name": tar_name,
        "prefix": prefix,
        "sample_ratio": sample_ratio,
        "tar_class_counts": tar_class_counts,
        "error_message": error_message,
        "skipped_existing": False,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference over TAR archives using the application_validation helpers")
    parser.add_argument("--account-url", default=os.environ.get("AZURE_STORAGE_ACCOUNT_URL", ""))
    parser.add_argument("--models-container", default="trainedmodels")
    parser.add_argument("--model-blob", default=os.environ.get("MODEL_BLOB", ""))
    parser.add_argument("--target-container", default=os.environ.get("TARGET_CONTAINER", ""))
    parser.add_argument("--target-prefix", default=os.environ.get("TARGET_PREFIX", ""))
    parser.add_argument("--day", default=os.environ.get("TARGET_DAY", ""))
    parser.add_argument("--run-key", default=os.environ.get("RUN_KEY", ""))
    parser.add_argument("--sample-ratio", default=os.environ.get("SAMPLE_RATIO", "1.0"))
    parser.add_argument("--output-container", default=os.environ.get("OUTPUT_CONTAINER", "ml-prediction-results"))
    parser.add_argument("--output-prefix", default=os.environ.get("OUTPUT_PREFIX", "runs"))
    parser.add_argument("--settings-blob", default=os.environ.get("SETTINGS_BLOB", ""))
    parser.add_argument("--resume-existing", default=os.environ.get("RESUME_EXISTING", "true"))
    parser.add_argument("--allow-empty-day", default=os.environ.get("ALLOW_EMPTY_DAY", "true"))
    parser.add_argument("--no-resume", action="store_true", help="Process TARs even if output JSON blobs already exist")
    parser.add_argument(
        "--parallel-streams",
        type=int,
        default=int(os.environ.get("PARALLEL_STREAMS", "0")),
        help="Number of TAR archives to process concurrently. 0 (default) auto-detects based on model architecture, matching the GUI heuristic.",
    )
    args = parser.parse_args()
    if not args.model_blob:
        parser.error("--model-blob is required (or set MODEL_BLOB)")
    if not args.target_container:
        parser.error("--target-container is required (or set TARGET_CONTAINER)")
    return args

def write_and_upload_empty_day_summary(
    *,
    args,
    account_url: str,
    run_dir: Path,
    bsc: BlobServiceClient,
    run_key: str,
    target_day: str,
    sample_ratio: float,
    resume_existing: bool,
    allow_empty_day: bool,
    message: str,
):
    output_prefix = args.output_prefix.strip("/") or "runs"
    output_slug = sanitize_slug(run_key or args.model_blob or "run")
    output_blob_service_client = None
    if args.output_container:
        output_account_url = os.environ.get("OUTPUT_AZURE_STORAGE_ACCOUNT_URL", "").strip()
        output_blob_service_client = build_blob_service_client(output_account_url) if output_account_url else bsc
    run_summary_path = run_dir / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "generated_utc": utc_now_iso(),
                "account_url": account_url,
                "target_container": args.target_container,
                "target_prefix": args.target_prefix or "",
                "target_day": target_day,
                "model_blob": args.model_blob,
                "tar_archives_discovered": 0,
                "tar_archives_processed": 0,
                "tar_archives_skipped_existing": 0,
                "selected_tars": [],
                "skipped_existing_tars": [],
                "class_counts": {},
                "sample_ratio_requested": sample_ratio,
                "sample_percent_requested": sample_ratio * 100.0,
                "resume_existing": resume_existing,
                "allow_empty_day": allow_empty_day,
                "empty_day_message": message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Wrote empty-day run summary: %s", run_summary_path)
    if args.output_container and output_blob_service_client is not None:
        run_summary_blob_name = build_output_blob_name(output_prefix, output_slug, f"run_summary__{sanitize_slug(target_day or 'empty-day')}.json")
        upload_json_blob(output_blob_service_client, args.output_container, run_summary_blob_name, run_summary_path)
        logging.info("Uploaded empty-day run summary to %s/%s", args.output_container, run_summary_blob_name)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("VIPS").setLevel(logging.WARNING)
    logging.getLogger("pyvips").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    args = parse_args()
    account_url = args.account_url
    if not account_url:
        raise RuntimeError("Missing Azure storage account URL")
    sample_ratio = parse_sample_ratio(args.sample_ratio)
    run_key = sanitize_slug(args.run_key or args.model_blob or "run")
    target_day = (args.day or "").strip()
    resume_existing = False if args.no_resume else parse_bool(args.resume_existing, default=True)
    allow_empty_day = parse_bool(args.allow_empty_day, default=True)
    run_dir = Path.cwd() / "workflow_outputs" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    bsc = build_blob_service_client(account_url)
    models_container = bsc.get_container_client(args.models_container)
    model_bytes = models_container.get_blob_client(args.model_blob).download_blob().readall()
    artifact = torch.load(BytesIO(model_bytes), map_location="cpu", weights_only=False)
    model = build_model_from_artifact(artifact)
    model_state = artifact.get("model_state_dict")
    if not model_state:
        raise RuntimeError("Model artifact missing 'model_state_dict'")
    model.load_state_dict(model_state)
    model.eval()
    logging.info("Torch version: %s", torch.__version__)
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    runner_environment = os.environ.get("RUNNER_ENVIRONMENT", "").strip() or "unknown"
    runner_name = os.environ.get("RUNNER_NAME", "").strip() or "unknown"
    runner_os = os.environ.get("RUNNER_OS", "").strip() or "unknown"
    self_hosted_runner = is_self_hosted_github_runner()
    logging.info("CUDA available: %s", cuda_available)
    logging.info("CUDA device count: %s", cuda_device_count)
    logging.info("GitHub runner environment: %s", runner_environment)
    logging.info("GitHub runner name: %s", runner_name)
    logging.info("GitHub runner OS: %s", runner_os)
    logging.info("Detected self-hosted GitHub runner: %s", self_hosted_runner)
    if cuda_available:
        logging.info("CUDA device 0: %s", torch.cuda.get_device_name(0))
        logging.info("CUDA version from torch: %s", torch.version.cuda)
    elif self_hosted_runner:
        raise RuntimeError(
            "CUDA is not available on a self-hosted GitHub runner. Refusing to continue on CPU because this runner is expected to provide GPU acceleration. "
            "Check nvidia-smi, NVIDIA drivers, CUDA runtime visibility, and whether the self-hosted runner service can see /dev/nvidia devices."
        )
    else:
        logging.warning(
            "CUDA is not available, but this runner is not detected as self-hosted. Continuing on CPU. "
            "This is expected for runs-on: ubuntu-latest unless a GPU-enabled runner is used."
        )
    device = torch.device("cuda:0" if cuda_available else "cpu")
    logging.info("Using inference device: %s", device)
    model.to(device)
    tfm = build_inference_transform(artifact)
    idx_to_label = idx_to_label_fn(artifact.get("idx_to_class") or {})
    runid = None
    model_name = Path(args.model_blob).name
    if model_name.startswith("model") and model_name.endswith(".pt"):
        runid = model_name[len("model"):-3]
    settings_blob = args.settings_blob or (f"modeltrainsettings{runid}.json" if runid else "")
    if settings_blob:
        try:
            settings_bytes = models_container.get_blob_client(settings_blob).download_blob().readall()
            settings_path = run_dir / settings_blob
            settings_path.write_bytes(settings_bytes)
            logging.info("Downloaded settings blob: %s", settings_blob)
        except Exception as exc:
            logging.warning("Could not fetch settings blob '%s': %s", settings_blob, exc)
    target_container_client = bsc.get_container_client(args.target_container)
    tar_names = list_tar_blobs_for_day(target_container_client, args.target_prefix or "", target_day)
    if not tar_names:
        day_hint = f" for day '{target_day}'" if target_day else ""
        message = f"No TAR blobs found{day_hint} in container '{args.target_container}' with prefix '{args.target_prefix}'"
        if target_day and allow_empty_day:
            logging.warning("%s; treating this day as complete because allow_empty_day is true", message)
            write_and_upload_empty_day_summary(
                args=args,
                account_url=account_url,
                run_dir=run_dir,
                bsc=bsc,
                run_key=run_key,
                target_day=target_day,
                sample_ratio=sample_ratio,
                resume_existing=resume_existing,
                allow_empty_day=allow_empty_day,
                message=message,
            )
            return
        raise RuntimeError(message)
    output_prefix = args.output_prefix.strip("/")
    if not output_prefix:
        output_prefix = "runs"
    output_slug = sanitize_slug(run_key or args.model_blob or "run")
    output_blob_service_client = None
    output_container_client = None
    if args.output_container:
        output_account_url = os.environ.get("OUTPUT_AZURE_STORAGE_ACCOUNT_URL", "").strip()
        output_blob_service_client = build_blob_service_client(output_account_url) if output_account_url else bsc
        output_container_client = output_blob_service_client.get_container_client(args.output_container)
    logging.info("Processing %s TAR archive(s) from %s/%s%s", len(tar_names), args.target_container, args.target_prefix or "", f" for day {target_day}" if target_day else "")
    logging.info("Resume existing outputs: %s", resume_existing)
    selected_tars = []
    skipped_existing = []
    class_counts = Counter()
    tars_to_process = []
    for tar_name in tar_names:
        if resume_existing and output_container_client is not None:
            pred_blob_name, summ_blob_name = expected_per_tar_output_blobs(run_dir, tar_name, output_prefix, output_slug)
            if blob_exists(output_container_client, pred_blob_name) and blob_exists(output_container_client, summ_blob_name):
                logging.info("Skipping existing TAR output: %s", tar_name)
                skipped_existing.append(
                    {
                        "container": args.target_container,
                        "source_tar_blob": tar_name,
                        "selected_prefix": args.target_prefix or "",
                        "sample_ratio": sample_ratio,
                        "prediction_blob": pred_blob_name,
                        "summary_blob": summ_blob_name,
                        "skipped_existing": True,
                    }
                )
                continue
        tars_to_process.append(tar_name)
    if args.parallel_streams and args.parallel_streams > 0:
        parallel_streams = args.parallel_streams
    else:
        parallel_streams = _default_parallel_stream_count(artifact)
    parallel_streams = max(1, min(len(tars_to_process), parallel_streams)) if tars_to_process else 1
    logging.info("Using %d parallel TAR stream(s) for inference", parallel_streams)

    def _collect_result(future, pending):
        pending.discard(future)
        try:
            result = future.result()
        except Exception as exc:
            logging.error("TAR inference task raised an unexpected exception: %s", exc)
            return
        selected_tars.append(
            {
                "container": args.target_container,
                "source_tar_blob": result["tar_name"],
                "selected_prefix": args.target_prefix or "",
                "sample_ratio": sample_ratio,
                "per_tar_output_dir": os.path.join("per_tar", safe_relative_stem_from_tar_name(result["tar_name"])),
                "skipped_existing": False,
                "error_message": result.get("error_message"),
            }
        )
        class_counts.update(result["tar_class_counts"])

    pending_futures = set()
    with ThreadPoolExecutor(max_workers=parallel_streams) as executor:
        for tar_name in tars_to_process:
            while len(pending_futures) >= parallel_streams:
                done, _ = wait(pending_futures, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    _collect_result(future, pending_futures)
            pending_futures.add(
                executor.submit(
                    infer_on_tar_archive,
                    container=args.target_container,
                    tar_name=tar_name,
                    prefix=args.target_prefix or "",
                    sample_ratio=sample_ratio,
                    model=model,
                    tfm=tfm,
                    idx_to_label=idx_to_label,
                    device=device,
                    model_blob=args.model_blob,
                    model_runid=runid,
                    run_dir=run_dir,
                    source_blob_service_client=bsc,
                    output_blob_service_client=output_blob_service_client,
                    output_container=args.output_container,
                    output_prefix=output_prefix,
                    output_slug=output_slug,
                    log_cb=logging.info,
                )
            )
        while pending_futures:
            done, _ = wait(pending_futures, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                _collect_result(future, pending_futures)
    run_summary_path = run_dir / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "generated_utc": utc_now_iso(),
                "account_url": account_url,
                "target_container": args.target_container,
                "target_prefix": args.target_prefix or "",
                "target_day": target_day,
                "model_blob": args.model_blob,
                "model_runid": runid,
                "tar_archives_discovered": len(tar_names),
                "tar_archives_processed": len(selected_tars),
                "tar_archives_skipped_existing": len(skipped_existing),
                "selected_tars": selected_tars,
                "skipped_existing_tars": skipped_existing,
                "class_counts": dict(class_counts),
                "sample_ratio_requested": sample_ratio,
                "sample_percent_requested": sample_ratio * 100.0,
                "parallel_streams": parallel_streams,
                "resume_existing": resume_existing,
                "allow_empty_day": allow_empty_day,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Wrote run summary: %s", run_summary_path)
    output_container = args.output_container
    run_summary_blob_name = build_output_blob_name(output_prefix, output_slug, f"run_summary__{sanitize_slug(target_day or settings_blob or args.model_blob or 'run-summary')}.json")
    if output_container:
        if output_blob_service_client is None:
            raise RuntimeError("Output blob service client is not configured")
        upload_json_blob(output_blob_service_client, output_container, run_summary_blob_name, run_summary_path)
        logging.info("Uploaded run summary to %s/%s", output_container, run_summary_blob_name)

if __name__ == "__main__":
    main()
