import json
import os
import shutil
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path

import pyvips
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

from .azure_utils import get_blob_service_client, utc_now_iso, extract_runid_from_model_blob
from .blob_listing import quick_count_blobs
from .constants import MAX_BLOBS_SAFETY, TAR_PROGRESS_EVERY_IMAGES
from .csv_utils import DEFAULT_TAR_SAMPLE_RATIO, parse_sample_ratio
from .model_utils import build_inference_transform, build_model_from_artifact, idx_to_label_fn
from .output_upload import (
    DEFAULT_OUTPUT_CONTAINER,
    DEFAULT_OUTPUT_PREFIX,
    build_gui_output_slug,
    build_output_blob_name,
    upload_json_blob,
)
from .tar_streaming import (
    build_tar_stream_segments,
    iter_tar_image_bytes_from_blob,
    iter_tar_members_from_blob,
    per_tar_paths,
    safe_file_id_from_tar_name,
    safe_relative_stem_from_tar_name,
)


def _normalize_selected_tar(entry):
    if isinstance(entry, dict):
        container = entry.get("container")
        tar_name = entry.get("tar_blob") or entry.get("source_tar_blob")
        prefix = entry.get("prefix") or entry.get("selected_prefix") or ""
        sample_ratio = entry.get("sample_ratio", DEFAULT_TAR_SAMPLE_RATIO)
    elif len(entry) >= 4:
        container, tar_name, prefix, sample_ratio = entry[:4]
    elif len(entry) == 3:
        container, tar_name, prefix = entry
        sample_ratio = DEFAULT_TAR_SAMPLE_RATIO
    else:
        raise RuntimeError(f"Unexpected TAR selection row: {entry!r}")

    return container, tar_name, prefix or "", parse_sample_ratio(sample_ratio)


def _default_parallel_stream_count(artifact):
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
    return 16 if arch == "mobilenet_v3_small" else 2


class ContainerListWorker(QThread):
    log = Signal(str)
    containers_ready = Signal(list)

    def __init__(self, account_url: str, parent=None):
        super().__init__(parent)
        self.account_url = account_url

    def run(self):
        try:
            bsc = get_blob_service_client(self.account_url, log_cb=self.log.emit)
            names = []
            for c in bsc.list_containers():
                try:
                    names.append(c["name"] if isinstance(c, dict) else c.name)
                except Exception:
                    pass
            self.containers_ready.emit(sorted(set(names)))
            self.log.emit(f"Loaded {len(names)} containers.")
        except Exception as e:
            self.log.emit(f"ERROR listing containers: {e}")


class ModelListWorker(QThread):
    log = Signal(str)
    models_ready = Signal(list)

    def __init__(self, account_url: str, models_container: str = "trainedmodels", parent=None):
        super().__init__(parent)
        self.account_url = account_url
        self.models_container = models_container

    def run(self):
        try:
            bsc = get_blob_service_client(self.account_url, log_cb=self.log.emit)
            cc = bsc.get_container_client(self.models_container)
            models = []
            for blob in cc.list_blobs():
                name = getattr(blob, "name", None) or (blob.get("name") if isinstance(blob, dict) else None)
                if name and name.lower().endswith(".pt") and os.path.basename(name).lower().startswith("model"):
                    models.append(name)
            self.models_ready.emit(sorted(models))
            self.log.emit(f"Loaded {len(models)} models from container '{self.models_container}'.")
        except Exception as e:
            self.log.emit(f"ERROR listing models: {e}")


class ApplicationWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        account_url: str,
        selected_tars: list,
        targets: list,
        model_blob: str,
        output_root: str,
        parent=None,
    ):
        super().__init__(parent)
        self.account_url = account_url
        self.targets = targets
        self.model_blob = model_blob
        self.output_root = output_root
        self.selected_tars = selected_tars

    def run(self):
        try:
            if not self.targets:
                raise RuntimeError("No targets selected")
            if not self.model_blob:
                raise RuntimeError("No model selected")

            runid = extract_runid_from_model_blob(self.model_blob)
            stamp = utc_now_iso().replace(":", "-")
            run_dir = os.path.join(self.output_root, f"application_{stamp}_model{runid or 'UNKNOWN'}")
            os.makedirs(run_dir, exist_ok=True)

            output_container = os.environ.get("GUI_OUTPUT_CONTAINER", DEFAULT_OUTPUT_CONTAINER)
            output_prefix = os.environ.get("GUI_OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX)
            output_slug = build_gui_output_slug(stamp)
            bsc = get_blob_service_client(self.account_url, log_cb=self.log.emit)

            for t in self.targets:
                cc = bsc.get_container_client(t["container"])
                n = quick_count_blobs(cc, t.get("prefix") or "", limit=MAX_BLOBS_SAFETY + 1)
                if n > MAX_BLOBS_SAFETY:
                    raise RuntimeError("over 100000 blobs found, please zip your tenbins up")

            mcc = bsc.get_container_client("trainedmodels")
            self.log.emit(f"Downloading model artifact: trainedmodels/{self.model_blob}")
            artifact = torch.load(BytesIO(mcc.get_blob_client(self.model_blob).download_blob().readall()), map_location="cpu")

            settings_path = None
            if runid:
                settings_blob = f"modeltrainsettings{runid}.json"
                try:
                    self.log.emit(f"Downloading {settings_blob}")
                    settings_bytes = mcc.get_blob_client(settings_blob).download_blob().readall()
                    settings_path = os.path.join(run_dir, settings_blob)
                    with open(settings_path, "wb") as f:
                        f.write(settings_bytes)
                except Exception as e:
                    self.log.emit(f"WARNING: could not fetch {settings_blob}: {e}")

            model = build_model_from_artifact(artifact)
            state = artifact.get("model_state_dict")
            if not state:
                raise RuntimeError("Model artifact missing 'model_state_dict'")
            model.load_state_dict(state)
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            tfm = build_inference_transform(artifact)

            cal = artifact.get("calibration") or {}
            temperature = float(cal.get("temperature", 1.0)) if cal.get("method") == "temperature_scaling" else 1.0
            temperature = temperature if temperature > 0 else 1.0
            idx_to_label = idx_to_label_fn(artifact.get("idx_to_class") or {})

            if not self.selected_tars:
                raise RuntimeError("No TARs selected (CSV empty or not loaded)")

            tar_blobs = [_normalize_selected_tar(row) for row in self.selected_tars]
            total = len(tar_blobs)
            unique_sample_ratios = sorted({sample_ratio for _, _, _, sample_ratio in tar_blobs})
            if len(unique_sample_ratios) == 1:
                self.log.emit(
                    f"Running application for {total} TAR(s) selected via CSV "
                    f"(within-TAR sample {unique_sample_ratios[0]:.2%})"
                )
            else:
                self.log.emit(
                    f"Running application for {total} TAR(s) selected via CSV "
                    f"(mixed within-TAR samples {unique_sample_ratios[0]:.2%} to {unique_sample_ratios[-1]:.2%})"
                )

            per_tar_root = os.path.join(run_dir, "per_tar")
            os.makedirs(per_tar_root, exist_ok=True)
            class_counts = Counter()
            parallel_streams = max(1, min(total, _default_parallel_stream_count(artifact)))
            self.log.emit(f"Using {parallel_streams} parallel TAR stream(s) for inference")
            run_real_dir = os.path.realpath(run_dir)

            def upload_output_json(local_path):
                if not local_path or not bsc:
                    return
                try:
                    local_real_path = os.path.realpath(local_path)
                    if not Path(local_real_path).is_relative_to(Path(run_real_dir)):
                        self.log.emit(
                            f"WARNING: skipped Azure upload for {os.path.basename(local_path)} because it is outside the run directory"
                        )
                        return
                    relative_path = os.path.relpath(local_real_path, run_real_dir)
                    blob_name = build_output_blob_name(output_prefix, output_slug, relative_path)
                    upload_json_blob(bsc, output_container, blob_name, local_real_path)
                    self.log.emit(
                        f"Uploaded {os.path.basename(local_path)} to {output_container}/{blob_name}"
                    )
                except Exception as exc:
                    self.log.emit(
                        f"WARNING: failed to upload {os.path.basename(local_path)} to {output_container}: {exc}"
                    )

            def upload_result_outputs(result):
                upload_output_json(result.get("pred_path"))
                upload_output_json(result.get("summ_path"))

            def process_tar_stream(index, container, tar_name, prefix, sample_ratio):
                self.log.emit(
                    f"[{index + 1}/{total}] Processing TAR {container}/{tar_name} "
                    f"(within-TAR sample {sample_ratio:.2%})"
                )
                cc = bsc.get_container_client(container)
                bc = cc.get_blob_client(tar_name)
                tar_predictions = []
                tar_class_counts = Counter()
                batch_size = 256
                progress_every = TAR_PROGRESS_EVERY_IMAGES
                imgs = []
                valid_members = []
                readable_count = 0
                inferred_count = 0
                next_progress_log = progress_every
                sampling_method = "full_tar_stream_all_images"
                sampling_bias_note = None
                max_stream_bytes = None
                segment_plan = []
                stream_images = True
                error_message = None

                if sample_ratio <= 0.0:
                    stream_images = False
                    sampling_method = "none_requested_zero_ratio"
                    self.log.emit("    Sampling ratio is 0.00%; skipping image streaming/inference for this TAR")
                elif 0.0 < sample_ratio < 1.0:
                    sampling_method = "five_segment_tar_stream"
                    sampling_bias_note = "spans_five_evenly_spaced_tar_segments"
                    try:
                        blob_size = int(getattr(bc.get_blob_properties(), "size", 0))
                    except Exception as e:
                        blob_size = 0
                        self.log.emit(
                            f"    WARNING: could not read blob size for streaming cap ({type(e).__name__}): {e}"
                        )
                    if blob_size > 0:
                        max_stream_bytes = max(1, int(blob_size * sample_ratio))
                        segment_plan = build_tar_stream_segments(blob_size, max_stream_bytes, segment_count=5)
                        self.log.emit(
                            f"    Sampling strategy: five-position TAR stream at {sample_ratio:.2%} "
                            f"of blob bytes ({len(segment_plan)} segments, ~{max_stream_bytes:,}/{blob_size:,} bytes total)"
                        )
                    else:
                        self.log.emit(
                            "    Sampling strategy fallback: blob size unavailable, streaming full TAR"
                        )

                def run_streamed_batch(batch_imgs, batch_members):
                    nonlocal inferred_count, next_progress_log
                    if not batch_imgs:
                        return
                    x = torch.stack(batch_imgs).to(device, non_blocking=True)
                    try:
                        with torch.no_grad():
                            logits = model(x) / temperature
                            probs = F.softmax(logits, dim=1)
                            confs, pred_idxs = torch.max(probs, dim=1)
                        for member_name, conf, pred_idx in zip(batch_members, confs, pred_idxs):
                            label = idx_to_label(int(pred_idx.item()))
                            conf = float(conf.item())
                            tar_class_counts[label] += 1
                            tar_predictions.append({
                                "source": "blob_tar_streamed_pyvips",
                                "container": container,
                                "source_tar_blob": tar_name,
                                "tar_member": member_name,
                                "local_path": None,
                                "blob_member_uri": f"{container}/{tar_name}!/{member_name}",
                                "selected_prefix": prefix,
                                "tar_sample_ratio": sample_ratio,
                                "predicted_label": label,
                                "confidence": conf,
                                "model_blob": self.model_blob,
                                "model_runid": runid,
                            })
                        inferred_count += len(batch_members)
                        while inferred_count >= next_progress_log:
                            self.log.emit(f"    {next_progress_log:,} images done within TAR: {tar_name}")
                            next_progress_log += progress_every
                    except Exception as e:
                        self.log.emit(f"ERROR during batched inference: {e}")

                try:
                    if stream_images:
                        if segment_plan:
                            for segment_index, (offset, length) in enumerate(segment_plan, start=1):
                                if length <= 0:
                                    continue
                                self.log.emit(
                                    f"    Segment {segment_index}/{len(segment_plan)} from offset {offset:,} for {length:,} bytes"
                                )
                                for member_name, image_bytes in iter_tar_image_bytes_from_blob(
                                    bc,
                                    tar_name,
                                    log_cb=self.log.emit,
                                    max_bytes=length,
                                    offset=offset,
                                ):
                                    try:
                                        img = tfm(image_bytes)
                                        readable_count += 1
                                        imgs.append(img)
                                        valid_members.append(member_name)
                                    except pyvips.Error as e:
                                        self.log.emit(f"Skipping unreadable image in TAR with pyvips: {member_name} - {e}")
                                        continue
                                    except Exception as e:
                                        self.log.emit(f"Skipping image in TAR ({type(e).__name__}): {member_name} - {e}")
                                        continue

                                    if len(imgs) >= batch_size:
                                        run_streamed_batch(imgs, valid_members)
                                        imgs = []
                                        valid_members = []
                        else:
                            for member_name, image_bytes in iter_tar_image_bytes_from_blob(
                                bc,
                                tar_name,
                                log_cb=self.log.emit,
                                max_bytes=max_stream_bytes,
                            ):
                                try:
                                    img = tfm(image_bytes)
                                    readable_count += 1
                                    imgs.append(img)
                                    valid_members.append(member_name)
                                except pyvips.Error as e:
                                    self.log.emit(f"Skipping unreadable image in TAR with pyvips: {member_name} - {e}")
                                    continue
                                except Exception as e:
                                    self.log.emit(f"Skipping image in TAR ({type(e).__name__}): {member_name} - {e}")
                                    continue

                                if len(imgs) >= batch_size:
                                    run_streamed_batch(imgs, valid_members)
                                    imgs = []
                                    valid_members = []

                    run_streamed_batch(imgs, valid_members)

                    if readable_count == 0:
                        self.log.emit(f"WARNING: no readable images found in {tar_name}")
                    else:
                        self.log.emit(
                            f"    Finished TAR: {tar_name} - {inferred_count:,}/{readable_count:,} "
                            f"readable images inferred"
                        )
                except Exception as e:
                    error_message = str(e)
                    self.log.emit(f"ERROR processing TAR {container}/{tar_name}: {e}")

                out_dir, pred_path, summ_path = per_tar_paths(run_dir, tar_name)
                relative_out_dir = os.path.relpath(out_dir, run_dir)
                self.log.emit(f"    Writing per-TAR outputs to: {relative_out_dir}")
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(tar_predictions, f, indent=2)
                with open(summ_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "generated_utc": utc_now_iso(),
                        "container": container,
                        "source_tar_blob": tar_name,
                        "selected_prefix": prefix,
                        "per_tar_output_dir": relative_out_dir,
                        "model_blob": self.model_blob,
                        "model_runid": runid,
                        "image_count": len(tar_predictions),
                        "sample_ratio_requested": sample_ratio,
                        "sample_percent_requested": sample_ratio * 100.0,
                        "readable_image_count": readable_count,
                        "sampled_image_count": inferred_count,
                        "sampling_method": sampling_method,
                        "sampling_bias_note": sampling_bias_note,
                        "stream_max_bytes": max_stream_bytes,
                        "class_counts": dict(tar_class_counts),
                        "temperature": temperature,
                        "decoder": "pyvips",
                        "within_tar_progress_every_images": TAR_PROGRESS_EVERY_IMAGES,
                        "error_message": error_message,
                    }, f, indent=2)
                return {
                    "index": index,
                    "container": container,
                    "tar_name": tar_name,
                    "prefix": prefix,
                    "sample_ratio": sample_ratio,
                    "tar_class_counts": tar_class_counts,
                    "error_message": error_message,
                    "pred_path": pred_path,
                    "summ_path": summ_path,
                }

            completed_count = 0
            pending_futures = set()
            with ThreadPoolExecutor(max_workers=parallel_streams) as executor:
                for index, (container, tar_name, prefix, sample_ratio) in enumerate(tar_blobs):
                    if self.isInterruptionRequested():
                        return
                    while len(pending_futures) >= parallel_streams:
                        done, _ = wait(pending_futures, timeout=0.2, return_when=FIRST_COMPLETED)
                        for future in done:
                            result = future.result()
                            pending_futures.remove(future)
                            completed_count += 1
                            class_counts.update(result["tar_class_counts"])
                            upload_result_outputs(result)
                            self.progress.emit(int((completed_count / total) * 100))
                    pending_futures.add(
                        executor.submit(process_tar_stream, index, container, tar_name, prefix, sample_ratio)
                    )

                while pending_futures:
                    done, _ = wait(pending_futures, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        pending_futures.remove(future)
                        completed_count += 1
                        class_counts.update(result["tar_class_counts"])
                        upload_result_outputs(result)
                        self.progress.emit(int((completed_count / total) * 100))

            self.progress.emit(100)

            run_summary_path = os.path.join(run_dir, "run_summary.json")
            with open(run_summary_path, "w", encoding="utf-8") as f:
                json.dump({
                    "generated_utc": utc_now_iso(),
                    "account_url": self.account_url,
                    "targets": self.targets,
                    "model_blob": self.model_blob,
                    "model_runid": runid,
                    "temperature": temperature,
                    "tar_archives": total,
                    "selected_tars": [
                        {
                            "container": container,
                            "source_tar_blob": tar_name,
                            "selected_prefix": prefix,
                            "sample_ratio": sample_ratio,
                            "per_tar_output_dir": os.path.join("per_tar", safe_relative_stem_from_tar_name(tar_name)),
                        }
                        for container, tar_name, prefix, sample_ratio in tar_blobs
                    ],
                    "sampling_method": "five_segment_tar_stream",
                    "sampling_method_note": "applies when 0% < sample_ratio < 100%; otherwise full TAR or zero-ratio skip",
                    "class_counts": dict(class_counts),
                    "per_tar_root": "per_tar",
                    "decoder": "pyvips",
                    "within_tar_progress_every_images": TAR_PROGRESS_EVERY_IMAGES,
                    "associated_modeltrainsettings": os.path.basename(settings_path) if settings_path else None,
                }, f, indent=2)

            upload_output_json(run_summary_path)

            self.log.emit(f"Wrote outputs to: {run_dir}")
            self.finished_ok.emit({"run_dir": run_dir, "predictions_dir": per_tar_root, "run_summary": run_summary_path})
        except Exception as e:
            self.failed.emit(str(e))


class MoveByClassWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, account_url: str, predictions: list, output_run_dir: str = None, parent=None):
        super().__init__(parent)
        self.account_url = account_url
        self.predictions = predictions
        self.output_run_dir = output_run_dir

    @staticmethod
    def _unique_dest_path(dest_dir: str, filename: str):
        os.makedirs(dest_dir, exist_ok=True)
        base = os.path.basename(filename) or "image"
        stem, ext = os.path.splitext(base)
        candidate = os.path.join(dest_dir, base)
        n = 1
        while os.path.exists(candidate):
            candidate = os.path.join(dest_dir, f"{stem}_{n}{ext}")
            n += 1
        return candidate

    def run(self):
        try:
            preds = [p for p in (self.predictions or []) if p.get("predicted_label") not in (None, "ERROR")]
            if not preds:
                raise RuntimeError("No usable predictions to download")

            grouped = {}
            for p in preds:
                container = p.get("container")
                tar_name = p.get("source_tar_blob")
                member = p.get("tar_member")
                label = p.get("predicted_label")
                if not (container and tar_name and member and label):
                    continue
                grouped.setdefault((container, tar_name), {})[member] = p

            if not grouped:
                raise RuntimeError("No blob TAR/member references found in predictions")

            bsc = get_blob_service_client(self.account_url, log_cb=self.log.emit)

            total = sum(len(members) for members in grouped.values())
            downloaded = 0
            skipped = 0
            failed = 0
            seen = 0
            out_root = self.output_run_dir or os.getcwd()

            for (container, tar_name), member_preds in grouped.items():
                remaining = set(member_preds.keys())
                cc = bsc.get_container_client(container)
                bc = cc.get_blob_client(tar_name)
                self.log.emit(f"Re-streaming {container}/{tar_name} to download {len(remaining)} predicted image(s)")
                try:
                    for member, fh in iter_tar_members_from_blob(bc, tar_name, log_cb=self.log.emit):
                        member_name = member.name
                        if member_name not in remaining:
                            continue
                        p = member_preds[member_name]
                        label = str(p.get("predicted_label"))
                        dest_dir = os.path.join(out_root, "by_class", label)
                        tar_stem = safe_file_id_from_tar_name(tar_name)
                        filename = f"{tar_stem}__{os.path.basename(member_name)}"
                        dest_path = self._unique_dest_path(dest_dir, filename)
                        try:
                            with open(dest_path, "wb") as out_fh:
                                shutil.copyfileobj(fh, out_fh, length=1024 * 1024)
                            downloaded += 1
                            remaining.remove(member_name)
                        except Exception as e:
                            self.log.emit(f"ERROR downloading {container}/{tar_name}!/{member_name} -> {dest_path}: {e}")
                            failed += 1
                        seen += 1
                        if seen % 2 == 0 or seen == total:
                            self.progress.emit(int((seen / total) * 100))
                        if not remaining:
                            break
                    if remaining:
                        skipped += len(remaining)
                        for member_name in sorted(remaining):
                            self.log.emit(f"WARNING: member not found on re-stream: {container}/{tar_name}!/{member_name}")
                except Exception as e:
                    failed += len(remaining)
                    self.log.emit(f"ERROR re-streaming {container}/{tar_name}: {e}")

            self.progress.emit(100)
            self.finished_ok.emit({
                "moved": downloaded,
                "downloaded": downloaded,
                "skipped": skipped,
                "failed": failed,
                "total": total,
                "mode": {"local": False, "blob_stream_redownload": True},
            })
        except Exception as e:
            self.failed.emit(str(e))
