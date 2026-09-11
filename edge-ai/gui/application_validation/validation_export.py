import io
import json
import os
import re
import tarfile
from collections import defaultdict
from datetime import datetime, timezone

try:
    from .tar_streaming import iter_tar_members_from_blob, safe_stem_from_tar_name
except ImportError:  # pragma: no cover - allows direct test imports
    from tar_streaming import iter_tar_members_from_blob, safe_stem_from_tar_name

try:
    from .validation_stopping import build_confusion_error_model, is_uncertain_label
except ImportError:  # pragma: no cover - allows direct test imports
    from validation_stopping import build_confusion_error_model, is_uncertain_label


TRAINING_LIBS_CONTAINER = "training-libs"
VALIDATION_EXPORT_PREFIX = "validationsession"
MAX_COLLISION_ATTEMPTS = 1000


def _utc_now(now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def utc_now_iso(now=None) -> str:
    return _utc_now(now).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_validation_session_folder_name(now=None) -> str:
    return f"{VALIDATION_EXPORT_PREFIX}_{_utc_now(now).strftime('%Y%m%d%H%M')}"


def _prefix_has_blobs(container_client, prefix: str) -> bool:
    return any(container_client.list_blobs(name_starts_with=f"{prefix.rstrip('/')}/"))


def choose_validation_session_folder(container_client, now=None) -> str:
    base = build_validation_session_folder_name(now=now)
    if not _prefix_has_blobs(container_client, base):
        return base
    for suffix in range(1, MAX_COLLISION_ATTEMPTS + 1):
        candidate = f"{base}_{suffix}"
        if not _prefix_has_blobs(container_client, candidate):
            return candidate
    raise RuntimeError(f"Could not allocate a unique validation export folder after {MAX_COLLISION_ATTEMPTS} attempts")


def _sanitize_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or ""))
    cleaned = cleaned.strip("._")
    return cleaned or "item"


def _metadata_value(metadata: dict, key: str, default="not provided"):
    value = (metadata or {}).get(key)
    if value in (None, ""):
        return default
    return str(value)


def _normalise_validation_export_row(row: dict) -> dict:
    if not isinstance(row, dict):
        return row
    normalised = dict(row)
    if "predicted_label" not in normalised and "original_label" in normalised:
        normalised["predicted_label"] = normalised["original_label"]
    if "prediction" not in normalised and "original_label" in normalised:
        normalised["prediction"] = normalised["original_label"]
    if "true_label" not in normalised and "final_label" in normalised:
        normalised["true_label"] = normalised["final_label"]
    if "label" not in normalised and "final_label" in normalised:
        normalised["label"] = normalised["final_label"]
    return normalised


def _error_model_for_validation_export(error_model: dict) -> dict:
    if not isinstance(error_model, dict):
        return {}
    excluded_keys = {"dirichlet_posterior_parameters", "posterior_credible_intervals"}
    return {
        key: value
        for key, value in error_model.items()
        if key not in excluded_keys
    }


def prepare_validation_summary_for_export(
    validation_summary: dict,
    *,
    rows: list = None,
    labels: list = None,
    metadata: dict = None,
    error_model: dict = None,
):
    if not isinstance(validation_summary, dict):
        return validation_summary

    summary = dict(validation_summary)
    export_rows = rows
    if export_rows is None:
        export_rows = summary.get("images")
    if export_rows is not None:
        summary["images"] = [_normalise_validation_export_row(row) for row in export_rows]

    if error_model is None:
        error_model = (
            summary.get("validation_error_model")
            or summary.get("classifier_error_model")
            or summary.get("error_model")
        )
    if isinstance(error_model, dict):
        export_error_model = _error_model_for_validation_export(error_model)
        summary["classifier_error_model"] = dict(export_error_model)
        summary["validation_error_model"] = dict(export_error_model)
        for key in (
            "class_order",
            "confusion_counts",
            "transition_probabilities",
            "class_example_counts",
            "predicted_class_counts",
            "prior_alpha",
            "sampling_model",
            "per_class_metrics",
        ):
            if key in export_error_model:
                summary[key] = export_error_model.get(key)

    if labels is not None:
        summary["labels"] = [str(label) for label in labels if label not in (None, "")]
    if metadata is not None:
        summary["validation_session"] = metadata
    return summary


def _choose_unknown_label(labels: list) -> str:
    for candidate in ("UNKNOWN", "unknown", "unclassified", "other", "OTHER"):
        if candidate in labels:
            return candidate
    return "UNKNOWN"


def _unique_parent_datasets(rows: list, account_url: str) -> list:
    parents = []
    seen = set()
    base_account = (account_url or "").rstrip("/")
    for row in rows:
        container = row.get("container")
        tar_name = row.get("source_tar_blob")
        if not (container and tar_name):
            continue
        dataset_url = f"{base_account}/{container}/{tar_name}"
        if dataset_url in seen:
            continue
        parents.append({
            "dataset": dataset_url,
            "relationship": "validated_subset",
        })
        seen.add(dataset_url)
    return parents


def build_validation_dataset_json(
    *,
    rows: list,
    labels: list,
    account_url: str,
    export_folder: str,
    generated_utc: str = None,
    config=None,
    metadata: dict = None,
):
    generated_utc = generated_utc or utc_now_iso()
    metadata = metadata or {}
    session_name = _metadata_value(metadata, "session_name", export_folder)
    validator_name = _metadata_value(metadata, "validator_name")
    validator_email = _metadata_value(metadata, "validator_email")
    organisation = _metadata_value(metadata, "organisation")
    project_name = _metadata_value(metadata, "project_name", "")
    dataset_name = _metadata_value(metadata, "dataset_name", session_name)
    instrument = _metadata_value(metadata, "instrument")
    taxonomy_name = _metadata_value(metadata, "taxonomy_name")
    validation_purpose = _metadata_value(metadata, "validation_purpose", "validation")
    validation_notes = _metadata_value(metadata, "validation_notes", "")
    model_label = _metadata_value(metadata, "model_label", "")

    usable_rows = [
        row
        for row in (rows or [])
        if row.get("final_label") not in (None, "")
        and row.get("original_label") not in (None, "")
        and not is_uncertain_label(row.get("final_label"))
        and not is_uncertain_label(row.get("original_label"))
    ]
    smoothing_prior_alpha = getattr(config, "smoothing_prior_alpha", 0.75)
    error_model = build_confusion_error_model(
        usable_rows,
        labels=labels,
        smoothing_prior_alpha=smoothing_prior_alpha,
    )
    class_labels = error_model.get("class_order") or []
    if not class_labels:
        class_labels = [str(x) for x in (labels or []) if x not in (None, "")]

    model_blobs = sorted({str(row.get("model_blob")) for row in rows if row.get("model_blob")})
    model_details = {}
    if model_blobs:
        model_details["model"] = [
            blob if str(blob).lower().startswith("http") else f"{(account_url or '').rstrip('/')}/trainedmodels/{blob.lstrip('/')}"
            for blob in model_blobs
        ]

    important_classes = list(getattr(config, "important_classes", []) or [])
    min_class_precision = getattr(config, "min_class_precision", None)
    ci_width_target = getattr(config, "ci_width_target", None)
    max_sample_size = getattr(config, "max_sample_size", None)
    min_class_examples = getattr(config, "min_class_examples", None)
    diag_tolerance = getattr(config, "diag_credible_interval_tolerance", None)
    offdiag_tolerance = getattr(config, "offdiag_credible_interval_tolerance", None)
    detailed_bits = [
        "Images were reviewed in the Rapid Plankton validation session GUI after model prediction.",
        "Each exported image has a manually confirmed final label.",
        "A multiclass confusion-matrix error model was exported for later uncertainty propagation.",
    ]
    if max_sample_size is not None:
        detailed_bits.append(f"The validation pool size cap was {max_sample_size}.")
    if min_class_examples is not None:
        detailed_bits.append(f"Validation stopped only after each class had at least {min_class_examples} validated examples.")
    if diag_tolerance is not None:
        detailed_bits.append(f"The diagonal confusion-entry posterior credible interval tolerance was {diag_tolerance:.2f}.")
    if offdiag_tolerance is not None:
        detailed_bits.append(f"The off-diagonal confusion-entry posterior credible interval tolerance was {offdiag_tolerance:.2f}.")
    if ci_width_target is not None:
        detailed_bits.append(f"The configured stopping target for overall accuracy was a Wilson CI half-width of ±{ci_width_target:.2f}.")
    if important_classes:
        threshold_text = f" at {min_class_precision:.2f} precision" if min_class_precision is not None else ""
        detailed_bits.append(
            f"Important classes checked during the session were {', '.join(important_classes)}{threshold_text}."
        )

    export_error_model = _error_model_for_validation_export(error_model)

    return {
        "schema_version": "1.0",
        "dataset_identity": {
            "dataset_id": export_folder,
            "dataset_name": dataset_name,
            "validation_session_name": session_name,
            "description": f"Validated plankton images exported from Rapid Plankton validation session {session_name}.",
            "organisation": organisation,
            "data_owner": validator_name,
            "project_name": project_name,
            "instrument": instrument,
            "contact": {
                "name": validator_name,
                "email": validator_email,
            },
        },
        "labelling_session": {
            "labelling_for_validation_or_improvement": "validation",
            "labelling_type": "manual validation of model predictions",
            "labelling_type_detailed_explanation": " ".join(detailed_bits),
            "labelling_100_percent_of_each_tenbin_assigned_to_a_label": False,
            "labelling_cherry_picked": False,
            "labellers": [
                {
                    "name": validator_name,
                    "email": validator_email,
                    "organisation": organisation,
                }
            ],
            "validation_session": {
                "session_name": session_name,
                "project_name": project_name,
                "purpose": validation_purpose,
                "model_label": model_label,
                "notes": validation_notes,
            },
        },
        "ml_assistance": {
            "used": bool(model_details),
            "model_details": model_details,
        },
        "taxonomy": {
            "taxonomy_name": taxonomy_name,
            "unknown_class_label": _choose_unknown_label(class_labels),
            "hierarchical": False,
        },
        "quality_control": {
            "qc_performed": True,
            "qc_methods": [
                "manual_validation_review",
            ],
            "qc_summary": {
                "estimated_spot_check_fraction": None,
                "estimated_label_accuracy": None,
                "issues_identified": [],
            },
            "known_limitations": [
                "This dataset is a validated subset exported from an application validation session.",
                "It may not represent every image available in the source tenbin archives.",
            ],
        },
        "class_options": {
            "classes": class_labels,
            "what_was_considered": "Validation class labels available in the Rapid Plankton validation session.",
        },
        "validation_error_model": {
            "class_order": class_labels,
            "confusion_counts": export_error_model.get("confusion_counts", []),
            "transition_probabilities": export_error_model.get("transition_probabilities", []),
            "prior_alpha": export_error_model.get("prior_alpha"),
            "sampling_model": export_error_model.get("sampling_model"),
        },
        "versioning_and_lineage": {
            "dataset_version": "1.0",
            "parent_datasets": _unique_parent_datasets(rows, account_url),
            "supersedes": [],
            "superseded_by": [],
            "change_log": [
                {
                    "version": "1.0",
                    "date_utc": generated_utc,
                    "changes": "Initial export from validation session to training-libs.",
                }
            ],
        },
        "generalnotes": (
            "Dataset created from manually validated application outputs for later training reuse. "
            f"Validation session: {session_name}. "
            f"Purpose: {validation_purpose}. "
            f"Project/survey: {project_name or 'not provided'}. "
            f"Notes: {validation_notes or 'not provided'}. "
            f"Export location: {TRAINING_LIBS_CONTAINER}/{export_folder}."
        ),
    }


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def _unique_blob_name(folder: str, label: str, tar_name: str, member_name: str, used_names: set) -> str:
    label_dir = _sanitize_fragment(label)
    tar_stem = safe_stem_from_tar_name(tar_name)
    member_fragment = _sanitize_fragment(member_name)
    base_name = f"{folder}/{label_dir}/{tar_stem}__{member_fragment}"
    stem, ext = os.path.splitext(base_name)
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    for suffix in range(1, MAX_COLLISION_ATTEMPTS + 1):
        candidate = f"{stem}_{suffix}{ext}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
    raise RuntimeError(f"Could not allocate a unique validation export blob name after {MAX_COLLISION_ATTEMPTS} attempts")


def export_validation_session_to_training_libs(
    *,
    blob_service_client,
    account_url: str,
    rows: list,
    validation_summary: dict,
    labels: list,
    config=None,
    metadata: dict = None,
    now=None,
    log_cb=None,
):
    validated_rows = [row for row in (rows or []) if row.get("validated")]
    if not validated_rows:
        raise RuntimeError("No validated images are available for export")
    uncertain_rows = []
    export_rows = []
    for row in validated_rows:
        if is_uncertain_label(row.get("final_label")) or is_uncertain_label(row.get("original_label")):
            uncertain_rows.append(row)
        else:
            export_rows.append(row)
    if not export_rows:
        raise RuntimeError("No non-uncertain validated images are available for export")

    def log(message: str):
        if log_cb:
            log_cb(message)

    generated_utc = utc_now_iso(now=now)
    if isinstance(validation_summary, dict):
        validation_summary = prepare_validation_summary_for_export(
            validation_summary,
            rows=validated_rows,
            labels=labels,
            metadata=metadata,
        )

    if isinstance(validation_summary, dict):
        smoothing_prior_alpha = getattr(config, "smoothing_prior_alpha", 0.75)
        error_model = build_confusion_error_model(
            export_rows,
            labels=labels,
            smoothing_prior_alpha=smoothing_prior_alpha,
        )
        validation_summary["classifier_error_model"] = error_model
        validation_summary["validation_error_model"] = error_model

    training_libs = blob_service_client.get_container_client(TRAINING_LIBS_CONTAINER)
    export_folder = choose_validation_session_folder(training_libs, now=now)
    if uncertain_rows:
        log(f"Skipping {len(uncertain_rows)} uncertain validated image(s) from training-libs export")

    dataset_json = build_validation_dataset_json(
        rows=export_rows,
        labels=labels,
        account_url=account_url,
        export_folder=export_folder,
        generated_utc=generated_utc,
        config=config,
        metadata=metadata,
    )

    grouped = defaultdict(dict)
    for row in export_rows:
        container = row.get("container")
        tar_name = row.get("source_tar_blob")
        member_name = row.get("tar_member")
        if not (container and tar_name and member_name):
            raise RuntimeError("Validated image is missing blob container, TAR, or member name")
        grouped[(container, tar_name)][member_name] = row

    used_blob_names = set()
    uploaded_images = 0
    for (container, tar_name), member_rows in grouped.items():
        remaining = dict(member_rows)
        source_blob = blob_service_client.get_container_client(container).get_blob_client(tar_name)
        log(f"Exporting {len(member_rows)} validated image(s) from {container}/{tar_name} to {TRAINING_LIBS_CONTAINER}/{export_folder}")
        for member, fh in iter_tar_members_from_blob(source_blob, tar_name, log_cb=log_cb):
            row = remaining.pop(member.name, None)
            if row is None:
                continue
            dest_name = _unique_blob_name(
                export_folder,
                str(row.get("final_label")),
                tar_name,
                member.name,
                used_names=used_blob_names,
            )
            training_libs.upload_blob(name=dest_name, data=fh.read(), overwrite=False)
            uploaded_images += 1
            if not remaining:
                break
        if remaining:
            missing = ", ".join(sorted(remaining))
            raise RuntimeError(f"Could not export all validated images from {container}/{tar_name}: {missing}")

    training_libs.upload_blob(
        name=f"{export_folder}/validation.json",
        data=_json_bytes(validation_summary),
        overwrite=False,
    )
    training_libs.upload_blob(
        name=f"{export_folder}/dataset.json",
        data=_json_bytes(dataset_json),
        overwrite=False,
    )
    log(
        f"Exported {uploaded_images} validated image(s) plus validation.json and dataset.json "
        f"to {TRAINING_LIBS_CONTAINER}/{export_folder}"
    )
    return {
        "export_folder": export_folder,
        "dataset_json": dataset_json,
        "uploaded_images": uploaded_images,
        "generated_utc": generated_utc,
    }
