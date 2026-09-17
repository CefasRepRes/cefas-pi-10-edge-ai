#!/usr/bin/env python3
"""Compare central 12x12 pixel sum and variance across PlanktoShare classes."""
from __future__ import annotations

import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from azure.core.exceptions import AzureError, ClientAuthenticationError
from azure.identity import InteractiveBrowserCredential
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

DEFAULT_ACCOUNT_URL = "https://citprodc8603uksa.blob.core.windows.net"
DEFAULT_CONTAINER = "training-libs"
DEFAULT_PREFIX = "PlanktoShare/"
DEFAULT_BUBBLE_CLASS = "bubbles"
TIFF_SUFFIXES = (".tif", ".tiff")


@dataclass(frozen=True)
class BlobImage:
    blob_name: str
    class_name: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate central 12x12 sums and variances for PlanktoShare TIFF classes."
    )
    parser.add_argument("--account-url", default=DEFAULT_ACCOUNT_URL)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--bubble-class", default=DEFAULT_BUBBLE_CLASS)
    parser.add_argument("--tenant-id")
    parser.add_argument("--client-id")
    parser.add_argument("--login-hint")
    parser.add_argument("--browser-timeout", type=int, default=300)
    parser.add_argument(
        "--workers", type=int,
        default=min(32, max(4, (os.cpu_count() or 4) * 2)),
    )
    parser.add_argument("--max-concurrency-per-blob", type=int, default=1)
    parser.add_argument("--output-prefix", default="planktoshare_central_12x12")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-alpha", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_concurrency_per_blob < 1:
        parser.error("--max-concurrency-per-blob must be at least 1")
    if args.browser_timeout < 1:
        parser.error("--browser-timeout must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def normalise_prefix(prefix: str) -> str:
    prefix = prefix.strip().lstrip("/")
    return prefix + "/" if prefix and not prefix.endswith("/") else prefix


def extract_class_name(blob_name: str, prefix: str) -> str:
    relative = blob_name[len(prefix):].lstrip("/")
    parts = PurePosixPath(relative).parts
    return parts[0] if len(parts) >= 2 else "__root__"


def create_credential(args: argparse.Namespace) -> InteractiveBrowserCredential:
    kwargs: dict[str, object] = {"timeout": args.browser_timeout}
    if args.tenant_id:
        kwargs["tenant_id"] = args.tenant_id
    if args.client_id:
        kwargs["client_id"] = args.client_id
    if args.login_hint:
        kwargs["login_hint"] = args.login_hint
    credential = InteractiveBrowserCredential(**kwargs)
    print("Opening browser authentication...")
    credential.authenticate(scopes=["https://storage.azure.com/.default"])
    return credential


def list_images(container_client, prefix: str, limit: int | None) -> list[BlobImage]:
    images: list[BlobImage] = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if not blob.name.lower().endswith(TIFF_SUFFIXES):
            continue
        images.append(BlobImage(blob.name, extract_class_name(blob.name, prefix)))
        if limit is not None and len(images) >= limit:
            break
    return sorted(images, key=lambda item: item.blob_name)


def central_12x12(image: np.ndarray, include_alpha: bool) -> np.ndarray:
    array = np.asarray(image)
    if not include_alpha and array.ndim >= 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim < 2:
        raise ValueError(f"Expected at least two dimensions, got {array.shape}")
    height, width = array.shape[:2]
    if height < 12 or width < 12:
        raise ValueError(f"Image is smaller than 12x12: {array.shape}")
    row = (height - 12) // 2
    column = (width - 12) // 2
    crop = array[row:row + 12, column:column + 12, ...].astype(np.float64, copy=False)
    if crop.shape[0:2] != (12, 12):
        raise RuntimeError(f"Unexpected crop shape: {crop.shape}")
    if np.iscomplexobj(crop) or not np.all(np.isfinite(crop)):
        raise ValueError("Central crop contains unsupported or non-finite values")
    return crop


def process_blob(
    service_client: BlobServiceClient,
    container_name: str,
    image: BlobImage,
    max_concurrency: int,
    include_alpha: bool,
) -> dict[str, object]:
    try:
        payload = service_client.get_blob_client(
            container=container_name, blob=image.blob_name
        ).download_blob(max_concurrency=max_concurrency).readall()
        with io.BytesIO(payload) as buffer:
            pixels = tifffile.imread(buffer)
        crop = central_12x12(pixels, include_alpha)
        return {
            "blob_name": image.blob_name,
            "class_name": image.class_name,
            "dtype": str(pixels.dtype),
            "image_shape": "x".join(map(str, pixels.shape)),
            "central_crop_shape": "x".join(map(str, crop.shape)),
            "central_sample_count": int(crop.size),
            "central_sum": float(np.sum(crop, dtype=np.float64)),
            "central_variance": float(np.var(crop, dtype=np.float64, ddof=0)),
            "compressed_blob_bytes": len(payload),
            "error": "",
        }
    except Exception as exc:
        return {
            "blob_name": image.blob_name,
            "class_name": image.class_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def class_summary(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("class_name", sort=True)
    output = grouped.size().rename("image_count").to_frame()
    for feature in ("central_sum", "central_variance"):
        output[f"{feature}_mean"] = grouped[feature].mean()
        output[f"{feature}_median"] = grouped[feature].median()
        output[f"{feature}_q25"] = grouped[feature].quantile(0.25)
        output[f"{feature}_q75"] = grouped[feature].quantile(0.75)
        output[f"{feature}_std"] = grouped[feature].std()
    return output.reset_index()


def plot_class_bars(data: pd.DataFrame, feature: str, output_path: Path) -> None:
    grouped = data.groupby("class_name", sort=True)[feature]
    medians = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    positions = np.arange(len(medians))
    errors = np.vstack([(medians - q25).to_numpy(), (q75 - medians).to_numpy()])
    fig, ax = plt.subplots(figsize=(max(10, min(26, 0.6 * len(medians))), 7))
    ax.bar(positions, medians.to_numpy(), yerr=errors, capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(medians.index, rotation=90)
    ax.set_xlabel("Training class")
    ax.set_ylabel(f"Median {feature.replace('_', ' ')} (IQR error bars)")
    ax.set_title(f"Central 12x12 {feature.removeprefix('central_')} by class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values[np.isfinite(values)])
    if unique.size == 0:
        return np.array([])
    if unique.size == 1:
        return np.array([np.nextafter(unique[0], -np.inf), unique[0], np.nextafter(unique[0], np.inf)])
    mids = unique[:-1] + (unique[1:] - unique[:-1]) / 2
    return np.r_[np.nextafter(unique[0], -np.inf), mids, np.nextafter(unique[-1], np.inf)]


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    balanced = (sensitivity + specificity) / 2
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": balanced}


def evaluate_thresholds(data: pd.DataFrame, bubble_class: str) -> pd.DataFrame:
    truth = data["class_name"].astype(str).str.casefold().eq(bubble_class.casefold()).to_numpy()
    rows: list[dict[str, object]] = []
    for feature in ("central_sum", "central_variance"):
        values = data[feature].to_numpy(dtype=float)
        for direction in ("low", "high"):
            best = None
            for threshold in candidate_thresholds(values):
                prediction = values <= threshold if direction == "low" else values >= threshold
                result = {"feature": feature, "bubble_when": direction,
                          "threshold": float(threshold), **metrics(truth, prediction)}
                if best is None or result["balanced_accuracy"] > best["balanced_accuracy"]:
                    best = result
            if best is not None:
                rows.append(best)
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)


def write_outputs(data: pd.DataFrame, bubble_class: str, output_prefix: str) -> None:
    prefix = Path(output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(f"{prefix}_images.csv", index=False)
    failures = data[data["error"].fillna("").ne("")]
    successes = data[data["error"].fillna("").eq("")].copy()
    failures.to_csv(f"{prefix}_errors.csv", index=False)
    if successes.empty:
        raise RuntimeError("No TIFF images were processed successfully")
    truth = successes["class_name"].astype(str).str.casefold().eq(bubble_class.casefold())
    if not truth.any() or truth.all():
        raise ValueError("Both bubble and non-bubble images are required for threshold evaluation")
    class_summary(successes).to_csv(f"{prefix}_class_summary.csv", index=False)
    evaluate_thresholds(successes, bubble_class).to_csv(f"{prefix}_thresholds.csv", index=False)
    plot_class_bars(successes, "central_sum", Path(f"{prefix}_central_sum_by_class.png"))
    plot_class_bars(successes, "central_variance", Path(f"{prefix}_central_variance_by_class.png"))
    print(f"Wrote outputs using prefix: {prefix}")


def main() -> int:
    args = parse_arguments()
    credential = None
    try:
        credential = create_credential(args)
        service = BlobServiceClient(args.account_url, credential=credential)
        images = list_images(
            service.get_container_client(args.container),
            normalise_prefix(args.prefix),
            args.limit,
        )
        if not images:
            print("No TIFF images found.", file=sys.stderr)
            return 2
        rows: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_blob, service, args.container, image,
                    args.max_concurrency_per_blob, args.include_alpha,
                )
                for image in images
            ]
            with tqdm(total=len(futures), unit="image", desc="Processing TIFFs") as progress:
                for future in as_completed(futures):
                    rows.append(future.result())
                    progress.update(1)
        data = pd.DataFrame(rows).sort_values("blob_name").reset_index(drop=True)
        write_outputs(data, args.bubble_class, args.output_prefix)
        failed = int(data["error"].fillna("").ne("").sum())
        print(f"Processed {len(data) - failed:,} successfully; {failed:,} failed.")
        return 0 if failed == 0 else 1
    except ClientAuthenticationError as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        return 3
    except AzureError as exc:
        print(f"Azure error: {exc}", file=sys.stderr)
        return 4
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 5
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    finally:
        if credential is not None:
            credential.close()


if __name__ == "__main__":
    raise SystemExit(main())
