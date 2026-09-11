# To get a confusion matrix from your training library without actually having to train a machine learning model
from pathlib import Path
from PIL import Image, ImageOps
import hashlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
ROOT = Path(r"C:\Users\JR13\Downloads\traininglib20260814")
OUT = ROOT / "_mobilenetv3_similarity_out"
VALID_EXTS = {".tif", ".tiff"}
BATCH_SIZE = 128
MAX_IMAGES_PER_CLASS = 50
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT.mkdir(exist_ok=True)
random.seed(RANDOM_SEED)
def stable_sample(paths, n):
    paths = sorted(paths)
    if len(paths) <= n:
        return paths
    keyed = [(hashlib.md5(str(p).encode("utf-8")).hexdigest(), p) for p in paths]
    keyed.sort()
    return [p for _, p in keyed[:n]]
def find_images_using_bottom_folder_as_classname(root):
    by_class = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            cls = p.parent.name
            by_class.setdefault(cls, []).append(p)
    return {k: sorted(v) for k, v in sorted(by_class.items())}    
def find_images(root):
    by_class = {}
    root = root.resolve()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        try:
            rel = p.resolve().relative_to(root)
        except Exception:
            continue
        parts = rel.parts
        if len(parts) < 2:
            continue
        cls = parts[0]
        by_class.setdefault(cls, []).append(p)
    return {k: sorted(v) for k, v in sorted(by_class.items())}
def build_model():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier = nn.Identity()
    model.eval()
    model.to(DEVICE)
    preprocess = weights.transforms()
    return model, preprocess
def load_image(path, preprocess):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return preprocess(im)
def embed_paths(paths, model, preprocess):
    embs = []
    kept = []
    failed = []
    batch = []
    batch_paths = []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            try:
                batch.append(load_image(p, preprocess))
                batch_paths.append(p)
            except Exception as e:
                failed.append((p, type(e).__name__, str(e)))
            if len(batch) >= BATCH_SIZE:
                x = torch.stack(batch).to(DEVICE, non_blocking=True)
                y = model(x)
                y = torch.nn.functional.normalize(y, p=2, dim=1)
                embs.append(y.cpu().numpy().astype(np.float32))
                kept.extend(batch_paths)
                batch = []
                batch_paths = []
            if i % 1000 == 0:
                print(f"EMBED_PROGRESS processed={i}/{len(paths)}")
        if batch:
            x = torch.stack(batch).to(DEVICE, non_blocking=True)
            y = model(x)
            y = torch.nn.functional.normalize(y, p=2, dim=1)
            embs.append(y.cpu().numpy().astype(np.float32))
            kept.extend(batch_paths)
    if embs:
        return np.vstack(embs), kept, failed
    return np.empty((0, 0), dtype=np.float32), kept, failed
def save_heatmap(matrix, labels, title, out_png, fmt=".2f"):
    fig_w = max(8, 0.45 * len(labels))
    fig_h = max(6, 0.40 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted / compared class")
    ax.set_ylabel("True / source class")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if len(labels) <= 30:
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                ax.text(c, r, format(matrix[r, c], fmt), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
def main():
    print(f"Device: {DEVICE}")
    by_class_raw = find_images(ROOT)
    if not by_class_raw:
        raise SystemExit(f"No tif/tiff files found under {ROOT}")
    by_class = {cls: stable_sample(paths, MAX_IMAGES_PER_CLASS) for cls, paths in by_class_raw.items()}
    labels = list(by_class.keys())
    print(f"Found {sum(len(v) for v in by_class_raw.values())} tif/tiff files across {len(labels)} classes")
    print(f"Using up to {MAX_IMAGES_PER_CLASS} images per class")
    model, preprocess = build_model()
    centroid_sums = {}
    centroid_counts = {}
    load_report = []
    failures = []
    print("Pass 1/2: building MobileNetV3 class centroids")
    for cls in labels:
        paths = by_class[cls]
        print(f"Class {cls}: embedding {len(paths)} of {len(by_class_raw[cls])}")
        emb, kept, failed = embed_paths(paths, model, preprocess)
        failures.extend([(str(p), cls, err, msg) for p, err, msg in failed])
        load_report.append({"class": cls, "found": len(by_class_raw[cls]), "used": len(kept), "failed": len(failed)})
        if emb.shape[0] > 0:
            centroid_sums[cls] = emb.sum(axis=0)
            centroid_counts[cls] = emb.shape[0]
    labels = [cls for cls in labels if cls in centroid_sums]
    if not labels:
        raise SystemExit("No images could be embedded")
    centroids = np.vstack([centroid_sums[cls] / centroid_counts[cls] for cls in labels]).astype(np.float32)
    centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    label_to_idx = {cls: i for i, cls in enumerate(labels)}
    cm_counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    per_image_rows = []
    print("Pass 2/2: assigning each image to nearest centroid")
    for true_cls in labels:
        paths = by_class[true_cls]
        emb, kept, failed = embed_paths(paths, model, preprocess)
        failures.extend([(str(p), true_cls, err, msg) for p, err, msg in failed])
        if emb.shape[0] == 0:
            continue
        sims = emb @ centroids.T
        pred_idx = sims.argmax(axis=1)
        nearest = sims.max(axis=1)
        for p, pi, ns in zip(kept, pred_idx, nearest):
            pred_cls = labels[int(pi)]
            cm_counts[label_to_idx[true_cls], label_to_idx[pred_cls]] += 1
            per_image_rows.append({"path": str(p), "true_class": true_cls, "predicted_nearest_class": pred_cls, "nearest_similarity": float(ns)})
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_frac = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=np.float32), where=row_sums != 0)
    centroid_sim = centroids @ centroids.T
    pd.DataFrame(load_report).to_csv(OUT / "load_report.csv", index=False)
    pd.DataFrame(failures, columns=["path", "class", "error_type", "message"]).to_csv(OUT / "failures.csv", index=False)
    pd.DataFrame(cm_counts, index=labels, columns=labels).to_csv(OUT / "mobilenetv3_nearest_centroid_confusion_counts.csv")
    pd.DataFrame(cm_frac, index=labels, columns=labels).to_csv(OUT / "mobilenetv3_nearest_centroid_confusion_fraction.csv")
    pd.DataFrame(centroid_sim, index=labels, columns=labels).to_csv(OUT / "mobilenetv3_class_centroid_similarity.csv")
    pd.DataFrame(per_image_rows).to_csv(OUT / "mobilenetv3_per_image_nearest_class.csv", index=False)
    save_heatmap(cm_frac, labels, "MobileNetV3 nearest-centroid confusion, fraction by true class", OUT / "mobilenetv3_nearest_centroid_confusion_fraction.png", fmt=".2f")
    save_heatmap(cm_counts, labels, "MobileNetV3 nearest-centroid confusion, counts", OUT / "mobilenetv3_nearest_centroid_confusion_counts.png", fmt="d")
    save_heatmap(centroid_sim, labels, "MobileNetV3 class centroid visual similarity", OUT / "mobilenetv3_class_centroid_similarity.png", fmt=".2f")
    print(f"Done. Outputs written to: {OUT}")
    print(OUT / "mobilenetv3_nearest_centroid_confusion_fraction.png")
    print(OUT / "mobilenetv3_class_centroid_similarity.png")
    print(OUT / "mobilenetv3_per_image_nearest_class.csv")
if __name__ == "__main__":
    main()