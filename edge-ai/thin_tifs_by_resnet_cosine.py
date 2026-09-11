#!/usr/bin/env python3
"""
thin_tifs_by_resnet_cosine_fast.py

Fast, rare-class-preserving TIFF thinning for plankton training libraries.

What it does
------------
1. Recursively finds bottom-level folders containing .tif/.tiff files.
2. Uses the bottom-level folder name as the class label.
3. Retains small / rare classes in full by default.
4. For large classes, extracts ResNet-18 or ResNet-50 embeddings in batches.
5. Caches embeddings so repeated runs do not re-run ResNet.
6. Optionally reduces embeddings to a smaller dimension before selection.
7. Selects a visually diverse subset using batched farthest-first cosine distance.

Default behaviour is deliberately conservative for minority classes:
    --retain-all-under 1000

So a class with 552 images will keep all 552, even if --keep-fraction 0.01.

Examples
--------
# Recommended first run: preserve rare classes, thin large class, cache embeddings.
python thin_tifs_by_resnet_cosine.py --input-root C:/Users/JR13/Downloads/traininglibs --output-root C:/Users/JR13/Downloads/traininglibsthinned --arch resnet50 --retention-rule fraction-floor --keep-fraction 0.01 --min-keep-per-class 5000 --retain-all-under 5000 --batch-size 128 --num-workers 8 --mode copy

# Re-run with a different keep fraction; cached embeddings are reused.
python thin_tifs_by_resnet_cosine.py ^
  --input-root C:/Users/JR13/Downloads/traininglibs ^
  --output-root C:/Users/JR13/Downloads/traininglibsthinned_2pct ^
  --arch resnet50 ^
  --retention-rule fraction-floor ^
  --keep-fraction 0.02 ^
  --mode copy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageSequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

TIF_EXTS = {".tif", ".tiff"}


class TifDataset(Dataset):
    def __init__(self, paths: Sequence[Path], image_size: int):
        self.paths = list(paths)
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = open_tif_as_rgb(path)
            return self.transform(img), str(path), ""
        except Exception as exc:
            dummy = torch.zeros(3, self.image_size, self.image_size)
            return dummy, str(path), f"{type(exc).__name__}: {exc}"


def open_tif_as_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        try:
            frame = next(ImageSequence.Iterator(im)).copy()
        except Exception:
            frame = im.copy()
        frame = ImageOps.exif_transpose(frame)
        if frame.mode not in ("RGB", "RGBA"):
            frame = frame.convert("RGB")
        elif frame.mode == "RGBA":
            bg = Image.new("RGB", frame.size, (255, 255, 255))
            bg.paste(frame, mask=frame.getchannel("A"))
            frame = bg
        return frame


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_feature_model(arch: str, device: torch.device) -> Tuple[nn.Module, int]:
    arch = arch.lower()
    if arch == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet18(pretrained=True)
        dim = model.fc.in_features
    elif arch == "resnet50":
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet50(pretrained=True)
        dim = model.fc.in_features
    else:
        raise ValueError("--arch must be resnet18 or resnet50")
    model.fc = nn.Identity()
    model.eval().to(device)
    return model, dim


def folder_has_tifs(folder: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in TIF_EXTS for p in folder.iterdir())
    except PermissionError:
        return False


def find_bottom_level_tif_folders(input_root: Path) -> List[Path]:
    tif_folders = [p for p in [input_root] + [q for q in input_root.rglob("*") if q.is_dir()] if folder_has_tifs(p)]
    tif_set = set(tif_folders)
    bottom = []
    for folder in tif_folders:
        has_tif_descendant = False
        for other in tif_folders:
            if other != folder and folder in other.parents:
                has_tif_descendant = True
                break
        if not has_tif_descendant:
            bottom.append(folder)
    return sorted(bottom)


def discover_classes(input_root: Path) -> Dict[str, List[Path]]:
    class_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for folder in find_bottom_level_tif_folders(input_root):
        class_name = folder.name
        tifs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TIF_EXTS)
        class_to_paths[class_name].extend(tifs)
    return {k: sorted(v) for k, v in sorted(class_to_paths.items())}


def retention_target(n: int, args) -> int:
    if n <= 0:
        return 0
    if args.retain_all_under is not None and n <= args.retain_all_under:
        return n
    if args.keep_count is not None:
        return min(n, max(1, int(args.keep_count)))

    fraction_n = int(math.ceil(n * args.keep_fraction)) if args.keep_fraction is not None else 0
    sqrt_n = int(math.ceil(math.sqrt(n) * args.sqrt_multiplier))
    floor_n = int(args.min_keep_per_class)

    if args.retention_rule == "fraction-floor":
        k = max(fraction_n, floor_n)
    elif args.retention_rule == "sqrt-floor":
        k = max(sqrt_n, floor_n)
    elif args.retention_rule == "hybrid":
        k = max(fraction_n, sqrt_n, floor_n)
    else:
        raise ValueError("unknown retention rule")
    return min(n, max(1, k))


def deterministic_sample(paths: Sequence[Path], sample_n: int, seed: int) -> List[Path]:
    if sample_n >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    keep = sorted(indices[:sample_n])
    return [paths[i] for i in keep]


def paths_signature(paths: Sequence[Path], input_root: Path, args, class_name: str) -> str:
    h = hashlib.sha256()
    h.update(str(input_root).encode("utf-8"))
    h.update(class_name.encode("utf-8"))
    h.update(args.arch.encode("utf-8"))
    h.update(str(args.image_size).encode("utf-8"))
    for p in paths:
        try:
            st = p.stat()
            rel = str(p.relative_to(input_root))
            h.update(rel.encode("utf-8"))
            h.update(str(st.st_size).encode("ascii"))
            h.update(str(int(st.st_mtime)).encode("ascii"))
        except FileNotFoundError:
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:24]


def cache_paths(args, output_root: Path, class_name: str, signature: str) -> Tuple[Path, Path]:
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_root / "_embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_class = "".join(c if c.isalnum() or c in "._-" else "_" for c in class_name)[:120]
    return cache_dir / f"{safe_class}_{signature}.npz", cache_dir / f"{safe_class}_{signature}_errors.json"


def embed_images_uncached(paths: Sequence[Path], model: nn.Module, device: torch.device, args) -> Tuple[np.ndarray, List[Path], List[Tuple[Path, str]]]:
    dataset = TifDataset(paths, image_size=args.image_size)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    features = []
    ok_paths: List[Path] = []
    errors: List[Tuple[Path, str]] = []

    with torch.inference_mode():
        for batch_i, (x, path_strs, err_strs) in enumerate(loader, start=1):
            good = [i for i, e in enumerate(err_strs) if not e]
            if good:
                x_good = x[good].to(device, non_blocking=True)
                if args.amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        z = model(x_good)
                else:
                    z = model(x_good)
                z = torch.nn.functional.normalize(z.float(), p=2, dim=1)
                arr = z.cpu().numpy()
                if args.embedding_dtype == "float16":
                    arr = arr.astype(np.float16)
                else:
                    arr = arr.astype(np.float32)
                features.append(arr)
                ok_paths.extend(Path(path_strs[i]) for i in good)
            for p, e in zip(path_strs, err_strs):
                if e:
                    errors.append((Path(p), e))
            if args.progress_every_batches and batch_i % args.progress_every_batches == 0:
                done = min(batch_i * args.batch_size, len(paths))
                print(f"    embedded {done:,}/{len(paths):,} candidate images")

    if not features:
        return np.empty((0, 0), dtype=np.float32), [], errors
    feats = np.vstack(features).astype(np.float32, copy=False)
    return feats, ok_paths, errors


def get_embeddings(paths: Sequence[Path], input_root: Path, output_root: Path, class_name: str, model: nn.Module, device: torch.device, args) -> Tuple[np.ndarray, List[Path], List[Tuple[Path, str]], str]:
    sig = paths_signature(paths, input_root, args, class_name)
    npz_path, err_path = cache_paths(args, output_root, class_name, sig)

    if args.use_cache and npz_path.exists():
        data = np.load(npz_path, allow_pickle=False)
        feats = data["features"].astype(np.float32, copy=False)
        ok_paths = [Path(x) for x in data["paths"].tolist()]
        errors = []
        if err_path.exists():
            try:
                errors = [(Path(e["path"]), e["error"]) for e in json.loads(err_path.read_text(encoding="utf-8"))]
            except Exception:
                errors = []
        print(f"  loaded cached embeddings: {npz_path.name} ({len(ok_paths):,} images)")
        return feats, ok_paths, errors, "cache_hit"

    feats, ok_paths, errors = embed_images_uncached(paths, model, device, args)
    if args.use_cache and len(ok_paths) > 0:
        save_feats = feats.astype(np.float16 if args.embedding_dtype == "float16" else np.float32)
        np.savez_compressed(npz_path, features=save_feats, paths=np.array([str(p) for p in ok_paths], dtype=str))
        err_path.write_text(json.dumps([{"path": str(p), "error": e} for p, e in errors], indent=2), encoding="utf-8")
        print(f"  saved embeddings cache: {npz_path.name}")
    return feats, ok_paths, errors, "cache_miss"


def reduce_features(features: np.ndarray, args, selection_device: torch.device) -> np.ndarray:
    n, d = features.shape
    if args.reduction == "none" or args.reduced_dim <= 0 or args.reduced_dim >= d or n <= args.reduced_dim:
        # Still enforce normalisation.
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)

    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(selection_device)
    x = x - x.mean(dim=0, keepdim=True)

    if args.reduction == "random-projection":
        gen = torch.Generator(device=selection_device)
        gen.manual_seed(args.random_seed)
        proj = torch.randn(d, args.reduced_dim, generator=gen, device=selection_device, dtype=x.dtype)
        proj = torch.nn.functional.normalize(proj, p=2, dim=0)
        y = x @ proj
    elif args.reduction == "pca":
        # torch.pca_lowrank is a good compromise here: fast enough and avoids sklearn dependency.
        q = min(args.reduced_dim + 16, min(n, d))
        try:
            _u, _s, v = torch.pca_lowrank(x, q=q, center=False, niter=args.pca_iter)
            y = x @ v[:, :args.reduced_dim]
        except RuntimeError as exc:
            if selection_device.type == "cuda":
                print(f"  PCA on CUDA failed ({type(exc).__name__}); retrying PCA on CPU")
                return reduce_features(features, replace_arg(args, "selection_device", "cpu"), torch.device("cpu"))
            raise
    else:
        raise ValueError("unknown reduction")

    y = torch.nn.functional.normalize(y.float(), p=2, dim=1)
    return y.cpu().numpy().astype(np.float32)


def replace_arg(args, key, value):
    class Obj:
        pass
    o = Obj()
    o.__dict__.update(args.__dict__)
    setattr(o, key, value)
    return o


def choose_selection_device(args, embed_device: torch.device) -> torch.device:
    if args.selection_device == "auto":
        return embed_device if embed_device.type == "cuda" else torch.device("cpu")
    return torch.device(args.selection_device)


def first_seed_index_torch(x: torch.Tensor, seed_mode: str, rng_seed: int) -> int:
    n = x.shape[0]
    if seed_mode == "first":
        return 0
    if seed_mode == "random":
        gen = torch.Generator(device=x.device)
        gen.manual_seed(rng_seed)
        return int(torch.randint(0, n, (1,), generator=gen, device=x.device).item())
    if seed_mode == "centroid":
        centroid = x.mean(dim=0, keepdim=True)
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)
        dist = 1.0 - (x @ centroid.T).squeeze(1)
        return int(torch.argmax(dist).item())
    raise ValueError("unknown seed mode")


def greedy_farthest_first_batched(features: np.ndarray, k: int, args, selection_device: torch.device) -> Tuple[List[int], List[Optional[float]]]:
    """Farthest-first selection using torch matvecs, optionally on GPU.

    This is still exact over the candidate set, but much faster than NumPy loops
    when the selection device is CUDA. For CPU-only machines, reducing dimensions
    and candidate count matters most.
    """
    n = features.shape[0]
    if n == 0:
        return [], []
    k = min(k, n)

    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(selection_device)
    x = torch.nn.functional.normalize(x, p=2, dim=1)

    selected = [first_seed_index_torch(x, args.seed, args.random_seed)]
    selected_dist: List[Optional[float]] = [None]

    nearest = 1.0 - (x @ x[selected[0]])
    nearest[selected[0]] = -float("inf")

    for step in range(1, k):
        idx = int(torch.argmax(nearest).item())
        selected.append(idx)
        selected_dist.append(float(nearest[idx].item()))

        new_dist = 1.0 - (x @ x[idx])
        nearest = torch.minimum(nearest, new_dist)
        nearest[torch.tensor(selected, device=selection_device)] = -float("inf")

        if args.progress_every_select and step % args.progress_every_select == 0:
            print(f"    selected {step:,}/{k:,} diverse representatives")

    return selected, selected_dist


def safe_copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "manifest-only":
        return
    else:
        raise ValueError("mode must be copy, symlink, move, or manifest-only")


def output_path_for(src: Path, input_root: Path, output_root: Path, class_name: str) -> Path:
    rel = src.relative_to(input_root)
    return output_root / class_name / rel


def row_dict(class_name, src, dst, selected, rank, distance, args, class_n, target_keep, note):
    return {
        "class_name": class_name,
        "source_path": str(src),
        "output_path": str(dst) if dst else "",
        "selected": bool(selected),
        "rank": rank,
        "selection_distance": distance,
        "class_total_files": class_n,
        "target_keep": target_keep,
        "retention_rule": args.retention_rule,
        "keep_fraction": args.keep_fraction,
        "sqrt_multiplier": args.sqrt_multiplier,
        "min_keep_per_class": args.min_keep_per_class,
        "retain_all_under": args.retain_all_under,
        "arch": args.arch,
        "reduction": args.reduction,
        "reduced_dim": args.reduced_dim,
        "candidate_note": note,
    }


def process_class(class_name: str, paths: List[Path], input_root: Path, output_root: Path, model: nn.Module, embed_device: torch.device, selection_device: torch.device, args) -> List[dict]:
    n_total = len(paths)
    k = retention_target(n_total, args)
    print(f"\nClass '{class_name}': {n_total:,} TIFF(s), target keep {k:,}")

    if k >= n_total:
        print("  retaining all files because class is at or below retention target")
        rows = []
        for rank, src in enumerate(paths, start=1):
            dst = output_path_for(src, input_root, output_root, class_name)
            safe_copy_or_link(src, dst, args.mode)
            rows.append(row_dict(class_name, src, dst, True, rank, None, args, n_total, k, "retained_all"))
        return rows

    candidate_paths = list(paths)
    candidate_note = "all_files"
    max_candidates = max(k, args.max_greedy_candidates)
    if len(candidate_paths) > max_candidates:
        candidate_paths = deterministic_sample(candidate_paths, max_candidates, args.random_seed)
        candidate_note = f"deterministic_candidate_sample_{max_candidates}_of_{n_total}"
        print(f"  using {len(candidate_paths):,} deterministic candidates for diversity search")

    features, ok_paths, errors, cache_note = get_embeddings(candidate_paths, input_root, output_root, class_name, model, embed_device, args)
    candidate_note = f"{candidate_note};{cache_note}"
    for p, e in errors[:20]:
        print(f"  WARNING: skipped unreadable TIFF: {p} ({e})")
    if len(errors) > 20:
        print(f"  WARNING: {len(errors) - 20:,} additional unreadable TIFFs omitted from console output")

    if len(ok_paths) == 0:
        print("  no readable candidate TIFFs")
        return [row_dict(class_name, p, None, False, "", "", args, n_total, k, "unreadable_or_not_embedded") for p in paths]

    effective_k = min(k, len(ok_paths))
    if args.reduction != "none":
        print(f"  reducing embeddings: {features.shape[1]} -> {min(args.reduced_dim, features.shape[1])} dims using {args.reduction}")
    reduced = reduce_features(features, args, selection_device)
    selected_idx, selected_dist = greedy_farthest_first_batched(reduced, effective_k, args, selection_device)

    selected_paths = [ok_paths[i] for i in selected_idx]
    selected_set = set(selected_paths)
    dist_by_path = {ok_paths[i]: selected_dist[j] for j, i in enumerate(selected_idx)}
    rank_by_path = {ok_paths[i]: j + 1 for j, i in enumerate(selected_idx)}

    print(f"  selected {len(selected_paths):,}/{n_total:,} ({candidate_note})")

    for src in selected_paths:
        dst = output_path_for(src, input_root, output_root, class_name)
        safe_copy_or_link(src, dst, args.mode)

    rows = []
    for src in paths:
        selected = src in selected_set
        dst = output_path_for(src, input_root, output_root, class_name) if selected and args.mode != "manifest-only" else None
        rows.append(row_dict(
            class_name, src, dst, selected, rank_by_path.get(src, ""), dist_by_path.get(src, ""),
            args, n_total, k, candidate_note,
        ))
    return rows


def write_manifests(rows: List[dict], output_root: Path, summary: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "thin_manifest.csv"
    json_path = output_root / "thin_manifest.json"
    summary_path = output_root / "thin_summary.json"

    fieldnames = [
        "class_name", "source_path", "output_path", "selected", "rank", "selection_distance",
        "class_total_files", "target_keep", "retention_rule", "keep_fraction", "sqrt_multiplier",
        "min_keep_per_class", "retain_all_under", "arch", "reduction", "reduced_dim", "candidate_note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Fast rare-class-preserving ResNet/cosine TIFF thinning by bottom-level folder class.")
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet50")

    p.add_argument("--retention-rule", choices=["hybrid", "fraction-floor", "sqrt-floor"], default="fraction-floor")
    p.add_argument("--keep-fraction", type=float, default=0.01)
    p.add_argument("--sqrt-multiplier", type=float, default=10.0)
    p.add_argument("--min-keep-per-class", type=int, default=500)
    p.add_argument("--retain-all-under", type=int, default=1000, help="Keep classes with <= this many files in full. Use 0 to disable.")
    p.add_argument("--keep-count", type=int, default=None, help="Override and keep this many per class, capped at class size.")

    p.add_argument("--max-greedy-candidates", type=int, default=20000, help="Candidate cap before greedy diversity search for huge classes.")
    p.add_argument("--reduction", choices=["pca", "random-projection", "none"], default="pca")
    p.add_argument("--reduced-dim", type=int, default=256)
    p.add_argument("--pca-iter", type=int, default=2)

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8, help="Use 0 if Windows multiprocessing causes problems.")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    p.add_argument("--selection-device", default="auto", help="auto, cpu, cuda, cuda:0")
    p.add_argument("--amp", action="store_true", help="Use mixed precision for ResNet inference on CUDA.")
    p.add_argument("--embedding-dtype", choices=["float32", "float16"], default="float16")

    p.add_argument("--cache-dir", type=Path, default=None)
    cache_group = p.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", dest="use_cache", action="store_true", default=True)
    cache_group.add_argument("--no-cache", dest="use_cache", action="store_false")

    p.add_argument("--seed", choices=["centroid", "first", "random"], default="centroid")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--mode", choices=["copy", "symlink", "move", "manifest-only"], default="copy")
    p.add_argument("--progress-every-batches", type=int, default=50)
    p.add_argument("--progress-every-select", type=int, default=100)

    args = p.parse_args(argv)
    if args.keep_fraction is not None and not (0 < args.keep_fraction <= 1):
        p.error("--keep-fraction must be > 0 and <= 1")
    if args.min_keep_per_class < 0:
        p.error("--min-keep-per-class must be >= 0")
    if args.retain_all_under is not None and args.retain_all_under < 0:
        p.error("--retain-all-under must be >= 0")
    if args.retain_all_under == 0:
        args.retain_all_under = None
    if args.max_greedy_candidates < 1:
        p.error("--max-greedy-candidates must be >= 1")
    if args.mode == "move":
        print("WARNING: --mode move is destructive; selected files are moved from input-root.", file=sys.stderr)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input-root does not exist: {input_root}")
    if input_root == output_root:
        raise ValueError("output-root must be different from input-root")

    embed_device = choose_device(args.device)
    selection_device = choose_selection_device(args, embed_device)
    print(f"Embedding device: {embed_device}")
    print(f"Selection device: {selection_device}")
    print(f"Architecture: {args.arch}")
    print(f"Retention rule: {args.retention_rule}")
    print(f"Retain all under: {args.retain_all_under}")
    print("Discovering bottom-level TIFF folders and class labels...")

    class_to_paths = discover_classes(input_root)
    if not class_to_paths:
        print("No bottom-level TIFF folders found.")
        return 1

    print(f"Found {len(class_to_paths)} class label(s):")
    for cls, paths in class_to_paths.items():
        print(f"  {cls}: {len(paths):,}")

    model, dim = build_feature_model(args.arch, embed_device)
    print(f"Raw feature dimension: {dim}")

    rows: List[dict] = []
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "arch": args.arch,
        "raw_feature_dimension": dim,
        "retention_rule": args.retention_rule,
        "keep_fraction": args.keep_fraction,
        "sqrt_multiplier": args.sqrt_multiplier,
        "min_keep_per_class": args.min_keep_per_class,
        "retain_all_under": args.retain_all_under,
        "max_greedy_candidates": args.max_greedy_candidates,
        "reduction": args.reduction,
        "reduced_dim": args.reduced_dim,
        "embedding_device": str(embed_device),
        "selection_device": str(selection_device),
        "classes": {},
    }

    for cls, paths in class_to_paths.items():
        class_rows = process_class(cls, paths, input_root, output_root, model, embed_device, selection_device, args)
        rows.extend(class_rows)
        summary["classes"][cls] = {
            "total_files": len(paths),
            "target_keep": retention_target(len(paths), args),
            "selected": sum(1 for r in class_rows if r.get("selected")),
        }

    write_manifests(rows, output_root, summary)
    selected = sum(1 for r in rows if r.get("selected"))
    total = len(rows)
    print(f"\nDone. Selected {selected:,}/{total:,} input TIFF row(s).")
    print(f"Manifest CSV:  {output_root / 'thin_manifest.csv'}")
    print(f"Manifest JSON: {output_root / 'thin_manifest.json'}")
    print(f"Summary JSON:  {output_root / 'thin_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
