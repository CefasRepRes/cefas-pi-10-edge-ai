#!/usr/bin/env python3
"""
train.py - local-cache-first trainer with probability calibration and robust
non-finite diagnostics.

Updated to support ResNet and MobileNet architectures:
- resnet18, resnet34, resnet50
- mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
"""
from __future__ import annotations
import argparse
import csv as _csv
import hashlib
import json
import os
import random
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix
SUPPORTED_ARCHITECTURES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
)
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
def safe_stamp() -> str:
    return utc_now_iso().replace(":", "-")
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None
def normalise_architecture(arch: str) -> str:
    arch = str(arch or "resnet18").strip().lower().replace("-", "_")
    aliases = {
        "resnet": "resnet18",
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "mobilenet": "mobilenet_v3_small",
        "mobilenetv2": "mobilenet_v2",
        "mobilenet_v2": "mobilenet_v2",
        "mobilenetv3": "mobilenet_v3_small",
        "mobilenetv3_small": "mobilenet_v3_small",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "mobilenetv3_large": "mobilenet_v3_large",
        "mobilenet_v3_large": "mobilenet_v3_large",
    }
    return aliases.get(arch, arch)
def tensor_stats(x: torch.Tensor, name: str = "tensor") -> dict:
    with torch.no_grad():
        x = x.detach()
        finite = torch.isfinite(x)
        finite_count = int(finite.sum().item())
        numel = int(x.numel())
        out = {
            "name": name,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "numel": numel,
            "finite_count": finite_count,
            "nonfinite_count": numel - finite_count,
            "all_finite": bool(finite_count == numel),
            "nan_count": int(torch.isnan(x).sum().item()) if x.is_floating_point() else 0,
            "posinf_count": int(torch.isposinf(x).sum().item()) if x.is_floating_point() else 0,
            "neginf_count": int(torch.isneginf(x).sum().item()) if x.is_floating_point() else 0,
        }
        if finite_count > 0:
            xf = x[finite]
            out.update({
                "min": _safe_float(xf.min().item()),
                "max": _safe_float(xf.max().item()),
                "mean": _safe_float(xf.float().mean().item()),
                "std": _safe_float(xf.float().std(unbiased=False).item()) if finite_count > 1 else 0.0,
                "abs_max": _safe_float(xf.abs().max().item()),
            })
        else:
            out.update({"min": None, "max": None, "mean": None, "std": None, "abs_max": None})
        return out
def print_tensor_stats(x: torch.Tensor, name: str) -> None:
    s = tensor_stats(x, name)
    print(
        f"{name}: shape={s['shape']} dtype={s['dtype']} finite={s['finite_count']}/{s['numel']} "
        f"nan={s['nan_count']} +inf={s['posinf_count']} -inf={s['neginf_count']} "
        f"min={s['min']} max={s['max']} mean={s['mean']} std={s['std']} abs_max={s['abs_max']}",
        flush=True,
    )
def first_nonfinite_model_state(model: nn.Module):
    for name, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            return name, p, "parameter"
    for name, b in model.named_buffers():
        if b is not None and torch.is_tensor(b) and not torch.isfinite(b).all():
            return name, b, "buffer"
    return None, None, None
def first_nonfinite_optimizer_state(optimizer):
    try:
        for param_idx, state in enumerate(optimizer.state.values()):
            for key, value in state.items():
                if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                    return f"state[{param_idx}].{key}", value
    except Exception:
        return None, None
    return None, None
def save_debug_batch(debug_dir: str, reason: str, epoch: int, step: int, paths, xb: torch.Tensor | None = None, yb: torch.Tensor | None = None, logits: torch.Tensor | None = None, extra: dict | None = None) -> None:
    try:
        os.makedirs(debug_dir, exist_ok=True)
        stamp = safe_stamp()
        base = f"debug_{reason}_epoch{epoch}_step{step}_{stamp}"
        pt_path = os.path.join(debug_dir, base + ".pt")
        json_path = os.path.join(debug_dir, base + ".json")
        path_list = [str(p) for p in paths] if isinstance(paths, (list, tuple)) else ([str(paths)] if paths is not None else [])
        payload = {
            "reason": reason,
            "epoch": int(epoch),
            "step": int(step),
            "paths": path_list,
            "labels": yb.detach().cpu() if torch.is_tensor(yb) else None,
            "images": xb.detach().cpu() if torch.is_tensor(xb) else None,
            "logits": logits.detach().cpu() if torch.is_tensor(logits) else None,
            "extra": extra or {},
        }
        torch.save(payload, pt_path)
        report = {
            "reason": reason,
            "epoch": int(epoch),
            "step": int(step),
            "paths_first_20": path_list[:20],
            "n_paths": len(path_list),
            "image_stats": tensor_stats(xb, "xb") if torch.is_tensor(xb) else None,
            "logit_stats": tensor_stats(logits, "logits") if torch.is_tensor(logits) else None,
            "extra": extra or {},
            "pt_file": pt_path,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved non-finite debug tensor bundle: {pt_path}", flush=True)
        print(f"Saved non-finite debug JSON report: {json_path}", flush=True)
        if torch.is_tensor(xb):
            try:
                from torchvision.utils import make_grid, save_image
                n = min(int(xb.shape[0]), 16)
                imgs = torch.nan_to_num(xb[:n].detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
                lo = imgs.amin(dim=(1, 2, 3), keepdim=True)
                hi = imgs.amax(dim=(1, 2, 3), keepdim=True)
                imgs = (imgs - lo) / (hi - lo).clamp_min(1e-6)
                png_path = os.path.join(debug_dir, base + "_images.png")
                save_image(make_grid(imgs, nrow=4), png_path)
                print(f"Saved quick-look image grid: {png_path}", flush=True)
            except Exception as e:
                print(f"WARNING: could not save quick-look image grid: {e}", flush=True)
    except Exception as e:
        print(f"WARNING: failed to save debug batch: {e}", flush=True)
def find_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from find_tensors(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from find_tensors(v)
def register_nonfinite_activation_hooks(model: nn.Module, debug_context: dict):
    handles = []
    def make_hook(module_name: str):
        def hook(_module, _inputs, output):
            for t in find_tensors(output):
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    epoch = int(debug_context.get("epoch", -1))
                    step = int(debug_context.get("step", -1))
                    paths = debug_context.get("paths")
                    xb = debug_context.get("xb")
                    yb = debug_context.get("yb")
                    debug_dir = debug_context.get("debug_dir", ".")
                    print("\n==================== NON-FINITE ACTIVATION ====================", flush=True)
                    print(f"First detected in layer: {module_name}", flush=True)
                    if isinstance(paths, (list, tuple)):
                        print("Example paths:", paths[:5], flush=True)
                    print_tensor_stats(t, f"activation[{module_name}]")
                    save_debug_batch(debug_dir, "activation", epoch, step, paths, xb, yb, t, {"layer": module_name, "activation_stats": tensor_stats(t, f"activation[{module_name}]")})
                    raise RuntimeError(f"Non-finite activation in layer: {module_name}")
        return hook
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles
@dataclass
class TrainConfig:
    arch: str = "resnet18"
    pretrained: bool = True
    image_size: Tuple[int, int] = (256, 256)
    normalisation: str = "imagenet"
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    num_workers: int = 8
    val_fraction: float = 0.2
    seed: int = 1337
    freeze_batchnorm: bool = True
    early_stopping_enabled: bool = True
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_warmup_epochs: int = 1
    curve_csv: str = "training_curve.csv"
    curve_png: str = "training_curve.png"
    confusion_matrix_csv: str = "confusion_matrix.csv"
    confusion_matrix_png: str = "confusion_matrix.png"
def _first_present(*values, default=None):
    for v in values:
        if v not in (None, ""):
            return v
    return default
def parse_train_config(settings: dict) -> TrainConfig:
    tp = settings.get("training_parameters", {}) or {}
    cfg = TrainConfig()
    top_arch = settings.get("model_architecture") or settings.get("arch")
    if isinstance(tp, dict) and ("model" in tp or "data" in tp or "training" in tp):
        model = tp.get("model", {}) or {}
        data = tp.get("data", {}) or {}
        trn = tp.get("training", {}) or {}
        cfg.arch = normalise_architecture(_first_present(model.get("architecture"), model.get("arch"), model.get("name"), tp.get("model_architecture"), tp.get("arch"), top_arch, default=cfg.arch))
        cfg.pretrained = bool(model.get("pretrained", cfg.pretrained))
        cfg.freeze_batchnorm = bool(model.get("freeze_batchnorm", trn.get("freeze_batchnorm", cfg.freeze_batchnorm)))
        size = _first_present(data.get("image_size"), tp.get("image_size"), default=cfg.image_size)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            cfg.image_size = (int(size[0]), int(size[1]))
        cfg.normalisation = str(_first_present(data.get("normalisation"), tp.get("normalisation"), default=cfg.normalisation)).lower()
        cfg.batch_size = int(trn.get("batch_size", cfg.batch_size))
        cfg.epochs = int(trn.get("epochs", cfg.epochs))
        cfg.learning_rate = float(trn.get("learning_rate", cfg.learning_rate))
        cfg.optimizer = str(trn.get("optimizer", cfg.optimizer)).lower()
        cfg.weight_decay = float(trn.get("weight_decay", cfg.weight_decay))
        cfg.num_workers = int(trn.get("num_workers", cfg.num_workers))
        cfg.val_fraction = float(trn.get("val_fraction", cfg.val_fraction))
        cfg.seed = int(trn.get("seed", cfg.seed))
        es = trn.get("early_stopping", {}) or {}
        cfg.early_stopping_enabled = bool(es.get("enabled", cfg.early_stopping_enabled))
        cfg.early_stopping_monitor = str(es.get("monitor", cfg.early_stopping_monitor)).lower()
        cfg.early_stopping_mode = str(es.get("mode", cfg.early_stopping_mode)).lower()
        cfg.early_stopping_patience = int(es.get("patience", cfg.early_stopping_patience))
        cfg.early_stopping_min_delta = float(es.get("min_delta", cfg.early_stopping_min_delta))
        cfg.early_stopping_warmup_epochs = int(es.get("warmup_epochs", cfg.early_stopping_warmup_epochs))
        lg = trn.get("logging", {}) or {}
        cfg.curve_csv = str(lg.get("curve_csv", cfg.curve_csv))
        cfg.curve_png = str(lg.get("curve_png", cfg.curve_png))
        cfg.confusion_matrix_csv = str(lg.get("confusion_matrix_csv", cfg.confusion_matrix_csv))
        cfg.confusion_matrix_png = str(lg.get("confusion_matrix_png", cfg.confusion_matrix_png))
        return cfg
    cfg.arch = normalise_architecture(_first_present(tp.get("model_architecture"), tp.get("arch"), top_arch, default=cfg.arch))
    size = tp.get("image_size", cfg.image_size)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        cfg.image_size = (int(size[0]), int(size[1]))
    cfg.batch_size = int(tp.get("batch_size", cfg.batch_size))
    cfg.epochs = int(tp.get("epochs", cfg.epochs))
    cfg.learning_rate = float(tp.get("learning_rate", cfg.learning_rate))
    cfg.optimizer = str(tp.get("optimizer", cfg.optimizer)).lower()
    cfg.pretrained = bool(tp.get("pretrained", cfg.pretrained))
    cfg.freeze_batchnorm = bool(tp.get("freeze_batchnorm", cfg.freeze_batchnorm))
    cfg.normalisation = str(tp.get("normalisation", cfg.normalisation)).lower()
    cfg.weight_decay = float(tp.get("weight_decay", cfg.weight_decay))
    cfg.num_workers = int(tp.get("num_workers", cfg.num_workers))
    cfg.val_fraction = float(tp.get("val_fraction", cfg.val_fraction))
    cfg.seed = int(tp.get("seed", cfg.seed))
    return cfg
def load_image_safe(path, log_cb=None):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except UnidentifiedImageError:
        if log_cb:
            log_cb(f"Skipping unreadable image (PIL): {path}")
        return None
    except Exception as e:
        if log_cb:
            log_cb(f"Skipping image ({type(e).__name__}): {path}")
        return None
class ImageListDataset(Dataset):
    def __init__(self, items, transform=None, return_path=False):
        self.items = items
        self.transform = transform
        self.return_path = return_path
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = load_image_safe(path)
        if img is None:
            raise RuntimeError(f"Could not load image: {path}")
        if self.transform is not None:
            try:
                img = self.transform(img)
            except Exception as e:
                raise RuntimeError(f"Transform failed for image: {path}\n{type(e).__name__}: {e}") from e
        if self.return_path:
            return img, y, path
        return img, y
class ValidateTensorStats:
    def __init__(self, max_abs=10.0, min_std=1e-3, name="post_normalisation"):
        self.max_abs = max_abs
        self.min_std = min_std
        self.name = name
    def __call__(self, x: torch.Tensor):
        if not torch.isfinite(x).all():
            raise RuntimeError(f"[{self.name}] Non-finite tensor values detected")
        max_val = x.abs().max().item()
        if max_val > self.max_abs:
            raise RuntimeError(f"[{self.name}] Extreme values after normalisation (abs max = {max_val:.2f} > {self.max_abs})")
        std = x.std().item()
        if std < self.min_std:
            raise RuntimeError(f"[{self.name}] Near-constant image (std = {std:.2e} < {self.min_std})")
        return x
def build_transforms(cfg: TrainConfig):
    tfms = [transforms.Resize(cfg.image_size), transforms.ToTensor()]
    if cfg.normalisation == "imagenet":
        tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif cfg.normalisation in ("none", "false", "no", "off"):
        pass
    else:
        print(f"WARNING: unknown normalisation={cfg.normalisation!r}; defaulting to imagenet", flush=True)
        tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tfms.append(ValidateTensorStats(max_abs=10.0, min_std=1e-3, name="after_normalisation"))
    return transforms.Compose(tfms)
def stratified_split(items: List[Tuple[str, int]], val_fraction: float, seed: int):
    rng = random.Random(seed)
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for p, y in items:
        by_class.setdefault(y, []).append((p, y))
    train_items: List[Tuple[str, int]] = []
    val_items: List[Tuple[str, int]] = []
    for _y, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_val = int(round(n * val_fraction))
        if n >= 2:
            n_val = max(1, min(n - 1, n_val))
        else:
            n_val = 0
        val_items.extend(lst[:n_val])
        train_items.extend(lst[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items
def build_model(cfg: TrainConfig, num_classes: int) -> nn.Module:
    arch = normalise_architecture(cfg.arch)
    cfg.arch = arch
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if cfg.pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet34":
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if cfg.pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if cfg.pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT if cfg.pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if cfg.pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    if arch == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT if cfg.pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {cfg.arch}. Supported: {', '.join(SUPPORTED_ARCHITECTURES)}")
def build_optimizer(cfg: TrainConfig, model: nn.Module):
    opt = cfg.optimizer.lower()
    if opt in ("adam", "adamw"):
        print("Using AdamW optimiser.", flush=True)
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, eps=1e-8)
    if opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}. Supported: adam/adamw/sgd")
def freeze_batchnorm(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        module.track_running_stats = False
        for p in module.parameters():
            p.requires_grad = False
def apply_batchnorm_freeze(model: nn.Module, enabled: bool) -> None:
    if enabled:
        model.apply(freeze_batchnorm)
def try_download_from_azure(settings: dict, blob_path: str, dest_path: str, log=print) -> bool:
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception as e:
        log(f"[azure] azure-storage-blob not available: {e}")
        return False
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    dataset_sel = settings.get("dataset_selection", {}) or {}
    account_url = dataset_sel.get("account_url")
    container_name = dataset_sel.get("container")
    try:
        if conn:
            bsc = BlobServiceClient.from_connection_string(conn)
        else:
            try:
                from azure.identity import DefaultAzureCredential
            except Exception as e:
                log(f"[azure] azure-identity not available: {e}")
                return False
            if not (account_url and container_name):
                log("[azure] Missing account_url/container in settings['dataset_selection']; cannot download.")
                return False
            bsc = BlobServiceClient(account_url, credential=DefaultAzureCredential())
        if not container_name:
            log("[azure] Missing container name; cannot download.")
            return False
        bc = bsc.get_container_client(container_name).get_blob_client(blob_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        downloader = bc.download_blob()
        with open(dest_path, "wb") as f:
            for ch in downloader.chunks():
                f.write(ch)
        return True
    except Exception as e:
        log(f"[azure] Download failed for {blob_path}: {e}")
        return False
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, debug_dir: str | None = None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(loader, start=1):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            xb, yb, paths = batch
        else:
            xb, yb = batch
            paths = ["<no path>"] * int(xb.shape[0])
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if yb.dtype != torch.long:
            yb = yb.long()
        state_name, state_tensor, state_kind = first_nonfinite_model_state(model)
        if state_name is not None:
            print(f"Non-finite model {state_kind} before validation forward: {state_name}", flush=True)
            if debug_dir:
                save_debug_batch(debug_dir, "val_model_state", -1, i, paths, xb, yb, extra={"state_name": state_name, "state_kind": state_kind, "state_stats": tensor_stats(state_tensor, state_name)})
            raise RuntimeError(f"Non-finite model {state_kind} before validation forward: {state_name}")
        if not torch.isfinite(xb).all():
            print("Non-finite validation inputs! Examples:", paths[:5], flush=True)
            if debug_dir:
                save_debug_batch(debug_dir, "val_input", -1, i, paths, xb, yb)
            raise RuntimeError("Non-finite validation input tensor")
        logits = model(xb)
        if not torch.isfinite(logits).all():
            print("Non-finite validation logits! Examples:", paths[:5], flush=True)
            if debug_dir:
                save_debug_batch(debug_dir, "val_logits", -1, i, paths, xb, yb, logits)
            raise RuntimeError("Non-finite validation logits")
        loss = criterion(logits, yb)
        if not torch.isfinite(loss).all():
            print("Non-finite validation loss! Examples:", paths[:5], flush=True)
            if debug_dir:
                save_debug_batch(debug_dir, "val_loss", -1, i, paths, xb, yb, logits)
            raise RuntimeError("Non-finite validation loss")
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)
def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer, epoch: int, epochs: int, debug_dir: str, debug_context: dict, freeze_bn: bool) -> float:
    model.train()
    apply_batchnorm_freeze(model, freeze_bn)
    criterion = nn.CrossEntropyLoss()
    running = 0.0
    n = 0
    for i, batch in enumerate(loader, start=1):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            xb, yb, paths = batch
        else:
            xb, yb = batch
            paths = ["<no path>"] * int(xb.shape[0])
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if yb.dtype != torch.long:
            yb = yb.long()
        debug_context.update({"epoch": epoch, "step": i, "paths": paths, "xb": xb, "yb": yb, "debug_dir": debug_dir})
        state_name, state_tensor, state_kind = first_nonfinite_model_state(model)
        if state_name is not None:
            print("\n==================== NON-FINITE MODEL STATE BEFORE FORWARD ====================", flush=True)
            print(f"{state_kind}: {state_name}", flush=True)
            print("Example paths:", paths[:5], flush=True)
            print_tensor_stats(state_tensor, state_name)
            save_debug_batch(debug_dir, "model_state_before_forward", epoch, i, paths, xb, yb, extra={"state_name": state_name, "state_kind": state_kind, "state_stats": tensor_stats(state_tensor, state_name)})
            raise RuntimeError(f"Non-finite model {state_kind} before forward: {state_name}")
        opt_name, opt_tensor = first_nonfinite_optimizer_state(optimizer)
        if opt_name is not None:
            print("\n==================== NON-FINITE OPTIMISER STATE BEFORE FORWARD ====================", flush=True)
            print("Optimiser state:", opt_name, flush=True)
            print_tensor_stats(opt_tensor, opt_name)
            save_debug_batch(debug_dir, "optimizer_before_forward", epoch, i, paths, xb, yb, extra={"optimizer_state": opt_name, "optimizer_state_stats": tensor_stats(opt_tensor, opt_name)})
            raise RuntimeError(f"Non-finite optimiser state before forward: {opt_name}")
        if not torch.isfinite(xb).all():
            print("\n==================== NON-FINITE INPUTS ====================", flush=True)
            print("Example paths:", paths[:5], flush=True)
            print_tensor_stats(xb, "train_xb")
            save_debug_batch(debug_dir, "input", epoch, i, paths, xb, yb)
            raise RuntimeError("Non-finite input tensor")
        optimizer.zero_grad(set_to_none=True)
        try:
            logits = model(xb)
        except Exception as e:
            print("\n==================== MODEL FORWARD FAILED ====================", flush=True)
            print("Example paths:", paths[:5], flush=True)
            print_tensor_stats(xb, "train_xb")
            save_debug_batch(debug_dir, "forward_exception", epoch, i, paths, xb, yb, extra={"exception": repr(e), "traceback": traceback.format_exc()})
            raise
        if not torch.isfinite(logits).all():
            print("\n==================== NON-FINITE LOGITS ====================", flush=True)
            print("Example paths:", paths[:5], flush=True)
            print_tensor_stats(xb, "train_xb")
            print_tensor_stats(logits, "logits")
            bad_rows = ~torch.isfinite(logits).all(dim=1)
            bad_idx = torch.where(bad_rows)[0].detach().cpu().tolist()
            print("Bad sample indices:", bad_idx, flush=True)
            for idx in bad_idx[:20]:
                print("\n--- BAD SAMPLE ---", flush=True)
                print("batch index:", idx, flush=True)
                if isinstance(paths, (list, tuple)):
                    print("path:", paths[idx], flush=True)
                print_tensor_stats(xb[idx], f"xb[{idx}]")
                print_tensor_stats(logits[idx], f"logits[{idx}]")
            save_debug_batch(debug_dir, "logits", epoch, i, paths, xb, yb, logits, {"bad_sample_indices": bad_idx})
            raise RuntimeError("Non-finite logits")
        loss = criterion(logits, yb)
        if not torch.isfinite(loss).all():
            print("\n==================== NON-FINITE LOSS ====================", flush=True)
            print("Example paths:", paths[:5], flush=True)
            print_tensor_stats(logits, "logits")
            save_debug_batch(debug_dir, "loss", epoch, i, paths, xb, yb, logits, {"loss": tensor_stats(loss, "loss")})
            raise RuntimeError("Non-finite loss")
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                print("\n==================== NON-FINITE GRADIENT ====================", flush=True)
                print("Gradient in:", name, flush=True)
                print("Example paths:", paths[:5], flush=True)
                print_tensor_stats(p.grad, f"grad[{name}]")
                save_debug_batch(debug_dir, "gradient", epoch, i, paths, xb, yb, logits, {"gradient_parameter": name, "gradient_stats": tensor_stats(p.grad, f"grad[{name}]")})
                raise RuntimeError(f"Non-finite gradients in {name}")
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        except TypeError:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                save_debug_batch(debug_dir, "grad_norm", epoch, i, paths, xb, yb, logits, {"grad_norm": str(grad_norm)})
                raise RuntimeError(f"Non-finite gradient norm before optimiser step: {grad_norm}")
        except RuntimeError as e:
            save_debug_batch(debug_dir, "grad_norm", epoch, i, paths, xb, yb, logits, {"exception": repr(e)})
            raise
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            save_debug_batch(debug_dir, "grad_norm", epoch, i, paths, xb, yb, logits, {"grad_norm": str(grad_norm)})
            raise RuntimeError(f"Non-finite gradient norm before optimiser step: {grad_norm}")
        optimizer.step()
        state_name, state_tensor, state_kind = first_nonfinite_model_state(model)
        if state_name is not None:
            print("\n==================== NON-FINITE MODEL STATE AFTER OPTIMISER STEP ====================", flush=True)
            print(f"{state_kind}: {state_name}", flush=True)
            print("Example paths:", paths[:5], flush=True)
            print_tensor_stats(state_tensor, state_name)
            save_debug_batch(debug_dir, "model_state_after_step", epoch, i, paths, xb, yb, logits, {"state_name": state_name, "state_kind": state_kind, "state_stats": tensor_stats(state_tensor, state_name), "grad_norm": _safe_float(grad_norm)})
            raise RuntimeError(f"Non-finite model {state_kind} after optimiser step: {state_name}")
        opt_name, opt_tensor = first_nonfinite_optimizer_state(optimizer)
        if opt_name is not None:
            print("\n==================== NON-FINITE OPTIMISER STATE AFTER OPTIMISER STEP ====================", flush=True)
            print("Optimiser state:", opt_name, flush=True)
            print_tensor_stats(opt_tensor, opt_name)
            save_debug_batch(debug_dir, "optimizer_after_step", epoch, i, paths, xb, yb, logits, {"optimizer_state": opt_name, "optimizer_state_stats": tensor_stats(opt_tensor, opt_name), "grad_norm": _safe_float(grad_norm)})
            raise RuntimeError(f"Non-finite optimiser state after step: {opt_name}")
        running += float(loss.item()) * yb.size(0)
        n += yb.size(0)
        if i % 10 == 0 or i == len(loader):
            print(f"[epoch {epoch}/{epochs}] step {i}/{len(loader)} loss={loss.item():.4f}", flush=True)
    return running / max(n, 1)
class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        init_temperature = float(init_temperature)
        if init_temperature <= 0:
            init_temperature = 1.0
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature().clamp(min=1e-6)
@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader: DataLoader, device: torch.device, debug_dir: str | None = None):
    model.eval()
    all_logits = []
    all_labels = []
    for i, batch in enumerate(loader, start=1):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            xb, yb, paths = batch
        else:
            xb, yb = batch
            paths = ["<no path>"] * int(xb.shape[0])
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        if not torch.isfinite(logits).all():
            print("Non-finite calibration logits! Examples:", paths[:5], flush=True)
            if debug_dir:
                save_debug_batch(debug_dir, "calibration_logits", -1, i, paths, xb, yb, logits)
            raise RuntimeError("Non-finite calibration logits")
        all_logits.append(logits.detach().cpu())
        all_labels.append(yb.detach().cpu())
    if not all_logits:
        return None, None
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels).detach().cpu().item())
def ece_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    acc = (preds == labels).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if mask.any():
            ece += torch.abs(acc[mask].mean() - conf[mask].mean()) * mask.float().mean()
    return float(ece.item())
def save_confusion_matrix_artifacts(confusion_matrix_values, class_labels: List[str], csv_path: str, png_path: str) -> dict:
    if not class_labels:
        return {"csv": None, "png": None}
    csv_path = str(csv_path)
    png_path = str(png_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    csv_written = False
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(["true_label", *[str(c) for c in class_labels]])
            for idx, label in enumerate(class_labels):
                writer.writerow([str(label), *[int(v) for v in confusion_matrix_values[idx]]])
        csv_written = True
    except Exception as e:
        print(f"WARNING: could not write confusion matrix CSV: {e}", flush=True)
    png_written = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if hasattr(confusion_matrix_values, "astype"):
            matrix = confusion_matrix_values.astype(int)
        else:
            matrix = [[int(v) for v in row] for row in confusion_matrix_values]
        fig, ax = plt.subplots(figsize=(max(4, len(class_labels) * 1.2), max(4, len(class_labels) * 1.2)))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels([str(c) for c in class_labels], rotation=45, ha="right")
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels([str(c) for c in class_labels])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion matrix")
        if matrix:
            row_count = len(matrix)
            col_count = len(matrix[0]) if matrix[0] else 0
            for i in range(row_count):
                for j in range(col_count):
                    val = int(matrix[i][j])
                    ax.text(j, i, str(val), ha="center", va="center", color="white" if val > (max(max(row) for row in matrix) / 2) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        png_written = True
    except Exception as e:
        print(f"WARNING: could not write confusion matrix PNG: {e}", flush=True)
    return {"csv": csv_path if csv_written else None, "png": png_path if png_written else None}

def fit_temperature_scaling(val_logits: torch.Tensor, val_labels: torch.Tensor) -> Tuple[float, dict]:
    if val_logits is None or val_labels is None:
        return 1.0, {"method": "none", "reason": "no_validation_data"}
    if val_logits.ndim != 2:
        return 1.0, {"method": "none", "reason": "unexpected_logits_shape"}
    n, c = val_logits.shape
    if n < 2 or c < 2:
        return 1.0, {"method": "none", "reason": "insufficient_samples_or_classes", "n": int(n), "c": int(c)}
    device = torch.device("cpu")
    logits = val_logits.to(device=device, dtype=torch.float32)
    labels = val_labels.to(device=device, dtype=torch.long)
    scaler = TemperatureScaler(init_temperature=1.0).to(device)
    pre_nll = nll_from_logits(logits, labels)
    pre_ece = ece_from_logits(logits, labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.5, max_iter=50, line_search_fn="strong_wolfe")
    def _closure():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss
    optimizer.step(_closure)
    with torch.no_grad():
        t = float(scaler.temperature().clamp(min=1e-6).item())
        calibrated_logits = logits / t
        post_nll = nll_from_logits(calibrated_logits, labels)
        post_ece = ece_from_logits(calibrated_logits, labels)
    return t, {"method": "temperature_scaling", "temperature": t, "fitted_on": "validation_split", "n_val": int(n), "pre": {"nll": pre_nll, "ece": pre_ece}, "post": {"nll": post_nll, "ece": post_ece}}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, help="Path to modeltrainsettings.json")
    ap.add_argument("--output", required=True, help="Path to write model artifact, e.g. model.pt")
    ap.add_argument("--allow-azure-download", action="store_true", help="Attempt Azure download for items missing local_path.")
    ap.add_argument("--cache-dir", default=None, help="Optional cache dir for Azure downloads if local_path missing.")
    ap.add_argument("--no-activation-hooks", action="store_true", help="Disable layer-wise NaN/Inf activation hooks if too slow/noisy.")
    ap.add_argument("--no-freeze-batchnorm", action="store_true", help="Override config and do not freeze BatchNorm layers.")
    args = ap.parse_args()
    settings_path = os.path.abspath(args.settings)
    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug_nonfinite")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Loading settings: {settings_path}", flush=True)
    print(f"Non-finite debug output directory: {debug_dir}", flush=True)
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    cfg = parse_train_config(settings)
    if args.no_freeze_batchnorm:
        cfg.freeze_batchnorm = False
    seed_everything(cfg.seed)
    files = settings.get("files", []) or []
    if not files:
        raise RuntimeError("settings['files'] is empty - nothing to train on.")
    class_labels = sorted({(x.get("class_label") or "UNKNOWN") for x in files})
    class_to_idx = {c: i for i, c in enumerate(class_labels)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    items: List[Tuple[str, int]] = []
    missing_local: List[dict] = []
    cache_dir = os.path.abspath(args.cache_dir or os.path.join(out_dir, "cache_images"))
    for entry in files:
        c = entry.get("class_label") or "UNKNOWN"
        y = class_to_idx.get(c, class_to_idx["UNKNOWN"] if "UNKNOWN" in class_to_idx else 0)
        lp = entry.get("local_path")
        if lp and os.path.exists(lp):
            items.append((lp, y))
            continue
        if args.allow_azure_download:
            bp = entry.get("blob_path")
            if not bp:
                missing_local.append(entry)
                continue
            base = os.path.basename(bp)
            pathhash = hashlib.sha256(bp.encode("utf-8")).hexdigest()[:12]
            dest = os.path.join(cache_dir, f"{pathhash}_{base}")
            if not os.path.exists(dest):
                ok = try_download_from_azure(settings, bp, dest, log=lambda s: print(s, flush=True))
                if not ok:
                    missing_local.append(entry)
                    continue
            entry["local_path"] = dest
            items.append((dest, y))
        else:
            missing_local.append(entry)
    if missing_local:
        raise RuntimeError(f"{len(missing_local)} items have no usable local_path. Rebuild the manifest with caching enabled or re-run with --allow-azure-download.")
    print(f"Discovered {len(items)} local images across {len(class_labels)} classes.", flush=True)
    train_items, val_items = stratified_split(items, cfg.val_fraction, cfg.seed)
    print(f"Split: train={len(train_items)} val={len(val_items)} (val_fraction={cfg.val_fraction})", flush=True)
    print(f"Training config: arch={cfg.arch} pretrained={cfg.pretrained} lr={cfg.learning_rate} optimizer={cfg.optimizer} weight_decay={cfg.weight_decay} freeze_batchnorm={cfg.freeze_batchnorm}", flush=True)
    tfm = build_transforms(cfg)
    train_ds = ImageListDataset(train_items, transform=tfm, return_path=True)
    val_ds = ImageListDataset(val_items, transform=tfm, return_path=True) if val_items else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")) if val_ds is not None else None
    if len(class_labels) < 2:
        raise RuntimeError(f"Need at least 2 classes for multiclass training, found {len(class_labels)}: {class_labels}")
    model = build_model(cfg, num_classes=len(class_labels)).to(device)
    apply_batchnorm_freeze(model, cfg.freeze_batchnorm)
    if cfg.freeze_batchnorm:
        print("BatchNorm layers frozen/eval during training. Use --no-freeze-batchnorm to disable.", flush=True)
    optimizer = build_optimizer(cfg, model)
    debug_context = {"debug_dir": debug_dir}
    hook_handles = []
    if not args.no_activation_hooks:
        hook_handles = register_nonfinite_activation_hooks(model, debug_context)
        print(f"Registered {len(hook_handles)} non-finite activation hooks.", flush=True)
    best_val_acc = -1.0
    best_state = None
    history = []
    monitor = (cfg.early_stopping_monitor or "val_loss").lower()
    mode = (cfg.early_stopping_mode or "").lower()
    if mode not in ("min", "max"):
        mode = "min" if "loss" in monitor else "max"
    best_metric = float("inf") if mode == "min" else -float("inf")
    best_metric_epoch = None
    no_improve = 0
    stopped_epoch = None
    try:
        for epoch in range(1, cfg.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, device, optimizer, epoch, cfg.epochs, debug_dir, debug_context, cfg.freeze_batchnorm)
            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, device, debug_dir=debug_dir)
            else:
                val_loss, val_acc = float("nan"), float("nan")
            print(f"[epoch {epoch}/{cfg.epochs}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc})
            if val_loader is not None and val_acc == val_acc and val_acc > best_val_acc:
                best_val_acc = val_acc
            if monitor == "val_loss":
                metric = val_loss
            elif monitor == "val_acc":
                metric = val_acc
            elif monitor == "train_loss":
                metric = tr_loss
            else:
                metric = val_loss if val_loader is not None else tr_loss
            if metric != metric:
                metric = tr_loss
            improved = metric < (best_metric - cfg.early_stopping_min_delta) if mode == "min" else metric > (best_metric + cfg.early_stopping_min_delta)
            if improved or best_state is None:
                best_metric = float(metric)
                best_metric_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
                print(f"  -> new best by {monitor} ({mode}) = {best_metric:.6f} at epoch {epoch}", flush=True)
            else:
                no_improve += 1
            if cfg.early_stopping_enabled and epoch >= cfg.early_stopping_warmup_epochs and no_improve >= cfg.early_stopping_patience:
                stopped_epoch = epoch
                print(f"Early stopping: no improvement in {monitor} for {no_improve} epoch(s) (patience={cfg.early_stopping_patience}, min_delta={cfg.early_stopping_min_delta}). Best epoch was {best_metric_epoch} with {monitor}={best_metric:.6f}.", flush=True)
                break
    finally:
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
    def _norm_out_path(p: str) -> str:
        if not p or str(p).lower() in ("none", "null", "false", "off"):
            return ""
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(out_dir, p)
    csv_path = _norm_out_path(getattr(cfg, "curve_csv", "training_curve.csv"))
    if csv_path:
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
                w.writeheader()
                for row in history:
                    w.writerow(row)
            print(f"Saved training curve CSV: {csv_path}", flush=True)
        except Exception as e:
            print(f"WARNING: could not write curve CSV: {e}", flush=True)
    png_path = _norm_out_path(getattr(cfg, "curve_png", "training_curve.png"))
    if png_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            epochs_x = [h["epoch"] for h in history]
            tr_loss_x = [h["train_loss"] for h in history]
            vl_loss_x = [h["val_loss"] for h in history]
            vl_acc_x = [h["val_acc"] for h in history]
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(epochs_x, tr_loss_x, label="train_loss")
            if any(v == v for v in vl_loss_x):
                ax1.plot(epochs_x, vl_loss_x, label="val_loss")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("loss")
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            if any(v == v for v in vl_acc_x):
                ax2.plot(epochs_x, vl_acc_x, color="tab:green", label="val_acc")
                ax2.set_ylabel("val_acc")
                ax2.set_ylim(0.0, 1.0)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"Saved training curve PNG: {png_path}", flush=True)
        except Exception as e:
            print(f"WARNING: could not write curve PNG: {e}", flush=True)
    calibration_info = {"method": "none", "reason": "no_validation_split"}
    confusion_matrix_info = {"csv": None, "png": None}
    if val_loader is not None and best_state is not None and len(val_items) > 0:
        best_model = build_model(cfg, num_classes=len(class_labels)).to(device)
        best_model.load_state_dict(best_state)
        if cfg.freeze_batchnorm:
            apply_batchnorm_freeze(best_model, True)
        val_logits, val_labels = collect_logits_labels(best_model, val_loader, device, debug_dir=debug_dir)
        _temperature, calibration_info = fit_temperature_scaling(val_logits, val_labels)
        if calibration_info.get("method") == "temperature_scaling":
            print(f"Calibration: fitted temperature={calibration_info.get('temperature'):.6f} (pre_nll={calibration_info.get('pre', {}).get('nll'):.4f} -> post_nll={calibration_info.get('post', {}).get('nll'):.4f}, pre_ece={calibration_info.get('pre', {}).get('ece'):.4f} -> post_ece={calibration_info.get('post', {}).get('ece'):.4f})", flush=True)
        else:
            print(f"Calibration skipped: {calibration_info}", flush=True)
        if val_logits is not None and val_labels is not None:
            val_preds = val_logits.argmax(dim=1).detach().cpu().tolist()
            val_true = val_labels.detach().cpu().tolist()
            cm = confusion_matrix(val_true, val_preds, labels=list(range(len(class_labels))))
            cm_csv = _norm_out_path(getattr(cfg, "confusion_matrix_csv", "confusion_matrix.csv"))
            cm_png = _norm_out_path(getattr(cfg, "confusion_matrix_png", "confusion_matrix.png"))
            confusion_matrix_info = save_confusion_matrix_artifacts(cm, class_labels, cm_csv, cm_png)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    settings_hash = sha256_file(settings_path)
    artifact = {
        "model_state_dict": best_state,
        "arch": cfg.arch,
        "pretrained": cfg.pretrained,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": list(cfg.image_size),
        "normalisation": cfg.normalisation,
        "trained_utc": utc_now_iso(),
        "training_run_id": settings.get("training_run_id"),
        "dataset_fingerprints": settings.get("dataset_fingerprints"),
        "settings_sha256": settings_hash,
        "history": history,
        "early_stopping": {"enabled": cfg.early_stopping_enabled, "monitor": monitor, "mode": mode, "patience": cfg.early_stopping_patience, "min_delta": cfg.early_stopping_min_delta, "warmup_epochs": cfg.early_stopping_warmup_epochs, "best_epoch": best_metric_epoch, "best_metric": best_metric, "stopped_epoch": stopped_epoch},
        "training_curve": {"csv": csv_path or None, "png": png_path or None},
        "confusion_matrix": confusion_matrix_info,
        "debugging": {"debug_nonfinite_dir": debug_dir, "freeze_batchnorm": cfg.freeze_batchnorm, "activation_hooks_enabled": not args.no_activation_hooks},
        "calibration": calibration_info,
    }
    torch.save(artifact, out_path)
    print(f"Saved model artifact: {out_path}", flush=True)
    report_path = os.path.join(out_dir, "training_report.json")
    report = {
        "trained_utc": artifact["trained_utc"],
        "training_run_id": artifact["training_run_id"],
        "arch": artifact["arch"],
        "pretrained": artifact["pretrained"],
        "num_classes": len(class_labels),
        "classes": class_labels,
        "image_size": artifact["image_size"],
        "normalisation": artifact["normalisation"],
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "optimizer": "adamw" if cfg.optimizer in ("adam", "adamw") else cfg.optimizer,
        "weight_decay": cfg.weight_decay,
        "num_workers": cfg.num_workers,
        "val_fraction": cfg.val_fraction,
        "seed": cfg.seed,
        "freeze_batchnorm": cfg.freeze_batchnorm,
        "best_val_acc": best_val_acc,
        "settings_path": settings_path,
        "settings_sha256": settings_hash,
        "model_path": out_path,
        "debug_nonfinite_dir": debug_dir,
        "calibration": calibration_info,
        "confusion_matrix": confusion_matrix_info,
        "inference_note": {"multiclass_probabilities": "probs = softmax(logits / temperature)", "temperature": calibration_info.get("temperature", 1.0)},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    settings["training_report"] = report
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    print("Updated modeltrainsettings.json with training_report", flush=True)
if __name__ == "__main__":
    main()
