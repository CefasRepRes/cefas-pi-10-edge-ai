import numpy as np
import pyvips
import torch
import torchvision

SUPPORTED_ARCHITECTURES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
)

def _normalise_arch(arch):
    arch = str(arch or "resnet18").strip().lower().replace("-", "_")
    aliases = {
        "mobilenet": "mobilenet_v3_small",
        "mobilenetv2": "mobilenet_v2",
        "mobilenet_v2": "mobilenet_v2",
        "mobilenetv3": "mobilenet_v3_small",
        "mobilenetv3_small": "mobilenet_v3_small",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "mobilenetv3_large": "mobilenet_v3_large",
        "mobilenet_v3_large": "mobilenet_v3_large",
        "resnet": "resnet18",
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
    }
    return aliases.get(arch, arch)

def build_model(arch="resnet18", num_classes=2, pretrained=True):
    arch = _normalise_arch(arch)
    num_classes = int(num_classes)
    if num_classes < 2:
        raise RuntimeError(f"num_classes must be >= 2, got {num_classes}")
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights=(torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None))
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet34":
        model = torchvision.models.resnet34(weights=(torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None))
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        model = torchvision.models.resnet50(weights=(torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None))
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(weights=(torchvision.models.MobileNet_V2_Weights.DEFAULT if pretrained else None))
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(weights=(torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None))
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    if arch == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large(weights=(torchvision.models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None))
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise RuntimeError(f"Unsupported architecture: {arch}. Supported: {', '.join(SUPPORTED_ARCHITECTURES)}")

def _num_classes_from_artifact(artifact):
    idx_to_class = artifact.get("idx_to_class") or {}
    num_classes = len(idx_to_class) if hasattr(idx_to_class, "__len__") else 0
    if num_classes < 2:
        c2i = artifact.get("class_to_idx") or {}
        num_classes = max(num_classes, len(c2i))
    if num_classes < 2:
        raise RuntimeError(f"Model artifact missing class mapping (num_classes={num_classes}).")
    return num_classes

def build_model_from_artifact(artifact: dict):
    arch = _normalise_arch(artifact.get("arch", "resnet18"))
    pretrained = bool(artifact.get("pretrained", True))
    return build_model(arch=arch, num_classes=_num_classes_from_artifact(artifact), pretrained=pretrained)

def build_inference_transform(artifact: dict):
    """Build a pyvips-backed inference transform for ResNet or MobileNet artifacts."""
    size = artifact.get("image_size", [256, 256])
    if isinstance(size, (list, tuple)) and len(size) == 2:
        image_size = (int(size[0]), int(size[1]))
    else:
        image_size = (256, 256)
    out_h, out_w = image_size
    norm = str(artifact.get("normalisation", "imagenet")).lower()
    use_imagenet_norm = norm not in ("none", "false", "no", "off", "null")
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    def _ensure_rgb_uchar(im: pyvips.Image) -> pyvips.Image:
        try:
            if im.hasalpha():
                im = im.flatten(background=[255, 255, 255])
        except Exception:
            pass
        if im.bands != 3:
            try:
                im = im.colourspace("srgb")
            except Exception:
                if im.bands == 1:
                    im = im.bandjoin([im, im])
                elif im.bands > 3:
                    im = im.extract_band(0, n=3)
                else:
                    raise
        if im.bands > 3:
            im = im.extract_band(0, n=3)
        if im.format != "uchar":
            im = im.cast("uchar")
        return im
    def tfm(image_bytes: bytes) -> torch.Tensor:
        im = pyvips.Image.thumbnail_buffer(image_bytes, out_w, height=out_h, size="force")
        im = _ensure_rgb_uchar(im)
        arr = np.frombuffer(im.write_to_memory(), dtype=np.uint8)
        arr = arr.reshape(im.height, im.width, im.bands)
        arr = np.ascontiguousarray(arr[:, :, :3].transpose(2, 0, 1)).copy()
        x = torch.from_numpy(arr).float().div_(255.0)
        if use_imagenet_norm:
            x = (x - mean) / std
        return x
    return tfm

def idx_to_label_fn(idx_to_class: dict):
    def idx_to_label(i: int):
        if i in idx_to_class:
            return idx_to_class[i]
        si = str(i)
        if si in idx_to_class:
            return idx_to_class[si]
        return str(i)
    return idx_to_label
