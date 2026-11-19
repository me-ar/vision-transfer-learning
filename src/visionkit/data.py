"""EuroSAT RGB: 27,000 Sentinel-2 satellite tiles (64x64) across 10 land-use classes.

All images are decoded once into a uint8 tensor (~330 MB) and cached, so training epochs do
batched tensor augmentation instead of slow per-image PIL work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE = DATA_DIR / "eurosat_uint8.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet statistics (pretrained backbones)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Splits:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    classes: list[str]


def load_eurosat() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if CACHE.exists():
        d = torch.load(CACHE)
        return d["x"], d["y"], d["classes"]
    from torchvision.datasets import EuroSAT
    from torchvision.transforms.functional import pil_to_tensor
    ds = EuroSAT(str(DATA_DIR), download=True)
    x = torch.stack([pil_to_tensor(ds[i][0].convert("RGB")) for i in range(len(ds))])
    y = torch.tensor(ds.targets)
    torch.save({"x": x, "y": y, "classes": ds.classes}, CACHE)
    return x, y, list(ds.classes)


def split(x: torch.Tensor, y: torch.Tensor, classes: list[str], seed: int = 42) -> Splits:
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.3, stratify=y.numpy(), random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y.numpy()[rest], random_state=seed)
    return Splits(x[tr], y[tr], x[va], y[va], x[te], y[te], classes)


def preprocess(x_uint8: torch.Tensor, size: int | None = None, augment: bool = False) -> torch.Tensor:
    """uint8 NCHW -> normalised float, optional resize and label-preserving augmentation.

    Satellite tiles have no canonical orientation, so flips and 90-degree rotations are free augmentation.
    """
    x = x_uint8.float() / 255.0
    if augment:
        if torch.rand(1) < 0.5:
            x = x.flip(3)
        if torch.rand(1) < 0.5:
            x = x.flip(2)
        x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))
    if size and size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


def batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, seed: int = 0):
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(len(y))
    for i in range(0, len(y), batch_size):
        j = order[i:i + batch_size]
        yield x[j], y[j]
