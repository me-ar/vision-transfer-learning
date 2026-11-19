"""Models, training loop, Grad-CAM and robustness perturbations."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import batches, preprocess


class SmallCNN(nn.Module):
    """A compact 4-block CNN trained from scratch (~0.6M parameters)."""

    def __init__(self, n_classes: int = 10, width: int = 32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        w = width
        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18(n_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights
    from torchvision.models import resnet18 as tv_resnet18
    m = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def n_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@dataclass
class TrainConfig:
    epochs: int = 8
    lr: float = 3e-3
    batch_size: int = 128
    size: int | None = None
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    seed: int = 42


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, size: int | None, batch_size: int = 256,
            perturb=None) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in batches(x, torch.zeros(len(x)), batch_size, shuffle=False):
        xb = preprocess(xb, size)
        if perturb is not None:
            xb = perturb(xb)
        out.append(F.softmax(model(xb), -1))
    return torch.cat(out).numpy()


def fit(model: nn.Module, sp, cfg: TrainConfig, log=print) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    steps = cfg.epochs * ((len(sp.y_train) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best, best_state, history, t0 = -1.0, None, [], time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in batches(sp.x_train, sp.y_train, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch):
            loss = loss_fn(model(preprocess(xb, cfg.size, augment=True)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        val_acc = float((predict(model, sp.x_val, cfg.size).argmax(1) == sp.y_val.numpy()).mean())
        history.append({"epoch": epoch, "train_loss": round(tot / n, 4), "val_acc": round(val_acc, 4),
                        "elapsed_s": round(time.time() - t0, 1)})
        log(f"  epoch {epoch}: loss={tot / n:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)")
        if val_acc > best:
            best, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"history": history, "train_seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def extract_features(backbone: nn.Module, x: torch.Tensor, size: int, batch_size: int = 256) -> np.ndarray:
    """Penultimate-layer features of a (frozen) ResNet."""
    body = nn.Sequential(*list(backbone.children())[:-1]).eval()
    feats = [body(preprocess(xb, size)).flatten(1) for xb, _ in batches(x, torch.zeros(len(x)), batch_size, False)]
    return torch.cat(feats).numpy()


# ---------------------------------------------------------------- explainability
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017): weight the last conv feature maps by the pooled gradient
    of the class score, ReLU, upsample — shows *where* the network looked."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        target = logits.argmax(1) if target is None else target
        self.model.zero_grad()
        logits.gather(1, target.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        return (cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)).detach()


# ---------------------------------------------------------------- robustness
def gaussian_noise(sigma: float):
    def f(x):
        g = torch.Generator().manual_seed(0)
        return x + sigma / 0.226 * torch.randn(x.shape, generator=g)   # sigma in [0,1] pixel units
    return f


def gaussian_blur(kernel: int):
    from torchvision.transforms.functional import gaussian_blur as gb
    return lambda x: gb(x, kernel_size=[kernel, kernel])


def brightness(delta: float):
    return lambda x: x + delta / 0.226


PERTURBATIONS = {"clean": None, "noise σ=0.05": gaussian_noise(0.05), "noise σ=0.1": gaussian_noise(0.1),
                 "noise σ=0.2": gaussian_noise(0.2), "blur k=3": gaussian_blur(3), "blur k=5": gaussian_blur(5),
                 "brightness +0.2": brightness(0.2)}
