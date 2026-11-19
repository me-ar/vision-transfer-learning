"""EuroSAT benchmark: CNN from scratch vs linear probe vs full fine-tuning, + Grad-CAM and robustness.

Usage: python -m visionkit.experiment          # full run (downloads EuroSAT + ImageNet weights)
       python -m visionkit.experiment --smoke  # synthetic data, random weights, seconds
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from . import data
from .data import MEAN, STD, preprocess
from .models import (
    PERTURBATIONS,
    GradCAM,
    SmallCNN,
    TrainConfig,
    extract_features,
    fit,
    n_params,
    predict,
    resnet18,
)

LABEL_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 1.0]


def synthetic(n_per_class: int = 60, n_classes: int = 10, size: int = 32):
    """Class-dependent colour tints + noise — enough signal to exercise the whole pipeline."""
    g = torch.Generator().manual_seed(0)
    tints = torch.rand(n_classes, 3, 1, 1, generator=g)
    xs, ys = [], []
    for c in range(n_classes):
        base = tints[c] * 200 + torch.rand(n_per_class, 3, size, size, generator=g) * 55
        xs.append(base.clamp(0, 255).to(torch.uint8))
        ys += [c] * n_per_class
    return torch.cat(xs), torch.tensor(ys), [f"class_{i}" for i in range(n_classes)]


def scores(y, proba) -> dict:
    pred = proba.argmax(1)
    return {"accuracy": round(float((pred == y).mean()), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4)}


def latency_ms(model, x, size, n=256) -> float:
    xs = x[:n]
    predict(model, xs, size)  # warm-up
    t = time.perf_counter()
    predict(model, xs, size)
    return 1000 * (time.perf_counter() - t) / len(xs)


def main(smoke: bool = False, out: Path | None = None) -> dict:
    out = out or data.ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    x, y, classes = synthetic() if smoke else data.load_eurosat()
    sp = data.split(x, y, classes)
    yt = sp.y_test.numpy()
    big = None if smoke else 96
    print(f"train={len(sp.y_train)} val={len(sp.y_val)} test={len(sp.y_test)} classes={len(classes)}", flush=True)
    rows, probas, models, extra = [], {}, {}, {}

    # 1) Small CNN from scratch
    print("== SmallCNN (from scratch)", flush=True)
    cnn = SmallCNN(len(classes))
    cnn, info = fit(cnn, sp, TrainConfig(epochs=1 if smoke else 15, lr=3e-3, size=None))
    p = predict(cnn, sp.x_test, None)
    rows.append({"model": "Small CNN (from scratch)", **scores(yt, p), "params": n_params(cnn),
                 "trainable_params": n_params(cnn, True), "train_seconds": info["train_seconds"],
                 "ms_per_image": round(latency_ms(cnn, sp.x_test, None), 3)})
    probas[rows[-1]["model"]], models["cnn"], extra["cnn"] = p, cnn, info

    # 2) Linear probe on frozen ImageNet ResNet18 features (+ label-efficiency curve)
    print("== ResNet18 linear probe", flush=True)
    backbone = resnet18(len(classes), pretrained=not smoke)
    t = time.perf_counter()
    f_tr, f_te = extract_features(backbone, sp.x_train, big), extract_features(backbone, sp.x_test, big)
    scaler = StandardScaler().fit(f_tr)
    probe = LogisticRegression(C=0.5, max_iter=3000).fit(scaler.transform(f_tr), sp.y_train.numpy())
    p = probe.predict_proba(scaler.transform(f_te))
    rows.append({"model": "ResNet18 linear probe (frozen)", **scores(yt, p), "params": n_params(backbone),
                 "trainable_params": int(probe.coef_.size + probe.intercept_.size),
                 "train_seconds": round(time.perf_counter() - t, 1), "ms_per_image": None})
    probas[rows[-1]["model"]] = p
    curve, rng = [], np.random.default_rng(0)
    for frac in LABEL_FRACTIONS:
        idx = np.concatenate([rng.permutation(np.where(sp.y_train.numpy() == c)[0])[:max(2, int(frac * (sp.y_train == c).sum()))]
                              for c in range(len(classes))])
        sc = StandardScaler().fit(f_tr[idx])
        lp = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(f_tr[idx]), sp.y_train.numpy()[idx])
        curve.append({"label_fraction": frac, "n_labels": int(len(idx)),
                      **scores(yt, lp.predict_proba(sc.transform(f_te)))})
        print(f"   probe with {len(idx)} labels: acc={curve[-1]['accuracy']:.4f}", flush=True)

    # 3) Full fine-tuning of the pretrained ResNet18
    print("== ResNet18 fine-tuned", flush=True)
    ft = resnet18(len(classes), pretrained=not smoke)
    ft, info = fit(ft, sp, TrainConfig(epochs=1 if smoke else 5, lr=5e-4, batch_size=64, size=big))
    p = predict(ft, sp.x_test, big)
    rows.append({"model": "ResNet18 fine-tuned (ImageNet init)", **scores(yt, p), "params": n_params(ft),
                 "trainable_params": n_params(ft, True), "train_seconds": info["train_seconds"],
                 "ms_per_image": round(latency_ms(ft, sp.x_test, big), 3)})
    probas[rows[-1]["model"]], models["ft"], extra["ft"] = p, ft, info

    results = pd.DataFrame(rows)
    print(results.to_string(index=False), flush=True)

    # Robustness to common corruptions
    rob = []
    for name, (m, size) in {"Small CNN (from scratch)": (models["cnn"], None),
                            "ResNet18 fine-tuned (ImageNet init)": (models["ft"], big)}.items():
        for pname, pert in PERTURBATIONS.items():
            acc = float((predict(m, sp.x_test, size, perturb=pert).argmax(1) == yt).mean())
            rob.append({"model": name, "perturbation": pname, "accuracy": round(acc, 4)})
    rob = pd.DataFrame(rob)
    print(rob.pivot(index="perturbation", columns="model", values="accuracy").to_string(), flush=True)

    best = results.sort_values("accuracy").iloc[-1]["model"]
    pred = probas[best].argmax(1)
    cm = confusion_matrix(yt, pred, labels=range(len(classes)))
    per_class = {c: round(float(cm[i, i] / cm[i].sum()), 4) for i, c in enumerate(classes)}
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    summary = {"best_model": best, "results": results.to_dict(orient="records"),
               "label_efficiency_linear_probe": curve, "robustness": rob.to_dict(orient="records"),
               "per_class_accuracy": per_class,
               "most_confused": {"true": classes[i], "pred": classes[j], "count": int(off[i, j])},
               "training_history": {k: v["history"] for k, v in extra.items()}}
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "results.md").write_text(results.to_markdown(index=False) + "\n\n" +
                                    rob.pivot(index="perturbation", columns="model", values="accuracy").to_markdown() + "\n")

    # ---------------- figures ----------------
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(results)), results["accuracy"], color=["#94a3b8", "#16a34a", "#2563eb"])
    for k, v in enumerate(results["accuracy"]):
        ax.text(k, v + 0.003, f"{v:.3f}", ha="center")
    ax.set_xticks(range(len(results)), [m.replace(" (", "\n(") for m in results["model"]], fontsize=8)
    ax.set(ylim=(max(0, results["accuracy"].min() - 0.1), 1.0), ylabel="test accuracy",
           title=f"EuroSAT test set ({len(yt):,} images, {len(classes)} classes)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "model_comparison.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    c = pd.DataFrame(curve)
    ax.plot(c["n_labels"], c["accuracy"], "o-", color="#16a34a", label="ResNet18 linear probe")
    ax.axhline(results.iloc[0]["accuracy"], ls="--", color="#94a3b8", label="Small CNN, all labels")
    ax.set(xscale="log", xlabel="labelled training images", ylabel="test accuracy",
           title="Pretrained features need far fewer labels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "label_efficiency.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    piv = rob.pivot(index="perturbation", columns="model", values="accuracy").loc[list(PERTURBATIONS)]
    piv.plot.bar(ax=ax, color=["#94a3b8", "#2563eb"])
    ax.set(ylabel="test accuracy", xlabel="", title="Robustness to image corruptions", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "robustness.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for a in range(len(classes)):
        for b in range(len(classes)):
            if cm[a, b]:
                ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=7,
                        color="white" if norm[a, b] > 0.5 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set(xlabel="predicted", ylabel="true", title=f"Confusion matrix — {best}")
    fig.tight_layout()
    fig.savefig(out / "figures" / "confusion_matrix.png")

    # Grad-CAM on the fine-tuned ResNet: 4 correct + 2 incorrect test images
    ft_pred = probas["ResNet18 fine-tuned (ImageNet init)"].argmax(1)
    right, wrong = np.where(ft_pred == yt)[0], np.where(ft_pred != yt)[0]
    pick = np.concatenate([np.random.default_rng(1).choice(right, 4, replace=False), wrong[:2]])
    xb = preprocess(sp.x_test[pick], big)
    cams = GradCAM(models["ft"], models["ft"].layer3)(xb)  # layer3: 6x6 maps at 96px, finer than layer4
    imgs = (xb * STD + MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 4.6))
    for k, idx in enumerate(pick):
        axes[0, k].imshow(imgs[k])
        axes[0, k].set_title(f"true: {classes[yt[idx]]}", fontsize=7)
        axes[1, k].imshow(imgs[k])
        axes[1, k].imshow(cams[k].numpy(), cmap="jet", alpha=0.45)
        ok = ft_pred[idx] == yt[idx]
        axes[1, k].set_title(f"pred: {classes[ft_pred[idx]]}", fontsize=7, color="green" if ok else "red")
        for a in axes[:, k]:
            a.axis("off")
    fig.suptitle("Grad-CAM: where the fine-tuned ResNet18 looks (bottom row)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figures" / "gradcam.png")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.smoke, Path("/tmp/vision_smoke") if a.smoke else None)
