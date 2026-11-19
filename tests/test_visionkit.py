"""Fast offline tests on synthetic images (no downloads)."""
import numpy as np
import pytest
import torch

from visionkit.data import preprocess, split
from visionkit.experiment import synthetic
from visionkit.models import (
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


@pytest.fixture(scope="module")
def sp():
    x, y, classes = synthetic(n_per_class=30, n_classes=4, size=32)
    return split(x, y, classes)


def test_split_is_stratified_and_disjoint(sp):
    assert len(sp.y_train) + len(sp.y_val) + len(sp.y_test) == 120
    counts = torch.bincount(sp.y_test).tolist()
    assert len(counts) == 4 and max(counts) - min(counts) <= 1


def test_preprocess_shapes_and_normalisation(sp):
    x = preprocess(sp.x_train[:8], size=48, augment=True)
    assert x.shape == (8, 3, 48, 48) and x.dtype == torch.float32
    assert abs(float(x.mean())) < 3


def test_small_cnn_learns_synthetic_classes(sp):
    model, info = fit(SmallCNN(4, width=8), sp, TrainConfig(epochs=4, lr=1e-2, batch_size=16), log=lambda *_: None)
    acc = (predict(model, sp.x_test, None).argmax(1) == sp.y_test.numpy()).mean()
    assert acc > 0.5 and len(info["history"]) == 4


def test_resnet_head_and_features(sp):
    m = resnet18(4, pretrained=False)
    assert m.fc.out_features == 4 and n_params(m) > 10_000_000
    f = extract_features(m, sp.x_test[:5], size=32)
    assert f.shape == (5, 512)


def test_gradcam_returns_normalised_maps(sp):
    m = resnet18(4, pretrained=False)
    cams = GradCAM(m, m.layer3)(preprocess(sp.x_test[:3], 64))
    assert cams.shape == (3, 64, 64) and float(cams.min()) >= 0 and float(cams.max()) <= 1 + 1e-6


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_perturbations_preserve_shape(sp, name):
    x = preprocess(sp.x_test[:4], 32)
    f = PERTURBATIONS[name]
    assert (x if f is None else f(x)).shape == x.shape


def test_predict_outputs_probabilities(sp):
    p = predict(SmallCNN(4, width=8), sp.x_test, None)
    assert np.allclose(p.sum(1), 1, atol=1e-5)
