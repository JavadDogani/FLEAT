"""Dataset loaders (no downloads) and model construction."""
from __future__ import annotations

import pytest
import torch

from fleat_project.data import DatasetFactory
from fleat_project.models import EdgeIotCNN, EdgeIotGRU, ModelFactory, SmallImageCNN

# ------------------------------------------------------------------------- data


def test_synthetic_dataset_shapes_and_determinism():
    tr, te, n_classes, in_ch = DatasetFactory.load_synthetic_image_dataset(n_train=20, n_test=8, seed=3)
    assert (len(tr), len(te), n_classes, in_ch) == (20, 8, 10, 1)
    x, y = tr[0]
    assert x.shape == (1, 28, 28)
    assert 0 <= int(y) < 10
    tr2, _, _, _ = DatasetFactory.load_synthetic_image_dataset(n_train=20, n_test=8, seed=3)
    assert torch.equal(tr.x, tr2.x) and torch.equal(tr.y, tr2.y)


def write_edge_csv(path, n=20):
    rows = ["f1,f2,proto,Attack_type"]
    for i in range(n):
        label = "Normal" if i % 2 == 0 else "DDoS"
        rows.append(f"{i},{i * 0.5},tcp,{label}")
    path.write_text("\n".join(rows) + "\n")


def test_edge_csv_loader_detects_label_and_splits(tmp_path):
    csv = tmp_path / "edge.csv"
    write_edge_csv(csv)
    tr, te, n_classes, n_features = DatasetFactory.load_edge_iiotset_csv(str(csv), test_ratio=0.25, seed=0)
    # No standard label name -> last non-numeric column ("Attack_type") is the label.
    assert n_classes == 2
    assert n_features == 3  # f1, f2 and the non-numeric "proto" (coerced to 0)
    assert (len(tr), len(te)) == (15, 5)
    x, y = tr[0]
    assert x.dtype == torch.float32 and x.shape == (3,)
    assert float(torch.cat([tr.x[:, 2], te.x[:, 2]]).abs().sum()) == 0.0


def test_edge_csv_loader_honours_explicit_label_column(tmp_path):
    csv = tmp_path / "edge.csv"
    write_edge_csv(csv)
    _, _, n_classes, n_features = DatasetFactory.load_edge_iiotset_csv(str(csv), label_col="proto", test_ratio=0.25)
    assert n_classes == 1
    assert n_features == 3


def test_edge_dataset_requires_csv_path(make_args):
    with pytest.raises(ValueError, match="edge_csv_path"):
        DatasetFactory.load_from_args(make_args("--dataset", "edge_iiotset"))


def test_sklearn_digits_loader():
    pytest.importorskip("sklearn")
    tr, te, n_classes, in_ch = DatasetFactory.load_sklearn_digits_as_tensors(test_ratio=0.2, seed=0)
    assert (n_classes, in_ch) == (10, 1)
    assert len(tr) + len(te) == 1797
    assert len(tr) == int(0.8 * 1797)
    x, _ = tr[0]
    assert x.shape == (1, 8, 8)
    assert 0.0 <= float(tr.x.min()) and float(tr.x.max()) <= 1.0


# ----------------------------------------------------------------------- models


def test_unknown_model_name_falls_back_to_small_cnn():
    model = ModelFactory.build("small_cnn", num_classes=10, in_channels=1)
    assert isinstance(model, SmallImageCNN)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.parametrize("name, cls", [("cnn_edge", EdgeIotCNN), ("rnn_edge", EdgeIotGRU)])
def test_tabular_models_accept_flat_feature_vectors(name, cls):
    model = ModelFactory.build(name, num_classes=4, in_channels=1, input_feature_dim=30)
    assert isinstance(model, cls)
    model.eval()
    assert model(torch.randn(5, 30)).shape == (5, 4)


@pytest.mark.parametrize("name", ["cnn_edge", "rnn_edge"])
def test_tabular_models_require_feature_dim(name):
    with pytest.raises(ValueError):
        ModelFactory.build(name, num_classes=4, in_channels=1)


@pytest.mark.filterwarnings("ignore:The default weight initialization of GoogleNet:FutureWarning")
@pytest.mark.parametrize("name", ["resnet18", "googlenet"])
def test_torchvision_models_adapt_to_grayscale_input(name):
    pytest.importorskip("torchvision")
    model = ModelFactory.build(name, num_classes=10, in_channels=1)
    assert not isinstance(model, SmallImageCNN)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_paper_defaults():
    assert ModelFactory.paper_defaults("googlenet", "mnist") == {"batch_size": 32, "lr": 5e-4, "tau_init": 2}
    assert ModelFactory.paper_defaults("resnet18", "cifar10") == {"batch_size": 64, "lr": 1e-3, "tau_init": 3}
    assert ModelFactory.paper_defaults("unknown", "x") == {"batch_size": 32, "lr": 1e-3, "tau_init": 2}
