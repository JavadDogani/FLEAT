from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from fleat_project.data import SimpleTransformDataset
from fleat_project.partitioning import PartitionManager


def labelled_dataset(n_per_class: int = 60, n_classes: int = 10) -> SimpleTransformDataset:
    y = torch.arange(n_per_class * n_classes) % n_classes
    x = torch.zeros(len(y), 1)
    return SimpleTransformDataset(x, y)


def assert_exact_cover(parts, n):
    flat = np.concatenate(parts)
    assert len(flat) == n, "every sample must be assigned exactly once"
    assert np.array_equal(np.sort(flat), np.arange(n))


def mean_majority_label_share(parts, y: np.ndarray, n_classes: int) -> float:
    """Average, over clients, of the fraction of samples in the client's most common class."""
    return float(np.mean([np.bincount(y[p], minlength=n_classes).max() / len(p) for p in parts]))


# --------------------------------------------------------------------------- IID


def test_iid_partition_is_a_balanced_disjoint_cover():
    parts = PartitionManager.partition_iid(103, 5, seed=0)
    assert len(parts) == 5
    assert_exact_cover(parts, 103)
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1


def test_iid_partition_is_seed_deterministic():
    a = PartitionManager.partition_iid(50, 4, seed=7)
    b = PartitionManager.partition_iid(50, 4, seed=7)
    c = PartitionManager.partition_iid(50, 4, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


# --------------------------------------------------------------------- Dirichlet


@pytest.mark.parametrize("alpha", [0.1, 0.3, 1.0])
def test_dirichlet_partition_is_a_disjoint_cover_respecting_min_size(alpha):
    ds = labelled_dataset()
    parts = PartitionManager.partition_dirichlet_label_skew(ds, num_clients=6, alpha=alpha, seed=0, min_size=10)
    assert len(parts) == 6
    assert_exact_cover(parts, len(ds))
    assert min(len(p) for p in parts) >= 10


def test_dirichlet_partition_is_seed_deterministic():
    ds = labelled_dataset()
    a = PartitionManager.partition_dirichlet_label_skew(ds, 6, 0.3, seed=3)
    b = PartitionManager.partition_dirichlet_label_skew(ds, 6, 0.3, seed=3)
    c = PartitionManager.partition_dirichlet_label_skew(ds, 6, 0.3, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_small_dirichlet_alpha_produces_label_skew(seed):
    ds = labelled_dataset()
    y = ds.y.numpy()
    skewed = PartitionManager.partition_dirichlet_label_skew(ds, 6, alpha=0.05, seed=seed, min_size=10)
    near_iid = PartitionManager.partition_dirichlet_label_skew(ds, 6, alpha=1000.0, seed=seed, min_size=10)
    share_skewed = mean_majority_label_share(skewed, y, 10)
    share_near_iid = mean_majority_label_share(near_iid, y, 10)
    # With 10 balanced classes a perfectly IID client has a majority share of 0.1.
    assert share_near_iid < 0.15
    assert share_skewed > share_near_iid + 0.2


# ------------------------------------------------------------------ targets / subsets


class _TargetsList(Dataset):
    targets = [0, 1, 2, 1]

    def __len__(self):
        return 4


class _LabelsTensor(Dataset):
    labels = torch.tensor([3, 2, 1])

    def __len__(self):
        return 3


@pytest.mark.parametrize(
    "dataset, expected",
    [
        (_TargetsList(), [0, 1, 2, 1]),  # torchvision-style .targets list (MNIST/CIFAR)
        (_LabelsTensor(), [3, 2, 1]),  # .labels attribute (SVHN)
        (SimpleTransformDataset(torch.zeros(3, 2), torch.tensor([2, 0, 1])), [2, 0, 1]),  # iterate items
    ],
)
def test_extract_targets_supports_common_dataset_layouts(dataset, expected):
    assert PartitionManager.extract_targets(dataset).tolist() == expected


@pytest.mark.parametrize("partition", ["iid", "dirichlet"])
def test_make_client_subsets_uses_cli_settings(make_args, partition):
    ds = labelled_dataset(n_per_class=20)
    args = make_args("--partition", partition, "--num_clients", "4", "--min_client_samples", "5")
    subsets = PartitionManager.make_client_subsets(ds, args)
    assert len(subsets) == 4
    assert all(isinstance(s, Subset) for s in subsets)
    assert sum(len(s) for s in subsets) == len(ds)
    assert min(len(s) for s in subsets) >= 5
