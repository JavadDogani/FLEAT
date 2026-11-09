"""Tests for pruning-group selection, masked deltas and (masked / FedAvg) aggregation."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fleat_project.aggregator import FederatedAggregator as FA

GROUP_ORDER = ["conv1.weight", "l1.weight", "l1.bias", "l2.weight", "l3.weight", "l4.weight", "fc.weight", "fc.bias"]
GROUP_BYTES = {
    "conv1.weight": 400, "l1.weight": 1000, "l1.bias": 40, "l2.weight": 2000,
    "l3.weight": 1500, "l4.weight": 3000, "fc.weight": 800, "fc.bias": 40,
}
TOTAL_BYTES = sum(GROUP_BYTES.values())  # 8780
IMPORTANCE = {
    "conv1.weight": 0.2, "l1.weight": 0.05, "l1.bias": 0.01, "l2.weight": 0.1,
    "l3.weight": 0.02, "l4.weight": 0.3, "fc.weight": 0.3, "fc.bias": 0.02,
}


# ------------------------------------------------------------------ size / groups


def test_model_state_bytes_counts_every_tensor():
    state = {"w": torch.zeros(10, dtype=torch.float32), "n": torch.zeros(3, dtype=torch.int64)}
    assert FA.model_state_bytes(state) == 10 * 4 + 3 * 8


def test_group_names_and_sizes_follow_named_parameters():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    order = FA.group_param_names(model)
    assert order == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert FA.group_sizes_bytes(model, order) == {"0.weight": 48, "0.bias": 12, "2.weight": 24, "2.bias": 8}


@pytest.mark.parametrize(
    "name, prunable",
    [
        ("layer1.0.conv2.weight", True),
        ("features.3.weight", True),
        ("layer1.0.conv2.bias", False),  # only weights are pruned
        ("layer1.0.bn1.weight", False),  # normalisation layers are kept
        ("features.1.norm.weight", False),
        ("conv1.weight", False),  # stem and head are kept
        ("fc.weight", False),
        ("classifier.1.weight", False),
        ("head.weight", False),
        ("layer2.0.downsample.1.weight", False),  # BN inside ResNet downsample
    ],
)
def test_is_prunable_group_name(name, prunable):
    assert FA.is_prunable_group_name(name) is prunable


def test_layer_importance_is_a_distribution_over_groups():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
    order = FA.group_param_names(model)
    batch = (torch.randn(8, 5), torch.randint(0, 3, (8,)))
    imp = FA.compute_layer_importance(model, batch, nn.CrossEntropyLoss(), torch.device("cpu"), order)
    assert list(imp) == order
    assert all(v >= 0 for v in imp.values())
    assert sum(imp.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- group selection


def test_zero_target_prunes_nothing():
    pruned, p_hat, gamma = FA.select_pruned_groups(IMPORTANCE, GROUP_ORDER, GROUP_BYTES, target_p=0.0)
    assert pruned == set()
    assert p_hat == 0.0
    assert gamma == pytest.approx(1.0)


def test_selection_prunes_least_important_groups_within_byte_budget():
    # Candidates are l1..l4 (conv1/fc and biases are never pruned). Sorted by
    # importance: l3 (0.02, 1500 B), l1 (0.05, 1000 B), l2, l4. With a 30 %
    # budget (2634 B) l3 and l1 fit, adding l2 would exceed it.
    pruned, p_hat, gamma = FA.select_pruned_groups(
        IMPORTANCE, GROUP_ORDER, GROUP_BYTES, target_p=0.3, protect_first_last=False, noncontiguous=False
    )
    assert pruned == {"l3.weight", "l1.weight"}
    assert p_hat == pytest.approx(2500 / TOTAL_BYTES)
    assert p_hat <= 0.3
    assert gamma == pytest.approx(1.0 - 0.02 - 0.05)


def test_selection_respects_protection_and_noncontiguity():
    # protect_first_last keeps l1 and l4 (first/last candidates); after pruning
    # l3, its neighbour l2 is blocked by the non-contiguity rule.
    pruned, p_hat, gamma = FA.select_pruned_groups(
        IMPORTANCE, GROUP_ORDER, GROUP_BYTES, target_p=0.3, protect_first_last=True, noncontiguous=True
    )
    assert pruned == {"l3.weight"}
    assert p_hat == pytest.approx(1500 / TOTAL_BYTES)
    assert gamma == pytest.approx(0.98)


def test_full_budget_never_prunes_protected_layer_types():
    pruned, p_hat, gamma = FA.select_pruned_groups(
        IMPORTANCE, GROUP_ORDER, GROUP_BYTES, target_p=1.0, protect_first_last=False, noncontiguous=False
    )
    assert pruned == {"l1.weight", "l2.weight", "l3.weight", "l4.weight"}
    assert p_hat == pytest.approx(7500 / TOTAL_BYTES)
    assert gamma == pytest.approx(1.0 - (0.05 + 0.1 + 0.02 + 0.3))


# ------------------------------------------------------------------ masked deltas


def test_state_dict_delta_and_masking():
    g = {"a.weight": torch.ones(2), "b.weight": torch.ones(2), "n": torch.tensor(1)}
    local = {"a.weight": torch.full((2,), 3.0), "b.weight": torch.full((2,), 0.5), "n": torch.tensor(4)}
    delta = FA.state_dict_delta(local, g)
    assert torch.equal(delta["a.weight"], torch.full((2,), 2.0))
    assert torch.equal(delta["n"], torch.tensor(3))

    masked = FA.make_masked_delta(delta, {"a.weight", "n"})
    assert masked["a.weight"] is None  # pruned float tensor is not uploaded
    assert masked["b.weight"] is delta["b.weight"]
    assert masked["n"] is delta["n"]  # non-float buffers are never masked


# -------------------------------------------------------------------- aggregation


def test_masked_aggregation_is_contributor_aware():
    g = {
        "w": torch.tensor([1.0, 1.0]),
        "v": torch.tensor([0.0, 0.0]),
        "z": torch.tensor([5.0]),
        "n": torch.tensor(5, dtype=torch.int64),
    }
    g_before = {k: v.clone() for k, v in g.items()}
    updates = [
        {"delta_masked": {"w": torch.tensor([1.0, 1.0]), "v": torch.tensor([4.0, 4.0]), "z": None, "n": torch.tensor(1)}, "tau": 2},
        {"delta_masked": {"w": torch.tensor([3.0, 3.0]), "v": None, "z": None, "n": torch.tensor(2)}, "tau": 2},
    ]
    new = FA.aggregate_masked_updates(g, updates, client_sizes=[1, 3])

    # Both clients upload "w": size-weighted mean of the deltas (0.25, 0.75).
    assert torch.allclose(new["w"], torch.tensor([3.5, 3.5]))
    # Only client 0 uploads "v": weights are renormalised over contributors,
    # so the update is client 0's full delta, not 0.25 of it.
    assert torch.allclose(new["v"], torch.tensor([4.0, 4.0]))
    # Nobody uploads "z": the global value is kept.
    assert torch.equal(new["z"], torch.tensor([5.0]))
    # Integer buffers: weighted mean of the clients' final values (6, 7), rounded.
    assert new["n"].dtype == torch.int64 and int(new["n"]) == 7
    # The input global state is not modified in place.
    assert all(torch.equal(g[k], g_before[k]) for k in g)


def test_unmasked_aggregation_equals_fedavg():
    torch.manual_seed(0)
    g = {"a": torch.randn(4, 3), "b": torch.randn(3)}
    states = [{k: v + torch.randn_like(v) for k, v in g.items()} for _ in range(3)]
    sizes = [10, 30, 60]
    updates = [{"delta_masked": FA.state_dict_delta(s, g), "tau": 1} for s in states]
    masked = FA.aggregate_masked_updates(g, updates, sizes)
    fedavg = FA.aggregate_fedavg(g, states, sizes)
    for k in g:
        assert torch.allclose(masked[k], fedavg[k], atol=1e-6)


def test_fedavg_weights_by_client_size():
    g = {"w": torch.zeros(2), "n": torch.tensor(0)}
    states = [{"w": torch.ones(2), "n": torch.tensor(6)}, {"w": torch.full((2,), 3.0), "n": torch.tensor(7)}]
    out = FA.aggregate_fedavg(g, states, [1, 3])
    assert torch.allclose(out["w"], torch.full((2,), 2.5))
    assert int(out["n"]) == 7  # round(0.25 * 6 + 0.75 * 7) = round(6.75)
