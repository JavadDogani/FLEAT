from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fleat_project.data import SimpleTransformDataset
from fleat_project.trainer import LocalTrainer

CPU = torch.device("cpu")


def tiny_dataset(n=10, d=5, classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return SimpleTransformDataset(torch.randn(n, d, generator=g), torch.randint(0, classes, (n,), generator=g))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3))


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.state_dict().items()}


def distance(model, state):
    return sum(float(torch.sum((p.detach() - state[n]) ** 2)) for n, p in model.named_parameters())


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "rmsprop"])
def test_runs_exactly_the_requested_number_of_steps(optimizer):
    loader = DataLoader(tiny_dataset(n=10), batch_size=4, shuffle=False)  # 3 batches per epoch
    model = tiny_model()
    before = snapshot(model)
    model, loss, phi, done = LocalTrainer.train_local_steps(
        model, loader, steps=7, lr=0.01, device=CPU, weight_decay=0.0, optimizer_name=optimizer
    )
    assert done == 7  # the loader is restarted when exhausted
    assert math.isfinite(loss)
    assert phi >= 0.0
    assert distance(model, before) > 0


def test_fedprox_term_keeps_model_closer_to_global():
    loader = DataLoader(tiny_dataset(n=32), batch_size=8, shuffle=False)
    global_state = snapshot(tiny_model())

    def train(mu):
        model = tiny_model()
        model, *_ = LocalTrainer.train_local_steps(
            model, loader, steps=10, lr=0.05, device=CPU, weight_decay=0.0, prox_mu=mu, global_state=global_state
        )
        return distance(model, global_state)

    assert train(mu=5.0) < train(mu=0.0)


class _Oracle(nn.Module):
    """Perfect classifier for SimpleTransformDataset(x, y) with x[:, 0] == y."""

    def forward(self, x):
        return nn.functional.one_hot(x[:, 0].long(), 3).float() * 10.0


def oracle_dataset(n):
    y = torch.arange(n) % 3
    return SimpleTransformDataset(torch.stack([y.float(), torch.zeros(n)], dim=1), y)


def test_evaluate_model_reports_accuracy_and_loss():
    res = LocalTrainer.evaluate_model(_Oracle(), oracle_dataset(10), batch_size=4, device=CPU)
    assert res["eval_acc"] == 1.0
    assert res["n_eval"] == 10
    assert 0.0 <= res["eval_loss"] < 1e-3


def test_evaluate_model_respects_sample_cap_on_batch_boundary():
    res = LocalTrainer.evaluate_model(_Oracle(), oracle_dataset(20), batch_size=4, device=CPU, max_eval_samples=8)
    assert res["n_eval"] == 8
    assert res["eval_acc"] == 1.0
