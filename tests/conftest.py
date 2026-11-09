"""Shared helpers for the FLEAT test suite.

All tests run on CPU with small synthetic data; nothing is downloaded.
"""
from __future__ import annotations

import pytest
import torch

from fleat_project.config import parse_args

# The tensors here are tiny; extra intra-op threads only add overhead (and the
# runner itself pins torch to --torch_num_threads=1 by default).
torch.set_num_threads(1)

# ModelFactory.build() falls back to SmallImageCNN for any name it does not
# recognise; tests use this name to get the small, BatchNorm-free CNN.
SMALL_CNN = "small_cnn"

METHODS = ["fleat", "fedavg", "fedprox", "local_update_only", "pruning_only"]

# A tiny synthetic run: 96 train / 48 test images (1x28x28, 10 classes),
# 3 clients x 32 samples, batch size 16, 2 rounds, forced onto the CPU.
TINY_ARGV = [
    "--dataset", "synthetic",
    "--synthetic_train", "96",
    "--synthetic_test", "48",
    "--num_clients", "3",
    "--rounds", "2",
    "--partition", "iid",
    "--batch_size", "16",
    "--tau_init", "2",
    "--tau_max", "3",
    "--device", "cpu",
    "--server_device", "cpu",
    "--quiet",
    "--disable_log_to_file",
]


@pytest.fixture
def make_args():
    """Build an argparse.Namespace from the real CLI parser (so all defaults
    match the command line), starting from TINY_ARGV plus ``extra`` flags,
    then apply keyword overrides."""

    def _make(*extra: str, **overrides):
        args = parse_args(TINY_ARGV + list(extra))
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    return _make
