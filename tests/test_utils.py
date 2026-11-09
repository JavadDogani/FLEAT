"""CLI defaults, device resolution, run naming and output helpers."""
from __future__ import annotations

import io
import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from fleat_project.config import parse_args
from fleat_project.io_utils import save_outputs
from fleat_project.utils import (
    TeeStream,
    format_alpha_for_name,
    make_default_run_name,
    parse_gpu_ids,
    resolve_execution_devices,
    sanitize_run_token,
)


def test_cli_defaults_are_stable():
    """Guards the documented experiment defaults against accidental changes."""
    a = parse_args([])
    assert (a.method, a.model, a.dataset) == ("fleat", "googlenet", "mnist")
    assert (a.num_clients, a.rounds, a.partition, a.dirichlet_alpha) == (10, 60, "dirichlet", 0.3)
    assert (a.lr, a.optimizer, a.batch_size, a.fedprox_mu) == (0.02, "sgd", None, 0.01)
    assert (a.alpha_weight, a.tau_min, a.tau_max, a.tau_init) == (0.5, 1, 10, 10)
    assert (a.p_init, a.p_max, a.T_max, a.phi_ema, a.base_phi_step) == (0.0, 0.5, 20.0, 0.3, 0.02)
    assert (a.device_profile_mode, a.seed, a.max_eval_samples) == ("sim_table3", 42, 2000)
    assert a.verbose is True


def test_quiet_flag_disables_verbose_output():
    assert parse_args(["--quiet"]).verbose is False


def test_parse_gpu_ids():
    assert parse_gpu_ids("0, 2,3") == [0, 2, 3]
    assert parse_gpu_ids("") == []
    assert parse_gpu_ids(None) == []


def device_args(**kw):
    base = dict(device="cuda", server_device="cuda", parallel_clients=True, max_gpus=3, gpu_ids="0,1,2")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def four_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "auto", "cpu"])
def test_devices_fall_back_to_cpu_without_cuda(no_cuda, device):
    assert resolve_execution_devices(device_args(device=device)) == ("cpu", ["cpu"])


def test_multi_gpu_worker_selection(four_gpus):
    assert resolve_execution_devices(device_args()) == ("cuda:0", ["cuda:0", "cuda:1", "cuda:2"])
    assert resolve_execution_devices(device_args(max_gpus=2))[1] == ["cuda:0", "cuda:1"]
    assert resolve_execution_devices(device_args(gpu_ids="", max_gpus=0))[1] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert resolve_execution_devices(device_args(gpu_ids="1,7"))[1] == ["cuda:1"]  # invisible IDs dropped
    assert resolve_execution_devices(device_args(parallel_clients=False)) == ("cuda:0", ["cuda:0"])


def test_run_name_helpers():
    assert sanitize_run_token("a b/c--d") == "a-b-c-d"
    assert sanitize_run_token("///") == "x"
    assert format_alpha_for_name(0.3) == "0p3"
    args = SimpleNamespace(method="fleat", model="googlenet", dataset="mnist", num_clients=10,
                           partition="dirichlet", dirichlet_alpha=0.3)
    assert re.fullmatch(r"fleat_googlenet_mnist_N10_dirichlet_a0p3_\d{8}-\d{6}", make_default_run_name(args))
    args.partition = "iid"
    assert re.fullmatch(r"fleat_googlenet_mnist_N10_iid_\d{8}-\d{6}", make_default_run_name(args))


def test_tee_stream_writes_to_all_streams():
    a, b = io.StringIO(), io.StringIO()
    tee = TeeStream(a, None, b)
    assert tee.write("hello") == 5
    tee.flush()
    assert a.getvalue() == b.getvalue() == "hello"


def test_save_outputs(tmp_path):
    df = pd.DataFrame({"round": [0, 1], "eval_acc": [0.1, 0.2]})
    csv_path, json_path = save_outputs(df, {"metrics": {"final_accuracy": 0.2}}, str(tmp_path / "out"), "run")
    assert csv_path.endswith("run_round_metrics.csv") and json_path.endswith("run_summary.json")
    assert pd.read_csv(csv_path)["eval_acc"].tolist() == [0.1, 0.2]
    assert json.loads(open(json_path).read())["metrics"]["final_accuracy"] == 0.2
