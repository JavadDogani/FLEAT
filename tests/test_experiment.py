"""End-to-end smoke tests of the experiment runner and CLI on tiny synthetic data (CPU only)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import METHODS, SMALL_CNN, TINY_ARGV
from fleat_project.experiment import ExperimentRunner
from fleat_project.main import main

SUMMARY_METRICS = {
    "final_accuracy", "final_loss", "best_round_accuracy", "best_round", "total_energy_ws",
    "total_comp_energy_ws", "total_comm_energy_ws", "total_time_s", "acc_at_common_wall_clock_half_total",
    "common_wall_clock_half_total_s", "avg_round_energy_ws", "avg_round_time_s", "mean_tau",
    "mean_pruning_ratio", "mean_gamma", "compute_time_share", "comm_time_share",
}


def check_accounting(df: pd.DataFrame, summary: dict, args) -> None:
    """Invariants of the per-round energy/time bookkeeping and the summary."""
    assert df["round"].tolist() == list(range(args.rounds + 1))
    assert df["eval_acc"].between(0.0, 1.0).all()
    assert np.isfinite(df["eval_loss"]).all()

    rounds = df.iloc[1:]
    assert (rounds["round_energy_ws"] > 0).all()
    assert np.allclose(rounds["round_energy_ws"], rounds["round_comp_energy_ws"] + rounds["round_comm_energy_ws"])
    assert np.allclose(df["cumulative_energy_ws"], df["round_energy_ws"].cumsum())
    assert np.allclose(df["cumulative_time_s"], df["round_time_s"].cumsum())
    for taus, ps in zip(rounds["tau_list"], rounds["p_achieved_list"]):
        assert len(taus) == len(ps) == args.num_clients
        assert all(0 <= t <= args.tau_max for t in taus)
        assert all(0.0 <= p <= args.p_max + 1e-9 for p in ps)

    m = summary["metrics"]
    assert set(m) == SUMMARY_METRICS
    assert summary["method"] == args.method and summary["rounds"] == args.rounds
    assert m["final_accuracy"] == df["eval_acc"].iloc[-1]
    assert m["total_energy_ws"] == pytest.approx(df["cumulative_energy_ws"].iloc[-1])
    assert m["total_energy_ws"] == pytest.approx(m["total_comp_energy_ws"] + m["total_comm_energy_ws"])
    assert m["total_time_s"] == pytest.approx(df["cumulative_time_s"].iloc[-1])
    assert m["compute_time_share"] + m["comm_time_share"] == pytest.approx(1.0)
    assert summary["execution"]["server_device"] == "cpu"


def test_fleat_end_to_end_with_resnet18_prunes_uploads(make_args):
    pytest.importorskip("torchvision")
    args = make_args("--model", "resnet18", "--method", "fleat")
    df, summary = ExperimentRunner(args).run()
    check_accounting(df, summary, args)
    # The controller drives p towards p_max and the masked upload actually drops
    # some ResNet layers, so the achieved pruning ratio becomes positive.
    assert max(max(ps) for ps in df["p_achieved_list"].iloc[1:]) > 0.0
    assert summary["metrics"]["mean_pruning_ratio"] > 0.0


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs_on_dirichlet_partition(make_args, method):
    args = make_args("--model", SMALL_CNN, "--method", method, "--partition", "dirichlet",
                     "--dirichlet_alpha", "0.5", "--min_client_samples", "5")
    df, summary = ExperimentRunner(args).run()
    check_accounting(df, summary, args)
    assert summary["dirichlet_alpha"] == 0.5
    if method in {"fedavg", "local_update_only"}:
        assert all(p == 0.0 for ps in df["p_target_list"].iloc[1:] for p in ps)
    if method in {"fedavg", "pruning_only"}:
        # Fixed local steps (the generous default T_max never binds here).
        assert all(t == args.tau_init for ts in df["tau_list"].iloc[1:] for t in ts)


def test_fedavg_accuracy_is_reproducible_for_a_fixed_seed(make_args):
    # T_max <= 0 disables the wall-clock-dependent time cap, so FedAvg's
    # training trajectory depends only on the seed.
    runs = [ExperimentRunner(make_args("--model", SMALL_CNN, "--method", "fedavg", "--T_max", "0")).run()[0] for _ in range(2)]
    pd.testing.assert_series_equal(runs[0]["eval_acc"], runs[1]["eval_acc"])
    pd.testing.assert_series_equal(runs[0]["eval_loss"], runs[1]["eval_loss"])


def cli_argv(tmp_path, *extra):
    argv = [a for a in TINY_ARGV if a != "--disable_log_to_file"]
    return argv + ["--model", SMALL_CNN, "--out_dir", str(tmp_path), "--run_name", "smoke", *extra]


def test_cli_writes_metrics_summary_and_log(tmp_path, capsys):
    main(cli_argv(tmp_path))
    df = pd.read_csv(tmp_path / "smoke_round_metrics.csv")
    summary = json.loads((tmp_path / "smoke_summary.json").read_text())
    assert len(df) == 3 and summary["metrics"]["final_accuracy"] == pytest.approx(df["eval_acc"].iloc[-1])
    log = (tmp_path / "logs" / "smoke.log").read_text()
    assert "SUMMARY (paper-style metrics)" in log
    assert "SUMMARY (paper-style metrics)" in capsys.readouterr().out


def test_cli_alpha_sweep(tmp_path):
    main(cli_argv(tmp_path, "--alpha_sweep", "0.25,0.75", "--disable_log_to_file"))
    sweep = pd.read_csv(tmp_path / "smoke_alpha_sweep.csv")
    assert sweep["alpha"].tolist() == [0.25, 0.75]
    for a in ("0.25", "0.75"):
        assert (tmp_path / f"smoke_alpha{a}_summary.json").exists()
