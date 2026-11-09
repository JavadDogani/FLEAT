"""Tests for the energy/time model, the per-client T_max cap and device profiles."""
from __future__ import annotations

import pytest

from fleat_project.aggregator import FederatedAggregator as FA
from fleat_project.profiles import REAL_DEVICE_PROFILES, SIM_TIERS, DeviceProfileFactory

# 8 Mbit/s up = 1e6 B/s, 16 Mbit/s down = 2e6 B/s.
PROFILE = {"bw_up": 8.0, "bw_down": 16.0, "p_comp": 10.0, "p_send": 4.0, "p_rec": 2.0}


# ------------------------------------------------------------- round energy/time


def test_round_energy_time_closed_form():
    et = FA.estimate_round_energy_time(PROFILE, tau=5, phi_step=0.1, b_upload_bytes=1_000_000, b_download_bytes=2_000_000)
    assert et["Y_comp_time_s"] == pytest.approx(0.5)  # tau * phi
    assert et["C_up_time_s"] == pytest.approx(1.0)  # 1e6 B / 1e6 B/s
    assert et["C_dn_time_s"] == pytest.approx(1.0)  # 2e6 B / 2e6 B/s
    assert et["round_time_s"] == pytest.approx(2.5)
    assert et["E_comp_ws"] == pytest.approx(10.0 * 0.5)
    assert et["E_send_ws"] == pytest.approx(4.0 * 1.0)
    assert et["E_rec_ws"] == pytest.approx(2.0 * 1.0)
    assert et["E_total_ws"] == pytest.approx(5.0 + 4.0 + 2.0)


def test_zero_local_steps_cost_only_communication():
    et = FA.estimate_round_energy_time(PROFILE, tau=0, phi_step=0.1, b_upload_bytes=1_000_000, b_download_bytes=2_000_000)
    assert et["Y_comp_time_s"] == 0.0
    assert et["E_comp_ws"] == 0.0
    assert et["E_total_ws"] == pytest.approx(et["E_send_ws"] + et["E_rec_ws"])


def test_masked_upload_reduces_upload_time_and_energy_linearly():
    b0 = 1_000_000
    full = FA.estimate_round_energy_time(PROFILE, 3, 0.1, b_upload_bytes=b0, b_download_bytes=b0)
    half = FA.estimate_round_energy_time(PROFILE, 3, 0.1, b_upload_bytes=int(0.5 * b0), b_download_bytes=b0)
    assert half["C_up_time_s"] == pytest.approx(0.5 * full["C_up_time_s"])
    assert half["E_send_ws"] == pytest.approx(0.5 * full["E_send_ws"])
    assert half["C_dn_time_s"] == full["C_dn_time_s"]  # download is always the full model


# ----------------------------------------------------------------- T_max cap

# b0 = 1e6 B -> full upload 1.0 s, full download 0.5 s; one local step = 0.1 s.
CAP = dict(profile=PROFILE, b0_bytes=1_000_000, phi_step=0.1, p_max=0.5, tau_min=1, tau_max=10)


def test_tmax_cap_disabled_when_tmax_not_positive():
    tau, p, info = FA.enforce_real_tmax_cap_per_client(tau=7, p_target=0.2, p_achieved_est=0.2, T_max=0, **CAP)
    assert (tau, p) == (7, 0.2)
    assert info["T_cap_used"] is False and info["T_cap_feasible"] is True


def test_tmax_cap_leaves_a_loose_budget_alone():
    tau, p, info = FA.enforce_real_tmax_cap_per_client(tau=4, p_target=0.2, p_achieved_est=0.0, T_max=100.0, **CAP)
    assert (tau, p) == (4, 0.2)
    assert info["T_cap_feasible"] is True
    assert info["T_cap_pred_time_s"] == pytest.approx(4 * 0.1 + 1.0 + 0.5)


def test_tmax_cap_reduces_local_steps():
    # 2.0 s budget - 0.5 s download - 1.0 s upload leaves 0.5 s = 5 steps.
    tau, p, info = FA.enforce_real_tmax_cap_per_client(tau=10, p_target=0.0, p_achieved_est=0.0, T_max=2.0, **CAP)
    assert tau == 5
    assert p == pytest.approx(0.0)
    assert info["T_cap_pred_time_s"] <= 2.0 + 1e-9


def test_tmax_cap_raises_pruning_ratio_when_upload_must_shrink():
    # With an estimated 40 % pruned upload (0.6 s), 1.65 - 0.5 - 0.6 = 0.55 s
    # leaves 5 steps; the remaining 0.65 s for upload requires p >= 0.35.
    tau, p, info = FA.enforce_real_tmax_cap_per_client(tau=10, p_target=0.0, p_achieved_est=0.4, T_max=1.65, **CAP)
    assert tau == 5
    assert p == pytest.approx(0.35)
    assert info["T_cap_feasible"] is True
    assert info["T_cap_pred_time_s"] == pytest.approx(5 * 0.1 + 0.6 + 0.5)


def test_tmax_cap_reports_infeasible_budget():
    # Even with zero steps and maximal pruning: 0.5 s upload + 0.5 s download > 0.8 s.
    tau, p, info = FA.enforce_real_tmax_cap_per_client(tau=10, p_target=0.0, p_achieved_est=0.0, T_max=0.8, **CAP)
    assert (tau, p) == (0, 0.5)
    assert info["T_cap_feasible"] is False
    assert info["T_cap_floor_time_s"] == pytest.approx(1.0)


def test_tmax_cap_can_keep_tau_min_as_floor():
    tau, _, info = FA.enforce_real_tmax_cap_per_client(
        tau=10, p_target=0.0, p_achieved_est=0.0, T_max=1.05, allow_tau_zero_if_needed=False, **CAP
    )
    assert tau == 1
    assert info["T_cap_floor_time_s"] == pytest.approx(0.1 + 0.5 + 0.5)


# ------------------------------------------------------------------ device profiles


def test_real5_profiles_cycle_through_the_five_devices():
    profiles = DeviceProfileFactory.sample(7, "real5")
    names = [p["name"] for p in profiles]
    assert names == ["pi0", "pi3", "pi4", "nano", "xavier", "pi0", "pi3"]
    assert [p["id"] for p in profiles] == list(range(7))
    assert profiles[0]["bw_up"] == REAL_DEVICE_PROFILES[0]["bw_up"]


def test_simulated_profiles_are_seeded_and_within_tier_ranges():
    a = DeviceProfileFactory.sample(50, "sim_table3", seed=1)
    b = DeviceProfileFactory.sample(50, "sim_table3", seed=1)
    c = DeviceProfileFactory.sample(50, "sim_table3", seed=2)
    assert a == b
    assert a != c
    tiers = {t["name"]: t for t in SIM_TIERS}
    for prof in a:
        tier = tiers[prof["name"]]
        for key in ("bw_up", "bw_down", "p_comp", "p_send", "p_rec"):
            lo, hi = tier[key]
            assert lo <= prof[key] <= hi
