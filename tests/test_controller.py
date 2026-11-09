"""Tests for the adaptive local-step / pruning-ratio controller (FLEATController)."""
from __future__ import annotations

import itertools

import pytest

from fleat_project.controller import FLEATController
from fleat_project.runtime import ClientRuntimeState

# A small hand-made "model": 8 parameter groups with byte sizes and
# normalised importances (sum to 1).
GROUP_ORDER = ["conv1.weight", "l1.weight", "l1.bias", "l2.weight", "l3.weight", "l4.weight", "fc.weight", "fc.bias"]
GROUP_BYTES = {
    "conv1.weight": 400, "l1.weight": 1000, "l1.bias": 40, "l2.weight": 2000,
    "l3.weight": 1500, "l4.weight": 3000, "fc.weight": 800, "fc.bias": 40,
}
IMPORTANCE = {
    "conv1.weight": 0.2, "l1.weight": 0.05, "l1.bias": 0.01, "l2.weight": 0.1,
    "l3.weight": 0.02, "l4.weight": 0.3, "fc.weight": 0.3, "fc.bias": 0.02,
}
PROFILE = {"bw_up": 50.0, "bw_down": 100.0, "p_comp": 8.0, "p_send": 5.0, "p_rec": 4.0}

TAU_MIN, TAU_MAX, P_MAX = 1, 10, 0.5


def run_controller(policy="fleat", *, alpha=0.5, T_max=20.0, tau=10, p=0.0, phi=0.05, b0=2_000_000,
                   F_curr=2.3, round_time_ref=20.0, data_frac=0.25, state=None):
    state = state or ClientRuntimeState(tau=tau, p=p, phi_step=phi)
    tau_k, p_k, aux = FLEATController.update_tau_p(
        client_state=state, profile=PROFILE, b0_bytes=b0, alpha_weight=alpha, p_max=P_MAX,
        tau_min=TAU_MIN, tau_max=TAU_MAX, T_max=T_max, eta=0.02, L_smooth=1.0, F_curr=F_curr, F_inf=0.0,
        data_frac=data_frac, round_time_ref=round_time_ref, importance=IMPORTANCE,
        group_order=GROUP_ORDER, group_bytes=GROUP_BYTES, policy=policy,
    )
    return tau_k, p_k, aux, state


def test_fedavg_policy_keeps_tau_and_disables_pruning():
    tau, p, aux, _ = run_controller("fedavg", tau=7, p=0.3)
    assert (tau, p) == (7, 0.0)
    assert aux["gamma"] == 1.0


def test_pruning_only_policy_keeps_tau_and_clamps_previous_p():
    tau, p, _, _ = run_controller("pruning_only", tau=4, p=0.9)
    assert tau == 4
    assert p == P_MAX


def test_local_update_only_policy_never_prunes():
    tau, p, aux, _ = run_controller("local_update_only")
    assert p == 0.0
    assert aux["gamma"] == pytest.approx(1.0)
    assert TAU_MIN <= tau <= TAU_MAX


def test_fedprox_uses_the_same_control_law_as_fleat():
    assert run_controller("fedprox")[:3] == run_controller("fleat")[:3]


@pytest.mark.parametrize(
    "policy, alpha, phi, b0, T_max",
    list(itertools.product(
        ["fleat", "fedprox", "local_update_only"],
        [0.0, 0.3, 1.0],
        [1e-4, 0.05, 1.0],
        [10_000, 2_000_000, 50_000_000],
        [0.5, 20.0],
    )),
)
def test_controller_outputs_stay_within_bounds(policy, alpha, phi, b0, T_max):
    tau, p, aux, _ = run_controller(policy, alpha=alpha, phi=phi, b0=b0, T_max=T_max)
    assert isinstance(tau, int)
    assert TAU_MIN <= tau <= TAU_MAX
    assert 0.0 <= p <= P_MAX
    assert 0.0 <= aux["gamma"] <= 1.0


def test_fleat_proxies_match_closed_form():
    """Ak, Bk, Ck and the retained-importance proxy gamma(p) for a hand-checked case."""
    tau, p, aux, state = run_controller("fleat")
    eta, L, df, F, Tr = 0.02, 1.0, 0.25, 2.3, 20.0
    assert aux["Ak"] == pytest.approx(2.0 * (F + 1e-4) / (eta * Tr) * df)
    assert aux["Bk"] == pytest.approx(eta * L * df)
    assert aux["Ck"] == pytest.approx(eta**2 * L**2 * df)
    # p saturates at p_max here. Groups sorted by (importance, bytes) give a
    # cumulative-byte share of 2580/8780 at importance 0.10 and 4580/8780 at
    # 0.20, so S(0.5) is linearly interpolated between those two points.
    assert p == P_MAX
    x0, x1 = 2580 / 8780, 4580 / 8780
    s_half = 0.10 + (0.5 - x0) / (x1 - x0) * (0.20 - 0.10)
    assert aux["gamma"] == pytest.approx(1.0 - s_half)
    assert aux["Mk"] == pytest.approx(0.1 + s_half * 1.0)
    assert tau == 1
    # The controller records the fitted pruning-curve parameters on the state.
    assert state.beta == pytest.approx(2.0)
    assert state.kappa > 0


def test_more_expensive_upload_leads_to_more_local_steps():
    """tau* grows with the upload time N_k (the sqrt rule), all else equal."""
    taus = [run_controller(alpha=0.8, b0=b0)[0] for b0 in (1_000_000, 10_000_000, 50_000_000)]
    assert taus == sorted(taus)
    assert taus[-1] > taus[0]


def test_cheaper_local_steps_lead_to_more_local_steps():
    taus = [run_controller(phi=phi)[0] for phi in (1.0, 0.05, 0.01, 0.001)]
    assert taus == sorted(taus)
    assert taus[-1] > taus[0]


def test_higher_accuracy_weight_does_not_reduce_local_steps():
    taus = [run_controller(alpha=a, phi=0.001)[0] for a in (0.1, 0.5, 0.9)]
    assert taus == sorted(taus)


def test_time_budget_caps_local_steps():
    loose = run_controller(phi=0.001, b0=50_000_000, T_max=1000.0)[0]
    # Upload alone (with p = p_max) takes 50e6 * 0.5 / (50e6 / 8) = 4 s > T_max = 3 s.
    tight = run_controller(phi=0.001, b0=50_000_000, T_max=3.0)[0]
    assert loose > TAU_MIN
    assert tight == TAU_MIN
