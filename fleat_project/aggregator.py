from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class FederatedAggregator:
    @staticmethod
    def model_state_bytes(state: Dict[str, torch.Tensor]) -> int:
        total = 0
        for v in state.values():
            total += v.numel() * v.element_size()
        return int(total)

    @staticmethod
    def is_prunable_group_name(name: str) -> bool:
        n = str(name).lower()
        if not n.endswith(".weight"):
            return False
        if ("bn" in n) or ("norm" in n):
            return False
        if n in {"conv1.weight", "fc.weight"}:
            return False
        if n.startswith("classifier.") or n.startswith("heads.") or n.startswith("head."):
            return False
        if ".downsample.1." in n:
            return False
        return True

    @staticmethod
    def group_param_names(model: nn.Module) -> List[str]:
        return [name for name, _ in model.named_parameters()]

    @staticmethod
    def param_group_of_name(param_name: str) -> str:
        return param_name

    @staticmethod
    def group_sizes_bytes(model: nn.Module, group_order: List[str]) -> Dict[str, int]:
        sizes = {g: 0 for g in group_order}
        for name, p in model.named_parameters():
            sizes[name] = int(p.numel() * p.element_size())
        return sizes

    @staticmethod
    def compute_layer_importance(model: nn.Module, batch, criterion, device: torch.device, group_order: List[str]) -> Dict[str, float]:
        model.train()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        imp = {g: 0.0 for g in group_order}
        tot = 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = FederatedAggregator.param_group_of_name(name)
            val = float(torch.sum(p.grad.detach() ** 2).item())
            imp[g] = imp.get(g, 0.0) + val
            tot += val

        if tot <= 0:
            u = 1.0 / max(1, len(group_order))
            return {g: u for g in group_order}

        imp = {g: (v / tot) for g, v in imp.items()}
        s = sum(imp.values())
        return {g: imp.get(g, 0.0) / s for g in group_order}

    @staticmethod
    def select_pruned_groups(
        importance: Dict[str, float],
        group_order: List[str],
        group_bytes: Dict[str, int],
        target_p: float,
        protect_first_last: bool = True,
        noncontiguous: bool = True,
    ) -> Tuple[set, float, float]:
        total_b_all = sum(int(group_bytes.get(g, 0)) for g in group_order)
        if total_b_all <= 0 or target_p <= 0:
            gamma = float(sum(importance.get(g, 0.0) for g in group_order))
            gamma = max(0.0, min(1.0, gamma))
            return set(), 0.0, gamma

        candidates = [g for g in group_order if FederatedAggregator.is_prunable_group_name(g) and group_bytes.get(g, 0) > 0]
        if not candidates:
            gamma = float(sum(importance.get(g, 0.0) for g in group_order))
            gamma = max(0.0, min(1.0, gamma))
            return set(), 0.0, gamma

        protected = set()
        if protect_first_last and len(candidates) >= 1:
            protected.add(candidates[0])
            protected.add(candidates[-1])

        cand_idx = {g: i for i, g in enumerate(candidates)}
        sorted_groups = sorted(candidates, key=lambda g: (importance.get(g, 0.0), group_bytes.get(g, 0)))

        pruned = []
        pruned_set = set()
        B = 0

        def feasible(g: str) -> bool:
            if g in protected or g in pruned_set:
                return False
            if noncontiguous and pruned:
                i = cand_idx[g]
                for pg in pruned:
                    j = cand_idx[pg]
                    if abs(i - j) == 1:
                        return False
            return True

        for g in sorted_groups:
            if not feasible(g):
                continue
            bg = int(group_bytes.get(g, 0))
            if bg <= 0:
                continue
            if (B + bg) / max(1, total_b_all) <= target_p + 1e-12:
                pruned.append(g)
                pruned_set.add(g)
                B += bg

        if (B / max(1, total_b_all)) < target_p:
            remain = [g for g in sorted(candidates, key=lambda x: group_bytes.get(x, 0)) if feasible(g)]
            for g in remain:
                bg = int(group_bytes.get(g, 0))
                if bg <= 0:
                    continue
                if (B + bg) / max(1, total_b_all) <= target_p + 1e-12:
                    pruned.append(g)
                    pruned_set.add(g)
                    B += bg
                if (B / max(1, total_b_all)) >= target_p:
                    break

        p_hat = float(B / max(1, total_b_all))
        gamma = float(sum(importance.get(g, 0.0) for g in group_order if g not in pruned_set))
        gamma = float(max(0.0, min(1.0, gamma)))
        return pruned_set, p_hat, gamma

    @staticmethod
    def state_dict_delta(local_state, global_state):
        return {k: local_state[k] - global_state[k] for k in global_state.keys()}

    @staticmethod
    def make_masked_delta(delta: Dict[str, torch.Tensor], pruned_groups: set) -> Dict[str, Optional[torch.Tensor]]:
        out = {}
        for k, v in delta.items():
            if k in pruned_groups and torch.is_tensor(v) and torch.is_floating_point(v):
                out[k] = None
            else:
                out[k] = v
        return out

    @staticmethod
    def _weighted_average_like_reference(tensors: List[torch.Tensor], weights: List[float], ref: torch.Tensor) -> torch.Tensor:
        if len(tensors) == 0:
            return ref.clone()

        acc_dtype = torch.float64 if ref.dtype in (torch.float64,) else torch.float32
        acc = torch.zeros_like(ref, dtype=acc_dtype)
        wsum = 0.0
        for w, t in zip(weights, tensors):
            if t is None:
                continue
            acc = acc + float(w) * t.to(dtype=acc_dtype)
            wsum += float(w)

        if wsum <= 0:
            return ref.clone()

        avg = acc / float(wsum)
        if torch.is_floating_point(ref):
            return avg.to(dtype=ref.dtype)
        if ref.dtype == torch.bool:
            return (avg > 0.5).to(dtype=ref.dtype)
        return torch.round(avg).to(dtype=ref.dtype)

    @staticmethod
    def aggregate_masked_updates(global_state, client_updates: List[dict], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        total_size = float(sum(client_sizes))
        weights = [sz / total_size for sz in client_sizes]
        new_state = {k: v.clone() for k, v in global_state.items()}

        for name in global_state.keys():
            contrib_tensors = []
            contrib_weights = []
            for w, upd in zip(weights, client_updates):
                d = upd["delta_masked"][name]
                tau = max(1, int(upd["tau"]))
                if d is None:
                    continue
                if torch.is_floating_point(d):
                    d_eff = d
                else:
                    d_eff = d
                contrib_tensors.append(d_eff)
                contrib_weights.append(float(w))

            if contrib_weights:
                ref = global_state[name]
                if torch.is_floating_point(ref):
                    acc = FederatedAggregator._weighted_average_like_reference(
                        contrib_tensors, contrib_weights, torch.zeros_like(ref)
                    )
                    new_state[name] = (ref.to(dtype=acc.dtype) + acc).to(dtype=ref.dtype)
                else:
                    final_vals = []
                    for upd in client_updates:
                        d = upd["delta_masked"][name]
                        if d is None:
                            continue
                        final_vals.append((global_state[name] + d).clone())
                    if final_vals:
                        new_state[name] = FederatedAggregator._weighted_average_like_reference(final_vals, contrib_weights, ref)
                    else:
                        new_state[name] = ref
            else:
                new_state[name] = global_state[name]

        return new_state

    @staticmethod
    def aggregate_fedavg(global_state, client_states: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        total = float(sum(client_sizes))
        out = {}
        for k, v in global_state.items():
            if torch.is_floating_point(v):
                acc_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
                acc = torch.zeros_like(v, dtype=acc_dtype)
                for st, sz in zip(client_states, client_sizes):
                    w = float(sz) / max(total, 1e-12)
                    acc = acc + st[k].to(dtype=acc_dtype) * w
                out[k] = acc.to(dtype=v.dtype)
            else:
                vals = []
                ws = []
                for st, sz in zip(client_states, client_sizes):
                    vals.append(st[k])
                    ws.append(float(sz))
                out[k] = FederatedAggregator._weighted_average_like_reference(vals, ws, v)
        return out

    @staticmethod
    def estimate_round_energy_time(profile: dict, tau: int, phi_step: float, b_upload_bytes: int, b_download_bytes: int) -> dict:
        bw_up_Bps = max(1e-9, profile["bw_up"] * 1e6 / 8.0)
        bw_dn_Bps = max(1e-9, profile["bw_down"] * 1e6 / 8.0)
        Y = float(tau) * float(phi_step)
        C_up = float(b_upload_bytes) / bw_up_Bps
        C_dn = float(b_download_bytes) / bw_dn_Bps
        T = Y + C_up + C_dn
        E_comp = profile["p_comp"] * Y
        E_send = profile["p_send"] * C_up
        E_rec = profile["p_rec"] * C_dn
        return {
            "Y_comp_time_s": Y,
            "C_up_time_s": C_up,
            "C_dn_time_s": C_dn,
            "round_time_s": T,
            "E_comp_ws": E_comp,
            "E_send_ws": E_send,
            "E_rec_ws": E_rec,
            "E_total_ws": E_comp + E_send + E_rec,
        }

    @staticmethod
    def enforce_real_tmax_cap_per_client(
        profile: dict,
        b0_bytes: int,
        tau: int,
        p_target: float,
        p_achieved_est: float,
        phi_step: float,
        T_max: float,
        p_max: float,
        tau_min: int,
        tau_max: int,
        allow_tau_zero_if_needed: bool = True,
    ):
        eps = 1e-12

        if T_max is None or float(T_max) <= 0:
            return int(tau), float(p_target), {
                "T_cap_feasible": True,
                "T_cap_used": False,
                "T_cap_floor_time_s": None,
                "T_cap_pred_time_s": None,
            }

        bw_up_Bps = max(1e-9, float(profile["bw_up"]) * 1e6 / 8.0)
        bw_dn_Bps = max(1e-9, float(profile["bw_down"]) * 1e6 / 8.0)
        phi = max(1e-9, float(phi_step))
        tau_floor = 0 if allow_tau_zero_if_needed else max(1, int(tau_min))

        tau = int(max(tau_floor, min(int(tau_max), int(tau))))
        p_target = float(max(0.0, min(float(p_max), float(p_target))))
        p_achieved_est = float(max(0.0, min(float(p_max), float(p_achieved_est))))

        full_up_time = float(b0_bytes) / bw_up_Bps
        full_dn_time = float(b0_bytes) / bw_dn_Bps

        def pred_time(tau_v: int, p_eff: float) -> float:
            tau_v = max(0, int(tau_v))
            p_eff = max(0.0, min(float(p_max), p_eff))
            up_t = (1.0 - p_eff) * full_up_time
            return tau_v * phi + up_t + full_dn_time

        floor_time = pred_time(tau_floor, float(p_max))
        if floor_time > float(T_max) + 1e-9:
            return int(tau_floor), float(p_max), {
                "T_cap_feasible": False,
                "T_cap_used": True,
                "T_cap_floor_time_s": float(floor_time),
                "T_cap_pred_time_s": float(floor_time),
            }

        remaining_for_compute = float(T_max) - full_dn_time - (1.0 - p_achieved_est) * full_up_time
        tau_cap = int(math.floor(remaining_for_compute / phi + 1e-12))
        tau = int(max(tau_floor, min(int(tau_max), min(int(tau), tau_cap))))

        remaining_for_upload = float(T_max) - full_dn_time - tau * phi
        if remaining_for_upload < 0:
            p_req = float(p_max)
        else:
            p_req = 1.0 - (remaining_for_upload / max(eps, full_up_time))

        p_target = float(max(p_target, p_req))
        p_target = float(max(0.0, min(float(p_max), p_target)))
        t_pred = pred_time(tau, p_achieved_est)

        return int(tau), float(p_target), {
            "T_cap_feasible": True,
            "T_cap_used": True,
            "T_cap_floor_time_s": float(floor_time),
            "T_cap_pred_time_s": float(t_pred),
        }
