from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .runtime import ClientRuntimeState


class FLEATController:
    @staticmethod
    def update_tau_p(
        client_state: ClientRuntimeState,
        profile: dict,
        b0_bytes: int,
        alpha_weight: float,
        p_max: float,
        tau_min: int,
        tau_max: int,
        T_max: float,
        eta: float,
        L_smooth: float,
        F_curr: float,
        F_inf: float,
        data_frac: float,
        round_time_ref: float,
        importance: Dict[str, float],
        group_order: List[str],
        group_bytes: Dict[str, int],
        policy: str = "fleat",
    ) -> Tuple[int, float, dict]:
        eps = 1e-8
        a = max(eps, min(1.0 - eps, alpha_weight))

        if policy == "fedavg":
            return int(client_state.tau), 0.0, {"gamma": 1.0, "Mk": client_state.last_G2}

        if policy == "local_update_only":
            p = 0.0
        elif policy == "pruning_only":
            p = client_state.p
            tau = int(client_state.tau)
            return tau, float(max(0.0, min(p_max, p))), {"gamma": client_state.last_gamma, "Mk": client_state.last_G2}
        else:
            p = float(max(0.0, min(p_max, client_state.p)))

        items = [(g, importance.get(g, 0.0), group_bytes.get(g, 0)) for g in group_order]
        total_b = sum(b for _, _, b in items) or 1
        sorted_items = sorted(items, key=lambda t: (t[1], t[2]))

        cum_b = 0
        cum_m = 0.0
        p_points = [0.0]
        s_points = [0.0]
        for g, imp, b in sorted_items:
            cum_b += b
            cum_m += imp
            p_points.append(min(1.0, cum_b / total_b))
            s_points.append(min(1.0, cum_m))

        def S_interp(pv: float) -> float:
            pv = float(max(0.0, min(1.0, pv)))
            for i in range(1, len(p_points)):
                if pv <= p_points[i]:
                    x0, x1 = p_points[i - 1], p_points[i]
                    y0, y1 = s_points[i - 1], s_points[i]
                    if x1 <= x0 + 1e-12:
                        return y1
                    t = (pv - x0) / (x1 - x0)
                    return y0 + t * (y1 - y0)
            return s_points[-1]

        gamma_p = 1.0 - S_interp(p)
        G2 = max(1e-6, client_state.last_G2)
        Mk = 0.1 + (1.0 - gamma_p) * G2

        Tr = max(1e-6, round_time_ref)
        Ak = max(1e-8, 2.0 * max(1e-6, (F_curr - F_inf + 1e-4)) / (max(1e-8, eta) * Tr) * data_frac)
        Bk = max(1e-8, eta * L_smooth * data_frac)
        Ck = max(1e-8, (eta ** 2) * (L_smooth ** 2) * data_frac)

        tau = int(max(tau_min, min(tau_max, client_state.tau)))
        for _ in range(2):
            if policy in {"fleat", "fedprox", "local_update_only"} and policy != "local_update_only":
                p_prev = max(1e-3, min(p_max, p if p > 0 else 0.1))
                beta = max(1.05, client_state.beta)
                Sp = max(1e-6, S_interp(p_prev))
                kappa = max(1e-6, Sp / (p_prev ** beta)) if p_prev > 0 else 1.0
                client_state.kappa = kappa
                client_state.beta = beta

                Rk = a * Ak * ((b0_bytes / max(1e-9, profile["bw_up"] * 1e6 / 8.0)) / max(1, tau)) + \
                     (1 - a) * profile["p_send"] * (b0_bytes / max(1e-9, profile["bw_up"] * 1e6 / 8.0))
                Dk = a * (Bk + Ck * (max(1, tau) - 1)) * G2

                if Dk <= 1e-12:
                    p_new = 0.0
                else:
                    base = max(1e-12, Rk / max(1e-12, Dk * kappa * beta))
                    p_new = float(base ** (1.0 / (beta - 1.0)))
                p = float(max(0.0, min(p_max, p_new)))
            elif policy == "local_update_only":
                p = 0.0

            gamma_p = 1.0 - S_interp(p)
            Mk = 0.1 + (1.0 - gamma_p) * G2

            if policy in {"fleat", "fedprox", "local_update_only"}:
                Nk = (b0_bytes * (1.0 - p)) / max(1e-9, profile["bw_up"] * 1e6 / 8.0)
                phi = max(1e-6, client_state.phi_step)
                Gamma = (Ck * Mk) + ((1.0 - a) / a) * (profile["p_comp"] * phi) / Ak
                denom = max(1e-8, phi + Gamma)
                tau_star = math.sqrt(max(0.0, Nk) / denom)
                tau_star = max(float(tau_min), min(float(tau_max), tau_star))
                tau_time_cap = max(float(tau_min), (T_max - Nk) / max(phi, 1e-8)) if T_max > Nk else float(tau_min)
                tau_star = min(tau_star, tau_time_cap, float(tau_max))
                tau = int(max(tau_min, min(tau_max, round(tau_star))))
            else:
                tau = int(client_state.tau)

        aux = {
            "gamma": float(max(0.0, min(1.0, gamma_p))),
            "Mk": float(Mk),
            "Ak": float(Ak),
            "Bk": float(Bk),
            "Ck": float(Ck),
        }
        return int(tau), float(p), aux
