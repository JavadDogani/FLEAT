from __future__ import annotations

import numpy as np

REAL_DEVICE_PROFILES = [
    {"name": "pi0", "bw_up": 35.0, "bw_down": 70.0, "p_comp": 3.0, "p_send": 2.0, "p_rec": 1.8},
    {"name": "pi3", "bw_up": 55.0, "bw_down": 110.0, "p_comp": 4.2, "p_send": 2.8, "p_rec": 2.4},
    {"name": "pi4", "bw_up": 70.0, "bw_down": 140.0, "p_comp": 6.5, "p_send": 4.2, "p_rec": 3.7},
    {"name": "nano", "bw_up": 930.0, "bw_down": 930.0, "p_comp": 9.0, "p_send": 6.0, "p_rec": 5.2},
    {"name": "xavier", "bw_up": 940.0, "bw_down": 940.0, "p_comp": 12.0, "p_send": 7.2, "p_rec": 6.0},
]

SIM_TIERS = [
    {"name": "wifi5_sbc", "prob": 0.55, "bw_up": (40, 90), "bw_down": (80, 180), "p_comp": (5.5, 11), "p_send": (3.5, 6.5), "p_rec": (3.0, 5.5)},
    {"name": "wifi6_sbc", "prob": 0.30, "bw_up": (120, 320), "bw_down": (250, 700), "p_comp": (8.0, 16), "p_send": (4.5, 8.5), "p_rec": (3.5, 7.0)},
    {"name": "gbe_mini", "prob": 0.15, "bw_up": (800, 950), "bw_down": (800, 950), "p_comp": (10, 30), "p_send": (7.0, 12), "p_rec": (5.0, 9.0)},
]


class DeviceProfileFactory:
    @staticmethod
    def sample(num_clients: int, mode: str, seed: int = 42):
        rng = np.random.default_rng(seed)
        profiles = []

        if mode == "real5":
            for i in range(num_clients):
                p = dict(REAL_DEVICE_PROFILES[i % len(REAL_DEVICE_PROFILES)])
                p["id"] = i
                profiles.append(p)
            return profiles

        probs = np.array([t["prob"] for t in SIM_TIERS], dtype=float)
        probs = probs / probs.sum()
        tier_ids = rng.choice(len(SIM_TIERS), size=num_clients, p=probs)
        for i, tid in enumerate(tier_ids):
            t = SIM_TIERS[int(tid)]
            profiles.append({
                "id": i,
                "name": t["name"],
                "bw_up": float(rng.uniform(*t["bw_up"])),
                "bw_down": float(rng.uniform(*t["bw_down"])),
                "p_comp": float(rng.uniform(*t["p_comp"])),
                "p_send": float(rng.uniform(*t["p_send"])),
                "p_rec": float(rng.uniform(*t["p_rec"])),
            })
        return profiles
