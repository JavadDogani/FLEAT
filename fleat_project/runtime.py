from __future__ import annotations

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass
class ClientRuntimeState:
    tau: int
    p: float
    beta: float = 2.0
    kappa: float = 1.0
    phi_step: float = 0.02
    last_gamma: float = 1.0
    last_G2: float = 1.0
    last_importance: Optional[Dict[str, float]] = None
