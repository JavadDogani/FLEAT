from __future__ import annotations

import random
import time
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def parse_gpu_ids(gpu_ids: str) -> List[int]:
    if gpu_ids is None or str(gpu_ids).strip() == "":
        return []
    out: List[int] = []
    for tok in str(gpu_ids).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def resolve_execution_devices(args) -> Tuple[str, List[str]]:
    requested = str(getattr(args, "device", "cpu")).strip().lower()
    server_req = str(getattr(args, "server_device", "auto")).strip().lower()
    parallel = bool(getattr(args, "parallel_clients", False))
    max_gpus = int(getattr(args, "max_gpus", 0) or 0)
    parsed_gpu_ids = parse_gpu_ids(getattr(args, "gpu_ids", ""))

    cuda_available = torch.cuda.is_available()
    visible_count = torch.cuda.device_count() if cuda_available else 0

    def cuda_worker_list() -> List[str]:
        if not cuda_available or visible_count <= 0:
            return []
        ids = parsed_gpu_ids if parsed_gpu_ids else list(range(visible_count))
        ids = [i for i in ids if 0 <= i < visible_count]
        if max_gpus > 0:
            ids = ids[:max_gpus]
        return [f"cuda:{i}" for i in ids]

    if requested == "auto":
        workers = cuda_worker_list() if cuda_available else []
        if not workers:
            workers = ["cpu"]
    elif requested.startswith("cuda"):
        if requested == "cuda":
            requested = "cuda:0"
        if parallel:
            workers = cuda_worker_list()
            if not workers:
                workers = [requested if cuda_available else "cpu"]
        else:
            workers = [requested if cuda_available else "cpu"]
    else:
        workers = [requested]

    if not parallel and len(workers) > 1:
        workers = [workers[0]]

    if server_req == "auto":
        server = workers[0] if workers else ("cuda:0" if cuda_available else "cpu")
    else:
        server = "cuda:0" if server_req == "cuda" else server_req
        if server.startswith("cuda") and not cuda_available:
            server = "cpu"

    return server, workers


def sanitize_run_token(x: str) -> str:
    s = str(x)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    s = "".join(out).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s or "x"


def format_alpha_for_name(a: float) -> str:
    try:
        return str(a).replace(".", "p")
    except Exception:
        return "na"


def make_default_run_name(args) -> str:
    parts = [
        sanitize_run_token(args.method),
        sanitize_run_token(args.model),
        sanitize_run_token(args.dataset),
        f"N{int(args.num_clients)}",
        sanitize_run_token(args.partition),
    ]
    if str(args.partition).lower() == "dirichlet":
        parts.append(f"a{format_alpha_for_name(args.dirichlet_alpha)}")
    parts.append(now_ts())
    return "_".join(parts)


class TeeStream:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
