from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_outputs(df: pd.DataFrame, summary: dict, out_dir: str, run_name: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{run_name}_round_metrics.csv"
    json_path = out / f"{run_name}_summary.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return str(csv_path), str(json_path)
