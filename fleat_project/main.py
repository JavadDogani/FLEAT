#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from .config import parse_args
from .experiment import ExperimentRunner
from .io_utils import save_outputs
from .utils import TeeStream, make_default_run_name


def main():
    args = parse_args()
    if getattr(args, "disable_log_to_file", False):
        args.log_to_file = False

    if not args.run_name:
        args.run_name = make_default_run_name(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else (out_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{args.run_name}.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_fh = None

    if getattr(args, "log_to_file", True):
        log_fh = open(log_path, "a", encoding="utf-8")
        sys.stdout = TeeStream(original_stdout, log_fh)
        sys.stderr = TeeStream(original_stderr, log_fh)
        print(f"[log] Writing execution log to: {log_path}")
        print(
            f"[run] method={args.method} model={args.model} dataset={args.dataset} clients={args.num_clients} "
            f"partition={args.partition} dirichlet_alpha={args.dirichlet_alpha if args.partition == 'dirichlet' else 'NA'}"
        )

    try:
        if args.alpha_sweep:
            alphas = [float(x.strip()) for x in args.alpha_sweep.split(",") if x.strip()]
            all_rows = []
            summaries = []

            for a in alphas:
                run_args = copy.deepcopy(args)
                run_args.alpha_weight = a
                run_args.run_name = f"{args.run_name}_alpha{a}"
                df, summary = ExperimentRunner(run_args).run()
                csv_path, json_path = save_outputs(df, summary, run_args.out_dir, run_args.run_name)
                srow = {
                    "alpha": a,
                    "final_accuracy": summary["metrics"]["final_accuracy"],
                    "total_energy_ws": summary["metrics"]["total_energy_ws"],
                    "total_time_s": summary["metrics"]["total_time_s"],
                    "mean_tau": summary["metrics"]["mean_tau"],
                    "mean_pruning_ratio": summary["metrics"]["mean_pruning_ratio"],
                    "round_csv": csv_path,
                    "summary_json": json_path,
                }
                summaries.append(summary)
                all_rows.append(srow)
                print(f"[alpha={a}] acc={srow['final_accuracy']:.4f} E={srow['total_energy_ws']:.3f}Ws T={srow['total_time_s']:.3f}s")

            sweep_df = pd.DataFrame(all_rows)
            sweep_csv = Path(args.out_dir) / f"{args.run_name}_alpha_sweep.csv"
            sweep_df.to_csv(sweep_csv, index=False)

            print("\nAlpha sweep summary:")
            print(sweep_df.to_string(index=False))
            print(f"\nSaved alpha sweep CSV: {sweep_csv}")
            if log_fh is not None:
                print(f"[log] Completed. Log file: {log_path}")
            return

        df, summary = ExperimentRunner(args).run()
        csv_path, json_path = save_outputs(df, summary, args.out_dir, args.run_name)

        print("\n=== SUMMARY (paper-style metrics) ===")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved round metrics CSV: {csv_path}")
        print(f"Saved summary JSON: {json_path}")
        if log_fh is not None:
            print(f"Saved execution log: {log_path}")

    except Exception as e:
        print(f"\n[error] {type(e).__name__}: {e}")
        print(traceback.format_exc())
        if log_fh is not None:
            print(f"[log] Error captured in: {log_path}")
        raise
    finally:
        if log_fh is not None:
            try:
                log_fh.flush()
                log_fh.close()
            except Exception:
                pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()
