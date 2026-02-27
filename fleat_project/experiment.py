from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .aggregator import FederatedAggregator
from .controller import FLEATController
from .data import DatasetFactory
from .models import ModelFactory
from .partitioning import PartitionManager
from .profiles import DeviceProfileFactory
from .runtime import ClientRuntimeState
from .trainer import LocalTrainer
from .utils import resolve_execution_devices, set_seed


class ExperimentRunner:
    def __init__(self, args):
        self.args = args

    def _run_one_client_round_task(
        self,
        k: int,
        subset: Dataset,
        client_size: int,
        total_client_samples: int,
        global_state,
        num_classes: int,
        in_ch: int,
        input_feature_dim: Optional[int],
        group_order,
        group_bytes,
        b0_bytes: int,
        profile: dict,
        runtime_state: ClientRuntimeState,
        F_curr: float,
        F_inf: float,
        round_time_ref: float,
        worker_device_str: str,
        round_idx: int,
    ):
        args = self.args
        device = torch.device(worker_device_str)
        if device.type == "cuda":
            try:
                torch.cuda.set_device(device)
            except Exception:
                pass

        dl_gen = torch.Generator()
        dl_gen.manual_seed(int(args.seed + 100000 * round_idx + 997 * k))
        bs = min(args.batch_size, max(1, len(subset)))
        local_loader = DataLoader(subset, batch_size=bs, shuffle=True, drop_last=False, generator=dl_gen)

        rt = copy.deepcopy(runtime_state)
        local_model = ModelFactory.build(args.model, num_classes=num_classes, in_channels=in_ch, input_feature_dim=input_feature_dim).to(device)
        local_model.load_state_dict(global_state)

        crit = nn.CrossEntropyLoss()
        try:
            imp_batch = next(iter(local_loader))
        except Exception:
            imp_batch = None

        if imp_batch is not None:
            importance = FederatedAggregator.compute_layer_importance(local_model, imp_batch, crit, device, group_order)
        else:
            importance = {g: 1.0 / max(1, len(group_order)) for g in group_order}

        data_frac = client_size / max(1, total_client_samples)
        tau_k, p_k, _aux = FLEATController.update_tau_p(
            client_state=rt,
            profile=profile,
            b0_bytes=b0_bytes,
            alpha_weight=args.alpha_weight,
            p_max=args.p_max,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            T_max=args.T_max,
            eta=args.lr,
            L_smooth=args.L_smooth,
            F_curr=F_curr,
            F_inf=F_inf,
            data_frac=data_frac,
            round_time_ref=round_time_ref,
            importance=importance,
            group_order=group_order,
            group_bytes=group_bytes,
            policy=args.method,
        )

        if args.method == "fedavg":
            tau_k, p_k = args.tau_init, 0.0
        elif args.method == "pruning_only":
            tau_k = args.tau_init
        elif args.method == "local_update_only":
            p_k = 0.0

        tcap_info = {
            "T_cap_feasible": True,
            "T_cap_used": False,
            "T_cap_floor_time_s": None,
            "T_cap_pred_time_s": None,
        }

        if getattr(args, "T_max", None) is not None and float(args.T_max) > 0:
            _, p_hat_est1, _ = FederatedAggregator.select_pruned_groups(
                importance=importance,
                group_order=group_order,
                group_bytes=group_bytes,
                target_p=float(p_k),
                protect_first_last=args.protect_first_last,
                noncontiguous=args.noncontiguous_pruning,
            )
            tau_k, p_k, tcap_info = FederatedAggregator.enforce_real_tmax_cap_per_client(
                profile=profile,
                b0_bytes=b0_bytes,
                tau=int(tau_k),
                p_target=float(p_k),
                p_achieved_est=float(p_hat_est1),
                phi_step=float(rt.phi_step),
                T_max=float(args.T_max),
                p_max=float(args.p_max),
                tau_min=int(args.tau_min),
                tau_max=int(args.tau_max),
                allow_tau_zero_if_needed=True,
            )

            _, p_hat_est2, _ = FederatedAggregator.select_pruned_groups(
                importance=importance,
                group_order=group_order,
                group_bytes=group_bytes,
                target_p=float(p_k),
                protect_first_last=args.protect_first_last,
                noncontiguous=args.noncontiguous_pruning,
            )
            tau_k, p_k, tcap_info = FederatedAggregator.enforce_real_tmax_cap_per_client(
                profile=profile,
                b0_bytes=b0_bytes,
                tau=int(tau_k),
                p_target=float(p_k),
                p_achieved_est=float(p_hat_est2),
                phi_step=float(rt.phi_step),
                T_max=float(args.T_max),
                p_max=float(args.p_max),
                tau_min=int(args.tau_min),
                tau_max=int(args.tau_max),
                allow_tau_zero_if_needed=True,
            )

        prox_mu = args.fedprox_mu if args.method == "fedprox" else 0.0

        if int(tau_k) > 0:
            local_model, train_loss, phi_meas, actual_steps = LocalTrainer.train_local_steps(
                local_model,
                local_loader,
                steps=tau_k,
                lr=args.lr,
                device=device,
                weight_decay=args.weight_decay,
                optimizer_name=args.optimizer,
                prox_mu=prox_mu,
                global_state=global_state,
            )
        else:
            train_loss = 0.0
            phi_meas = float(rt.phi_step)
            actual_steps = 0

        rt.phi_step = (1 - args.phi_ema) * rt.phi_step + args.phi_ema * phi_meas
        rt.last_importance = importance

        local_state = {n: p.detach().cpu().clone() for n, p in local_model.state_dict().items()}
        delta = FederatedAggregator.state_dict_delta(local_state, global_state)

        pruned_groups, p_hat, gamma = FederatedAggregator.select_pruned_groups(
            importance=importance,
            group_order=group_order,
            group_bytes=group_bytes,
            target_p=p_k,
            protect_first_last=args.protect_first_last,
            noncontiguous=args.noncontiguous_pruning,
        )
        delta_masked = FederatedAggregator.make_masked_delta(delta, pruned_groups)
        upload_bytes = int(round((1.0 - p_hat) * b0_bytes))

        et = FederatedAggregator.estimate_round_energy_time(
            profile,
            tau=max(0, int(actual_steps)),
            phi_step=rt.phi_step,
            b_upload_bytes=upload_bytes,
            b_download_bytes=b0_bytes,
        )

        G2_proxy = max(1e-4, 1.0 + (1.0 - gamma))
        rt.last_G2 = G2_proxy
        rt.last_gamma = gamma
        rt.tau = int(tau_k)
        rt.p = float(p_k)

        client_update = {
            "delta_masked": delta_masked,
            "tau": int(actual_steps),
            "size": client_size,
            "gamma": float(gamma),
            "p_target": float(p_k),
            "p_achieved": float(p_hat),
        }
        client_metric = {
            "client": k,
            "tau": int(tau_k),
            "tau_executed": int(actual_steps),
            "p_target": float(p_k),
            "p_achieved": float(p_hat),
            "gamma": float(gamma),
            "worker_device": worker_device_str,
            "T_cap_feasible": bool(tcap_info.get("T_cap_feasible", True)),
            "T_cap_used": bool(tcap_info.get("T_cap_used", False)),
            "T_cap_floor_time_s": tcap_info.get("T_cap_floor_time_s", None),
            "T_cap_pred_time_s": tcap_info.get("T_cap_pred_time_s", None),
            **et,
        }

        del local_model
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return {
            "k": k,
            "client_update": client_update,
            "client_state": local_state,
            "client_metric": client_metric,
            "runtime_state": rt,
            "train_loss": float(train_loss),
        }

    def run(self) -> Tuple[pd.DataFrame, dict]:
        args = self.args
        set_seed(args.seed)

        try:
            torch.set_num_threads(int(getattr(args, "torch_num_threads", 1)))
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(int(getattr(args, "torch_num_threads", 1)))
        except Exception:
            pass

        server_device_str, worker_device_strs = resolve_execution_devices(args)
        if args.device == "auto" and len(worker_device_strs) > 1:
            args.parallel_clients = True
        if not getattr(args, "parallel_clients", False):
            worker_device_strs = [worker_device_strs[0]]

        device = torch.device(server_device_str)
        if getattr(args, "verbose", False):
            print(f"[exec] server_device={server_device_str} workers={worker_device_strs} parallel_clients={args.parallel_clients}")

        train_ds, test_ds, num_classes, aux_dim = DatasetFactory.load_from_args(args)
        client_subsets = PartitionManager.make_client_subsets(train_ds, args)
        client_sizes = [len(s) for s in client_subsets]

        defaults = ModelFactory.paper_defaults(args.model, args.dataset)
        if args.batch_size is None:
            args.batch_size = defaults["batch_size"]
        if args.lr is None:
            args.lr = defaults["lr"]
        if args.tau_init is None:
            args.tau_init = defaults["tau_init"]

        image_like = args.dataset.lower() not in {"edge_iiotset"}
        if image_like:
            x0, _ = train_ds[0]
            if x0.dim() == 3:
                in_ch = int(x0.shape[0])
            elif x0.dim() == 2:
                in_ch = 1
            else:
                in_ch = 1
            input_feature_dim = None
        else:
            in_ch = 1
            input_feature_dim = int(aux_dim)

        global_model = ModelFactory.build(args.model, num_classes=num_classes, in_channels=in_ch, input_feature_dim=input_feature_dim).to(device)
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        group_order = FederatedAggregator.group_param_names(global_model)
        group_bytes = FederatedAggregator.group_sizes_bytes(global_model, group_order)
        b0_bytes = FederatedAggregator.model_state_bytes(global_state)

        profiles = DeviceProfileFactory.sample(args.num_clients, args.device_profile_mode, seed=args.seed)

        client_runtimes = []
        for k in range(args.num_clients):
            prof = profiles[k]
            base_phi = args.base_phi_step * (8.0 / max(1.0, prof["p_comp"]))
            client_runtimes.append(ClientRuntimeState(tau=args.tau_init, p=args.p_init, phi_step=base_phi))

        history = []
        cumulative_time = 0.0
        cumulative_energy = 0.0
        cumulative_comp_e = 0.0
        cumulative_comm_e = 0.0
        cumulative_comp_t = 0.0
        cumulative_comm_t = 0.0

        global_model.load_state_dict(global_state)
        eval0 = LocalTrainer.evaluate_model(
            global_model,
            test_ds,
            batch_size=args.eval_batch_size,
            device=device,
            max_eval_samples=args.max_eval_samples,
        )
        history.append({
            "round": 0,
            "eval_acc": eval0["eval_acc"],
            "eval_loss": eval0["eval_loss"],
            "round_time_s": 0.0,
            "cumulative_time_s": cumulative_time,
            "round_energy_ws": 0.0,
            "cumulative_energy_ws": cumulative_energy,
            "round_comp_energy_ws": 0.0,
            "round_comm_energy_ws": 0.0,
            "cumulative_comp_energy_ws": cumulative_comp_e,
            "cumulative_comm_energy_ws": cumulative_comm_e,
            "round_comp_time_s": 0.0,
            "round_comm_time_s": 0.0,
            "tau_mean": np.mean([rt.tau for rt in client_runtimes]),
            "p_mean": np.mean([rt.p for rt in client_runtimes]),
            "tau_list": [int(rt.tau) for rt in client_runtimes],
            "p_target_list": [float(rt.p) for rt in client_runtimes],
            "p_achieved_list": [float(rt.p) for rt in client_runtimes],
            "method": args.method,
            "alpha": args.alpha_weight,
            "partition": args.partition,
            "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
        })

        for r in range(1, args.rounds + 1):
            client_updates = []
            client_states = []
            client_metrics = []

            F_curr = float(history[-1]["eval_loss"])
            F_inf = float(args.F_inf)
            prev_rt = history[-1]["round_time_s"] if history[-1]["round_time_s"] > 0 else args.T_max
            round_time_ref = float(prev_rt)
            total_client_samples = int(sum(client_sizes))

            ordered_results = [None] * args.num_clients
            use_parallel = bool(getattr(args, "parallel_clients", False)) and len(worker_device_strs) > 1

            if use_parallel:
                max_workers = int(getattr(args, "max_parallel_clients", 0) or 0)
                if max_workers <= 0:
                    max_workers = len(worker_device_strs)
                max_workers = max(1, min(max_workers, args.num_clients))

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {}
                    for k in range(args.num_clients):
                        worker_dev = worker_device_strs[k % len(worker_device_strs)]
                        fut = ex.submit(
                            self._run_one_client_round_task,
                            k,
                            client_subsets[k],
                            client_sizes[k],
                            total_client_samples,
                            global_state,
                            num_classes,
                            in_ch,
                            input_feature_dim,
                            group_order,
                            group_bytes,
                            b0_bytes,
                            profiles[k],
                            client_runtimes[k],
                            F_curr,
                            F_inf,
                            round_time_ref,
                            worker_dev,
                            r,
                        )
                        futs[fut] = k
                    for fut in as_completed(futs):
                        res = fut.result()
                        ordered_results[int(res["k"])] = res
            else:
                single_dev = worker_device_strs[0] if worker_device_strs else str(device)
                for k in range(args.num_clients):
                    res = self._run_one_client_round_task(
                        k,
                        client_subsets[k],
                        client_sizes[k],
                        total_client_samples,
                        global_state,
                        num_classes,
                        in_ch,
                        input_feature_dim,
                        group_order,
                        group_bytes,
                        b0_bytes,
                        profiles[k],
                        client_runtimes[k],
                        F_curr,
                        F_inf,
                        round_time_ref,
                        single_dev,
                        r,
                    )
                    ordered_results[k] = res

            for k in range(args.num_clients):
                res = ordered_results[k]
                if res is None:
                    raise RuntimeError(f"Missing client result for client {k} in round {r}")
                client_runtimes[k] = res["runtime_state"]
                client_updates.append(res["client_update"])
                client_states.append(res["client_state"])
                client_metrics.append(res["client_metric"])

            if args.method == "fedavg":
                new_state = FederatedAggregator.aggregate_fedavg(global_state, client_states, client_sizes)
            else:
                new_state = FederatedAggregator.aggregate_masked_updates(global_state, client_updates, client_sizes)

            global_state = new_state
            global_model.load_state_dict(global_state)

            ev = LocalTrainer.evaluate_model(
                global_model,
                test_ds,
                batch_size=args.eval_batch_size,
                device=device,
                max_eval_samples=args.max_eval_samples,
            )

            round_time = float(max(cm["round_time_s"] for cm in client_metrics)) if client_metrics else 0.0
            round_energy_total = float(sum(cm["E_total_ws"] for cm in client_metrics))
            round_comp_e = float(sum(cm["E_comp_ws"] for cm in client_metrics))
            round_comm_e = float(sum(cm["E_send_ws"] + cm["E_rec_ws"] for cm in client_metrics))
            round_comp_t = float(sum(cm["Y_comp_time_s"] for cm in client_metrics))
            round_comm_t = float(sum(cm["C_up_time_s"] + cm["C_dn_time_s"] for cm in client_metrics))
            cumulative_time += round_time
            cumulative_energy += round_energy_total
            cumulative_comp_e += round_comp_e
            cumulative_comm_e += round_comm_e
            cumulative_comp_t += round_comp_t
            cumulative_comm_t += round_comm_t

            client_metrics_sorted = sorted(client_metrics, key=lambda x: int(x["client"]))
            tau_list = [int(cm["tau"]) for cm in client_metrics_sorted]
            tau_executed_list = [int(cm.get("tau_executed", cm["tau"])) for cm in client_metrics_sorted]
            p_target_list = [float(cm["p_target"]) for cm in client_metrics_sorted]
            p_achieved_list = [float(cm["p_achieved"]) for cm in client_metrics_sorted]
            gamma_list = [float(cm["gamma"]) for cm in client_metrics_sorted]
            tcap_ok_frac = float(np.mean([1.0 if bool(cm.get("T_cap_feasible", True)) else 0.0 for cm in client_metrics_sorted])) if client_metrics_sorted else 1.0

            history.append({
                "round": r,
                "eval_acc": ev["eval_acc"],
                "eval_loss": ev["eval_loss"],
                "round_time_s": round_time,
                "cumulative_time_s": cumulative_time,
                "round_energy_ws": round_energy_total,
                "cumulative_energy_ws": cumulative_energy,
                "round_comp_energy_ws": round_comp_e,
                "round_comm_energy_ws": round_comm_e,
                "cumulative_comp_energy_ws": cumulative_comp_e,
                "cumulative_comm_energy_ws": cumulative_comm_e,
                "round_comp_time_s": round_comp_t,
                "round_comm_time_s": round_comm_t,
                "tau_mean": float(np.mean(tau_list)) if tau_list else 0.0,
                "p_mean": float(np.mean(p_achieved_list)) if p_achieved_list else 0.0,
                "gamma_mean": float(np.mean(gamma_list)) if gamma_list else 1.0,
                "tau_list": tau_list,
                "tau_executed_list": tau_executed_list,
                "p_target_list": p_target_list,
                "p_achieved_list": p_achieved_list,
                "T_cap_feasible_frac": tcap_ok_frac,
                "method": args.method,
                "alpha": args.alpha_weight,
                "partition": args.partition,
                "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
            })

            if args.verbose:
                tau_list_str = "[" + ", ".join(str(x) for x in tau_list) + "]"
                tau_exec_list_str = "[" + ", ".join(str(x) for x in tau_executed_list) + "]"
                p_tgt_list_str = "[" + ", ".join(f"{x:.3f}" for x in p_target_list) + "]"
                p_hat_list_str = "[" + ", ".join(f"{x:.3f}" for x in p_achieved_list) + "]"
                print(
                    f"[r={r:03d}] acc={ev['eval_acc']:.4f} loss={ev['eval_loss']:.4f} "
                    f"E_round={round_energy_total:.3f}Ws E_cum={cumulative_energy:.3f}Ws "
                    f"T_round={round_time:.3f}s T_cum={cumulative_time:.3f}s "
                    f"tau_mean={history[-1]['tau_mean']:.2f} p_mean={history[-1]['p_mean']:.3f} "
                    f"Tcap_ok_frac={tcap_ok_frac:.2f}"
                )
                print(f"         tau_target={tau_list_str}")
                print(f"         tau_exec  ={tau_exec_list_str}")
                print(f"         p_target  ={p_tgt_list_str}")
                print(f"         p_achieved={p_hat_list_str}")

        df = pd.DataFrame(history)
        final_row = df.iloc[-1].to_dict()
        best_acc_idx = int(df["eval_acc"].idxmax())
        best_acc_row = df.loc[best_acc_idx].to_dict()

        def acc_at_time(target_t: float) -> float:
            sub = df[df["cumulative_time_s"] <= target_t]
            if len(sub) == 0:
                return float(df.iloc[0]["eval_acc"])
            return float(sub.iloc[-1]["eval_acc"])

        common_wall_clock = float(df["cumulative_time_s"].max() * 0.5)

        summary = {
            "method": args.method,
            "model": args.model,
            "dataset": args.dataset,
            "partition": args.partition,
            "dirichlet_alpha": args.dirichlet_alpha if args.partition == "dirichlet" else None,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "alpha_weight": args.alpha_weight,
            "execution": {
                "server_device": server_device_str,
                "worker_devices": worker_device_strs,
                "parallel_clients": bool(getattr(args, "parallel_clients", False)),
            },
            "metrics": {
                "final_accuracy": float(final_row["eval_acc"]),
                "final_loss": float(final_row["eval_loss"]),
                "best_round_accuracy": float(best_acc_row["eval_acc"]),
                "best_round": int(best_acc_row["round"]),
                "total_energy_ws": float(final_row["cumulative_energy_ws"]),
                "total_comp_energy_ws": float(final_row["cumulative_comp_energy_ws"]),
                "total_comm_energy_ws": float(final_row["cumulative_comm_energy_ws"]),
                "total_time_s": float(final_row["cumulative_time_s"]),
                "acc_at_common_wall_clock_half_total": acc_at_time(common_wall_clock),
                "common_wall_clock_half_total_s": common_wall_clock,
                "avg_round_energy_ws": float(df["round_energy_ws"].iloc[1:].mean()) if len(df) > 1 else 0.0,
                "avg_round_time_s": float(df["round_time_s"].iloc[1:].mean()) if len(df) > 1 else 0.0,
                "mean_tau": float(df["tau_mean"].iloc[1:].mean()) if len(df) > 1 else float(df["tau_mean"].iloc[0]),
                "mean_pruning_ratio": float(df["p_mean"].iloc[1:].mean()) if "p_mean" in df and len(df) > 1 else 0.0,
                "mean_gamma": float(df["gamma_mean"].iloc[1:].mean()) if "gamma_mean" in df and len(df) > 1 else 1.0,
                "compute_time_share": float(
                    df["round_comp_time_s"].sum() / max(1e-9, (df["round_comp_time_s"].sum() + df["round_comm_time_s"].sum()))
                ),
                "comm_time_share": float(
                    df["round_comm_time_s"].sum() / max(1e-9, (df["round_comp_time_s"].sum() + df["round_comm_time_s"].sum()))
                ),
            },
        }
        return df, summary
