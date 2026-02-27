from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="FLEAT simulation/evaluation runner")

    p.add_argument("--method", type=str, default="fleat",
                   choices=["fleat", "fedavg", "fedprox", "local_update_only", "pruning_only"],
                   help="Training/control method")
    p.add_argument("--model", type=str, default="googlenet",
                   help="Model: efficientnet_b0, mobilenet_v2, resnet18, googlenet, cnn_edge, rnn_edge")
    p.add_argument("--dataset", type=str, default="mnist",
                   help="Dataset: mnist, svhn, cifar10, fashionmnist/fmnist, edge_iiotset, sklearn_digits, synthetic")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--edge_csv_path", type=str, default="", help="CSV path for Edge-IIoTset-like tabular dataset")
    p.add_argument("--edge_label_col", type=str, default=None)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha for label-skew non-IID")
    p.add_argument("--min_client_samples", type=int, default=10)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "rmsprop"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--fedprox_mu", type=float, default=0.01)

    p.add_argument("--alpha_weight", type=float, default=0.5, help="Accuracy-energy tradeoff weight alpha")
    p.add_argument("--tau_min", type=int, default=1)
    p.add_argument("--tau_max", type=int, default=10)
    p.add_argument("--tau_init", type=int, default=10)
    p.add_argument("--p_init", type=float, default=0.0)
    p.add_argument("--p_max", type=float, default=0.5)
    p.add_argument("--T_max", type=float, default=20, help="Per-round target time budget proxy (s) for controller")
    p.add_argument("--L_smooth", type=float, default=1.0, help="Smoothness constant proxy")
    p.add_argument("--F_inf", type=float, default=0.0, help="Lower bound proxy for loss")
    p.add_argument("--phi_ema", type=float, default=0.3)
    p.add_argument("--base_phi_step", type=float, default=0.02, help="Base per-step time proxy (s)")

    p.add_argument("--protect_first_last", action="store_true", default=True)
    p.add_argument("--noncontiguous_pruning", action="store_true", default=True)

    p.add_argument("--device_profile_mode", type=str, default="sim_table3", choices=["sim_table3", "real5"],
                   help="Use simulation tier sampling (Table III style) or paper real-device profiles cyclically")

    p.add_argument("--device", type=str, default="cuda", help="Execution device for workers: cpu, cuda, cuda:N, or auto")
    p.add_argument("--server_device", type=str, default="cuda", help="Server/eval device: auto, cpu, cuda, or cuda:N")
    p.add_argument("--parallel_clients", action="store_true", default=True, help="Parallelize client local updates across available GPUs")
    p.add_argument("--gpu_ids", type=str, default="0,1,2", help="Comma-separated CUDA device IDs to use (default: all visible)")
    p.add_argument("--max_gpus", type=int, default=3, help="Maximum number of GPUs to use (0 = all selected GPUs)")
    p.add_argument("--max_parallel_clients", type=int, default=3, help="Thread workers for client parallelism (0 = number of worker GPUs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch_num_threads", type=int, default=1)
    p.add_argument("--max_eval_samples", type=int, default=2000)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--out_dir", type=str, default="./outputs_fleat")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--log_to_file", action="store_true", default=True, help="Write console output and errors to a run-specific log file")
    p.add_argument("--disable_log_to_file", action="store_true", default=False, help="Disable run log file")
    p.add_argument("--log_dir", type=str, default="", help="Directory for execution logs (default: out_dir/logs)")

    p.add_argument("--synthetic_train", type=int, default=2000)
    p.add_argument("--synthetic_test", type=int, default=500)

    p.add_argument("--alpha_sweep", type=str, default="", help='Comma-separated alpha values (e.g., "0,0.25,0.5,0.75,1")')

    args = p.parse_args()
    if args.quiet:
        args.verbose = False
    return args
