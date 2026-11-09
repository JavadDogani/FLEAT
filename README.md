# FLEAT Project

[![CI](https://github.com/JavadDogani/FLEAT/actions/workflows/ci.yml/badge.svg)](https://github.com/JavadDogani/FLEAT/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Paper: IEEE TMC 2026](https://img.shields.io/badge/paper-IEEE%20TMC%202026-00629B.svg)](https://doi.org/10.1109/TMC.2026.3710334)

A modular, simulation-first implementation of the FLEAT experimental runner for federated learning benchmarking.  
This refactoring keeps the original **context, execution flow, controller logic, energy/time accounting, aggregation behavior, and CLI semantics**, but splits the code into maintainable modules that are easier to extend, debug, and publish on GitHub.

The code accompanies the paper *FLEAT: Energy-Accuracy Trade-Off in Federated Learning Over Heterogeneous Edge Devices* (IEEE Transactions on Mobile Computing, 2026, [doi:10.1109/TMC.2026.3710334](https://doi.org/10.1109/TMC.2026.3710334)); see [Paper and citation](#paper-and-citation).

## What this project does

This project implements a practical approximation of the FLEAT workflow for reproducible experiments in synchronous federated learning. It supports:

- synchronous FL with local training on multiple clients
- per-client adaptive local-step control
- masked upload with contributor-aware aggregation
- simulation-based communication and energy accounting
- FLEAT-style alternating update for local steps and pruning ratio
- baseline modes:
  - `fleat`
  - `fedavg`
  - `fedprox`
  - `local_update_only`
  - `pruning_only`
- IID and Dirichlet non-IID partitioning
- image datasets and tabular Edge-IIoTset-like data
- single-GPU, multi-GPU, or CPU execution
- alpha-sweep experiments
- CSV, JSON, and log-file outputs

## Quickstart

A short CPU-only run on scikit-learn's bundled 8x8 digits dataset (nothing is downloaded besides the Python packages):

```bash
git clone https://github.com/JavadDogani/FLEAT.git
cd FLEAT
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only PyTorch (smaller download)
pip install -e .
fleat --dataset sklearn_digits --model resnet18 --num_clients 5 --rounds 5 --device cpu --server_device cpu
```

This trains FLEAT for 5 rounds with 5 simulated clients on a Dirichlet non-IID split, prints per-round accuracy, energy, time, `tau`, and pruning ratios, and writes the round-level CSV, summary JSON, and log to `./outputs_fleat/`. `fleat` is the same CLI as `python -m fleat_project.main`; run `fleat --help` for all options.

## Project structure

```text
FLEAT/
├── .github/workflows/ci.yml   # lint + tests on Python 3.10-3.12
├── fleat_project/
│   ├── __init__.py
│   ├── aggregator.py
│   ├── config.py
│   ├── controller.py
│   ├── data.py
│   ├── experiment.py
│   ├── io_utils.py
│   ├── main.py
│   ├── models.py
│   ├── partitioning.py
│   ├── profiles.py
│   ├── runtime.py
│   ├── trainer.py
│   └── utils.py
├── scripts/
│   └── run_example.sh         # the "Basic usage" run below (20 rounds)
├── tests/                     # pytest suite (CPU-only, synthetic data)
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Module overview

### `config.py`
Contains the CLI argument parser.  
All original experiment flags are preserved so you can keep the same command-line workflow.

### `models.py`
Contains:
- `SmallImageCNN`
- `EdgeIotCNN`
- `EdgeIotGRU`
- `ModelFactory`

This file centralizes all model construction and paper-style per-model defaults.

`--model` accepts `googlenet` (default), `resnet18`, `mobilenet_v2`, `efficientnet_b0` (torchvision, adapted to 1-channel input when needed) and `cnn_edge` / `rnn_edge` (tabular Edge-IIoTset data). Any other name, or a torchvision model that cannot be built, falls back to `SmallImageCNN`.  
Of the per-model defaults, only the batch size is applied from the command line (when `--batch_size` is omitted); `--lr` and `--tau_init` always have CLI defaults (`0.02` and `10`), so the per-model learning rate and initial `tau` are not used by `fleat` / `python -m fleat_project.main`.

### `data.py`
Contains:
- `SimpleTransformDataset`
- `DatasetFactory`

Responsible for loading:
- MNIST
- FashionMNIST
- CIFAR-10
- SVHN
- sklearn digits
- synthetic image data
- Edge-IIoTset-like CSV data

### `partitioning.py`
Contains `PartitionManager`, which handles:
- IID partitioning
- Dirichlet label-skew partitioning
- target extraction from different dataset types
- creation of per-client subsets

### `profiles.py`
Contains device profile definitions and `DeviceProfileFactory`, which generates:
- fixed real-device profiles
- sampled simulated device tiers

### `runtime.py`
Contains the `ClientRuntimeState` dataclass that stores each client's controller state, including:
- current tau
- current pruning ratio
- per-step time estimate
- controller statistics such as gamma and G2 proxies

### `aggregator.py`
Contains `FederatedAggregator`, which implements:
- state-size accounting
- importance computation
- exact-parameter grouping
- pruning-group selection
- masked delta generation
- masked aggregation
- FedAvg aggregation
- energy/time estimation
- strict real `T_max` cap enforcement

### `trainer.py`
Contains `LocalTrainer`, which provides:
- local training for a fixed number of steps
- evaluation on the test set

### `controller.py`
Contains `FLEATController`, which performs the approximate FLEAT update for:
- local training steps `tau`
- upload pruning ratio `p`

### `experiment.py`
Contains `ExperimentRunner`, which is the main orchestration class.  
It performs:
- dataset loading
- model creation
- client partitioning
- local round execution
- multi-GPU client scheduling
- aggregation
- evaluation
- metric tracking
- summary generation

### `io_utils.py`
Handles saving:
- round-level CSV metrics
- summary JSON

### `main.py`
Entry point of the project.  
It supports:
- standard single-run execution
- alpha sweeps
- log file generation
- final summary printing

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

Installing the package also provides the `fleat` command, equivalent to `python -m fleat_project.main`. Python 3.10-3.12 is tested in CI. On machines without a GPU you can first install the smaller CPU-only PyTorch wheels with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Basic usage

Run a standard experiment:

```bash
python -m fleat_project.main \
  --method fleat \
  --model googlenet \
  --dataset mnist \
  --num_clients 10 \
  --rounds 60 \
  --partition dirichlet \
  --dirichlet_alpha 0.3 \
  --device cuda \
  --server_device cuda
```

This downloads MNIST to `--data_root` (default `./data`). If CUDA is not available, `--device cuda` and `--server_device cuda` fall back to the CPU.

## Output files

Each run writes to `--out_dir` (default `./outputs_fleat`), using `--run_name` (default: method, model, dataset, number of clients, partition, Dirichlet alpha if applicable, and a timestamp) as the file prefix:

- `*_round_metrics.csv`  
  Per-round metrics including accuracy, loss, energy, time, tau, and pruning information.

- `*_summary.json`  
  Final paper-style summary metrics.

- `logs/*.log`  
  Full execution log, written to `<out_dir>/logs/` by default (change with `--log_dir`, disable with `--disable_log_to_file`).

- `*_alpha_sweep.csv`  
  One summary row per alpha value when `--alpha_sweep` is used (each alpha also gets its own CSV/JSON files).

## Testing

The test suite runs on the CPU in well under a minute, uses small synthetic tensors and data, and downloads nothing:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

It covers IID and Dirichlet non-IID partitioning, the adaptive `tau` / pruning-ratio controller, pruning-group selection and masked, contributor-aware aggregation (including its equivalence with FedAvg when nothing is masked), the energy/time model and the per-client `T_max` cap, the dataset loaders and model factory, and end-to-end runs of every method plus the CLI on tiny synthetic data. CI runs the linter and the tests on Python 3.10, 3.11, and 3.12 with CPU-only PyTorch.

## Paper and citation

This repository accompanies:

> Javad Dogani, Reza Namvar, Masoumeh Khodarahmi, Farshad Khunjush, and Nikolaos Laoutaris. "FLEAT: Energy-Accuracy Trade-Off in Federated Learning Over Heterogeneous Edge Devices." *IEEE Transactions on Mobile Computing*, vol. 25, no. 11, pp. 19797-19811, Nov. 2026. doi: [10.1109/TMC.2026.3710334](https://doi.org/10.1109/TMC.2026.3710334)

If you use this code, please cite the paper:

```bibtex
@article{dogani2026fleat,
  author  = {Dogani, Javad and Namvar, Reza and Khodarahmi, Masoumeh and Khunjush, Farshad and Laoutaris, Nikolaos},
  title   = {{FLEAT}: Energy-Accuracy Trade-Off in Federated Learning Over Heterogeneous Edge Devices},
  journal = {IEEE Transactions on Mobile Computing},
  year    = {2026},
  volume  = {25},
  number  = {11},
  pages   = {19797--19811},
  month   = nov,
  doi     = {10.1109/TMC.2026.3710334}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

## License

Released under the [MIT License](LICENSE).
