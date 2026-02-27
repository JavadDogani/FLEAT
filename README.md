# FLEAT Project

A modular, simulation-first implementation of the FLEAT experimental runner for federated learning benchmarking.  
This refactoring keeps the original **context, execution flow, controller logic, energy/time accounting, aggregation behavior, and CLI semantics**, but splits the code into maintainable modules that are easier to extend, debug, and publish on GitHub.

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

## Project structure

```text
fleat_refactored_project/
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

## Output files

Each run generates:

- `*_round_metrics.csv`  
  Per-round metrics including accuracy, loss, energy, time, tau, and pruning information.

- `*_summary.json`  
  Final paper-style summary metrics.

- `logs/*.log`  
  Full execution log if logging is enabled.
