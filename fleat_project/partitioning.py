from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class PartitionManager:
    @staticmethod
    def extract_targets(dataset: Dataset) -> np.ndarray:
        if hasattr(dataset, "targets"):
            t = getattr(dataset, "targets")
            if isinstance(t, list):
                return np.array(t)
            if torch.is_tensor(t):
                return t.cpu().numpy()
            return np.array(t)

        if hasattr(dataset, "labels"):
            t = getattr(dataset, "labels")
            if torch.is_tensor(t):
                return t.cpu().numpy()
            return np.array(t)

        ys = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            ys.append(int(y))
        return np.array(ys)

    @staticmethod
    def partition_iid(n: int, num_clients: int, seed: int = 42) -> List[np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        return [arr.astype(int) for arr in np.array_split(idx, num_clients)]

    @staticmethod
    def partition_dirichlet_label_skew(
        dataset: Dataset,
        num_clients: int,
        alpha: float,
        seed: int = 42,
        min_size: int = 10,
    ) -> List[np.ndarray]:
        y = PartitionManager.extract_targets(dataset)
        n_classes = int(np.max(y)) + 1
        rng = np.random.default_rng(seed)
        class_indices = [np.where(y == c)[0] for c in range(n_classes)]
        for c in range(n_classes):
            rng.shuffle(class_indices[c])

        while True:
            client_bins = [[] for _ in range(num_clients)]
            for c in range(n_classes):
                idx_c = class_indices[c]
                if len(idx_c) == 0:
                    continue
                proportions = rng.dirichlet([alpha] * num_clients)
                cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                splits = np.split(idx_c, cuts)
                for k, part in enumerate(splits):
                    client_bins[k].extend(part.tolist())
            sizes = [len(b) for b in client_bins]
            if min(sizes) >= min_size:
                break

        for k in range(num_clients):
            rng.shuffle(client_bins[k])
        return [np.array(b, dtype=int) for b in client_bins]

    @staticmethod
    def make_client_subsets(train_dataset: Dataset, args) -> List[Subset]:
        if args.partition == "iid":
            idx_parts = PartitionManager.partition_iid(len(train_dataset), args.num_clients, seed=args.seed)
        else:
            idx_parts = PartitionManager.partition_dirichlet_label_skew(
                train_dataset,
                args.num_clients,
                args.dirichlet_alpha,
                seed=args.seed,
                min_size=args.min_client_samples,
            )
        return [Subset(train_dataset, ids.tolist()) for ids in idx_parts]
