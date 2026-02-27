from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import torchvision
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except Exception:
    torchvision = None
    T = None
    HAS_TORCHVISION = False

try:
    from sklearn.datasets import load_digits
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class SimpleTransformDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DatasetFactory:
    @staticmethod
    def load_sklearn_digits_as_tensors(test_ratio: float = 0.2, seed: int = 42):
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn is not installed")
        ds = load_digits()
        X = ds.images.astype(np.float32) / 16.0
        X = X[:, None, :, :]
        y = ds.target.astype(np.int64)

        rng = np.random.default_rng(seed)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        split = int((1 - test_ratio) * len(y))
        tr, te = idx[:split], idx[split:]

        xtr = torch.tensor(X[tr])
        ytr = torch.tensor(y[tr])
        xte = torch.tensor(X[te])
        yte = torch.tensor(y[te])
        return SimpleTransformDataset(xtr, ytr), SimpleTransformDataset(xte, yte), 10, 1

    @staticmethod
    def load_torchvision_dataset(name: str, root: str):
        if not HAS_TORCHVISION:
            raise RuntimeError("torchvision unavailable in this environment")

        name = name.lower()
        tf_gray = T.Compose([T.ToTensor()])
        tf_rgb32 = T.Compose([T.ToTensor()])

        if name == "mnist":
            train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tf_gray)
            test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tf_gray)
            return train, test, 10, 1
        if name in {"fashionmnist", "fmnist"}:
            train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=tf_gray)
            test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=tf_gray)
            return train, test, 10, 1
        if name == "cifar10":
            train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf_rgb32)
            test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf_rgb32)
            return train, test, 10, 3
        if name == "svhn":
            train = torchvision.datasets.SVHN(root=root, split="train", download=True, transform=tf_rgb32)
            test = torchvision.datasets.SVHN(root=root, split="test", download=True, transform=tf_rgb32)
            return train, test, 10, 3

        raise ValueError(f"Unsupported torchvision dataset: {name}")

    @staticmethod
    def load_edge_iiotset_csv(csv_path: str, label_col: Optional[str] = None, test_ratio: float = 0.2, seed: int = 42):
        df = pd.read_csv(csv_path)

        if label_col is None:
            candidates = ["label", "Label", "attack_label", "class", "target", "y"]
            for c in candidates:
                if c in df.columns:
                    label_col = c
                    break

        if label_col is None:
            non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            label_col = non_numeric[-1] if non_numeric else df.columns[-1]

        y_raw = df[label_col]
        X_df = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)

        X = X_df.values.astype(np.float32)
        labels, uniques = pd.factorize(y_raw.astype(str))
        y = labels.astype(np.int64)

        rng = np.random.default_rng(seed)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        split = int((1 - test_ratio) * len(y))
        tr, te = idx[:split], idx[split:]

        xtr = torch.tensor(X[tr])
        ytr = torch.tensor(y[tr])
        xte = torch.tensor(X[te])
        yte = torch.tensor(y[te])
        return SimpleTransformDataset(xtr, ytr), SimpleTransformDataset(xte, yte), len(uniques), X.shape[1]

    @staticmethod
    def load_synthetic_image_dataset(n_train=5000, n_test=1000, n_classes=10, in_ch=1, size=28, seed=42):
        g = torch.Generator().manual_seed(seed)
        xtr = torch.randn(n_train, in_ch, size, size, generator=g)
        ytr = torch.randint(0, n_classes, (n_train,), generator=g)
        xte = torch.randn(n_test, in_ch, size, size, generator=g)
        yte = torch.randint(0, n_classes, (n_test,), generator=g)
        return SimpleTransformDataset(xtr, ytr), SimpleTransformDataset(xte, yte), n_classes, in_ch

    @staticmethod
    def load_from_args(args) -> Tuple[Dataset, Dataset, int, int]:
        name = args.dataset.lower()
        if name == "edge_iiotset":
            if not args.edge_csv_path:
                raise ValueError("--edge_csv_path is required for dataset=edge_iiotset")
            return DatasetFactory.load_edge_iiotset_csv(
                args.edge_csv_path,
                label_col=args.edge_label_col,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        if name == "sklearn_digits":
            return DatasetFactory.load_sklearn_digits_as_tensors(test_ratio=args.test_ratio, seed=args.seed)
        if name == "synthetic":
            return DatasetFactory.load_synthetic_image_dataset(
                n_train=args.synthetic_train,
                n_test=args.synthetic_test,
                in_ch=1,
                size=28,
                seed=args.seed,
            )
        return DatasetFactory.load_torchvision_dataset(name, root=args.data_root)
