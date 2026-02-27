from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    HAS_TORCHVISION = True
except Exception:
    torchvision = None
    HAS_TORCHVISION = False


class SmallImageCNN(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)


class EdgeIotCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        side = int(math.ceil(math.sqrt(n_features)))
        self.side = side
        self.n_features = n_features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        dummy = torch.zeros(1, 1, side, side)
        with torch.no_grad():
            z = self.pool(F.relu(self.conv1(dummy)))
            z = self.pool(F.relu(self.conv2(z)))
            flat = z.numel()
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def _pad_and_reshape(self, x):
        b = x.shape[0]
        if x.shape[1] < self.side * self.side:
            pad = self.side * self.side - x.shape[1]
            x = F.pad(x, (0, pad))
        x = x[:, : self.side * self.side]
        return x.view(b, 1, self.side, self.side)

    def forward(self, x):
        if x.dim() == 2:
            x = self._pad_and_reshape(x)
        z = self.pool(F.relu(self.conv1(x)))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.flatten(1)
        z = F.relu(self.fc1(z))
        return self.fc2(z)


class EdgeIotGRU(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden1: int = 64, hidden2: int = 32, step_size: int = 8):
        super().__init__()
        self.n_features = n_features
        self.step_size = step_size
        self.input_size = step_size
        self.seq_len = int(math.ceil(n_features / step_size))
        self.gru1 = nn.GRU(input_size=self.input_size, hidden_size=hidden1, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden2, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def _to_seq(self, x):
        b = x.shape[0]
        total = self.seq_len * self.step_size
        if x.shape[1] < total:
            x = F.pad(x, (0, total - x.shape[1]))
        return x[:, :total].view(b, self.seq_len, self.step_size)

    def forward(self, x):
        if x.dim() == 2:
            x = self._to_seq(x)
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = out[:, -1, :]
        out = self.drop(out)
        out = F.relu(self.fc1(out))
        return self.fc2(out)


class ModelFactory:
    @staticmethod
    def build(model_name: str, num_classes: int, in_channels: int, input_feature_dim: Optional[int] = None) -> nn.Module:
        name = model_name.lower()
        if name in {"cnn_edge", "edge_cnn"}:
            if input_feature_dim is None:
                raise ValueError("edge CNN requires input_feature_dim")
            return EdgeIotCNN(input_feature_dim, num_classes)

        if name in {"rnn_edge", "edge_rnn", "gru_edge"}:
            if input_feature_dim is None:
                raise ValueError("edge RNN requires input_feature_dim")
            return EdgeIotGRU(input_feature_dim, num_classes)

        if HAS_TORCHVISION:
            try:
                if name == "resnet18":
                    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
                    if in_channels == 1:
                        m.conv1 = nn.Conv2d(
                            1, m.conv1.out_channels, kernel_size=m.conv1.kernel_size,
                            stride=m.conv1.stride, padding=m.conv1.padding, bias=False
                        )
                    return m
                if name == "googlenet":
                    m = torchvision.models.googlenet(weights=None, num_classes=num_classes, aux_logits=False)
                    if in_channels == 1:
                        m.conv1.conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    return m
                if name in {"mobilenetv2", "mobilenet_v2"}:
                    m = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
                    if in_channels == 1:
                        first = m.features[0][0]
                        m.features[0][0] = nn.Conv2d(
                            1, first.out_channels, kernel_size=first.kernel_size,
                            stride=first.stride, padding=first.padding, bias=False
                        )
                    return m
                if name in {"efficientnet_b0", "efficientnetb0"}:
                    m = torchvision.models.efficientnet_b0(weights=None, num_classes=num_classes)
                    if in_channels == 1:
                        first = m.features[0][0]
                        m.features[0][0] = nn.Conv2d(
                            1, first.out_channels, kernel_size=first.kernel_size,
                            stride=first.stride, padding=first.padding, bias=False
                        )
                    return m
            except Exception:
                pass

        return SmallImageCNN(in_ch=in_channels, num_classes=num_classes)

    @staticmethod
    def paper_defaults(model_name: str, dataset_name: str):
        model_name = model_name.lower()
        dataset_name = dataset_name.lower()
        if model_name in {"efficientnet_b0", "efficientnetb0"}:
            return {"batch_size": 32, "lr": 1e-3, "tau_init": 1}
        if model_name in {"mobilenetv2", "mobilenet_v2"}:
            return {"batch_size": 64, "lr": 1e-3, "tau_init": 2}
        if model_name == "resnet18":
            return {"batch_size": 64, "lr": 1e-3, "tau_init": 3}
        if model_name == "googlenet":
            return {"batch_size": 32, "lr": 5e-4, "tau_init": 2}
        if model_name in {"cnn_edge", "edge_cnn"}:
            return {"batch_size": 64, "lr": 1e-3, "tau_init": 1}
        if model_name in {"rnn_edge", "edge_rnn", "gru_edge"}:
            return {"batch_size": 32, "lr": 5e-4, "tau_init": 2}
        return {"batch_size": 32, "lr": 1e-3, "tau_init": 2}
