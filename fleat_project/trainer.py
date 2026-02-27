from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LocalTrainer:
    @staticmethod
    def train_local_steps(
        model: nn.Module,
        loader: DataLoader,
        steps: int,
        lr: float,
        device: torch.device,
        weight_decay: float,
        optimizer_name: str = "sgd",
        prox_mu: float = 0.0,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[nn.Module, float, float, int]:
        model.train()

        if optimizer_name == "rmsprop":
            opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.99)
        elif optimizer_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()
        it = iter(loader)
        total_loss = 0.0
        done = 0
        import time
        wall_t0 = time.perf_counter()

        for _ in range(int(steps)):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)

            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)

            if prox_mu > 0.0 and global_state is not None:
                prox = 0.0
                for n, p in model.named_parameters():
                    prox = prox + torch.sum((p - global_state[n].to(device)) ** 2)
                loss = loss + 0.5 * prox_mu * prox

            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            done += 1

        wall_t1 = time.perf_counter()
        phi_step_meas = (wall_t1 - wall_t0) / max(1, done)

        avg_loss = total_loss / max(1, done)
        return model, avg_loss, phi_step_meas, done

    @staticmethod
    @torch.no_grad()
    def evaluate_model(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device, max_eval_samples: int = 0):
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss(reduction="sum")

        total = 0
        correct = 0
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            if max_eval_samples > 0 and total >= max_eval_samples:
                break

        if max_eval_samples > 0:
            total = min(total, max_eval_samples)

        return {
            "eval_loss": total_loss / max(1, total),
            "eval_acc": correct / max(1, total),
            "n_eval": total,
        }
