from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tfo_sensor_selection.models.base import BaseModel


@dataclass
class _NNParams:
    hidden_size: int = 64
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100


class MLPModel(BaseModel):
    def __init__(self, seed: int = 42, device: str = "cpu", **kwargs: Any) -> None:
        self.seed = seed
        self.device = torch.device(device)
        self.params = _NNParams(**{k: v for k, v in kwargs.items() if hasattr(_NNParams, k)})
        self._net = None

    def suggest_params(self, trial: Any) -> dict[str, Any]:
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 50, 50),
        }

    def set_params(self, **params: Any) -> "MLPModel":
        merged = self.params.__dict__.copy()
        merged.update(params)
        self.params = _NNParams(**merged)
        self._net = None
        return self

    def _build_net(self, input_dim: int):
        torch.manual_seed(self.seed)
        hidden = self.params.hidden_size
        net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        net = net.to(self.device)
        return net

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPModel":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        net = self._build_net(x.shape[1])
        optimizer = torch.optim.Adam(net.parameters(), lr=self.params.lr)
        loss_fn = nn.L1Loss()

        dataset = TensorDataset(torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device))
        loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True)   


        net.train()
        for _ in range(self.params.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = net(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

        self._net = net
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model has not been fit yet")

        x = np.asarray(x, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(torch.from_numpy(x).to(self.device)).cpu().numpy().reshape(-1)
        return pred.astype(float)
