from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def suggest_params(self, trial: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_params(self, **params: Any) -> "BaseModel":
        raise NotImplementedError
