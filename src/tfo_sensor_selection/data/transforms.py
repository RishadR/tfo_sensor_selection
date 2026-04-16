from __future__ import annotations

from typing import Protocol

import numpy as np


class BaseTransform(Protocol):
    def fit(self, x: np.ndarray) -> "BaseTransform":
        ...

    def transform(self, x: np.ndarray) -> np.ndarray:
        ...


class IdentityTransform:
    def fit(self, x: np.ndarray) -> "IdentityTransform":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class TransformPipeline:
    def __init__(self, transforms: list[BaseTransform] | None = None) -> None:
        self.transforms = transforms or [IdentityTransform()]

    def fit(self, x: np.ndarray) -> "TransformPipeline":
        z = x
        for transform in self.transforms:
            transform.fit(z)
            z = transform.transform(z)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = x
        for transform in self.transforms:
            z = transform.transform(z)
        return z

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
