from typing import Literal

from .base import BaseModel
from .neural import MLPModel
from .tree import GradientBoostingModel, RandomForestModel


ModelName = Literal[
    "random_forest",
    "rf",
    "gradient_boosting",
    "gb",
    "neural_network",
    "mlp",
    "nn",
]


def build_model(model_name: ModelName, seed: int = 42, device: str = "cpu") -> BaseModel:
    name = model_name.lower()
    if name in {"random_forest", "rf"}:
        return RandomForestModel(random_state=seed)
    if name in {"gradient_boosting", "gb"}:
        return GradientBoostingModel(random_state=seed)
    if name in {"neural_network", "mlp", "nn"}:
        return MLPModel(seed=seed, device=device)
    raise KeyError(f"Unknown model_name '{model_name}'")


__all__ = [
    "BaseModel",
    "GradientBoostingModel",
    "MLPModel",
    "ModelName",
    "RandomForestModel",
    "build_model",
]
