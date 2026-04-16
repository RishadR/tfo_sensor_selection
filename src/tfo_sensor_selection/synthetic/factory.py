from __future__ import annotations

from pathlib import Path

from tfo_sensor_selection.synthetic.arf_generator import ARFGenerator
from tfo_sensor_selection.synthetic.base import ConditionalSyntheticGenerator
from tfo_sensor_selection.synthetic.sdv_gaussian_copula import SDVGaussianCopulaGenerator


def build_generator(
    generator_name: str,
    seed: int = 42,
    extra_params: dict | None = None,
    model_path: Path | None = None,
) -> ConditionalSyntheticGenerator:
    """
    Factory function to build a synthetic data generator based on the provided name.
    """
    name = generator_name.lower()
    params = extra_params if extra_params is not None else {}

    if model_path is not None and not isinstance(model_path, Path):
        raise TypeError("model_path must be a pathlib.Path or None")

    if name in {"gaussian_copula", "sdv", "sdv_gaussian_copula"}:
        if model_path is not None:
            raise ValueError("model_path is only supported for ARFGenerator")
        return SDVGaussianCopulaGenerator(seed=seed)
    if name in {"arf", "adversarial_random_forest"}:
        return ARFGenerator(seed=seed, arf_params=params, model_path=model_path)
    raise KeyError(f"Unknown generator_name '{generator_name}'")
