from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from tfo_sensor_selection.config import load_metadata
from tfo_sensor_selection.data.loader import load_dataset
from tfo_sensor_selection.evaluation import compute_synthetic_match_metrics
from tfo_sensor_selection.results import record_synthetic_metrics
from tfo_sensor_selection.synthetic.arf_generator import ARFGenerator


def fit_and_save_simulation_arf(
    output_path: Path = Path("models/simulation_arf.pkl"),
    seed: int = 42,
    *,
    arf_params: dict[str, Any],
) -> None:
    cfg = load_metadata("simulation")
    df = load_dataset(cfg)

    raw_fit_df = df[cfg.features].copy()
    fit_df = raw_fit_df.to_numpy()
    scaler = StandardScaler()
    fit_df = pd.DataFrame(scaler.fit_transform(fit_df), columns=cfg.features)

    start = perf_counter()
    generator = ARFGenerator(seed=seed, arf_params=arf_params)
    generator.fit(train_df=fit_df, fit_columns=cfg.features)
    fit_seconds = perf_counter() - start

    scaled_synth_df = generator.generate_conditional(
        conditional_df=pd.DataFrame(index=fit_df.index),
        conditional_column_names=[],
        target_column_names=cfg.features,
    )
    synth_df = pd.DataFrame(
        scaler.inverse_transform(scaled_synth_df[cfg.features].to_numpy()),
        columns=cfg.features,
        index=fit_df.index,
    )
    synthetic_metrics = compute_synthetic_match_metrics(
        x_true_test=raw_fit_df.to_numpy(),
        x_synth_test=synth_df.to_numpy(),
        feature_names=cfg.features,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_model(output_path)
    record_synthetic_metrics(
        dataset=cfg.name,
        model=output_path.stem,
        case_name=f"{cfg.name}_arf_seed_{seed}",
        metrics=synthetic_metrics,
        generator_name=generator.name(),
        case_mode="unconditional",
        conditioning_cols=[],
        generated_cols=cfg.features,
    )

    print(f"Dataset: {cfg.name}")
    print(f"Rows: {len(fit_df)}")
    print(f"Features: {len(cfg.features)}")
    print(f"Fit time (s): {fit_seconds:.3f}")
    print(f"Saved model: {output_path}")
    print(f"ARF params: {arf_params}")
    print(f"Mean KS: {synthetic_metrics['summary']['mean_ks']:.4f}")
    print(f"Mean Wasserstein: {synthetic_metrics['summary']['mean_wasserstein']:.4f}")
    print(f"Mean KL divergence: {synthetic_metrics['summary']['mean_kl_divergence']:.4f}")
    print(f"Mean correlation difference: {synthetic_metrics['summary']['mean_corr_diff']:.4f}")


if __name__ == "__main__":
    fit_and_save_simulation_arf(
        output_path=Path("models/simulation_arf.pkl"),
        seed=42,
        arf_params={
            "num_trees": 50,
            "delta": 0.0,
            "max_iters": 15,
            "min_node_size": 2,
            "early_stop": True,
            "verbose": True,
        },
    )
