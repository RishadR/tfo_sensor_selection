"""
Modular computational time benchmarks for the invivo pipeline steps.
Reports mean ± std (seconds) for: ARF fit, MLP training, _synthesize_one.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from invivo_arf_fit import fit_and_save_invivo_arf
from invivo_model_training import run_experiment

from tfo_sensor_selection.feature_selection.rfi_greedy import (
    _synthesize_one,
    build_wavelength_groups,
)
from tfo_sensor_selection.results import load_pretrained_seed_contexts
from tfo_sensor_selection.synthetic.factory import build_generator
from tfo_sensor_selection.training.trainer import compute_absolute_errors

OUTPUT_PATH = Path(__file__).parent.parent / "results" / "computational_time.txt"
N_RUNS = 20
_MLP_SEED = 42  # not in {0, 1, 2}
_ARF_PARAMS = {
    "num_trees": 50,
    "delta": 0.0,
    "max_iters": 15,
    "min_node_size": 2,
    "early_stop": True,
    "verbose": True,
}


def _time_arf_fit() -> tuple[float, float]:
    times = []
    for _ in range(N_RUNS):
        tmp = Path(tempfile.mktemp(suffix=".pkl"))
        t0 = time.perf_counter()
        fit_and_save_invivo_arf(output_path=tmp, seed=42, arf_params=_ARF_PARAMS)
        times.append(time.perf_counter() - t0)
        tmp.unlink(missing_ok=True)
    return float(np.mean(times)), float(np.std(times))


def _time_mlp_training() -> tuple[float, float]:
    times = []
    model_path = Path(f"models/trained/invivo_mlp_seed{_MLP_SEED}.pkl")
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        run_experiment(model_name="mlp", seed=_MLP_SEED, record_results=False)
        times.append(time.perf_counter() - t0)
        model_path.unlink(missing_ok=True)
    return float(np.mean(times)), float(np.std(times))


def _time_synthesize_one() -> tuple[float, float]:
    # Setup (not timed)
    seed_contexts = load_pretrained_seed_contexts("invivo", "mlp", seed_values=[0])
    ctx = seed_contexts[0]
    label_name = ctx.label_name
    feature_names = ctx.feature_names
    source_columns = [label_name] + feature_names
    wavelength_groups = build_wavelength_groups(ctx.config)
    target_feature_names = list(wavelength_groups[sorted(wavelength_groups)[0]])
    baseline_absolute_errors = compute_absolute_errors(ctx.model, ctx.x_test, ctx.y_test)

    generator = build_generator("arf", seed=0, extra_params=_ARF_PARAMS, model_path=Path("models/invivo_arf.pkl"))
    run_contexts = [
        {
            "model_context": ctx,
            "split_df": ctx.test_df,
            "x_split": ctx.x_test,
            "y_split": ctx.y_test,
            "baseline_split_mae": float(ctx.maes["test"]),
            "baseline_absolute_errors": baseline_absolute_errors,
            "generator": generator,
        }
    ]

    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _synthesize_one(
            synth_idx=i,
            run_contexts=run_contexts,
            source_columns=source_columns,
            feature_names=feature_names,
            conditioned_feature_names=[],
            target_feature_names=target_feature_names,
        )
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


if __name__ == "__main__":
    print(f"Running {N_RUNS} iterations each.\n")

    print("Step 1/3: ARF fit...")
    arf_mean, arf_std = _time_arf_fit()
    print(f"  Done. {arf_mean:.3f} ± {arf_std:.3f} s\n")

    print(f"Step 2/3: MLP training (seed={_MLP_SEED})...")
    mlp_mean, mlp_std = _time_mlp_training()
    print(f"  Done. {mlp_mean:.3f} ± {mlp_std:.3f} s\n")

    print("Step 3/3: _synthesize_one...")
    synth_mean, synth_std = _time_synthesize_one()
    print(f"  Done. {synth_mean:.6f} ± {synth_std:.6f} s\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        f.write(f"n_runs: {N_RUNS}\n\n")
        f.write(f"{'Step':<25} {'Mean (s)':>12} {'Std (s)':>12}\n")
        f.write(f"{'-'*25} {'-'*12} {'-'*12}\n")
        f.write(f"{'ARF fit':<25} {arf_mean:>12.3f} {arf_std:>12.3f}\n")
        f.write(f"{'MLP training':<25} {mlp_mean:>12.3f} {mlp_std:>12.3f}\n")
        f.write(f"{'synthesize_one':<25} {synth_mean:>12.6f} {synth_std:>12.6f}\n")
    print(f"Results written to {OUTPUT_PATH}")
