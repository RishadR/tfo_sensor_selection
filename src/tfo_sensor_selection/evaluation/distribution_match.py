from __future__ import annotations

from typing import Any

import numpy as np


def _kl_divergence_from_hist(x_true: np.ndarray, x_synth: np.ndarray, bins: int = 64) -> float:
    lo = float(min(np.min(x_true), np.min(x_synth)))
    hi = float(max(np.max(x_true), np.max(x_synth)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0

    true_hist, edges = np.histogram(x_true, bins=bins, range=(lo, hi), density=False)
    synth_hist, _ = np.histogram(x_synth, bins=edges, density=False)

    p = true_hist.astype(float)
    q = synth_hist.astype(float)
    p /= max(p.sum(), 1.0)
    q /= max(q.sum(), 1.0)

    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def compute_synthetic_match_metrics(
    x_true_test: np.ndarray,
    x_synth_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except Exception as exc:
        raise ImportError("scipy is required for synthetic match metrics") from exc

    true_x = np.asarray(x_true_test, dtype=float)
    synth_x = np.asarray(x_synth_test, dtype=float)

    if true_x.shape != synth_x.shape:
        raise ValueError(f"Shape mismatch: true={true_x.shape}, synth={synth_x.shape}")
    if true_x.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match number of columns")

    per_feature: dict[str, dict[str, float]] = {}
    ks_values: list[float] = []
    wd_values: list[float] = []
    kl_values: list[float] = []

    true_corr = np.corrcoef(true_x, rowvar=False)
    synth_corr = np.corrcoef(synth_x, rowvar=False)
    true_corr = np.nan_to_num(true_corr, nan=0.0, posinf=0.0, neginf=0.0)
    synth_corr = np.nan_to_num(synth_corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr_abs_diff = np.abs(true_corr - synth_corr)
    np.fill_diagonal(corr_abs_diff, 0.0)
    corr_per_feature = corr_abs_diff.mean(axis=1)
    corr_frobenius = float(np.linalg.norm(corr_abs_diff, ord="fro"))
    corr_mean_abs = float(corr_abs_diff.mean())

    for idx, feature in enumerate(feature_names):
        ks_result = ks_2samp(true_x[:, idx], synth_x[:, idx])
        ks = float(ks_result.statistic)
        ks_pvalue = float(ks_result.pvalue)
        wd = float(wasserstein_distance(true_x[:, idx], synth_x[:, idx]))
        kl = _kl_divergence_from_hist(true_x[:, idx], synth_x[:, idx])
        corr_diff = float(corr_per_feature[idx])

        per_feature[feature] = {
            "ks_stat": ks,
            "ks_pvalue": ks_pvalue,
            "wasserstein": wd,
            "kl_divergence": kl,
            "corr_diff": corr_diff,
        }
        ks_values.append(ks)
        wd_values.append(wd)
        kl_values.append(kl)

    return {
        "per_feature": per_feature,
        "summary": {
            "mean_ks": float(np.mean(ks_values)) if ks_values else 0.0,
            "mean_wasserstein": float(np.mean(wd_values)) if wd_values else 0.0,
            "mean_kl_divergence": float(np.mean(kl_values)) if kl_values else 0.0,
            "mean_corr_diff": corr_mean_abs,
            "corr_frobenius": corr_frobenius,
        },
    }
