from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tfo_sensor_selection.config import DatasetName


SYNTHETIC_COLUMNS = [
    "timestamp",
    "dataset",
    "model",
    "case_name",
    "generator_name",
    "case_mode",
    "conditioning_cols",
    "generated_cols",
    "feature_name_or__aggregate__",
    "ks_stat",
    "ks_pvalue",
    "wasserstein",
    "kl_divergence",
    "corr_diff",
    "corr_frobenius",
]


def record_synthetic_metrics(
    dataset: DatasetName,
    model: str,
    case_name: str,
    metrics: dict[str, Any],
    generator_name: str | None = None,
    case_mode: str | None = None,
    conditioning_cols: list[str] | None = None,
    generated_cols: list[str] | None = None,
    timestamp: str | None = None,
    output_path: str | Path = "results/synthetic_match_metrics.csv",
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_timestamp = timestamp or datetime.now(tz=timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    for feature_name, values in metrics.get("per_feature", {}).items():
        rows.append(
            {
                "timestamp": run_timestamp,
                "dataset": dataset,
                "model": model,
                "case_name": case_name,
                "generator_name": generator_name,
                "case_mode": case_mode,
                "conditioning_cols": "|".join(conditioning_cols or []),
                "generated_cols": "|".join(generated_cols or []),
                "feature_name_or__aggregate__": feature_name,
                "ks_stat": values.get("ks_stat"),
                "ks_pvalue": values.get("ks_pvalue"),
                "wasserstein": values.get("wasserstein"),
                "kl_divergence": values.get("kl_divergence"),
                "corr_diff": values.get("corr_diff"),
                "corr_frobenius": None,
            }
        )

    summary = metrics.get("summary", {})
    rows.append(
        {
            "timestamp": run_timestamp,
            "dataset": dataset,
            "model": model,
            "case_name": case_name,
            "generator_name": generator_name,
            "case_mode": case_mode,
            "conditioning_cols": "|".join(conditioning_cols or []),
            "generated_cols": "|".join(generated_cols or []),
            "feature_name_or__aggregate__": "__aggregate__",
            "ks_stat": summary.get("mean_ks"),
            "ks_pvalue": None,
            "wasserstein": summary.get("mean_wasserstein"),
            "kl_divergence": summary.get("mean_kl_divergence"),
            "corr_diff": summary.get("mean_corr_diff"),
            "corr_frobenius": summary.get("corr_frobenius"),
        }
    )

    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SYNTHETIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
