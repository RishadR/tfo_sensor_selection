from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_METRIC_COLUMNS = [
    "timestamp",
    "dataset",
    "model",
    "seed",
    "n_val_groups",
    "n_test_groups",
    "best_params",
    "train_mae",
    "val_mae",
    "test_mae",
]


def record_model_metrics(result: dict[str, Any], output_path: str | Path = "results/results.csv") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "dataset": result["dataset"],
        "model": result["model"],
        "seed": result["seed"],
        "n_val_groups": result["n_val_groups"],
        "n_test_groups": result["n_test_groups"],
        "best_params": json.dumps(result["best_params"], sort_keys=True),
        "train_mae": result["maes"]["train"],
        "val_mae": result["maes"]["val"],
        "test_mae": result["maes"]["test"],
    }

    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODEL_METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
