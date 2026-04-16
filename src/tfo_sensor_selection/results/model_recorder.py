from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tfo_sensor_selection.data.transforms import TransformPipeline
from tfo_sensor_selection.models import BaseModel, ModelName

if TYPE_CHECKING:
    from tfo_sensor_selection.training.pipeline import PipelineResult


PERSISTENCE_VERSION = 2
REQUIRED_PAYLOAD_KEYS = {
    "version",
    "timestamp",
    "dataset",
    "model_name",
    "seed",
    "n_val_groups",
    "n_test_groups",
    "best_params",
    "maes",
    "feature_names",
    "label_name",
    "transform",
    "model",
}


@dataclass(frozen=True)
class TrainedModelArtifact:
    dataset: str
    model_name: ModelName
    seed: int
    n_val_groups: int
    n_test_groups: int
    best_params: dict[str, Any]
    maes: dict[str, float]
    feature_names: list[str]
    label_name: str
    transform: TransformPipeline
    model: BaseModel
    timestamp: str
    path: Path | None = None


def default_model_artifact_path(
    result: "PipelineResult",
    output_dir: str | Path = "models/trained",
) -> Path:
    return Path(output_dir) / f"{result.dataset}_{result.model_name}_seed{result.seed}.pkl"


def save_trained_model_artifact(
    result: "PipelineResult",
    output_path: str | Path | None = None,
    output_dir: str | Path = "models/trained",
) -> Path:
    artifact_path = Path(output_path) if output_path is not None else default_model_artifact_path(
        result=result,
        output_dir=output_dir,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": PERSISTENCE_VERSION,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "dataset": str(result.dataset),
        "model_name": str(result.model_name),
        "seed": int(result.seed),
        "n_val_groups": int(result.n_val_groups),
        "n_test_groups": int(result.n_test_groups),
        "best_params": dict(result.best_params),
        "maes": {
            "train": float(result.maes["train"]),
            "val": float(result.maes["val"]),
            "test": float(result.maes["test"]),
        },
        "feature_names": list(result.feature_names),
        "label_name": str(result.label_name),
        "transform": result.transform,
        "model": result.model,
    }

    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    return artifact_path


def load_trained_model_artifact(path: str | Path) -> TrainedModelArtifact:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Trained model artifact does not exist: {artifact_path}")

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Trained model artifact payload must be a dict")

    missing_keys = sorted(REQUIRED_PAYLOAD_KEYS - set(payload.keys()))
    if missing_keys:
        raise KeyError(f"Trained model artifact payload is missing keys: {missing_keys}")

    version = payload["version"]
    if version != PERSISTENCE_VERSION:
        raise ValueError(
            f"Unsupported trained model artifact version {version}. "
            f"Expected {PERSISTENCE_VERSION}."
        )

    raw_maes = payload["maes"]
    if not isinstance(raw_maes, dict):
        raise ValueError("Trained model artifact payload field 'maes' must be a dict")
    maes = {
        "train": float(raw_maes["train"]),
        "val": float(raw_maes["val"]),
        "test": float(raw_maes["test"]),
    }

    return TrainedModelArtifact(
        dataset=str(payload["dataset"]),
        model_name=str(payload["model_name"]),  # type: ignore
        seed=int(payload["seed"]),
        n_val_groups=int(payload["n_val_groups"]),
        n_test_groups=int(payload["n_test_groups"]),
        best_params=dict(payload["best_params"]),
        maes=maes,
        feature_names=[str(name) for name in payload["feature_names"]],
        label_name=str(payload["label_name"]),
        transform=payload["transform"],
        model=payload["model"],
        timestamp=str(payload["timestamp"]),
        path=artifact_path,
    )


def list_trained_model_artifacts(
    model_name: ModelName,
    dataset_name: str | None = None,
    artifacts_dir: str | Path = "models/trained",
) -> list[Path]:
    base_dir = Path(artifacts_dir)
    if not base_dir.exists():
        return []

    pattern = f"*_{model_name}_seed*.pkl" if dataset_name is None else f"{dataset_name}_{model_name}_seed*.pkl"
    return sorted(base_dir.glob(pattern))


__all__ = [
    "TrainedModelArtifact",
    "default_model_artifact_path",
    "list_trained_model_artifacts",
    "load_trained_model_artifact",
    "save_trained_model_artifact",
]