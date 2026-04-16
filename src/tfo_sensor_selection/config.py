from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


DATASET_PATHS = {
    "invivo": "data/invivo_data.csv",
    "simulation": "data/sim_data.csv",
}

DatasetName = Literal["invivo", "simulation"]


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetName
    path: Path
    features: list[str]
    labels: list[str]
    grouping_col: str
    detector_distances: list[int]
    wavelength: list[int]
    ignored_group: Any | None = None


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    metadata_path: Path


def get_project_config() -> ProjectConfig:
    root = Path(__file__).resolve().parents[2]
    return ProjectConfig(project_root=root, metadata_path=root / "data" / "dataset_metadata.yaml")


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must parse to dict, got {type(data).__name__}")
    return data


def load_metadata(dataset_name: DatasetName) -> DatasetConfig:
    cfg = get_project_config()
    metadata = _read_metadata(cfg.metadata_path)
    if dataset_name not in metadata:
        valid = ", ".join(sorted(metadata.keys()))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid}")

    entry = metadata[dataset_name]
    if not isinstance(entry, dict):
        raise ValueError(f"Dataset entry '{dataset_name}' must be a dict")

    if dataset_name not in DATASET_PATHS:
        valid = ", ".join(sorted(DATASET_PATHS.keys()))
        raise KeyError(f"No data path configured for '{dataset_name}'. Expected one of: {valid}")

    path = cfg.project_root / DATASET_PATHS[dataset_name]

    features = entry.get("features", [])
    labels = entry.get("labels", [])
    grouping_col = entry.get("grouping_col")
    detector_distances = entry.get("detector_distances", [])
    wavelength = entry.get("wavelength", [])
    ignored_group = entry.get("ignored_group", None)

    if not features or not isinstance(features, list):
        raise ValueError(f"Dataset '{dataset_name}' must define non-empty feature list")
    if not labels or not isinstance(labels, list):
        raise ValueError(f"Dataset '{dataset_name}' must define non-empty label list")
    if not grouping_col or not isinstance(grouping_col, str):
        raise ValueError(f"Dataset '{dataset_name}' must define grouping_col")

    return DatasetConfig(
        name=dataset_name,
        path=path,
        features=list(features),
        labels=list(labels),
        grouping_col=grouping_col,
        detector_distances=[int(v) for v in detector_distances],
        wavelength=[int(v) for v in wavelength],
        ignored_group=ignored_group,
    )
