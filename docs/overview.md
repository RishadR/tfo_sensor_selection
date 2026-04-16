# Pipeline Overview

The pipeline is modular and function-first so it can be orchestrated by scripts or agents.

## Components

- Metadata config: [data/dataset_metadata.yaml](data/dataset_metadata.yaml) defines dataset-specific features, labels, and grouping column.
- Data I/O: loaders read invivo CSV or simulation PKL and validate required columns.
- Holdout splitter: group-aware split by `grouping_col` into train/val/test (no random row-level leakage).
- Transform stage: pluggable preprocessing pipeline, fit on train and reused for val/test/synthetic inputs.
- Model layer: interchangeable backends (gradient boosting, random forest, 3-layer neural net).
- HPO stage: Optuna tunes on validation MAE, then best params are retrained on train+val.
- Evaluation stage: reports MAE on train/val/test and synthetic-impact deltas.
- Synthetic match stage: quantifies synthetic-vs-true similarity (KS, Wasserstein) separately from model MAE.
- Result recorders:
	- [results/results.csv](results/results.csv): model performance metrics.
	- [results/synthetic_match_metrics.csv](results/synthetic_match_metrics.csv): synthetic quality metrics.
