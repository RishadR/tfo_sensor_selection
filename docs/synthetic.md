# Conditional Synthetic Generation

This module generates synthetic values for a selected subset of columns (`target_column_names`) while preserving all non-target columns exactly as they appear in the input (`source_df`).

The workflow is:

1. Choose a backend generator (`arf` or `sdv` Gaussian copula).
2. Fit that generator on training data, using only conditional + target columns.
3. For each row in `source_df`, generate target values conditioned on the row's conditional values.
4. Return a completed DataFrame where untouched columns remain unchanged.

Primary entry points are exported in `tfo_sensor_selection.synthetic`:

- `build_generator(...)`
- `generate_missing_features(...)`
- `ConditionalSyntheticGenerator` (interface)

## Quick Start

```python
from tfo_sensor_selection.synthetic import build_generator, generate_missing_features

generator = build_generator(generator_name="arf", seed=42)

completed_df = generate_missing_features(
    generator=generator,
    train_df=train_df,
    source_df=test_df.copy(),
    all_column_names=["fSaO2", "ratio_1", "ratio_2", "ratio_3", "ratio_4"],
    conditional_column_names=["fSaO2", "ratio_1", "ratio_2"],
    target_column_names=["ratio_3", "ratio_4"],
)
```

Result:

- `ratio_3` and `ratio_4` are replaced with synthetic values.
- All other columns in `all_column_names` stay untouched.

## Interface Overview

### `ConditionalSyntheticGenerator` (`src/tfo_sensor_selection/synthetic/base.py`)

Each backend implements:

- `fit(train_df, conditional_column_names, target_column_names, all_column_names)`
	- Trains the model on conditional + target columns.
	- Ignores unaffected columns.
- `generate_conditional(conditional_df, conditional_column_names, target_column_names)`
	- Produces synthetic values for target columns only, conditioned on `conditional_df`.
- `name()`
	- Returns a backend identifier (`"arf"`, `"gaussian_copula"`, etc.).

The contract is intentionally narrow so different synthesis engines can be swapped without changing experiment code.

### Factory: `build_generator(...)` (`src/tfo_sensor_selection/synthetic/factory.py`)

```python
generator = build_generator(generator_name="arf", seed=42)
```

Supported aliases:

- ARF backend: `"arf"`, `"adversarial_random_forest"`
- SDV Gaussian Copula backend: `"gaussian_copula"`, `"sdv"`, `"sdv_gaussian_copula"`

Unknown names raise `KeyError`.

### Orchestration: `generate_missing_features(...)` (`src/tfo_sensor_selection/synthetic/conditional.py`)

This is the high-level API most callers should use.

Behavior:

1. Validates column roles.
2. Fits the provided generator on `train_df`.
3. Calls `generator.generate_conditional(...)` using conditional columns from `source_df`.
4. Replaces only target columns in a copy of `source_df`.
5. Returns full output in `all_column_names` order.

Role rules:

- `conditional_column_names` and `target_column_names` must be disjoint.
- All requested columns must exist in `all_column_names`.
- `source_df` must include all `all_column_names`.
- If `target_column_names` is empty, the function returns a copy with no fitting/sampling.

## Backend Implementations

### 1) ARF backend (`src/tfo_sensor_selection/synthetic/arf_generator.py`)

Class: `ARFGenerator`

### How it fits

On `fit(...)`:

- Builds model columns as:
	- all conditional columns, then
	- target columns not already present.
- Trains `arfpy` adversarial random forest on that reduced table.
- Runs `forde()` and stores the resulting per-leaf distribution statistics.

### How conditional sampling works

On `generate_conditional(...)`:

1. Converts FORDE output to a per-leaf wide format (robust to schema variants in `arfpy`).
2. Gets per-leaf coverage weights.
3. For each input row:
   - computes log-likelihood of the row's conditional values under each leaf's truncated normal stats,
   - combines with coverage prior,
   - normalizes to posterior probabilities over leaves.
4. Samples one leaf index per row.
5. For each target column, samples from that leaf's truncated normal:
   - mean/std/min/max read from FORDE stats,
   - clipping std to avoid numerical collapse.

So ARF is effectively doing latent leaf assignment conditioned on known features, then drawing target values from leaf-level parametric distributions.

### Notes and constraints

- Assumes numeric conditional/target columns (`to_numpy(dtype=float)` is used).
- Handles multiple possible FORDE column names (`sd` vs `std`, `coverage` aliases, etc.).
- Raises explicit `KeyError` if required feature stats are missing from FORDE output.
- Uses a seeded NumPy RNG for reproducibility.

### 2) SDV Gaussian Copula backend (`src/tfo_sensor_selection/sdv_gaussian_copula.py`)

Class: `SDVGaussianCopulaGenerator`

### How it fits

On `fit(...)`:

- Dynamically imports SDV modules.
- Builds training table from conditional + target columns.
- Infers metadata with `Metadata.detect_from_dataframe(...)`.
- Trains `GaussianCopulaSynthesizer` on that table.

### How conditional sampling works

On `generate_conditional(...)`:

1. Builds one SDV `Condition` per input row using provided conditional columns.
2. Calls `sample_from_conditions(...)`.
3. Returns only requested target columns and aligns output index to the input `conditional_df` index.

### Notes and constraints

- Supports SDV's native conditional sampling API.
- Per-row condition objects can be slower for very large batches.
- Also expects conditional/target columns to exist and be compatible with SDV detection/sampling.

## Data Flow and Column Semantics

Given:

- `all_column_names = [A, B, C, D, E]`
- `conditional_column_names = [A, B]`
- `target_column_names = [D, E]`

Then:

- `C` is unaffected and passed through unchanged.
- `A`, `B` are used as conditioning inputs.
- `D`, `E` are synthesized and overwritten in output.
- Returned frame is ordered exactly as `all_column_names`.

## Typical Calling Pattern in This Repo

The same pattern is used in experiment scripts (for example `experiments/try_synthetic_gen.py` and greedy RFI flow):

```python
generator = build_generator(generator_name="arf", seed=seed)

completed_df = generate_missing_features(
    generator=generator,
    train_df=train_df,
    source_df=split_df[source_columns].copy(),
    all_column_names=source_columns,
    conditional_column_names=conditioned_feature_names,
    target_column_names=target_feature_names,
)
```

In your use case, this gives synthetic replacements for the "missing"/target feature subset while preserving already-known columns.

## Error Handling Summary

Common exceptions raised by the interface:

- `KeyError`: unknown requested columns, missing input columns, missing generated target columns.
- `ValueError`: overlap between conditional and target columns.
- `RuntimeError`: attempting sampling before `fit`, or invalid ARF FORDE output.

## Installation Notes

Core package dependencies are in the base install, but synthetic backends are optional extras.

Install with synthetic extras:

```bash
pip install -e .[synthetic]
```

This brings in:

- `sdv`
- `arfpy`

## Practical Guidance

- Use `arf` when you want leaf-wise conditional generation with explicit per-feature truncated bounds from FORDE.
- Use `sdv` Gaussian copula when you want a mature tabular synthesizer with native condition objects.
- Keep conditional and target sets disjoint.
- Include label in conditional columns only if you explicitly want label-conditioned generation.
- Reuse `generate_missing_features(...)` as the stable API; swap backends through `build_generator(...)`.
