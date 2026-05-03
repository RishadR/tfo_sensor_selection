from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from load_config import load_plot_config

FIGURE_DIR = Path(__file__).parents[1] / "figures"
RESULT_DIR = Path(__file__).parents[1] / "results"
METHOD_LABELS = {"OURS": "ROWS"}


def _load_error_evolution_experiments() -> dict[str, dict]:
    result_path = RESULT_DIR / "error_evolution.yaml"
    with result_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("error_evolution.yaml must contain a top-level mapping")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("error_evolution.yaml must contain an 'experiments' list")

    return {
        str(experiment["experiment_id"]): experiment
        for experiment in experiments
    }


def _load_mi_evolution_experiments() -> dict[str, dict]:
    result_path = RESULT_DIR / "mi_evolution.yaml"
    with result_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("mi_evolution.yaml must contain a top-level mapping")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("mi_evolution.yaml must contain an 'experiments' list")

    return {
        str(experiment["experiment_id"]): experiment
        for experiment in experiments
    }


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = pd.to_numeric(out["candidate_group_value"], errors="coerce")
    if numeric.notna().all():
        out["_sort_key"] = numeric
    else:
        out["_sort_key"] = out["candidate_group_value"].astype(str)
    out = out.sort_values("_sort_key").drop(columns=["_sort_key"])
    return out


def _candidate_color_map(result_df: pd.DataFrame) -> dict[str, str]:
    ordered = _sort_candidates(result_df[["candidate_group_value"]].drop_duplicates())
    candidates = ordered["candidate_group_value"].astype(str).tolist()

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle_colors:
        cycle_colors = [f"C{i}" for i in range(max(len(candidates), 1))]

    return {candidate: cycle_colors[idx % len(cycle_colors)] for idx, candidate in enumerate(candidates)}


def _plot_grouping_rfi_all_rounds(
    grouping_type: str,
    dataset_name: Literal["invivo", "simulation"],
    seed: int = 42,
) -> None:    
    result_filename = (
        "invivo_rfi_greedy_results.csv"
        if dataset_name == "invivo"
        else "simulation_rfi_greedy_results.csv"
    )
    result_df = pd.read_csv(RESULT_DIR / result_filename)
    result_df = result_df[(result_df["dataset"] == dataset_name) & (result_df["seed"] == seed)]
    result_df = result_df[result_df["grouping_type"] == grouping_type]

    if result_df.empty:
        raise ValueError(
            f"No {dataset_name} rows found for grouping_type={grouping_type}, seed={seed}"
        )

    rfi_col = "rfi_mean"
    var_col = "rfi_variance"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rounds = sorted(result_df["round_index"].unique())[:-1]
    if not rounds:
        raise ValueError(
            f"Not enough rounds to plot after excluding the final stage for {dataset_name}, grouping_type={grouping_type}, seed={seed}"
        )
    candidate_to_color = _candidate_color_map(result_df)
    grouping_axis_label = "Wavelength (nm)" if grouping_type == "wavelength" else "Detector Distance (mm)"

    # Make the width based on the number of rounds x 2.5
    fig, ax = plt.subplots(figsize=(len(rounds) * 2.5, 4))
    ax.grid(False)
    
    x_positions: list[float] = []
    x_labels: list[str] = []
    round_centers: list[float] = []
    round_selected_groups: list[str] = []
    round_conditioned_groups: list[str] = []
    cursor = 0.0
    stage_gap = 0.6
    trailing_stage_gap = 0.25

    for idx, round_index in enumerate(rounds):
        round_df = result_df[result_df["round_index"] == round_index]
        round_df = _sort_candidates(round_df)

        labels = round_df["candidate_group_value"].astype(str).tolist()
        y = round_df[rfi_col].to_numpy(dtype=float)
        is_negative = y < 0
        yerr = round_df[var_col].to_numpy(dtype=float)
        round_max = float(y.max())
        denom = round_max if round_max > 0 else 1.0
        y = y / denom
        y = y.clip(min=0.01)  # Avoid bars disappearing due to very small/negative values
        yerr = yerr / denom
        # Kill error bar if the RFI itself is negative - its meaningless
        yerr[is_negative] = 0.0
        n_bars = len(labels)
        bar_x = [cursor + i for i in range(n_bars)]
        bar_colors = [candidate_to_color[label] for label in labels]

        ax.bar(
            bar_x,
            y,
            color=bar_colors,
            alpha=0.9,
        )
        ax.errorbar(
            bar_x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            capsize=4,
            zorder=3,
        )

        x_positions.extend(bar_x)
        x_labels.extend(labels)
        round_centers.append(cursor + (n_bars - 1) / 2)
        if "selected_group_value" in round_df.columns:
            selected_values = (
                round_df["selected_group_value"].dropna().astype(str).drop_duplicates().tolist()
            )
            if len(selected_values) == 1:
                selected_label = selected_values[0]
            elif len(selected_values) > 1:
                selected_label = ", ".join(selected_values)
            else:
                selected_label = ""
        else:
            selected_label = ""
        round_selected_groups.append(selected_label)

        if "conditioned_group_values" in round_df.columns:
            conditioned_values = (
                round_df["conditioned_group_values"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            conditioned_values = conditioned_values[conditioned_values != ""].drop_duplicates().tolist()
            conditioned_label = (
                conditioned_values[0].replace("|", ", ")
                if conditioned_values
                else "None"
            )
        else:
            conditioned_label = "None"
        round_conditioned_groups.append(conditioned_label)

        cursor += n_bars
        if idx < len(rounds) - 1:
            separator_x = cursor + (stage_gap / 2) - 0.5
            # Put the divider in the middle of the stage gap for clearer grouping.
            ax.axvline(
                separator_x,
                color="#000000",
                alpha=1.0,
            )
            ax.text(
                separator_x - 0.12,
                0.5,
                f"Conditioned on: {conditioned_label}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="center",
                va="center",
                fontsize=9,
                color="#222222",
            )
            cursor += stage_gap

    if round_conditioned_groups:
        last_conditioned_label = round_conditioned_groups[-1]
        trailing_separator_x = cursor + (trailing_stage_gap / 2) - 0.5
        ax.text(
            trailing_separator_x - 0.02,
            0.5,
            f"Conditioned on: {last_conditioned_label}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            color="#222222",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Normalized RFI")
    if x_positions:
        ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5 + trailing_stage_gap)
    ax.margins(x=0)

    if x_positions:
        group_center = (min(x_positions) + max(x_positions)) / 2
        ax.text(group_center, -0.16, grouping_axis_label, transform=ax.get_xaxis_transform(), ha="center", va="top")

    last_stage = len(round_centers)
    for idx, (center, selected_group) in enumerate(
        zip(round_centers, round_selected_groups), start=1
    ):
        selected_text = f"(Selected : {selected_group})"
        ax.text(
            center,
            1.02,
            f"Stage {idx}\n{selected_text}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            linespacing=1.1,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.84)

    stem = FIGURE_DIR / f"{dataset_name}_rfi_{grouping_type}"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)


def plot_invivo_wavelength_rfi(seed: int = 42) -> None:
    _plot_grouping_rfi_all_rounds(grouping_type="wavelength", dataset_name="invivo", seed=seed)


def plot_invivo_detector_distance_rfi(seed: int = 42) -> None:
    _plot_grouping_rfi_all_rounds(grouping_type="detector_distance", dataset_name="invivo", seed=seed)


def plot_simulation_wavelength_rfi(seed: int = 42) -> None:
    _plot_grouping_rfi_all_rounds(grouping_type="wavelength", dataset_name="simulation", seed=seed)


def plot_simulation_detector_distance_rfi(seed: int = 42) -> None:
    _plot_grouping_rfi_all_rounds(grouping_type="detector_distance", dataset_name="simulation", seed=seed)


def plot_error_evolution_test_mae(
    experiment_ids: list[str],
    plot_labels: list[str],
    output_name: str = "error_evolution_test_mae",
) -> None:
    if len(experiment_ids) != len(plot_labels):
        raise ValueError("experiment_ids and plot_labels must have the same length")
    if len(experiment_ids) == 0:
        raise ValueError("experiment_ids must contain at least one value")

    experiments_by_id = _load_error_evolution_experiments()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    for experiment_id, plot_label in zip(experiment_ids, plot_labels):
        if experiment_id not in experiments_by_id:
            raise ValueError(f"Unknown experiment_id: {experiment_id}")

        experiment = experiments_by_id[experiment_id]
        steps = experiment.get("steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError(f"Experiment {experiment_id} does not contain any steps")

        x = [int(step["step_index"]) for step in steps]
        y = [float(step["mae_mean"]["test"]) for step in steps]
        yerr = [float(np.sqrt(step["mae_variance"]["test"])) for step in steps]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            capsize=4,
            label=plot_label,
        )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sensor Groups Count")
    ax.set_ylabel("Test MAE")
    ax.legend()
    ax.set_xticks(sorted({tick for experiment_id in experiment_ids for tick in [int(step["step_index"]) for step in experiments_by_id[experiment_id]["steps"]]}))

    fig.tight_layout()
    stem = FIGURE_DIR / output_name
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)


def plot_mi_evolution(
    dataset_name: Literal["invivo", "simulation"],
    evolution_type: Literal["wavelength", "detector_distance"],
    output_name: str,
) -> None:
    experiments_by_id = _load_mi_evolution_experiments()
    matching_experiments = [
        experiment
        for experiment in experiments_by_id.values()
        if experiment.get("dataset") == dataset_name
        and experiment.get("evolution_type") == evolution_type
    ]
    if not matching_experiments:
        raise ValueError(
            f"No MI evolution experiments found for dataset={dataset_name}, evolution_type={evolution_type}"
        )

    method_order = {"SAGE": 0, "PFI": 1, "OURS": 2}
    matching_experiments = sorted(
        matching_experiments,
        key=lambda experiment: (
            method_order.get(str(experiment.get("method")), 999),
            str(experiment.get("method")),
        ),
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(False)

    all_ticks: set[int] = set()

    for experiment in matching_experiments:
        steps = experiment.get("steps", [])
        method = str(experiment.get("method", "unknown"))
        display_method = METHOD_LABELS.get(method, method)
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError(
                f"Experiment for dataset={dataset_name}, evolution_type={evolution_type}, method={method} does not contain any steps"
            )

        x = np.array([int(step["step_index"]) for step in steps], dtype=int)
        y = np.array([float(step["mi"]) for step in steps], dtype=float)
        ci_lower = np.array(
            [float(step["confidence_interval_95_lower"]) for step in steps],
            dtype=float,
        )
        ci_upper = np.array(
            [float(step["confidence_interval_95_upper"]) for step in steps],
            dtype=float,
        )

        line = ax.plot(x, y, marker="o", label=display_method)[0]
        ax.fill_between(
            x,
            y + ci_lower,
            y + ci_upper,
            color=line.get_color(),
            alpha=0.2,
        )
        all_ticks.update(int(value) for value in x)

    ax.set_xlabel("Sensor Groups Count")
    ax.set_ylabel("Mutual Information")
    ax.legend()
    ax.set_xticks(sorted(all_ticks))

    fig.tight_layout()
    stem = FIGURE_DIR / output_name
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)


def plot_invivo_wavelength_mi() -> None:
    plot_mi_evolution(
        dataset_name="invivo",
        evolution_type="wavelength",
        output_name="invivo_mi_wavelength",
    )


def plot_invivo_detector_distance_mi() -> None:
    plot_mi_evolution(
        dataset_name="invivo",
        evolution_type="detector_distance",
        output_name="invivo_mi_detector_distance",
    )


def plot_simulation_wavelength_mi() -> None:
    plot_mi_evolution(
        dataset_name="simulation",
        evolution_type="wavelength",
        output_name="simulation_mi_wavelength",
    )


def plot_simulation_detector_distance_mi() -> None:
    plot_mi_evolution(
        dataset_name="simulation",
        evolution_type="detector_distance",
        output_name="simulation_mi_detector_distance",
    )


if __name__ == "__main__":
    load_plot_config()
    # plot_invivo_wavelength_rfi(seed=0)
    # plot_invivo_detector_distance_rfi(seed=0)
    # plot_simulation_wavelength_rfi(seed=0)
    # plot_simulation_detector_distance_rfi(seed=0)
    # plot_error_evolution_test_mae(['error_evolution_0004'], ['ROWS'])
    plot_invivo_wavelength_mi()
    plot_invivo_detector_distance_mi()
    plot_simulation_wavelength_mi()
    plot_simulation_detector_distance_mi()
