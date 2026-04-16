from pathlib import Path

import ast
import matplotlib.pyplot as plt
import pandas as pd

from load_config import load_plot_config

FIGURE_DIR = Path(__file__).parents[1] / "figures"
RESULT_DIR = Path(__file__).parents[1] / "results"
STRATEGY_ORDER = ["OURS", "PFI", "SAGE"]


def plot_error_evolution_alt(
    dataset_name: str = "simulation",
    evolution_type: str = "wavelength",
    output_name: str | None = None,
) -> None:
    load_plot_config()

    df = pd.read_csv(RESULT_DIR / "error_evolution_alt.csv")
    df = df[(df["dataset_name"] == dataset_name) & (df["evolution_type"] == evolution_type)]

    if dataset_name == "simulation":
        df["test_error"] = df["test_error"] * 100

    df["sequence_length"] = df["current_sequence"].apply(lambda s: len(ast.literal_eval(s)))

    grouped = (
        df.groupby(["selection_strategy", "sequence_length"])["test_error"]
        .agg(mean="mean", std="std", var="var")
        .reset_index()
    )

    fig, ax = plt.subplots()

    strategies_present = [s for s in STRATEGY_ORDER if s in grouped["selection_strategy"].unique()]
    x_values = sorted(grouped["sequence_length"].unique())
    offset_step = 0.08
    strategy_offsets = {
        strategy: idx * offset_step
        for idx, strategy in enumerate(strategies_present)
    }

    for strategy in strategies_present:
        sub = grouped[grouped["selection_strategy"] == strategy].sort_values("sequence_length")
        x_shifted = sub["sequence_length"] + strategy_offsets[strategy]
        ax.errorbar(
            x_shifted,
            sub["mean"],
            yerr=sub["std"] / 2,
            fmt="-o",
            capsize=3,
            elinewidth=1,
            label=strategy,
        )

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))   # type: ignore
    ax.set_xticks(x_values)
    xlabel = "Wavelength Count" if evolution_type == "wavelength" else "Detector Count"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Error (MAE % Saturation)")
    ax.set_ylim(bottom=0)
    ax.legend()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / (output_name or "error_evolution_alt")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    plt.close(fig)
    print(f"Saved to {stem}.pdf / .svg")


if __name__ == "__main__":
    combinations = [
        ("simulation", "detector_distance", "sim_detector"),
        ("simulation", "wavelength", "sim_wavelength"),
        ("invivo", "detector_distance", "invivo_detector"),
        ("invivo", "wavelength", "invivo_wavelength"),
    ]

    for dataset_name, evolution_type, short_name in combinations:
        plot_error_evolution_alt(
            dataset_name=dataset_name,
            evolution_type=evolution_type,
            output_name=f"error_evolution_alt_{short_name}",
        )
