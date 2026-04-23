from pathlib import Path

import ast
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from load_config import load_plot_config

FIGURE_DIR = Path(__file__).parents[1] / "figures"
RESULT_DIR = Path(__file__).parents[1] / "results"
DATA_DIR = Path(__file__).parents[1] / "data"
STRATEGY_ORDER = ["OURS", "PFI", "SAGE"]


def _load_sequences() -> dict:
    with (DATA_DIR / "sequences.yaml").open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("sequences.yaml must contain a top-level mapping")

    return payload


def _parse_sequence(value: str) -> tuple[int, ...]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list-form sequence, got {value!r}")
    return tuple(sorted(int(item) for item in parsed))


def _build_strategy_curve(
    df: pd.DataFrame,
    sequence: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for length in range(1, len(sequence) + 1):
        prefix = tuple(sorted(sequence[:length]))
        matching = df[(df["parsed_sequence"] == prefix)]

        if matching.empty:
            raise ValueError(f"No rows found for prefix={list(prefix)}")

        variance = float(matching["test_error"].var())
        rows.append(
            {
                "sequence_length": length,
                "mean": float(matching["test_error"].mean()),
                "variance": 0.0 if pd.isna(variance) else variance,
                "std": 0.0 if pd.isna(variance) else variance**0.5,
            }
        )

    return pd.DataFrame(rows)


def plot_error_evolution(
    dataset_name: str,
    evolution_type: str,
    output_name: str | None = None,
) -> None:
    load_plot_config()

    df = pd.read_csv(RESULT_DIR / "error_evolution.csv")
    df = df[(df["dataset_name"] == dataset_name) & (df["evolution_type"] == evolution_type)].copy()

    if df.empty:
        raise ValueError(f"No rows found for dataset_name={dataset_name}, evolution_type={evolution_type}")

    if dataset_name == "simulation":
        df["test_error"] = df["test_error"] * 100

    df["parsed_sequence"] = df["current_sequence"].apply(_parse_sequence)

    sequences = _load_sequences()
    strategy_sequences = sequences[dataset_name][evolution_type]

    fig, ax = plt.subplots()

    offsets = [0.0, 0.05, 0.1]
    for strategy, offset in zip(STRATEGY_ORDER, offsets):
        curve = _build_strategy_curve(df, strategy_sequences[strategy])
        ax.errorbar(
            curve["sequence_length"] + offset,
            curve["mean"],
            yerr=curve["std"],
            fmt="-o",
            capsize=3,
            elinewidth=1,
            label=strategy,
        )

    x_values = list(range(1, len(next(iter(strategy_sequences.values()))) + 1))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore
    ax.set_xticks(x_values)
    ax.set_xlabel("Wavelength Count" if evolution_type == "wavelength" else "Detector Count")
    ax.set_ylabel("Test MAE (%Saturation)")
    # ax.set_ylim(bottom=0)
    ax.legend()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / (output_name or f"error_evolution_{dataset_name}_{evolution_type}")
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

    for ds_name, evo_type, short_name in combinations:
        plot_error_evolution(
            dataset_name=ds_name,
            evolution_type=evo_type,
            output_name=f"error_evolution_{short_name}",
        )
