"""Plot four per-circuit comparisons from final_comparison.csv."""

import argparse
import csv
import math
from pathlib import Path
import sys


REQUIRED_MODELS = (
    "heuristic",
    "smartatpg_mean",
    "smartatpg_gat_gru",
)
MEAN_MODEL = "smartatpg_mean"
RELATIVE_MODELS = (
    "heuristic",
    "smartatpg_gat_gru",
)
MODEL_COLORS = {
    "heuristic": "#6B7280",
    "smartatpg_mean": "#F59E0B",
    "smartatpg_gat_gru": "#2563EB",
}
METRICS = (
    {
        "column": "backtrace_steps",
        "title": "Relative Backtrace Steps by Circuit",
        "ylabel": "Ratio to smartatpg_mean",
        "filename": "backtrace_steps_by_circuit.png",
        "relative_to_mean": True,
    },
    {
        "column": "backtracks",
        "title": "Relative Backtracks by Circuit",
        "ylabel": "Ratio to smartatpg_mean",
        "filename": "backtracks_by_circuit.png",
        "relative_to_mean": True,
    },
    {
        "column": "atpg_seconds",
        "title": "Relative ATPG Runtime by Circuit",
        "ylabel": "Ratio to smartatpg_mean",
        "filename": "runtime_by_circuit.png",
        "relative_to_mean": True,
    },
    {
        "column": "fault_coverage",
        "title": "Fault Coverage by Circuit",
        "ylabel": "Fault coverage (%)",
        "filename": "fault_coverage_by_circuit.png",
        "relative_to_mean": False,
    },
)
REQUIRED_COLUMNS = {
    "circuit",
    "model",
    *(metric["column"] for metric in METRICS),
}


def read_comparison_csv(csv_path):
    """Read and validate the benchmark's long-form comparison CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    records = {}
    circuits = []
    models = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {csv_path}")
        missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        for line_number, row in enumerate(reader, start=2):
            circuit = (row.get("circuit") or "").strip()
            model = (row.get("model") or "").strip()
            if not circuit or not model:
                raise ValueError(
                    f"Line {line_number} has an empty circuit or model value."
                )
            key = (circuit, model)
            if key in records:
                raise ValueError(
                    f"Duplicate circuit/model pair on line {line_number}: "
                    f"{circuit}/{model}"
                )

            values = {}
            for metric in METRICS:
                column = metric["column"]
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Line {line_number} has a non-numeric {column} value: "
                        f"{row[column]!r}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Line {line_number} has a non-finite {column} value."
                    )
                if value < 0.0:
                    raise ValueError(
                        f"Line {line_number} has a negative {column} value."
                    )
                if column == "fault_coverage" and value > 1.0:
                    raise ValueError(
                        f"Line {line_number} has fault_coverage outside [0, 1]."
                    )
                values[column] = value

            records[key] = values
            if circuit not in circuits:
                circuits.append(circuit)
            if model not in models:
                models.append(model)

    if not records:
        raise ValueError(f"Input CSV contains no data rows: {csv_path}")

    missing_models = [model for model in REQUIRED_MODELS if model not in models]
    if missing_models:
        raise ValueError(
            "Input CSV is missing required models: " + ", ".join(missing_models)
        )

    for circuit in circuits:
        missing_for_circuit = [
            model for model in REQUIRED_MODELS if (circuit, model) not in records
        ]
        if missing_for_circuit:
            raise ValueError(
                f"Circuit {circuit!r} is missing models: "
                + ", ".join(missing_for_circuit)
            )

    model_order = list(REQUIRED_MODELS)
    model_order.extend(
        model for model in models
        if model not in REQUIRED_MODELS
        and all((circuit, model) in records for circuit in circuits)
    )

    return circuits, model_order, records


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required. Install python-requirements.txt first."
        ) from error
    return plt


def models_for_metric(metric):
    """Return the model bars shown for one metric."""
    return RELATIVE_MODELS if metric["relative_to_mean"] else REQUIRED_MODELS


def metric_values(circuits, model, records, metric):
    """Return one model's relative ratios or fault-coverage percentages."""
    column = metric["column"]
    if not metric["relative_to_mean"]:
        return [records[(circuit, model)][column] * 100.0 for circuit in circuits]

    values = []
    for circuit in circuits:
        denominator = records[(circuit, MEAN_MODEL)][column]
        if denominator == 0.0:
            raise ValueError(
                f"Cannot normalize {column} for circuit {circuit!r}: "
                f"{MEAN_MODEL} is zero."
            )
        values.append(records[(circuit, model)][column] / denominator)
    return values


def plot_metric(plt, circuits, records, metric, output_path):
    """Write one grouped bar chart for a configured metric."""
    figure_width = max(10.0, 0.8 * len(circuits) + 2.5)
    figure, axis = plt.subplots(figsize=(figure_width, 6.0))
    models = models_for_metric(metric)
    group_width = 0.72 if metric["relative_to_mean"] else 0.82
    bar_width = group_width / len(models)
    centers = list(range(len(circuits)))

    for model_index, model in enumerate(models):
        positions = [
            center - group_width / 2.0 + bar_width * (model_index + 0.5)
            for center in centers
        ]
        values = metric_values(circuits, model, records, metric)
        axis.bar(
            positions,
            values,
            width=bar_width * 0.92,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
        )

    axis.set_title(metric["title"], fontsize=15, pad=12)
    axis.set_xlabel("Circuit")
    axis.set_ylabel(metric["ylabel"])
    axis.set_xticks(centers, circuits, rotation=35, ha="right")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if metric["relative_to_mean"]:
        axis.axhline(
            1.0,
            color=MODEL_COLORS[MEAN_MODEL],
            linestyle="--",
            linewidth=1.6,
            label="smartatpg_mean baseline (1.0)",
        )
    else:
        axis.set_ylim(0.0, 100.0)
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_plots(csv_path, output_dir=None):
    """Read one final comparison CSV and generate all four plot files."""
    csv_path = Path(csv_path).resolve()
    circuits, _models, records = read_comparison_csv(csv_path)
    output_dir = (
        Path(output_dir).resolve() if output_dir is not None else csv_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()

    output_paths = []
    for metric in METRICS:
        output_path = output_dir / metric["filename"]
        plot_metric(plt, circuits, records, metric, output_path)
        output_paths.append(output_path)
    return output_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_comparison_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to the input CSV directory.",
    )
    args = parser.parse_args(argv)
    try:
        output_paths = generate_plots(
            args.final_comparison_csv,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    for output_path in output_paths:
        print(f"WROTE {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
