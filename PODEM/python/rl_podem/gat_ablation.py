"""Fresh, identity-checked validation and reporting for GAT ablations."""

import argparse
from pathlib import Path

from .artifact_io import atomic_json, atomic_json_lines
from .data_split import discover_validation_dataset
from .validation import (
    _atomic_csv, _evaluate_model_round, _read_json, _validate_summary,
)
from .validation_core import _load_validation_catalogs, validation_score


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("full", "no_gru", "no_lock", "no_object_value")
VARIANTS = {
    "full": "level_gat_gru",
    "no_gru": "level_gat_no_gru",
    "no_lock": "level_gat_gru",
    "no_object_value": "level_gat_gru_no_object_value",
}
METRICS = (
    ("detected_faults", "detected_faults"),
    ("redundant_faults", "redundant_faults"),
    ("aborted_faults", "aborted_faults"),
    ("backtracks", "backtracks_total"),
    ("backtrace_steps", "backtrace_steps_total"),
    ("runtime_atpg_s", "atpg_seconds"),
)


def _row(summary, circuit):
    return summary if circuit == "TOTAL" else next(
        item for item in summary["circuits"] if item["circuit"] == circuit
    )


def comparison_row(circuit, results):
    row = {"circuit": circuit}
    full = _row(results["full"], circuit)
    for group in GROUPS:
        entry = _row(results[group], circuit)
        for label, source in METRICS:
            row[f"{group}_{label}"] = entry[source]
        row[f"{group}_fault_coverage_pct"] = 100.0 * entry["fault_coverage"]
        if group != "full":
            for label in ("detected_faults", "redundant_faults", "aborted_faults"):
                row[f"{group}_{label}_delta_vs_full"] = entry[label] - full[label]
            row[f"{group}_backtracks_delta_vs_full"] = (
                entry["backtracks_total"] - full["backtracks_total"]
            )
            row[f"{group}_backtrace_steps_delta_vs_full"] = (
                entry["backtrace_steps_total"] - full["backtrace_steps_total"]
            )
            row[f"{group}_runtime_delta_pct_vs_full"] = (
                100.0 * (entry["atpg_seconds"] / full["atpg_seconds"] - 1.0)
                if full["atpg_seconds"] else None
            )
            row[f"{group}_coverage_delta_pp_vs_full"] = (
                100.0 * (entry["fault_coverage"] - full["fault_coverage"])
            )
    return row


def run_validation(run_dir, output_dir=None, seed=None):
    run_dir = Path(run_dir).resolve()
    summary = _read_json(run_dir / "train_summary.json", "GAT ablation training summary")
    if summary.get("format") != "SMARTATPG_GAT_ABLATION_TRAINING_V1":
        raise ValueError("Incompatible GAT ablation training summary")
    if set(summary.get("groups", {})) != set(GROUPS):
        raise ValueError("GAT ablation summary must contain four ordered groups")
    if summary.get("rounds") != 1 or summary.get("backtrack_limit") != 100:
        raise ValueError("GAT ablation training protocol changed")
    if seed is None:
        seed = summary["seed"]
    if seed != summary["seed"]:
        raise ValueError("Validation seed must match training")
    output_dir = (run_dir / "validation" if output_dir is None else Path(output_dir).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(summary["manifest"])
    manifest_data = _read_json(manifest, "GAT training manifest")
    if manifest_data.get("encoder_variant") != "level_gat_gru":
        raise ValueError("A GAT training manifest is required")
    validation_paths = discover_validation_dataset(Path(summary["dataset_root"]))
    circuits = _load_validation_catalogs(
        {"format": "fresh-validation"},
        [{"name": path.stem, "circuit": str(path)} for path in validation_paths],
    )
    results = {}
    selection = {}
    for group in GROUPS:
        group_info = summary["groups"][group]
        expected_lock = group != "no_lock"
        if (group_info.get("encoder_variant") != VARIANTS[group] or
                group_info.get("backtrace_lock") is not expected_lock):
            raise ValueError(f"Training group identity mismatch: {group}")
        round_results = [
            _evaluate_model_round(
                group, VARIANTS[group], manifest, group_info["model_dir"],
                circuits, output_dir, number, seed, expected_lock, group,
            )
            for number in range(1, summary["rounds"] + 1)
        ]
        records, best = min(
            round_results,
            key=lambda item: validation_score(item[1], item[1]["round"]),
        )
        _validate_summary(best, circuits, best["round"], group)
        atomic_json_lines(output_dir / "detailed" / f"{group}_records.jsonl", records)
        results[group] = best
        selection[group] = {
            "best_round": best["round"],
            "score": list(validation_score(best, best["round"])),
        }
    rows = [
        comparison_row(name, results)
        for name in [item["name"] for item in circuits] + ["TOTAL"]
    ]
    report = {
        "format": "SMARTATPG_GAT_ABLATION_VALIDATION_V1",
        "seed": seed,
        "backtrack_limit": 100,
        "runtime_definition": "sum(per_fault.atpg_seconds)",
        "manifest": str(manifest),
        "model_selection": selection,
        "rows": rows,
    }
    atomic_json(output_dir / "validation_gat_ablation.json", report)
    _atomic_csv(output_dir / "validation_gat_ablation.csv", rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/gat_ablation")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    run_validation(args.run_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
