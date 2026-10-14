"""Validate three independently trained pre-fanin GAT ablations."""

import argparse
from pathlib import Path

from .artifact_io import atomic_json, atomic_json_lines
from .data_split import discover_validation_dataset
from .smartatpg_artifacts import snapshot_id
from .validation import (
    _atomic_csv, _evaluate_model_round, _load_inference_checkpoint,
    _read_json, _validate_summary,
)
from .validation_core import (
    _load_validation_catalogs, validation_score,
)


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("no_cache", "no_lock", "no_object_value")
VARIANTS = {
    "no_cache": "level_gat_gru",
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
    for group in GROUPS:
        entry = _row(results[group], circuit)
        for label, source in METRICS:
            row[f"{group}_{label}"] = entry[source]
        row[f"{group}_fault_coverage_pct"] = 100.0 * entry["fault_coverage"]
    return row


def _check_group(summary, group, manifest, seed):
    info = summary["groups"][group]
    expected_lock = group != "no_lock"
    if (info.get("encoder_variant") != VARIANTS[group] or
            info.get("backtrace_lock") is not expected_lock):
        raise ValueError(f"Training group identity mismatch: {group}")
    model_dir = Path(info["model_dir"]).resolve()
    if any(
        model_dir == Path(other["model_dir"]).resolve()
        for name, other in summary["groups"].items() if name != group
    ):
        raise ValueError(f"{group} must have its own trained model directory")
    snapshots = []
    for number in range(1, summary["rounds"] + 1):
        checkpoint = model_dir / f"inference_round_{number:02d}.pth"
        _, state = _load_inference_checkpoint(
            checkpoint, VARIANTS[group], manifest, number,
            expected_ablation=group,
            expected_backtrace_lock=expected_lock,
            expected_seed=seed,
        )
        snapshots.append(snapshot_id(state))
    return model_dir, snapshots


def run_validation(run_dir, output_dir=None, seed=None):
    run_dir = Path(run_dir).resolve()
    summary = _read_json(run_dir / "train_summary.json", "pre-fanin training summary")
    if summary.get("format") != "SMARTATPG_PRE_FANIN_ABLATION_TRAINING_V3_THREE_GROUPS":
        raise ValueError("Incompatible pre-fanin training summary")
    if set(summary.get("groups", {})) != set(GROUPS):
        raise ValueError("Expected exactly three ablation groups")
    if summary.get("rounds") != 2 or summary.get("backtrack_limit") != 100:
        raise ValueError("Pre-fanin training protocol changed")
    if seed is None:
        seed = summary["seed"]
    if seed != summary["seed"]:
        raise ValueError("Validation seed must match training")
    output_dir = run_dir / "validation" if output_dir is None else Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(summary["manifest"]).resolve()
    manifest_data = _read_json(manifest, "pre-fanin training manifest")
    if (manifest_data.get("encoder_variant") != "level_gat_gru" or
            manifest_data.get("normal_rounds") != 2):
        raise ValueError("A two-round GAT training manifest is required")
    validation_paths = discover_validation_dataset(Path(summary["dataset_root"]))
    circuits = _load_validation_catalogs(
        {"format": "fresh-validation"},
        [{"name": path.stem, "circuit": str(path)} for path in validation_paths],
    )
    group_dirs = {}
    snapshots = {}
    for group in GROUPS:
        group_dirs[group], snapshots[group] = _check_group(
            summary, group, manifest, seed,
        )
    results = {}
    selection = {}
    for group in GROUPS:
        round_results = [
            _evaluate_model_round(
                group, VARIANTS[group], manifest, group_dirs[group], circuits,
                output_dir, number, seed, group != "no_lock",
                expected_ablation=group, logits_cache=group != "no_cache",
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
            "actor_snapshot": snapshots[group][best["round"] - 1],
        }
    rows = [
        comparison_row(name, results)
        for name in [item["name"] for item in circuits] + ["TOTAL"]
    ]
    report = {
        "format": "SMARTATPG_PRE_FANIN_ABLATION_VALIDATION_V2_THREE_GROUPS",
        "seed": seed,
        "backtrack_limit": 100,
        "runtime_definition": "sum(per_fault.atpg_seconds)",
        "manifest": str(manifest),
        "logits_cache": {group: group != "no_cache" for group in GROUPS},
        "independently_trained_groups": list(GROUPS),
        "model_selection": selection,
        "rows": rows,
    }
    atomic_json(output_dir / "validation_pre_fanin_ablation.json", report)
    _atomic_csv(output_dir / "validation_pre_fanin_ablation.csv", rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/pre_fanin_ablation")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    run_validation(args.run_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
