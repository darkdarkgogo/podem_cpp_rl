"""Validate three GAT ablations against an existing full GAT result."""

import argparse
import json
import math
from pathlib import Path

from .artifact_io import atomic_json, atomic_json_lines
from .data_split import discover_validation_dataset
from .smartatpg_artifacts import snapshot_id
from .validation import (
    _atomic_csv, _evaluate_model_round, _load_inference_checkpoint,
    _read_json, _validate_records,
    _validate_summary,
)
from .validation_core import (
    _load_validation_catalogs, _summarize_validation, validation_score,
)


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("full", "no_cache", "no_lock", "no_object_value")
VARIANTS = {
    "full": "level_gat_gru",
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


def _existing_full_result(full_run_dir, circuits, seed, manifest):
    summary = _read_json(full_run_dir / "train_summary.json", "existing full GAT training summary")
    if summary.get("format") != "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION":
        raise ValueError("Existing full GAT training summary has an incompatible format")
    if summary.get("seed") != seed or summary.get("backtrack_limit") != 100:
        raise ValueError("Existing full GAT seed or backtrack limit differs from ablations")
    if summary.get("backtrace_lock") not in ("auto", "on"):
        raise ValueError("Existing full GAT must have the backtrace lock enabled")
    if Path(summary["manifests"]["gat"]).resolve() != manifest.resolve():
        raise ValueError("Existing full GAT manifest differs from ablation manifest")
    validation_dir = full_run_dir / "validation"
    report = _read_json(
        validation_dir / "validation_three_way_comparison.json",
        "existing full GAT validation report",
    )
    if (report.get("format") != "SMARTATPG_THREE_WAY_VALIDATION_V1" or
            report.get("seed") != seed or report.get("backtrack_limit") != 100 or
            report.get("backtrace_lock") not in ("auto", "on")):
        raise ValueError("Existing full GAT validation protocol differs from ablations")
    selection = report.get("model_selection", {}).get("gat")
    if not isinstance(selection, dict):
        raise ValueError("Existing full GAT validation lacks model selection")
    round_number = int(selection["best_round"])
    if round_number < 1 or round_number > summary["rounds"]:
        raise ValueError("Existing full GAT selected round is invalid")
    checkpoint = Path(summary["training_dirs"]["gat"]) / f"inference_round_{round_number:02d}.pth"
    full_payload, full_state = _load_inference_checkpoint(
        checkpoint, "level_gat_gru", manifest, round_number,
    )
    if full_payload.get("seed") != seed or full_payload.get("backtrace_lock") is not True:
        raise ValueError("Existing full GAT checkpoint seed or backtrace lock differs from ablations")
    records_path = validation_dir / "detailed" / "gat_records.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"Existing full GAT validation records are missing: {records_path}")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    _validate_records(records, "full")
    result = _summarize_validation(records, circuits, round_number)
    _validate_summary(result, circuits, round_number, "full")
    if list(validation_score(result, round_number)) != selection.get("score"):
        raise ValueError("Existing full GAT records disagree with model selection")
    report_rows = report.get("rows", [])
    names = [item["name"] for item in circuits] + ["TOTAL"]
    if [row.get("circuit") for row in report_rows] != names:
        raise ValueError("Existing full GAT report has a different circuit order")
    for name, report_row in zip(names, report_rows):
        full_row = _row(result, name)
        expected = {
            "gat_backtracks": full_row["backtracks_total"],
            "gat_backtrace_steps": full_row["backtrace_steps_total"],
            "gat_runtime_atpg_s": full_row["atpg_seconds"],
            "gat_fault_coverage_pct": 100.0 * full_row["fault_coverage"],
        }
        if any(
            key not in report_row or not math.isclose(
                float(report_row[key]), float(value), rel_tol=1e-9, abs_tol=1e-6,
            )
            for key, value in expected.items()
        ):
            raise ValueError(f"Existing full GAT records disagree with report: {name}")
    return records, result, selection, snapshot_id(full_state)


def run_validation(run_dir, output_dir=None, seed=None):
    run_dir = Path(run_dir).resolve()
    summary = _read_json(run_dir / "train_summary.json", "GAT ablation training summary")
    if summary.get("format") != "SMARTATPG_GAT_ABLATION_TRAINING_V2":
        raise ValueError("Incompatible GAT ablation training summary")
    if set(summary.get("groups", {})) != set(GROUPS[1:]):
        raise ValueError("GAT ablation summary must contain exactly three trained groups")
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
    full_run_dir = Path(summary["full_run_dir"]).resolve()
    full_records, full_result, full_selection, full_actor_snapshot = _existing_full_result(
        full_run_dir, circuits, seed, manifest,
    )
    no_cache_checkpoint = (
        Path(summary["groups"]["no_cache"]["model_dir"])
        / "inference_round_01.pth"
    )
    _, no_cache_state = _load_inference_checkpoint(
        no_cache_checkpoint, "level_gat_gru", manifest, 1,
        expected_ablation="no_cache", expected_backtrace_lock=True,
        expected_seed=seed,
    )
    if snapshot_id(no_cache_state) != full_actor_snapshot:
        raise ValueError(
            "The no_cache and full Actor snapshots differ; a cache-only "
            "comparison requires identical model weights"
        )
    results = {"full": full_result}
    selection = {"full": full_selection}
    atomic_json_lines(output_dir / "detailed" / "full_records.jsonl", full_records)
    for group in GROUPS[1:]:
        group_info = summary["groups"][group]
        expected_lock = group != "no_lock"
        if (group_info.get("encoder_variant") != VARIANTS[group] or
                group_info.get("backtrace_lock") is not expected_lock):
            raise ValueError(f"Training group identity mismatch: {group}")
        round_results = [
            _evaluate_model_round(
                group, VARIANTS[group], manifest, group_info["model_dir"],
                circuits, output_dir, number, seed, expected_lock, group,
                logits_cache=False if group == "no_cache" else None,
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
        "format": "SMARTATPG_GAT_ABLATION_VALIDATION_V2",
        "seed": seed,
        "backtrack_limit": 100,
        "runtime_definition": "sum(per_fault.atpg_seconds)",
        "manifest": str(manifest),
        "full_run_dir": str(full_run_dir),
        "logits_cache": {group: group != "no_cache" for group in GROUPS},
        "no_cache_actor_matches_full": True,
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
