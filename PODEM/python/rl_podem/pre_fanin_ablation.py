"""Identity-checked validation of the original two-output GAT Actor."""

import argparse
import json
import math
from pathlib import Path

from .artifact_io import atomic_json, atomic_json_lines
from .data_split import discover_validation_dataset
from .smartatpg_artifacts import snapshot_id
from .validation import (
    _atomic_csv, _evaluate_model_round, _load_inference_checkpoint,
    _read_json, _validate_records, _validate_summary,
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
            for label, source in METRICS:
                if label != "runtime_atpg_s":
                    row[f"{group}_{label}_delta_vs_full"] = entry[source] - full[source]
            row[f"{group}_runtime_delta_pct_vs_full"] = (
                100.0 * (entry["atpg_seconds"] / full["atpg_seconds"] - 1.0)
                if full["atpg_seconds"] else None
            )
            row[f"{group}_coverage_delta_pp_vs_full"] = (
                100.0 * (entry["fault_coverage"] - full["fault_coverage"])
            )
    return row


def _validate_record_order(records, circuits):
    expected = [
        (item["name"], fault_id)
        for item in circuits for fault_id in item["episode_fault_ids"]
    ]
    actual = [(record["circuit"], record["fault_id"]) for record in records]
    if actual != expected:
        raise ValueError("Existing full GAT fault catalog differs from ablations")


def _check_group(summary, group, manifest, seed):
    info = summary["groups"][group]
    expected_lock = group != "no_lock"
    if (info.get("encoder_variant") != VARIANTS[group] or
            info.get("backtrace_lock") is not expected_lock):
        raise ValueError(f"Training group identity mismatch: {group}")
    model_dir = Path(info["model_dir"]).resolve()
    if group == "no_cache":
        full_dir = Path(summary["groups"]["full"]["model_dir"]).resolve()
        if model_dir != full_dir or info.get("source_group") != "full":
            raise ValueError("no_cache must reuse the full model directory")
        checkpoint_group = None
    elif group == "full":
        checkpoint_group = None
    else:
        checkpoint_group = group
    snapshots = []
    for number in range(1, summary["rounds"] + 1):
        checkpoint = model_dir / f"inference_round_{number:02d}.pth"
        _, state = _load_inference_checkpoint(
            checkpoint, VARIANTS[group], manifest, number,
            expected_ablation=checkpoint_group,
            expected_backtrace_lock=expected_lock if checkpoint_group else None,
            expected_seed=seed,
        )
        snapshots.append(snapshot_id(state))
    return model_dir, snapshots


def _existing_full_result(full_run_dir, circuits, seed, manifest):
    summary = _read_json(full_run_dir / "train_summary.json", "existing full GAT summary")
    if (summary.get("format") != "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION" or
            summary.get("seed") != seed or summary.get("rounds") != 2 or
            summary.get("backtrack_limit") != 100 or
            summary.get("backtrace_lock") not in ("auto", "on")):
        raise ValueError("Existing full GAT training protocol differs from ablations")
    if Path(summary["manifests"]["gat"]).resolve() != manifest:
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
    if round_number not in (1, 2):
        raise ValueError("Existing full GAT selected round is invalid")
    checkpoint = Path(summary["training_dirs"]["gat"]) / f"inference_round_{round_number:02d}.pth"
    full_payload, full_state = _load_inference_checkpoint(
        checkpoint, "level_gat_gru", manifest, round_number,
        expected_seed=seed,
    )
    if full_payload.get("backtrace_lock", True) is not True:
        raise ValueError("Existing full GAT checkpoint disables backtrace lock")
    records_path = validation_dir / "detailed" / "gat_records.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"Existing full GAT records are missing: {records_path}")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    _validate_records(records, "full")
    _validate_record_order(records, circuits)
    result = _summarize_validation(records, circuits, round_number)
    _validate_summary(result, circuits, round_number, "full")
    if list(validation_score(result, round_number)) != selection.get("score"):
        raise ValueError("Existing full GAT records disagree with model selection")
    names = [item["name"] for item in circuits] + ["TOTAL"]
    report_rows = report.get("rows", [])
    if [row.get("circuit") for row in report_rows] != names:
        raise ValueError("Existing full GAT report has a different circuit order")
    for name, report_row in zip(names, report_rows):
        expected = _row(result, name)
        fields = {
            "gat_backtracks": expected["backtracks_total"],
            "gat_backtrace_steps": expected["backtrace_steps_total"],
            "gat_runtime_atpg_s": expected["atpg_seconds"],
            "gat_fault_coverage_pct": 100.0 * expected["fault_coverage"],
        }
        if any(
            key not in report_row or not math.isclose(
                float(report_row[key]), float(value), rel_tol=1e-9, abs_tol=1e-6,
            )
            for key, value in fields.items()
        ):
            raise ValueError(f"Existing full GAT records disagree with report: {name}")
    return records, result, selection, snapshot_id(full_state)


def run_validation(run_dir, output_dir=None, seed=None):
    run_dir = Path(run_dir).resolve()
    summary = _read_json(run_dir / "train_summary.json", "pre-fanin training summary")
    if summary.get("format") != "SMARTATPG_PRE_FANIN_ABLATION_TRAINING_V1":
        raise ValueError("Incompatible pre-fanin training summary")
    if set(summary.get("groups", {})) != set(GROUPS):
        raise ValueError("Expected full and exactly three ablation groups")
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
    if snapshots["full"] != snapshots["no_cache"]:
        raise ValueError("full and no_cache Actor snapshots differ")

    full_run_dir = Path(summary["full_run_dir"]).resolve()
    full_records, full_result, full_selection, full_snapshot = _existing_full_result(
        full_run_dir, circuits, seed, manifest,
    )
    if snapshots["full"][full_selection["best_round"] - 1] != full_snapshot:
        raise ValueError("Existing full GAT checkpoint snapshot changed")
    results = {"full": full_result}
    selection = {"full": full_selection}
    atomic_json_lines(output_dir / "detailed" / "full_records.jsonl", full_records)
    for group in ("no_lock", "no_object_value"):
        round_results = [
            _evaluate_model_round(
                group, VARIANTS[group], manifest, group_dirs[group], circuits,
                output_dir, number, seed, group != "no_lock",
                expected_ablation=group,
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

    full_round = selection["full"]["best_round"]
    records, no_cache = _evaluate_model_round(
        "no_cache", VARIANTS["no_cache"], manifest, group_dirs["no_cache"],
        circuits, output_dir, full_round, seed, True,
        logits_cache=False,
    )
    _validate_summary(no_cache, circuits, full_round, "no_cache")
    atomic_json_lines(output_dir / "detailed" / "no_cache_records.jsonl", records)
    results["no_cache"] = no_cache
    selection["no_cache"] = {
        "best_round": full_round,
        "score": list(validation_score(no_cache, full_round)),
        "actor_snapshot": snapshots["no_cache"][full_round - 1],
    }
    rows = [
        comparison_row(name, results)
        for name in [item["name"] for item in circuits] + ["TOTAL"]
    ]
    report = {
        "format": "SMARTATPG_PRE_FANIN_ABLATION_VALIDATION_V1",
        "seed": seed,
        "backtrack_limit": 100,
        "runtime_definition": "sum(per_fault.atpg_seconds)",
        "manifest": str(manifest),
        "full_run_dir": str(full_run_dir),
        "logits_cache": {group: group != "no_cache" for group in GROUPS},
        "no_cache_actor_matches_full_checkpoint": True,
        "historical_full_actor_hash_available": False,
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
