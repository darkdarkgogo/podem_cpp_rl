"""Run fresh SmartATPG/SCOAP validation and write a three-way comparison."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from .smartatpg_artifacts import export_descriptors, policy_from_state
from .smartatpg_features import load_circuit_graph
from .smartatpg_rewards import reward_scheme_for_encoder
from .data_split import discover_validation_dataset
from .training import (
    INFERENCE_CHECKPOINT_FORMAT,
    _atomic_json,
    _atomic_json_lines,
    _load_validation_catalogs,
    _manifest_hash,
    _native_validation_batch,
    _summarize_validation,
    validation_score,
)
from .validation_tables import build_three_way_row


BACKTRACK_LIMIT = 100
RAW_METRICS = (
    "episodes", "detected_faults", "redundant_faults", "aborted_faults",
    "test_vectors", "backtracks_total", "backtracks_mean",
    "backtrace_steps_total", "backtrace_steps_mean", "return_total",
    "return_mean", "atpg_seconds", "fault_coverage",
)
ADDITIVE_METRICS = (
    "episodes", "detected_faults", "redundant_faults", "aborted_faults",
    "test_vectors", "backtracks_total", "backtrace_steps_total",
    "return_total", "atpg_seconds",
)


class ScoapValidationEvaluator:
    """Expose the SCOAP heuristic through the Python evaluator protocol."""

    @staticmethod
    def decision_callback(request):
        return int(request["heuristic_action"])

    def run(
        self, circuit_path, *, backtrack_limit, seed, fault_ids,
        use_scoap, event_callback,
    ):
        try:
            import cpp_podem
        except ImportError as error:
            raise ImportError(
                "Cannot import cpp_podem. Install with 'python -m pip install -e .'"
            ) from error
        return dict(cpp_podem.run_stuck_at(
            str(Path(circuit_path).resolve()), self.decision_callback,
            event_callback, backtrack_limit, seed, fault_ids, True,
            "backtrace_rl", "", True,
        ))


def _read_json(path, description):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {description}: {path}") from error


def _atomic_csv(path, rows):
    if not rows:
        raise ValueError("Three-way comparison requires at least one row")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("Three-way comparison rows have inconsistent columns")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_records(records, method):
    for record in records:
        location = f"{record.get('circuit')} {record.get('fault_id')}"
        for key in ("backtracks", "backtrace_steps", "detected", "test_vectors"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid {method} {key} for {location}")
        for key in ("return", "atpg_seconds"):
            value = record.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (key == "atpg_seconds" and value < 0)
            ):
                raise ValueError(f"Invalid {method} {key} for {location}")


def _validate_summary(summary, circuits, round_number, method):
    expected_names = [item["name"] for item in circuits]
    if summary.get("round") != round_number:
        raise ValueError(f"{method} summary has the wrong round")
    if [row.get("circuit") for row in summary.get("circuits", [])] != expected_names:
        raise ValueError(f"{method} summary has the wrong circuit order")
    scopes = [summary, *summary["circuits"]]
    if any(any(key not in row for key in RAW_METRICS) for row in scopes):
        raise ValueError(f"{method} summary is incomplete")
    expected_episodes = sum(len(item["episode_fault_ids"]) for item in circuits)
    if summary["episodes"] != expected_episodes:
        raise ValueError(f"{method} summary does not cover the full fault catalog")
    for item, row in zip(circuits, summary["circuits"]):
        if row["episodes"] != len(item["episode_fault_ids"]):
            raise ValueError(f"{method} circuit summary is incomplete: {item['name']}")
    for key in ADDITIVE_METRICS:
        expected = sum(row[key] for row in summary["circuits"])
        if not math.isclose(float(summary[key]), float(expected), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{method} TOTAL {key} does not equal circuit rows")


def _evaluate_scoap_batch(item, fault_ids, seed):
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError(
            "Native SCOAP validation requires the rebuilt cpp_podem extension"
        ) from error
    if not hasattr(cpp_podem, "run_native_scoap_validation"):
        raise RuntimeError(
            "cpp_podem is stale; rebuild it with: python -m pip install -e ."
        )
    reward_scheme = reward_scheme_for_encoder("level_gat_gru")
    native_records = cpp_podem.run_native_scoap_validation(
        str(Path(item["circuit"]).resolve()), BACKTRACK_LIMIT, seed, fault_ids,
        reward_scheme, item["name"],
    )
    if len(native_records) != len(fault_ids):
        raise RuntimeError("Native SCOAP validation returned the wrong fault count")
    records = []
    for fault_id, raw in zip(fault_ids, native_records):
        if raw["fault_id"] != fault_id:
            raise RuntimeError("Native SCOAP validation returned faults out of order")
        outcome = int(raw["outcome"])
        records.append({
            "circuit": item["name"],
            "fault_id": fault_id,
            "outcome": outcome,
            "detected": int(outcome == 1),
            "redundant": int(outcome == 0),
            "aborted": int(outcome not in (0, 1)),
            "backtracks": int(raw["backtracks"]),
            "backtrace_steps": int(raw["backtrace_steps"]),
            "return": float(raw["return"]),
            "test_vectors": int(outcome == 1),
            "atpg_seconds": float(raw["atpg_seconds"]),
        })
    _validate_records(records, "SCOAP")
    return records


def _load_inference_checkpoint(path, encoder, manifest_path, round_number):
    payload = torch.load(path, map_location="cpu")
    expected = {
        "format": INFERENCE_CHECKPOINT_FORMAT,
        "manifest_hash": _manifest_hash(manifest_path),
        "encoder_variant": encoder,
        "reward_scheme": reward_scheme_for_encoder(encoder),
        "backtrack_limit": BACKTRACK_LIMIT,
        "round": round_number,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Inference checkpoint identity mismatch: {path}")
    state = payload.get("policy_old")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Inference checkpoint has no policy state: {path}")
    return payload, state


def _evaluate_model_round(
    name, encoder, manifest_path, model_dir, circuits, output_dir,
    round_number, seed,
):
    checkpoint = Path(model_dir) / f"inference_round_{round_number:02d}.pth"
    actor_path = Path(model_dir) / f"model_round_{round_number:02d}.txt"
    payload, state = _load_inference_checkpoint(
        checkpoint, encoder, manifest_path, round_number,
    )
    if not actor_path.is_file():
        raise FileNotFoundError(f"Missing native actor for round {round_number}: {actor_path}")
    policy = policy_from_state(state)
    embeddings = {}
    for item in circuits:
        graph = load_circuit_graph(item["circuit"])
        embedding_path = (
            Path(output_dir) / "embeddings" / name
            / f"round_{round_number:02d}" / f"{item['name']}.emb"
        )
        export_descriptors(state, graph, embedding_path, policy)
        embeddings[item["name"]] = embedding_path

    journal = Path(output_dir) / "native_journals" / name / f"round_{round_number:02d}.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    if journal.is_file():
        journal.unlink()
    records = []
    for item in circuits:
        records.extend(_native_validation_batch(
            item, item["episode_fault_ids"], embeddings[item["name"]], actor_path,
            journal, seed, payload["reward_scheme"],
        ))
    _validate_records(records, name)
    summary = _summarize_validation(records, circuits, round_number)
    _validate_summary(summary, circuits, round_number, name)
    return records, summary


def run_fresh_validation(run_dir, dataset_root=None, output_dir=None, seed=2026):
    run_dir = Path(run_dir).resolve()
    output_dir = (
        run_dir / "validation" if output_dir is None else Path(output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = _read_json(run_dir / "train_summary.json", "training summary")
    if run_summary.get("format") != "SMARTATPG_DUAL_TRAINING_V3_SEPARATED_VALIDATION":
        raise ValueError("Training summary is incompatible with separated validation")
    training_dataset_root = Path(run_summary["dataset_root"]).resolve()
    if dataset_root is not None and Path(dataset_root).resolve() != training_dataset_root:
        raise ValueError("Validation dataset root differs from the training run")
    if int(seed) != int(run_summary["seed"]):
        raise ValueError("Validation seed must match the training seed")

    manifests = {key: Path(value) for key, value in run_summary["manifests"].items()}
    model_dirs = {key: Path(value) for key, value in run_summary["training_dirs"].items()}
    model_specs = {"gat": "level_gat_gru", "mean": "fanin_mean"}
    validation_paths = discover_validation_dataset(training_dataset_root)
    circuits = [
        {"name": path.stem, "circuit": str(path)} for path in validation_paths
    ]
    circuits = _load_validation_catalogs({"format": "fresh-validation"}, circuits)
    for name in ("gat", "mean"):
        _read_json(manifests[name], f"{name} training manifest")

    model_results = {}
    selection = {}
    detailed = output_dir / "detailed"
    rounds = int(run_summary["rounds"])
    for name, encoder in model_specs.items():
        round_results = [
            _evaluate_model_round(
                name, encoder, manifests[name], model_dirs[name], circuits,
                output_dir, round_number, seed,
            )
            for round_number in range(1, rounds + 1)
        ]
        best_records, best_summary = min(
            round_results,
            key=lambda value: validation_score(value[1], value[1]["round"]),
        )
        score = validation_score(best_summary, best_summary["round"])
        selection[name] = {
            "best_round": int(best_summary["round"]),
            "score": list(score),
        }
        model_results[name] = best_summary
        _atomic_json_lines(detailed / f"{name}_records.jsonl", best_records)

    scoap_records = []
    for item in circuits:
        scoap_records.extend(_evaluate_scoap_batch(
            item, item["episode_fault_ids"], seed,
        ))
    scoap_summary = _summarize_validation(scoap_records, circuits, 0)
    _validate_summary(scoap_summary, circuits, 0, "SCOAP")
    _atomic_json_lines(detailed / "scoap_records.jsonl", scoap_records)

    names = [item["name"] for item in circuits] + ["TOTAL"]
    rows = [
        build_three_way_row(
            circuit, scoap_summary, model_results["gat"], model_results["mean"],
        )
        for circuit in names
    ]
    result = {
        "format": "SMARTATPG_THREE_WAY_VALIDATION_V1",
        "seed": seed,
        "backtrack_limit": BACKTRACK_LIMIT,
        "runtime_definition": "sum(per_fault.atpg_seconds)",
        "model_selection": selection,
        "rows": rows,
    }
    _atomic_json(output_dir / "model_selection.json", selection)
    _atomic_json(output_dir / "validation_three_way_comparison.json", result)
    _atomic_csv(output_dir / "validation_three_way_comparison.csv", rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    run_fresh_validation(
        args.run_dir, args.dataset_root, args.output_dir, args.seed,
    )


if __name__ == "__main__":
    main()
