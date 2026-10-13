"""Run fresh SCOAP/GAT/Mean/NN-All validation from trained artifacts."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "artifacts" / "smartatpg_dual"
DEFAULT_NN_ALL_RUN_DIR = ROOT / "artifacts" / "nn_all_podem"

from .artifact_io import (
    INFERENCE_CHECKPOINT_FORMAT,
    atomic_json as _atomic_json,
    atomic_json_lines as _atomic_json_lines,
    manifest_hash as _manifest_hash,
)
from .cpp_bridge import resolve_backtrace_lock
from .smartatpg_artifacts import export_descriptors, policy_from_state
from .smartatpg_features import load_circuit_graph
from .smartatpg_rewards import (
    reward_scheme_for_encoder,
    smartatpg_backtrace_step_reward,
    smartatpg_circuit_depth,
)
from .data_split import discover_validation_dataset
from .validation_core import (
    _load_validation_catalogs,
    _native_validation_batch,
    _summarize_validation,
    validation_score,
)
from .validation_tables import build_four_way_row, build_three_way_row
from .nn_all_features import FEATURE_SCHEMA, load_nn_all_features, write_native_features
from .nn_all_training import (
    MODEL_FORMAT, TRAINING_FORMAT, PAPER_TRAIN_CIRCUITS, TRAINING_EPOCHS,
)
from .native_io import _native_circuit_path


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
        raise ValueError("Validation comparison requires at least one row")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("Validation comparison rows have inconsistent columns")
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
    graph = load_circuit_graph(item["circuit"])
    backtrace_step_reward = smartatpg_backtrace_step_reward(
        reward_scheme, smartatpg_circuit_depth(graph.levels),
    )
    native_records = cpp_podem.run_native_scoap_validation(
        str(Path(item["circuit"]).resolve()), BACKTRACK_LIMIT, seed, fault_ids,
        reward_scheme, backtrace_step_reward, item["name"],
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


def _load_inference_checkpoint(
    path, encoder, manifest_path, round_number,
    expected_ablation=None, expected_backtrace_lock=None, expected_seed=None,
):
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
    if expected_ablation is not None and payload.get("ablation") != expected_ablation:
        raise ValueError(f"Inference checkpoint ablation mismatch: {path}")
    if (expected_backtrace_lock is not None and
            payload.get("backtrace_lock") != expected_backtrace_lock):
        raise ValueError(f"Inference checkpoint backtrace lock mismatch: {path}")
    if expected_seed is not None and payload.get("seed") != expected_seed:
        raise ValueError(f"Inference checkpoint seed mismatch: {path}")
    state = payload.get("policy_old")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Inference checkpoint has no policy state: {path}")
    return payload, state


def _evaluate_model_round(
    name, encoder, manifest_path, model_dir, circuits, output_dir,
    round_number, seed, backtrace_lock, expected_ablation=None,
):
    checkpoint = Path(model_dir) / f"inference_round_{round_number:02d}.pth"
    actor_path = Path(model_dir) / f"model_round_{round_number:02d}.txt"
    payload, state = _load_inference_checkpoint(
        checkpoint, encoder, manifest_path, round_number,
        expected_ablation, backtrace_lock if expected_ablation is not None else None,
        seed,
    )
    if not actor_path.is_file():
        raise FileNotFoundError(f"Missing native actor for round {round_number}: {actor_path}")
    policy = policy_from_state(state)
    embeddings = {}
    backtrace_step_rewards = {}
    for item in circuits:
        graph = load_circuit_graph(item["circuit"])
        embedding_path = (
            Path(output_dir) / "embeddings" / name
            / f"round_{round_number:02d}" / f"{item['name']}.emb"
        )
        export_descriptors(state, graph, embedding_path, policy)
        embeddings[item["name"]] = embedding_path
        backtrace_step_rewards[item["name"]] = (
            smartatpg_backtrace_step_reward(
                payload["reward_scheme"],
                smartatpg_circuit_depth(graph.levels),
            )
        )

    journal = Path(output_dir) / "native_journals" / name / f"round_{round_number:02d}.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    if journal.is_file():
        journal.unlink()
    records = []
    for item in circuits:
        records.extend(_native_validation_batch(
            item, item["episode_fault_ids"], embeddings[item["name"]], actor_path,
            journal, seed, payload["reward_scheme"],
            backtrace_step_rewards[item["name"]],
            backtrace_lock,
        ))
    _validate_records(records, name)
    summary = _summarize_validation(records, circuits, round_number)
    _validate_summary(summary, circuits, round_number, name)
    return records, summary


def _load_nn_all_model(run_dir, dataset_root, seed):
    run_dir = Path(run_dir).resolve()
    summary = _read_json(run_dir / "train_summary.json", "NN-All training summary")
    if summary.get("format") != TRAINING_FORMAT:
        raise ValueError("Incompatible NN-All training summary")
    if summary.get("epochs") != TRAINING_EPOCHS or summary.get("checkpoint_policy") != "final_epoch":
        raise ValueError("NN-All training must use 30,000 epochs and final weights")
    if (summary.get("training_circuits") != list(PAPER_TRAIN_CIRCUITS)
            or summary.get("fault_selection") != "all_catalog_stuck_at"):
        raise ValueError("NN-All paper training circuit or fault scope mismatch")
    if summary.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("NN-All feature schema mismatch")
    if Path(summary["dataset_root"]).resolve() != dataset_root:
        raise ValueError("NN-All training dataset differs from validation dataset")
    if int(summary["seed"]) != int(seed):
        raise ValueError("NN-All training seed differs from validation seed")
    manifest = _read_json(run_dir / "train_manifest.json", "NN-All training manifest")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != summary.get("manifest_sha256"):
        raise ValueError("NN-All training manifest hash mismatch")
    if not manifest or len(manifest) != summary.get("circuits"):
        raise ValueError("NN-All training manifest circuit count mismatch")
    if any("split" in item for item in manifest):
        raise ValueError("NN-All training manifest must not contain circuit holdouts")
    if [item.get("circuit") for item in manifest] != list(PAPER_TRAIN_CIRCUITS):
        raise ValueError("NN-All training manifest must contain c1908 and c2670 only")
    for item in manifest:
        circuit = dataset_root / "nn_all_paper_train" / item["circuit"]
        if not circuit.is_file() or hashlib.sha256(circuit.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError(f"NN-All training circuit identity mismatch: {circuit}")
    model_path = run_dir / "model_final.txt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing NN-All model: {model_path}")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != summary.get("model_sha256"):
        raise ValueError("NN-All model SHA-256 mismatch")
    with model_path.open(encoding="utf-8") as stream:
        header = [stream.readline().strip() for _ in range(5)]
    if header != [MODEL_FORMAT, f"feature_schema {FEATURE_SCHEMA}",
                  "hidden_dim 60", f"manifest_sha256 {digest}",
                  f"seed {seed}"]:
        raise ValueError("NN-All model identity mismatch")
    return model_path, summary


def _evaluate_nn_all(circuits, model_path, output_dir, seed):
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError("Native NN-All validation requires cpp_podem") from error
    if not hasattr(cpp_podem, "run_native_nn_all_validation"):
        raise RuntimeError("cpp_podem is stale; rebuild with python -m pip install -e .")
    records = []
    for item in circuits:
        features = load_nn_all_features(item["circuit"])
        feature_path = output_dir / "features" / "nn_all" / f"{item['name']}.txt"
        write_native_features(feature_path, features)
        fault_ids = item["episode_fault_ids"]
        native = cpp_podem.run_native_nn_all_validation(
            _native_circuit_path(item["circuit"]),
            _native_circuit_path(feature_path), _native_circuit_path(model_path),
            BACKTRACK_LIMIT, seed, fault_ids, item["name"],
        )
        if len(native) != len(fault_ids):
            raise RuntimeError("Native NN-All validation returned the wrong fault count")
        for fault_id, raw in zip(fault_ids, native):
            if raw["fault_id"] != fault_id:
                raise RuntimeError("Native NN-All validation returned faults out of order")
            outcome = int(raw["outcome"])
            records.append({
                "circuit": item["name"], "fault_id": fault_id,
                "outcome": outcome, "detected": int(outcome == 1),
                "redundant": int(outcome == 0), "aborted": int(outcome not in (0, 1)),
                "backtracks": int(raw["backtracks"]),
                "backtrace_steps": int(raw["backtrace_steps"]),
                "return": float(raw["return"]),
                "test_vectors": int(outcome == 1),
                "atpg_seconds": float(raw["atpg_seconds"]),
            })
    _validate_records(records, "NN-All")
    summary = _summarize_validation(records, circuits, 0)
    _validate_summary(summary, circuits, 0, "NN-All")
    return records, summary


def run_fresh_validation(
    run_dir, dataset_root=None, output_dir=None, seed=2026,
    backtrace_lock="auto", nn_all_run_dir=None,
):
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
    nn_model = nn_training = None
    if nn_all_run_dir is not None:
        nn_model, nn_training = _load_nn_all_model(
            nn_all_run_dir, training_dataset_root, seed,
        )

    manifests = {key: Path(value) for key, value in run_summary["manifests"].items()}
    model_dirs = {key: Path(value) for key, value in run_summary["training_dirs"].items()}
    model_specs = {"gat": "level_gat_gru", "mean": "fanin_mean"}
    model_backtrace_locks = {
        name: resolve_backtrace_lock(backtrace_lock, encoder)
        for name, encoder in model_specs.items()
    }
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
                output_dir, round_number, seed, model_backtrace_locks[name],
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
        "backtrace_lock": backtrace_lock,
        "runtime_definition": "sum(per_fault.atpg_seconds)",
        "model_selection": selection,
        "rows": rows,
    }
    _atomic_json(output_dir / "model_selection.json", selection)
    _atomic_json(output_dir / "validation_three_way_comparison.json", result)
    _atomic_csv(output_dir / "validation_three_way_comparison.csv", rows)
    if nn_all_run_dir is not None:
        nn_records, nn_summary = _evaluate_nn_all(circuits, nn_model, output_dir, seed)
        _atomic_json_lines(detailed / "nn_all_podem_records.jsonl", nn_records)
        four_rows = [
            build_four_way_row(
                circuit, scoap_summary, model_results["gat"],
                model_results["mean"], nn_summary,
            ) for circuit in names
        ]
        four_result = {
            "format": "SMARTATPG_FOUR_WAY_VALIDATION_V1",
            "seed": seed,
            "backtrack_limit": BACKTRACK_LIMIT,
            "backtrace_lock": backtrace_lock,
            "runtime_definition": "sum(per_fault.atpg_seconds)",
            "model_selection": selection,
            "nn_all_training": {
                "run_dir": str(Path(nn_all_run_dir).resolve()),
                "manifest_sha256": nn_training["manifest_sha256"],
                "model_sha256": nn_training["model_sha256"],
                "epochs": nn_training["epochs"],
                "checkpoint_policy": nn_training["checkpoint_policy"],
                "feature_schema": FEATURE_SCHEMA,
            },
            "rows": four_rows,
        }
        _atomic_json(output_dir / "validation_four_way_comparison.json", four_result)
        _atomic_csv(output_dir / "validation_four_way_comparison.csv", four_rows)
        return four_result
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    nn_all_options = parser.add_mutually_exclusive_group()
    nn_all_options.add_argument(
        "--nn-all-run-dir", type=Path, default=DEFAULT_NN_ALL_RUN_DIR,
    )
    nn_all_options.add_argument("--three-way-only", action="store_true")
    parser.add_argument(
        "--backtrace-lock", choices=("auto", "on", "off"), default="auto",
    )
    args = parser.parse_args(argv)
    nn_all_run_dir = None if args.three_way_only else args.nn_all_run_dir
    run_fresh_validation(
        args.run_dir, args.dataset_root, args.output_dir, args.seed,
        args.backtrace_lock, nn_all_run_dir=nn_all_run_dir,
    )


if __name__ == "__main__":
    main()
