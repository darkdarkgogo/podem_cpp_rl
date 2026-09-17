"""Compare two V8 SmartATPG validation runs against one SCOAP baseline."""

import argparse
import csv
import json
import math
from pathlib import Path

from train_smartatpg import (
    _atomic_json,
    _evaluate_fault,
    _load_validation_catalogs,
    _manifest_hash,
    _resolve_circuit_records,
    _summarize_validation,
    _validation_catalog_hash,
    _validation_order,
    validation_score,
)


COMPARISON_FORMAT = "SMARTATPG_DUAL_VALIDATION_COMPARISON_V1"
SCOAP_FORMAT = "SMARTATPG_SCOAP_VALIDATION_V1"
EXPECTED_ENCODERS = {
    "smartatpg_gat_gru": "level_gat_gru",
    "smartatpg_mean": "fanin_mean",
}
IDENTITY_KEYS = {
    "format",
    "manifest_hash",
    "encoder_variant",
    "normal_rounds",
    "faults_per_update",
    "k_epochs",
    "backtrack_limit",
    "validation_catalog_hash",
    "validation_circuits",
    "seed",
}
SHARED_IDENTITY_FIELDS = (
    "manifest_hash",
    "normal_rounds",
    "faults_per_update",
    "k_epochs",
    "backtrack_limit",
    "validation_catalog_hash",
    "validation_circuits",
    "seed",
)
RAW_METRICS = (
    "episodes",
    "detected_faults",
    "redundant_faults",
    "aborted_faults",
    "test_vectors",
    "backtracks_total",
    "backtracks_mean",
    "backtrace_steps_total",
    "backtrace_steps_mean",
    "return_total",
    "return_mean",
    "atpg_seconds",
    "fault_coverage",
)
REDUCTION_METRICS = (
    "backtracks_total",
    "backtrace_steps_total",
    "atpg_seconds",
)
ADDITIVE_METRICS = (
    "episodes",
    "detected_faults",
    "redundant_faults",
    "aborted_faults",
    "test_vectors",
    "backtracks_total",
    "backtrace_steps_total",
    "return_total",
    "atpg_seconds",
)
DERIVED_METRICS = {
    "fault_coverage": "detected_faults",
    "backtracks_mean": "backtracks_total",
    "backtrace_steps_mean": "backtrace_steps_total",
    "return_mean": "return_total",
}
NONNEGATIVE_INTEGER_METRICS = (
    "episodes",
    "detected_faults",
    "redundant_faults",
    "aborted_faults",
    "backtracks_total",
    "backtrace_steps_total",
    "test_vectors",
)
FINITE_NUMBER_METRICS = (
    "return_total",
    "fault_coverage",
    "backtracks_mean",
    "backtrace_steps_mean",
    "return_mean",
)


class ScoapValidationEvaluator:
    """Expose the native SCOAP heuristic through the learned evaluator protocol."""

    @staticmethod
    def decision_callback(request):
        return int(request["heuristic_action"])

    def run(
        self,
        circuit_path,
        *,
        backtrack_limit,
        seed,
        fault_ids,
        use_scoap,
        event_callback,
    ):
        try:
            import cpp_podem
        except ImportError as error:
            raise ImportError(
                "Cannot import cpp_podem. Install this project in the active "
                "environment with 'python -m pip install -e .'."
            ) from error
        return dict(cpp_podem.run_stuck_at(
            str(Path(circuit_path).resolve()),
            self.decision_callback,
            event_callback,
            backtrack_limit,
            seed,
            fault_ids,
            True,
            "backtrace_rl",
            "",
            True,
        ))


def _read_json(path, description):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {description}: {path}") from error


def _load_run(name, directory):
    directory = Path(directory)
    identity = _read_json(
        directory / "validation_identity.json", f"validation identity for {name}"
    )
    rounds = _read_json(
        directory / "validation_metrics.json", f"validation metrics for {name}"
    )
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise ValueError(f"Wrong V8 validation identity fields for {name}")
    if identity.get("format") != "SMARTATPG_VALIDATION_IDENTITY_V1":
        raise ValueError(f"Wrong validation identity format for {name}")
    if identity["encoder_variant"] != EXPECTED_ENCODERS[name]:
        raise ValueError(f"Wrong encoder variant for {name}")
    for key in ("normal_rounds", "faults_per_update", "k_epochs", "backtrack_limit", "seed"):
        if type(identity[key]) is not int:
            raise ValueError(f"Invalid V8 validation identity {key} for {name}")
    if (
        identity["normal_rounds"],
        identity["faults_per_update"],
        identity["k_epochs"],
        identity["backtrack_limit"],
    ) != (2, 8, 1, 200):
        raise ValueError(f"Wrong V8 training protocol for {name}")
    if (
        not isinstance(rounds, list)
        or any(not isinstance(item, dict) for item in rounds)
        or [item.get("round") for item in rounds] != [1, 2]
    ):
        raise ValueError(f"Incomplete validation rounds for {name}")
    circuit_order = identity["validation_circuits"]
    if not isinstance(circuit_order, list) or not circuit_order or any(
        not isinstance(item, str) or not item for item in circuit_order
    ) or len(circuit_order) != len(set(circuit_order)):
        raise ValueError(f"Invalid validation circuit order for {name}")
    for item in rounds:
        if [row.get("circuit") for row in item.get("circuits", [])] != circuit_order:
            raise ValueError(f"Validation circuit order mismatch for {name}")
        missing = [key for key in RAW_METRICS if key not in item]
        if missing or any(
            any(key not in row for key in RAW_METRICS)
            for row in item["circuits"]
        ):
            raise ValueError(f"Incomplete validation metrics for {name}")
    best = [item for item in rounds if item.get("is_best") is True]
    if len(best) != 1:
        raise ValueError(f"Expected exactly one best validation round for {name}")
    return identity, rounds


def _metrics_equal(actual, expected):
    if not _is_finite_number(actual) or not _is_finite_number(expected):
        return False
    return math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-9)


def _is_finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and (isinstance(value, int) or math.isfinite(value))
    )


def _validate_scope_metrics(name, round_number, scope, row):
    """Reject malformed raw and derived metrics before arithmetic checks."""
    for key in NONNEGATIVE_INTEGER_METRICS:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Invalid validation metrics for {name} round {round_number}: "
                f"{key} for {scope} must be a nonnegative integer"
            )
    for key in FINITE_NUMBER_METRICS:
        if not _is_finite_number(row[key]):
            raise ValueError(
                f"Invalid validation metrics for {name} round {round_number}: "
                f"{key} for {scope} must be a finite number"
            )
    if not _is_finite_number(row["atpg_seconds"]) or row["atpg_seconds"] < 0:
        raise ValueError(
            f"Invalid validation metrics for {name} round {round_number}: "
            f"atpg_seconds for {scope} must be a finite nonnegative number"
        )
    episodes = row["episodes"]
    if sum(row[key] for key in (
        "detected_faults", "redundant_faults", "aborted_faults"
    )) != episodes:
        raise ValueError(
            f"Inconsistent validation metrics for {name} round {round_number}: "
            f"outcome counts for {scope}"
        )
    if row["test_vectors"] != row["detected_faults"]:
        raise ValueError(
            f"Inconsistent validation metrics for {name} round {round_number}: "
            f"test vectors for {scope}"
        )
    divisor = max(1, episodes)
    for derived, total in DERIVED_METRICS.items():
        expected = row[total] / divisor
        if not _metrics_equal(row[derived], expected):
            raise ValueError(
                f"Inconsistent validation metrics for {name} round "
                f"{round_number}: {derived} for {scope}"
            )


def _validate_run_metrics(name, rounds, circuits):
    expected_total_episodes = len(_validation_order(circuits))
    expected_circuit_episodes = {
        item["name"]: len(item["episode_fault_ids"]) for item in circuits
    }
    for summary in rounds:
        round_number = summary["round"]
        _validate_scope_metrics(name, round_number, "TOTAL", summary)
        if summary["episodes"] != expected_total_episodes:
            raise ValueError(
                f"Incomplete validation metrics for {name} round {round_number}: "
                "total episodes do not cover the runtime fault catalog"
            )
        for row in summary["circuits"]:
            circuit = row["circuit"]
            _validate_scope_metrics(name, round_number, circuit, row)
            if row["episodes"] != expected_circuit_episodes[circuit]:
                raise ValueError(
                    f"Incomplete validation metrics for {name} round "
                    f"{round_number}: {circuit} episodes do not cover the "
                    "runtime fault catalog"
                )
        for key in ADDITIVE_METRICS:
            expected = sum(row[key] for row in summary["circuits"])
            if not _metrics_equal(summary[key], expected):
                raise ValueError(
                    f"Inconsistent validation metrics for {name} round "
                    f"{round_number}: TOTAL {key} does not equal circuit rows"
                )


def percentage_reduction(baseline, candidate):
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (baseline - candidate) / baseline * 100.0


def _scope_rows(summary):
    return [("total", "TOTAL", summary)] + [
        ("circuit", row["circuit"], row) for row in summary["circuits"]
    ]


def _raw_values(summary):
    return {key: summary[key] for key in RAW_METRICS}


def _comparison_row(model, round_number, scope, circuit, model_row, baseline,
                    score, best_round, seed):
    result = {
        "scope": scope,
        "circuit": circuit,
        "round": round_number,
        "model": model,
        "seed": seed,
        "validation_score": score,
        "best_round": best_round,
        "is_best": round_number == best_round,
        **_raw_values(model_row),
        **{f"scoap_{key}": baseline[key] for key in RAW_METRICS},
        "fault_coverage_delta": (
            model_row["fault_coverage"] - baseline["fault_coverage"]
        ),
    }
    for key in REDUCTION_METRICS:
        label = key.removesuffix("_total") + "_reduction_percent"
        result[label] = percentage_reduction(baseline[key], model_row[key])
    return result


def _direct_row(round_number, scope, circuit, gat_row, mean_row):
    result = {"scope": scope, "circuit": circuit, "round": round_number}
    for key in RAW_METRICS:
        result[f"gat_{key}"] = gat_row[key]
        result[f"mean_{key}"] = mean_row[key]
        result[f"{key}_gat_minus_mean"] = gat_row[key] - mean_row[key]
    return result


def _validate_summary(summary, circuits, round_number):
    if not isinstance(summary, dict) or summary.get("round") != round_number:
        raise ValueError("SCOAP validation summary has the wrong round")
    if [row.get("circuit") for row in summary.get("circuits", [])] != [
        item["name"] for item in circuits
    ]:
        raise ValueError("SCOAP validation summary has the wrong circuit order")
    if summary.get("episodes") != len(_validation_order(circuits)):
        raise ValueError("SCOAP validation summary is incomplete")
    for item, row in zip(circuits, summary["circuits"]):
        if row.get("episodes") != len(item["episode_fault_ids"]):
            raise ValueError("SCOAP validation circuit summary is incomplete")
        if any(key not in row for key in RAW_METRICS):
            raise ValueError("SCOAP validation circuit summary is incomplete")
    if any(key not in summary for key in RAW_METRICS):
        raise ValueError("SCOAP validation summary is incomplete")


def _scoap_identity(identity, seed):
    return {
        "manifest_hash": identity["manifest_hash"],
        "validation_catalog_hash": identity["validation_catalog_hash"],
        "validation_circuits": identity["validation_circuits"],
        "backtrack_limit": identity["backtrack_limit"],
        "seed": seed,
    }


def _load_or_run_scoap(output_dir, identity, circuits, seed):
    path = Path(output_dir) / "scoap_validation.json"
    expected_identity = _scoap_identity(identity, seed)
    if path.is_file():
        saved = _read_json(path, "SCOAP validation cache")
        if saved.get("format") != SCOAP_FORMAT:
            raise ValueError("SCOAP validation cache has the wrong format")
        if saved.get("identity") != expected_identity:
            raise ValueError("SCOAP validation cache identity mismatch")
        records = saved.get("records")
        if not isinstance(records, list):
            raise ValueError("SCOAP validation cache records are missing")
        recalculated = _summarize_validation(records, circuits, 0)
        if saved.get("summary") != recalculated:
            raise ValueError("SCOAP validation cache summary mismatch")
        _validate_summary(recalculated, circuits, 0)
        return saved

    evaluator = ScoapValidationEvaluator()
    by_name = {item["name"]: item for item in circuits}
    records = [
        _evaluate_fault(
            evaluator,
            by_name[circuit_name],
            fault_id,
            identity["backtrack_limit"],
            seed,
        )
        for circuit_name, fault_id in _validation_order(circuits)
    ]
    summary = _summarize_validation(records, circuits, 0)
    _validate_summary(summary, circuits, 0)
    payload = {
        "format": SCOAP_FORMAT,
        "identity": expected_identity,
        "records": records,
        "summary": summary,
    }
    _atomic_json(path, payload)
    return payload


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_validation_comparison(
    manifest_path, gat_dir, mean_dir, output_dir, seed=None
):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir)
    runs = {}
    for name, directory in (
        ("smartatpg_gat_gru", gat_dir),
        ("smartatpg_mean", mean_dir),
    ):
        identity, rounds = _load_run(name, directory)
        runs[name] = {"identity": identity, "rounds": rounds}

    gat_identity = runs["smartatpg_gat_gru"]["identity"]
    mean_identity = runs["smartatpg_mean"]["identity"]
    for key in SHARED_IDENTITY_FIELDS:
        if gat_identity[key] != mean_identity[key]:
            raise ValueError(f"Validation run identity mismatch: {key}")
    if seed is None:
        seed = gat_identity["seed"]
    elif type(seed) is not int or seed != gat_identity["seed"]:
        raise ValueError("SCOAP seed conflicts with validation run seed")
    if gat_identity["manifest_hash"] != _manifest_hash(manifest_path):
        raise ValueError("Validation run identity mismatch: manifest_hash")

    manifest = _read_json(manifest_path, "training manifest")
    _, circuits = _resolve_circuit_records(manifest, manifest_path)
    circuits = _load_validation_catalogs(manifest, circuits)
    if [item["name"] for item in circuits] != gat_identity["validation_circuits"]:
        raise ValueError("Validation run identity mismatch: validation_circuits")
    if _validation_catalog_hash(circuits) != gat_identity["validation_catalog_hash"]:
        raise ValueError("Validation run identity mismatch: validation_catalog_hash")
    for name, data in runs.items():
        _validate_run_metrics(name, data["rounds"], circuits)
        scores = [validation_score(item, item["round"]) for item in data["rounds"]]
        if data["rounds"][scores.index(min(scores))].get("is_best") is not True:
            raise ValueError(
                f"Best validation round disagrees with validation score for {name}"
            )

    baseline_payload = _load_or_run_scoap(
        output_dir, gat_identity, circuits, seed
    )
    baseline_summary = baseline_payload["summary"]
    baseline_rows = {
        (scope, circuit): row
        for scope, circuit, row in _scope_rows(baseline_summary)
    }

    comparisons = []
    csv_rows = []
    for model in ("smartatpg_gat_gru", "smartatpg_mean"):
        best_round = next(item["round"] for item in runs[model]["rounds"]
                          if item.get("is_best") is True)
        for round_summary in runs[model]["rounds"]:
            round_number = round_summary["round"]
            score = list(validation_score(round_summary, round_number))
            rows = [
                _comparison_row(
                    model,
                    round_number,
                    scope,
                    circuit,
                    row,
                    baseline_rows[(scope, circuit)],
                    score,
                    best_round,
                    seed,
                )
                for scope, circuit, row in _scope_rows(round_summary)
            ]
            comparisons.append({
                "model": model,
                "round": round_number,
                "rows": rows,
            })
            csv_rows.extend({"row_type": "model_vs_scoap", **row,
                             "validation_score": json.dumps(row["validation_score"])}
                            for row in rows)

    direct_comparisons = []
    gat_by_round = {
        item["round"]: item for item in runs["smartatpg_gat_gru"]["rounds"]
    }
    mean_by_round = {
        item["round"]: item for item in runs["smartatpg_mean"]["rounds"]
    }
    for round_number in (1, 2):
        gat_scopes = _scope_rows(gat_by_round[round_number])
        mean_scopes = _scope_rows(mean_by_round[round_number])
        if [(scope, circuit) for scope, circuit, _ in gat_scopes] != [
            (scope, circuit) for scope, circuit, _ in mean_scopes
        ]:
            raise ValueError("Model validation scopes do not match")
        rows = [
            _direct_row(round_number, scope, circuit, gat_row, mean_row)
            for (scope, circuit, gat_row), (_, _, mean_row)
            in zip(gat_scopes, mean_scopes)
        ]
        direct_comparisons.append({"round": round_number, "rows": rows})
        csv_rows.extend({"row_type": "gat_minus_mean", "seed": seed, **row} for row in rows)

    model_summaries = {}
    for name, data in runs.items():
        rounds = data["rounds"]
        best = next(item for item in rounds if item.get("is_best") is True)
        model_summaries[name] = {
            "encoder_variant": data["identity"]["encoder_variant"],
            "best_round": best["round"],
            "scores": [list(validation_score(item, item["round"]))
                       for item in rounds],
            "best_score": list(validation_score(best, best["round"])),
        }
    result = {
        "format": COMPARISON_FORMAT,
        "identity": {
            key: gat_identity[key] for key in SHARED_IDENTITY_FIELDS
        },
        "scoap": baseline_summary,
        "models": model_summaries,
        "comparisons": comparisons,
        "direct_comparisons": direct_comparisons,
    }
    _atomic_json(output_dir / "validation_comparison.json", result)
    _atomic_csv(output_dir / "validation_comparison.csv", csv_rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("gat_dir", type=Path)
    parser.add_argument("mean_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    build_validation_comparison(
        args.manifest, args.gat_dir, args.mean_dir, args.output_dir, args.seed
    )


if __name__ == "__main__":
    main()
