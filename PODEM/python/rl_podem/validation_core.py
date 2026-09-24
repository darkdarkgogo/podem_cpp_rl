"""Torch-free validation primitives shared by SmartATPG evaluators."""

import hashlib
import json
import math

from .native_io import _native_circuit_path, catalog_cpp_podem
from .data_split import (
    BACKTRACK_LIMIT,
    LEGACY_MANIFEST_FORMAT,
    validation_fault_ids,
)
from .smartatpg_rewards import (
    BACKTRACK_MAX,
    GAT_REWARD_SCHEME,
    MEAN_REWARD_SCHEME,
    PAPER_REWARD,
    reward_scheme_for_encoder,
    smartatpg_backtrack_reward,
    smartatpg_pi_reward,
)


def _native_validation_batch(
    item, fault_ids, embedding_path, actor_path, records_path, seed,
    reward_scheme,
):
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError(
            "Native validation requires the rebuilt cpp_podem extension"
        ) from error
    if not hasattr(cpp_podem, "run_native_validation"):
        raise RuntimeError(
            "cpp_podem is stale; rebuild it with: python -m pip install -e ."
        )
    native_records = cpp_podem.run_native_validation(
        _native_circuit_path(item["circuit"]),
        _native_circuit_path(embedding_path),
        _native_circuit_path(actor_path),
        BACKTRACK_LIMIT, seed, fault_ids, reward_scheme,
        _native_circuit_path(records_path), item["name"],
    )
    if len(native_records) != len(fault_ids):
        raise RuntimeError("Native validation returned the wrong fault count")
    records = []
    for fault_id, raw in zip(fault_ids, native_records):
        if raw["fault_id"] != fault_id:
            raise RuntimeError("Native validation returned faults out of order")
        outcome = int(raw["outcome"])
        if not math.isfinite(float(raw["return"])):
            raise ValueError(
                f"Non-finite validation return for {item['name']} {fault_id}"
            )
        if not math.isfinite(float(raw["atpg_seconds"])):
            raise ValueError(
                f"Non-finite validation time for {item['name']} {fault_id}"
            )
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
    return records


def validation_score(summary, round_number):
    return (
        -int(summary["detected_faults"]),
        int(summary["backtracks_total"]),
        int(summary["backtrace_steps_total"]),
        -float(summary["return_total"]),
        int(round_number),
    )


def _catalog_fault_ids(circuit_path):
    return validation_fault_ids(catalog_cpp_podem(circuit_path), circuit_path)


def _load_validation_catalogs(manifest, circuits):
    if manifest.get("format") == LEGACY_MANIFEST_FORMAT:
        return circuits
    for index, item in enumerate(circuits, 1):
        item["episode_fault_ids"] = _catalog_fault_ids(item["circuit"])
        print(
            f"CATALOG split=validation index={index}/{len(circuits)} "
            f"circuit={item['name']} faults={len(item['episode_fault_ids'])}",
            flush=True,
        )
    return circuits


def _validation_catalog_hash(circuits):
    payload = [
        {"name": item["name"], "fault_ids": item["episode_fault_ids"]}
        for item in circuits
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_order(circuits):
    return [
        (item["name"], fault_id)
        for item in circuits
        for fault_id in item["episode_fault_ids"]
    ]


def _evaluate_fault(
    evaluator, item, fault_id, backtrack_limit, seed, reward_scheme,
):
    if backtrack_limit != BACKTRACK_MAX:
        raise ValueError(
            f"SmartATPG validation requires backtrack_limit={BACKTRACK_MAX}"
        )
    if reward_scheme not in (GAT_REWARD_SCHEME, MEAN_REWARD_SCHEME):
        raise ValueError(f"Unknown SmartATPG reward scheme: {reward_scheme}")
    agent = getattr(evaluator, "agent", None)
    if agent is not None:
        expected_scheme = reward_scheme_for_encoder(agent.encoder_variant)
        if reward_scheme != expected_scheme:
            raise ValueError(
                "SmartATPG reward scheme does not match evaluator encoder"
            )
    extrinsic_return = 0.0
    terminal = None
    backtrack_count = 0

    def event_callback(event):
        nonlocal extrinsic_return, terminal, backtrack_count
        if event["event"] == "backtrace_step":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if (
                decision_sequences is None
                or int(event["decision_sequence"]) in decision_sequences
            ):
                extrinsic_return += PAPER_REWARD["non_pi"]
        elif event["event"] == "backtrack":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if (
                reward_scheme == GAT_REWARD_SCHEME
                and (
                    decision_sequences is None
                    or int(event["decision_sequence"]) in decision_sequences
                )
            ):
                backtrack_count += 1
                extrinsic_return += smartatpg_backtrack_reward(backtrack_count)
        elif event["event"] == "pi_not_done":
            decision_sequences = getattr(evaluator, "decision_sequences", None)
            if (
                reward_scheme == MEAN_REWARD_SCHEME
                and (
                    decision_sequences is None
                    or int(event["decision_sequence"]) in decision_sequences
                )
            ):
                extrinsic_return += smartatpg_pi_reward(
                    int(event["backtracks"]),
                    int(event["pi_visits"]),
                    PAPER_REWARD["alpha"],
                    PAPER_REWARD["beta"],
                )
        elif event["event"] == "episode_end":
            if terminal is not None:
                raise RuntimeError("Validation fault produced multiple terminal events")
            terminal = dict(event)
            extrinsic_return += (
                PAPER_REWARD["detected"]
                if int(event["outcome"]) == 1
                else PAPER_REWARD["undetected"]
            )

    summary = evaluator.run(
        item["circuit"],
        backtrack_limit=backtrack_limit,
        seed=seed,
        fault_ids=[fault_id],
        use_scoap=True,
        event_callback=event_callback,
    )
    if int(summary["episodes"]) != 1 or terminal is None:
        raise RuntimeError("Validation fault did not produce exactly one episode")
    if terminal.get("fault_id") != fault_id:
        raise RuntimeError("Validation terminal fault does not match the request")
    outcome = int(terminal["outcome"])
    if not math.isfinite(extrinsic_return):
        raise ValueError(
            f"Non-finite validation return for {item['name']} {fault_id}"
        )
    atpg_seconds = float(summary["atpg_seconds"])
    if not math.isfinite(atpg_seconds):
        raise ValueError(
            f"Non-finite validation time for {item['name']} {fault_id}"
        )
    return {
        "circuit": item["name"],
        "fault_id": fault_id,
        "outcome": outcome,
        "detected": int(outcome == 1),
        "redundant": int(outcome == 0),
        "aborted": int(outcome not in (0, 1)),
        "backtracks": int(terminal["backtracks"]),
        "backtrace_steps": int(terminal["backtrace_steps"]),
        "return": float(extrinsic_return),
        "test_vectors": int(outcome == 1),
        "atpg_seconds": atpg_seconds,
    }


def _summarize_fault_records(records):
    for item in records:
        if not math.isfinite(float(item["return"])):
            raise ValueError(
                "Non-finite validation return for "
                f"{item.get('circuit', '<unknown>')} "
                f"{item.get('fault_id', '<unknown>')}"
            )
        if not math.isfinite(float(item["atpg_seconds"])):
            raise ValueError(
                "Non-finite validation time for "
                f"{item.get('circuit', '<unknown>')} "
                f"{item.get('fault_id', '<unknown>')}"
            )
    count = len(records)
    totals = {
        "episodes": count,
        "detected_faults": sum(int(item["detected"]) for item in records),
        "redundant_faults": sum(int(item["redundant"]) for item in records),
        "aborted_faults": sum(int(item["aborted"]) for item in records),
        "backtracks_total": sum(int(item["backtracks"]) for item in records),
        "backtrace_steps_total": sum(
            int(item["backtrace_steps"]) for item in records
        ),
        "return_total": sum(float(item["return"]) for item in records),
        "test_vectors": sum(int(item["test_vectors"]) for item in records),
        "atpg_seconds": sum(float(item["atpg_seconds"]) for item in records),
    }
    divisor = max(1, count)
    totals.update(
        fault_coverage=totals["detected_faults"] / divisor,
        backtracks_mean=totals["backtracks_total"] / divisor,
        backtrace_steps_mean=totals["backtrace_steps_total"] / divisor,
        return_mean=totals["return_total"] / divisor,
    )
    if not all(math.isfinite(float(totals[key])) for key in (
        "return_total", "return_mean",
    )):
        raise ValueError("Non-finite validation return summary")
    return totals


def _summarize_validation(records, circuits, round_number):
    expected = _validation_order(circuits)
    actual = [(item.get("circuit"), item.get("fault_id")) for item in records]
    if actual != expected:
        raise ValueError("Validation must cover the full fault catalog exactly once")
    totals = _summarize_fault_records(records)
    per_circuit = [
        {
            "circuit": item["name"],
            **_summarize_fault_records([
                record for record in records if record["circuit"] == item["name"]
            ]),
        }
        for item in circuits
    ]
    return {"round": round_number, **totals, "circuits": per_circuit}
