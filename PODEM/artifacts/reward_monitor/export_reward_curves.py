"""Reconstruct historical TensorBoard scalars; never modify training artifacts."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from collections import defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_curriculum import _stage_fault_ids

RUNS = {
    "DeepGate-GAE": ROOT / "artifacts/paper_v7_gae_20260831",
    "SmartATPG-GAE": ROOT / "artifacts/paper_v8_smartatpg",
}
COLORS = {"DeepGate-GAE": "#346c9c", "SmartATPG-GAE": "#c36623"}
STAGES = {"easy": 0, "medium": 1, "hard": 2}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_run(directory):
    manifest_path = directory / ("experiment_manifest.json" if directory.name == "paper_v8_smartatpg" else "training_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = directory / "training_state.pth"
    log_path = directory / "train.log"
    hashes = {str(p): sha(p) for p in (manifest_path, checkpoint_path, log_path)}
    state = torch.load(checkpoint_path, map_location="cpu")
    assert state["manifest_hash"] == sha(manifest_path)
    circuits = {item["name"]: item for item in manifest["circuits"]}
    difficulty = {(item["name"], f["fault_id"]): f["difficulty"]
                  for item in circuits.values() for f in item["training_faults"]}
    expected = set()
    for unit in state["progress"]:
        ids = _stage_fault_ids(circuits[unit["circuit"]], unit["stage"], unit["sweep"], state["config"]["seed"])[unit["difficulty"]]
        expected.update((unit["stage_name"], unit["sweep"], unit["circuit"], fault) for fault in ids)
    records = {}
    malformed = 0
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines()):
        if not line.startswith("ROLLOUT_RESULT "):
            continue
        try:
            record = json.loads(line.split(" ", 1)[1])
        except json.JSONDecodeError:
            malformed += 1
            continue
        key = (record["stage"], record["sweep"], record["circuit"], record["fault_id"])
        if key in expected:
            records[key] = (line_number, record)
    assert set(records) == expected
    rows = [record for _, record in sorted(records.values(), key=lambda pair: pair[0])]
    assert len(rows) == 4200
    assert sum(bool(r["updated"]) for r in rows) == state["agent"]["update_count"]
    assert sum(r["steps"] for r in rows) == sum(u["summary"]["decisions"] for u in state["progress"])
    decisions = 0
    for episode, row in enumerate(rows, 1):
        row["episode"] = episode
        decisions += row["steps"]
        row["cumulative_decisions"] = decisions
        row["difficulty"] = difficulty[(row["circuit"], row["fault_id"])]
        assert np.isfinite(row["extrinsic_reward_sum"])
        assert np.isclose(row["reward_sum"], row["extrinsic_reward_sum"] + row["scaled_intrinsic_reward_sum"])
    return rows, state, hashes, malformed


def mean(rows, key):
    return statistics.mean(r[key] for r in rows)


def group_summary(rows):
    updated = [r for r in rows if r["updated"]]
    return {"episodes": len(rows), "end_episode": max(r["episode"] for r in rows),
            "reward_mean": mean(rows, "extrinsic_reward_sum"),
            "reward_std": statistics.pstdev(r["extrinsic_reward_sum"] for r in rows),
            "detected": sum(r["outcome"] == 1 for r in rows),
            "entropy_mean": mean(updated, "entropy"),
            "critic_loss_mean": mean(updated, "value_loss")}


class Writer:
    def __init__(self, path, wall_time):
        self.writer = EventFileWriter(str(path))
        self.wall_time = wall_time

    def add(self, step, values):
        assert all(np.isfinite(v) for v in values.values())
        self.writer.add_event(Event(wall_time=self.wall_time, step=int(step),
            summary=Summary(value=[Summary.Value(tag=k, simple_value=float(v)) for k, v in values.items()])))

    def close(self):
        self.writer.flush()
        self.writer.close()


def export_run(name, rows, state, output, wall_time):
    writer = Writer(output / "tensorboard" / name, wall_time)
    reward_window, entropy_window, critic_window = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    local_windows = defaultdict(lambda: deque(maxlen=20))
    curve = []
    for row in rows:
        reward_window.append(row["extrinsic_reward_sum"])
        values = {
            "reward/episode_extrinsic_raw": row["extrinsic_reward_sum"],
            "reward/episode_combined_raw": row["reward_sum"],
            "reward/rnd_weighted_bonus": row["scaled_intrinsic_reward_sum"],
            "reward/episode_extrinsic_scaled": row["extrinsic_reward_sum"] / state["config"]["return_scale"],
            "rollout/decision_steps": row["steps"],
            "rollout/cumulative_decisions": row["cumulative_decisions"],
            "curriculum/stage_id": STAGES[row["stage"]],
        }
        if len(reward_window) == 100:
            value = statistics.mean(reward_window)
            values["reward/rolling100_mixed_tasks"] = value
            curve.append((row["episode"], value))
        local = local_windows[(row["circuit"], row["difficulty"])]
        local.append(row["extrinsic_reward_sum"])
        if len(local) == 20:
            values[f"per_circuit/{row['circuit']}/{row['difficulty']}_reward_ma20"] = statistics.mean(local)
        if row["updated"]:
            entropy_window.append(row["entropy"])
            critic_window.append(row["value_loss"])
            values.update({"diagnostics/policy_entropy": row["entropy"],
                           "diagnostics/critic_loss": row["value_loss"],
                           "diagnostics/actor_loss": row["policy_loss"],
                           "diagnostics/raw_advantage_std": row["raw_adv_std"]})
            if len(entropy_window) == 100:
                values["diagnostics/entropy_ma100_updates"] = statistics.mean(entropy_window)
                values["diagnostics/critic_loss_ma100_updates"] = statistics.mean(critic_window)
        writer.add(row["episode"], values)

    sweeps = []
    end_steps = {"behavior_cloning": 0}
    for stage, sweep in dict.fromkeys((r["stage"], r["sweep"]) for r in rows):
        selected = [r for r in rows if (r["stage"], r["sweep"]) == (stage, sweep)]
        summary = {"stage": stage, "sweep": sweep, **group_summary(selected)}
        sweeps.append(summary)
        end_steps[f"{stage}_sweep_{sweep}"] = summary["end_episode"]
        writer.add(summary["end_episode"], {"sweep/reward_mean_mixed_tasks": summary["reward_mean"],
                   "sweep/reward_std_mixed_tasks": summary["reward_std"],
                   "sweep/train_detection_fraction": summary["detected"] / len(selected)})

    fixed_hard = []
    cohort = None
    for sweep in (1, 2, 3):
        selected = [r for r in rows if r["stage"] == "hard" and r["sweep"] == sweep and r["difficulty"] == "hard"]
        keys = {(r["circuit"], r["fault_id"]) for r in selected}
        assert len(keys) == len(selected) == 200
        if cohort is None:
            cohort = keys
        assert keys == cohort
        summary = {"sweep": sweep, **group_summary(selected)}
        summary["by_circuit"] = {c: group_summary([r for r in selected if r["circuit"] == c])
                                  for c in dict.fromkeys(r["circuit"] for r in selected)}
        fixed_hard.append(summary)
        step = end_steps[f"hard_sweep_{sweep}"]
        writer.add(step, {"same_200_hard_train_faults/reward_mean": summary["reward_mean"],
                         "same_200_hard_train_faults/reward_std": summary["reward_std"],
                         "same_200_hard_train_faults/detected": summary["detected"]})
        writer.add(step, {f"same_hard_by_circuit/{c}/reward_mean": s["reward_mean"]
                          for c, s in summary["by_circuit"].items()})

    for validation in state["validation_history"]:
        writer.add(end_steps[validation["label"]], {
            "validation/detected_of_500": validation["aggregate"]["detected"],
            "validation/aborted_of_500": validation["aggregate"]["aborted"],
            "validation/backtrack_ratio_macro": validation["aggregate"]["mean_backtrack_ratio"],
            "validation/backtrace_ratio_macro": validation["aggregate"]["mean_backtrace_ratio"],
        })
    writer.close()
    reader = EventAccumulator(str(output / "tensorboard" / name), size_guidance={"scalars": 0})
    reader.Reload()
    saved = reader.Scalars("reward/episode_extrinsic_raw")
    assert [event.step for event in saved] == list(range(1, 4201))
    np.testing.assert_allclose([e.value for e in saved], [r["extrinsic_reward_sum"] for r in rows], rtol=1e-6, atol=1e-5)
    assert len(reader.Scalars("diagnostics/policy_entropy")) == 4001
    assert len(reader.Scalars("validation/detected_of_500")) == 8
    return {"sweeps": sweeps, "fixed_hard": fixed_hard, "moving_average": curve,
            "validation": [{"label": v["label"], "episode": end_steps[v["label"]], **v["aggregate"]}
                           for v in state["validation_history"]],
            "scalar_tags": len(reader.Tags()["scalars"]), "best_label": state["best_label"]}


def plots(summaries, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#faf9f5", "axes.facecolor": "#faf9f5"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for name, summary in summaries.items():
        color = COLORS[name]
        sweeps = summary["sweeps"]
        axes[0, 0].plot([s["end_episode"] for s in sweeps], [s["reward_mean"] for s in sweeps], "o-", color=color, label=name)
        cohort = summary["fixed_hard"]
        axes[0, 1].plot([1, 2, 3], [s["reward_mean"] for s in cohort], "o-", color=color, label=name)
        for index, row in enumerate(cohort, 1):
            axes[0, 1].annotate(f"{row['reward_mean']:.2f}", (index, row["reward_mean"]),
                                xytext=(0, 10 if name == "DeepGate-GAE" else -17), textcoords="offset points", ha="center", color=color)
        validation = summary["validation"]
        axes[1, 0].plot([v["episode"] for v in validation], [v["detected"] for v in validation], "o-", color=color, label=name)
        values = np.array(summary["moving_average"])
        axes[1, 1].plot(values[:, 0], values[:, 1], color=color, alpha=0.85, linewidth=1.2, label=name)
    axes[0, 0].set(title="Mean external return per sweep\nTask mix changes with curriculum", ylabel="Mean episode return", xlabel="Completed training faults")
    axes[0, 1].set(title="The SAME 200 hard training faults\nComparable cohort; still training, not validation", ylabel="Mean episode return", xlabel="Hard-stage sweep", xticks=[1, 2, 3])
    axes[1, 0].set(title="Fixed validation set: 500 faults\nValidation reward was not recorded", ylabel="Detected faults", xlabel="Completed training faults", ylim=(477, 500))
    axes[1, 1].set(title="External return: trailing 100-fault mean\nMixed circuits/difficulties; descriptive only", ylabel="Mean episode return", xlabel="Completed training faults")
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        for boundary in (800, 2400):
            ax.axvline(boundary, color="#9c9c94", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 4250)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Reward audit | SmartATPG vs DeepGate, GAE seed 2026", fontsize=17, fontweight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.025, "Raw external reward, before /100 scaling; RND excluded. Dashed lines: curriculum changes.\nHistorical reconstruction: x-axis is episode count, NOT elapsed training time. Only one seed; three hard sweeps do not establish convergence.", fontsize=9, color="#565b5d")
    fig.tight_layout(rect=(0.03, 0.08, 1, 0.93))
    fig.savefig(output / "reward_overview.png", dpi=160)
    plt.close(fig)

    names = sorted(summaries["SmartATPG-GAE"]["fixed_hard"][0]["by_circuit"])
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True)
    for ax, circuit in zip(axes.flat, names):
        for name, summary in summaries.items():
            ax.plot([1, 2, 3], [s["by_circuit"][circuit]["reward_mean"] for s in summary["fixed_hard"]], "o-", color=COLORS[name], label=name)
        ax.set(title=circuit, xticks=[1, 2, 3], xlabel="Hard sweep")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("Mean external return")
    axes[1, 0].set_ylabel("Mean external return")
    fig.suptitle("Same 20 hard training faults per circuit | no fault-by-fault plot required", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.025, "Each circuit uses the identical fault IDs across all three sweeps. These are exploratory training returns, not held-out fixed-policy evaluation.", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.92))
    fig.savefig(output / "reward_by_circuit.png", dpi=160)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    output = HOME / ("export_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    summaries, hashes, all_rows = {}, {}, []
    exported_at = time.time()
    for name, directory in RUNS.items():
        rows, state, source_hashes, malformed = read_run(directory)
        summaries[name] = export_run(name, rows, state, output, exported_at)
        hashes.update(source_hashes)
        all_rows.extend({"run": name, **row} for row in rows)
        print(f"EXPORTED {name} episodes={len(rows)} tags={summaries[name]['scalar_tags']} malformed_skipped={malformed}", flush=True)
    plots(summaries, output)
    csv_keys = ("run", "episode", "cumulative_decisions", "circuit", "fault_id", "stage", "sweep", "difficulty", "extrinsic_reward_sum", "reward_sum", "scaled_intrinsic_reward_sum", "steps", "outcome", "updated", "entropy", "value_loss")
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    save_json(output / "summary.json", {name: {k: v for k, v in s.items() if k != "moving_average"} for name, s in summaries.items()})
    assert all(sha(path) == digest for path, digest in hashes.items())
    save_json(output / "provenance.json", {"source_sha256": hashes, "sources_unchanged": True,
              "event_wall_time_is_export_time": True, "episode_axis": "one complete fault; includes 199 zero-decision episodes per run",
              "validation_reward_available": False, "tensorboard_url": "http://127.0.0.1:6006",
              "window_definitions": {"mixed_reward": 100, "per_circuit_difficulty": 20, "diagnostic_updates": 100},
              "same_hard_cohort": "identical 200 training faults across hard sweeps, not fixed-policy validation"})
    save_json(HOME / "latest_export.json", {"directory": str(output), "logdir": str(output / "tensorboard")})
    print("OUTPUT " + str(output), flush=True)


if __name__ == "__main__":
    main()
