"""Paired native check of the actual final PPO actor, separate from BC fallback."""

import json

import torch

import run_experiment as experiment


def main():
    state = torch.load(experiment.NEW_CHECKPOINT, map_location="cpu")
    if len(state["progress"]) != 150 or state["validation_history"][-1]["label"] != "hard_sweep_3":
        raise RuntimeError("Complete the original training budget before evaluating the final policy.")
    selected_report = experiment.RUN / "comparison.json"
    if not selected_report.exists():
        raise RuntimeError("Complete the initial selected-policy benchmark first.")
    directory = experiment.Path(json.loads(selected_report.read_text(encoding="utf-8"))["benchmark_directory"])
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not any(row["model"] == "gae_final" for row in rows):
        manifest = json.loads(experiment.MANIFEST.read_text(encoding="utf-8"))
        rows, directory = experiment.benchmark(manifest, 5, models={
            "heuristic": None,
            "old_v6": experiment.OLD_ACTOR,
            "gae_final": experiment.RUN / "actor_v2_latest.txt",
        })
    totals = {}
    for model in ("heuristic", "old_v6", "gae_final"):
        selected = [row for row in rows if row["model"] == model]
        totals[model] = {key: sum(row[key] for row in selected) for key in (
            "detected", "total_faults", "equivalent_detected", "equivalent_faults",
            "aborted", "backtracks", "backtrace_steps", "atpg_seconds", "wall_seconds")}
    result = {
        "benchmark_directory": str(directory), "totals": totals,
        "final_validation": state["validation_history"][-1],
        "note": "Actual final GAE actor, not the validation-selected BC fallback. Old V6 and the heuristic are remeasured in this same paired run.",
    }
    experiment.save_json(experiment.RUN / "final_policy_comparison.json", result)
    experiment.save_json(directory / "final_policy_comparison.json", result)
    lines = ["# Actual Final GAE Policy vs Old V6", "",
             result["note"],
             "Same circuits, seed 14, backtrack limit 500. One warmup plus five measured interleaved runs; elapsed times are medians.",
             "One training seed; same-circuit evaluation includes training/validation faults. This is not a GAE-only ablation.", "",
             "| Circuit | V6 detected | Final GAE detected | V6 abort | Final GAE abort | V6 ATPG s | Final GAE ATPG s | Time change |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in dict.fromkeys(row["circuit"] for row in rows):
        before = next(row for row in rows if row["circuit"] == name and row["model"] == "old_v6")
        after = next(row for row in rows if row["circuit"] == name and row["model"] == "gae_final")
        delta = (after["atpg_seconds"] / before["atpg_seconds"] - 1) * 100
        lines.append(f"| {name} | {before['detected']} | {after['detected']} | {before['aborted']} | {after['aborted']} | {before['atpg_seconds']:.4f} | {after['atpg_seconds']:.4f} | {delta:+.1f}% |")
    lines += ["", "Aggregate (sum of per-circuit medians for time):", "", "```json",
              json.dumps(totals, indent=2), "```", "",
              "Detected counts are uncollapsed weighted fault counts; aborted counts are collapsed fault attempts."]
    report = "\n".join(lines) + "\n"
    (experiment.RUN / "final_policy_comparison.md").write_text(report, encoding="utf-8")
    (directory / "final_policy_comparison.md").write_text(report, encoding="utf-8")
    print("FINAL_POLICY_EVALUATION_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
