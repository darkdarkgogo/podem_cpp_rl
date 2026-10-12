"""Torch-free construction of SmartATPG three-way comparison rows."""


def percentage_reduction(baseline, candidate):
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (baseline - candidate) / baseline * 100.0


def _method_row(summary, circuit):
    if circuit == "TOTAL":
        return summary
    return next(item for item in summary["circuits"] if item["circuit"] == circuit)


def build_three_way_row(circuit, scoap, gat, mean):
    rows = {
        "scoap": _method_row(scoap, circuit),
        "gat": _method_row(gat, circuit),
        "mean": _method_row(mean, circuit),
    }
    result = {"circuit": circuit}
    for method, row in rows.items():
        result.update({
            f"{method}_backtracks": int(row["backtracks_total"]),
            f"{method}_backtrace_steps": int(row["backtrace_steps_total"]),
            f"{method}_runtime_atpg_s": float(row["atpg_seconds"]),
            f"{method}_fault_coverage_pct": float(row["fault_coverage"]) * 100.0,
        })
    for method in ("gat", "mean"):
        result.update({
            f"{method}_backtracks_reduction_pct_vs_scoap": percentage_reduction(
                rows["scoap"]["backtracks_total"], rows[method]["backtracks_total"],
            ),
            f"{method}_backtrace_reduction_pct_vs_scoap": percentage_reduction(
                rows["scoap"]["backtrace_steps_total"],
                rows[method]["backtrace_steps_total"],
            ),
            f"{method}_runtime_reduction_pct_vs_scoap": percentage_reduction(
                rows["scoap"]["atpg_seconds"], rows[method]["atpg_seconds"],
            ),
            f"{method}_fault_coverage_delta_pp_vs_scoap": (
                rows[method]["fault_coverage"] - rows["scoap"]["fault_coverage"]
            ) * 100.0,
        })
    return result


def build_four_way_row(circuit, scoap, gat, mean, nn_all):
    """Extend the established three-way table with the NN-All PODEM run."""
    result = build_three_way_row(circuit, scoap, gat, mean)
    baseline = _method_row(scoap, circuit)
    candidate = _method_row(nn_all, circuit)
    for method, row in (("scoap", baseline), ("gat", _method_row(gat, circuit)),
                        ("mean", _method_row(mean, circuit)), ("nn_all", candidate)):
        result[f"{method}_detected_faults"] = int(row["detected_faults"])
        result[f"{method}_redundant_faults"] = int(row["redundant_faults"])
        result[f"{method}_aborted_faults"] = int(row["aborted_faults"])
    result.update({
        "nn_all_backtracks": int(candidate["backtracks_total"]),
        "nn_all_backtrace_steps": int(candidate["backtrace_steps_total"]),
        "nn_all_runtime_atpg_s": float(candidate["atpg_seconds"]),
        "nn_all_fault_coverage_pct": float(candidate["fault_coverage"]) * 100.0,
        "nn_all_backtracks_reduction_pct_vs_scoap": percentage_reduction(
            baseline["backtracks_total"], candidate["backtracks_total"],
        ),
        "nn_all_backtrace_reduction_pct_vs_scoap": percentage_reduction(
            baseline["backtrace_steps_total"], candidate["backtrace_steps_total"],
        ),
        "nn_all_runtime_reduction_pct_vs_scoap": percentage_reduction(
            baseline["atpg_seconds"], candidate["atpg_seconds"],
        ),
        "nn_all_fault_coverage_delta_pp_vs_scoap": (
            candidate["fault_coverage"] - baseline["fault_coverage"]
        ) * 100.0,
    })
    return result
