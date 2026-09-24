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
