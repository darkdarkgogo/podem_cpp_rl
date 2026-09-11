import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_final_comparison import (
    METRICS,
    metric_values,
    models_for_metric,
    read_comparison_csv,
    warn_for_clipped_coverage,
)


CSV_COLUMNS = (
    "circuit",
    "model",
    "detected",
    "redundant_uncollapsed",
    "successful_faults",
    "total_faults",
    "equivalent_detected",
    "equivalent_faults",
    "aborted",
    "redundant",
    "backtracks",
    "backtrace_steps",
    "test_vectors",
    "rl_select_calls",
    "actor_forward_calls",
    "atpg_seconds",
    "rl_select_seconds",
    "actor_forward_seconds",
    "average_rl_select_microseconds",
    "average_actor_forward_microseconds",
    "fault_coverage",
)
MODELS = ("scoap_heuristic", "smartatpg_gat_gru")


def comparison_rows():
    rows = []
    for circuit_index, circuit in enumerate(("c17", "c432"), start=1):
        for model_index, model in enumerate(MODELS, start=1):
            rows.append({
                "circuit": circuit,
                "model": model,
                "detected": 95 + model_index,
                "redundant_uncollapsed": 3,
                "successful_faults": 98 + model_index,
                "total_faults": 100,
                "equivalent_detected": 0,
                "equivalent_faults": 0,
                "aborted": 2,
                "redundant": 3,
                "backtracks": circuit_index * 10 + model_index,
                "backtrace_steps": circuit_index * 20 + model_index,
                "test_vectors": 8,
                "rl_select_calls": 4,
                "actor_forward_calls": 3,
                "atpg_seconds": circuit_index + model_index / 10.0,
                "rl_select_seconds": 0.1,
                "actor_forward_seconds": 0.05,
                "average_rl_select_microseconds": 25000,
                "average_actor_forward_microseconds": 16666.7,
                "fault_coverage": (98 + model_index) / 100.0,
            })
    return rows


def write_comparison_csv(path, *, rows=None, fieldnames=CSV_COLUMNS):
    rows = comparison_rows() if rows is None else rows
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


class FinalComparisonPlotTests(unittest.TestCase):
    def test_relative_metrics_use_scoap_as_baseline_without_a_baseline_bar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "final_comparison.csv"
            write_comparison_csv(csv_path)
            circuits, models, records = read_comparison_csv(csv_path)

        self.assertEqual(circuits, ["c17", "c432"])
        self.assertEqual(models, list(MODELS))
        backtracks_metric = next(
            metric for metric in METRICS if metric["column"] == "backtracks"
        )
        self.assertEqual(
            models_for_metric(backtracks_metric),
            ("smartatpg_gat_gru",),
        )
        self.assertEqual(
            metric_values(
                circuits,
                "smartatpg_gat_gru",
                records,
                backtracks_metric,
            ),
            [12 / 11, 22 / 21],
        )

    def test_fault_coverage_keeps_both_models_as_percentages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "final_comparison.csv"
            write_comparison_csv(csv_path)
            circuits, _models, records = read_comparison_csv(csv_path)

        coverage_metric = next(
            metric for metric in METRICS
            if metric["column"] == "fault_coverage"
        )
        self.assertEqual(models_for_metric(coverage_metric), MODELS)
        self.assertEqual(coverage_metric["ylim"], (98.0, 100.0))
        self.assertEqual(
            metric_values(circuits, "smartatpg_gat_gru", records, coverage_metric),
            [100.0, 100.0],
        )

    def test_warns_when_coverage_is_clipped_below_98_percent(self):
        records = {
            ("c17", "scoap_heuristic"): {"fault_coverage": 0.97},
            ("c17", "smartatpg_gat_gru"): {"fault_coverage": 0.99},
        }
        with self.assertWarnsRegex(RuntimeWarning, "c17/scoap_heuristic=97.000%"):
            warn_for_clipped_coverage(records)

    def test_rejects_zero_scoap_denominator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "final_comparison.csv"
            rows = comparison_rows()
            baseline_row = next(
                row for row in rows
                if row["circuit"] == "c17"
                and row["model"] == "scoap_heuristic"
            )
            baseline_row["atpg_seconds"] = 0
            write_comparison_csv(csv_path, rows=rows)
            circuits, _models, records = read_comparison_csv(csv_path)
            runtime_metric = next(
                metric for metric in METRICS
                if metric["column"] == "atpg_seconds"
            )

            with self.assertRaisesRegex(
                ValueError,
                "Cannot normalize atpg_seconds for circuit 'c17'",
            ):
                metric_values(
                    circuits, "smartatpg_gat_gru", records, runtime_metric
                )

    def test_rejects_a_circuit_with_a_missing_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "final_comparison.csv"
            rows = [
                row for row in comparison_rows()
                if (row["circuit"], row["model"])
                != ("c432", "smartatpg_gat_gru")
            ]
            write_comparison_csv(csv_path, rows=rows)
            with self.assertRaisesRegex(
                ValueError,
                "Circuit 'c432' is missing models: smartatpg_gat_gru",
            ):
                read_comparison_csv(csv_path)

    def test_optional_models_do_not_change_the_plotted_series(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "final_comparison.csv"
            rows = comparison_rows()
            rows.append({**rows[0], "model": "partial_model"})
            rows.extend([
                {**rows[0], "model": "complete_model"},
                {**rows[3], "model": "complete_model"},
            ])
            write_comparison_csv(csv_path, rows=rows)
            _circuits, models, _records = read_comparison_csv(csv_path)

        self.assertEqual(models, [*MODELS, "complete_model"])
        for metric in METRICS:
            self.assertNotIn("complete_model", models_for_metric(metric))

    def test_rejects_missing_column_nonnumeric_value_and_duplicate_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cases = []

            missing_column_path = temp_path / "missing_column.csv"
            write_comparison_csv(
                missing_column_path,
                fieldnames=tuple(
                    column for column in CSV_COLUMNS
                    if column != "fault_coverage"
                ),
            )
            cases.append((
                missing_column_path,
                "missing required columns: fault_coverage",
            ))

            nonnumeric_path = temp_path / "nonnumeric.csv"
            nonnumeric_rows = comparison_rows()
            nonnumeric_rows[0]["backtracks"] = "not-a-number"
            write_comparison_csv(nonnumeric_path, rows=nonnumeric_rows)
            cases.append((nonnumeric_path, "non-numeric backtracks value"))

            duplicate_path = temp_path / "duplicate.csv"
            duplicate_rows = comparison_rows()
            duplicate_rows.append(dict(duplicate_rows[0]))
            write_comparison_csv(duplicate_path, rows=duplicate_rows)
            cases.append((duplicate_path, "Duplicate circuit/model pair"))

            for csv_path, message in cases:
                with self.subTest(csv_path=csv_path.name):
                    with self.assertRaisesRegex(ValueError, message):
                        read_comparison_csv(csv_path)

    def test_short_row_reports_concise_cli_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "short_row.csv"
            csv_path.write_text(
                "circuit,model,backtrace_steps,backtracks,atpg_seconds,"
                "fault_coverage\nc17\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "plot_final_comparison.py"),
                    str(csv_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("empty circuit or model value", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "Matplotlib is not installed",
    )
    def test_command_writes_four_png_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "final_comparison.csv"
            output_dir = temp_path / "plots"
            write_comparison_csv(csv_path)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "plot_final_comparison.py"),
                    str(csv_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output_paths = [
                output_dir / metric["filename"] for metric in METRICS
            ]

            self.assertEqual(len(output_paths), 4)
            self.assertEqual(
                {path.name for path in output_paths},
                {metric["filename"] for metric in METRICS},
            )
            for output_path in output_paths:
                self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertIn(f"WROTE {output_path}", result.stdout)


if __name__ == "__main__":
    unittest.main()
