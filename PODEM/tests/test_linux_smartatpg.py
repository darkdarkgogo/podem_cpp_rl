import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_smartatpg import _stage_circuit_copy, _summarize, percentage_change
from rl_podem.artifact_paths import matches_artifact_hash
from rl_podem.backends import MANIFEST_V5, smartatpg_metadata
from run_smartatpg_linux import main as run_linux_main, relocate_manifest
from train_curriculum import _validate_manifest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LinuxManifestTests(unittest.TestCase):
    def test_launcher_enables_tensorboard_in_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            source_manifest = root / "source.json"
            with (
                patch("run_smartatpg_linux.sys.platform", "linux"),
                patch("run_smartatpg_linux._check_cpp_extension"),
                patch("run_smartatpg_linux.relocate_manifest"),
                patch("run_smartatpg_linux._tee_command", return_value=0) as tee,
            ):
                run_linux_main([
                    "--source-manifest", str(source_manifest),
                    "--output-dir", str(output),
                    "--skip-benchmark",
                ])
            command = tee.call_args.args[0]
            option_index = command.index("--tensorboard-log-dir")
            self.assertEqual(
                Path(command[option_index + 1]),
                (output / "tensorboard").resolve(),
            )

    def test_json_hash_accepts_only_lf_crlf_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "artifact.json"
            crlf = b'{\r\n  "value": 1\r\n}\r\n'
            expected = hashlib.sha256(crlf).hexdigest()
            json_path.write_bytes(crlf.replace(b"\r\n", b"\n"))
            self.assertTrue(matches_artifact_hash(json_path, expected))

            json_path.write_bytes(b'{\n  "value": 2\n}\n')
            self.assertFalse(matches_artifact_hash(json_path, expected))

            circuit_path = root / "artifact.bench"
            circuit_path.write_bytes(crlf.replace(b"\r\n", b"\n"))
            self.assertFalse(matches_artifact_hash(circuit_path, expected))

    def test_training_validator_accepts_portable_json_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            circuit = root / "test.bench"
            fault_map = root / "test.faultmap"
            profile = root / "profile.json"
            teacher_training = root / "teacher_training.json"
            teacher_validation = root / "teacher_validation.json"
            circuit.write_bytes(b"INPUT(a)\n")
            fault_map.write_bytes(b"fault map\n")

            json_crlf = b'{\r\n  "value": 1\r\n}\r\n'
            json_lf = json_crlf.replace(b"\r\n", b"\n")
            for path in (profile, teacher_training, teacher_validation):
                path.write_bytes(json_lf)
            portable_hash = hashlib.sha256(json_crlf).hexdigest()

            manifest = {
                "format": MANIFEST_V5,
                "backtrack_limit": 500,
                **smartatpg_metadata(),
                "circuits": [{
                    "name": "test",
                    "circuit": str(circuit),
                    "fault_map": str(fault_map),
                    "profile": str(profile),
                    "artifact_sha256": {
                        "circuit": sha256(circuit),
                        "fault_map": sha256(fault_map),
                        "profile": portable_hash,
                    },
                    "training_faults": [{
                        "fault_id": "train", "difficulty": "easy",
                        "outcome": 1, "backtracks": 0, "backtrace_steps": 1,
                    }],
                    "validation_faults": [{
                        "fault_id": "validation", "difficulty": "easy",
                        "outcome": 1, "backtracks": 0, "backtrace_steps": 1,
                    }],
                }],
                "teacher_training": str(teacher_training),
                "teacher_validation": str(teacher_validation),
                "teacher_sha256": {
                    "training": portable_hash,
                    "validation": portable_hash,
                },
            }
            self.assertEqual(_validate_manifest(manifest)[0]["name"], "test")

            profile.write_bytes(b'{\n  "value": 2\n}\n')
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                _validate_manifest(manifest)

    def test_windows_paths_are_relocated_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            data = root / "data"
            data.mkdir(parents=True)
            files = {}
            for name in (
                "teacher_training.json",
                "teacher_validation.json",
                "test.bench",
                "test.faultmap",
                "test_profile.json",
            ):
                path = data / name
                path.write_text(name, encoding="utf-8")
                files[name] = path
            teacher_crlf = b'{\r\n  "teacher": true\r\n}\r\n'
            files["teacher_training.json"].write_bytes(teacher_crlf)
            windows = lambda name: f"C:\\old\\PODEM\\data\\{name}"
            manifest = {
                "teacher_training": windows("teacher_training.json"),
                "teacher_validation": windows("teacher_validation.json"),
                "teacher_sha256": {
                    "training": sha256(files["teacher_training.json"]),
                    "validation": sha256(files["teacher_validation.json"]),
                },
                "circuits": [{
                    "name": "test",
                    "circuit": windows("test.bench"),
                    "fault_map": windows("test.faultmap"),
                    "profile": windows("test_profile.json"),
                    "artifact_sha256": {
                        "circuit": sha256(files["test.bench"]),
                        "fault_map": sha256(files["test.faultmap"]),
                        "profile": sha256(files["test_profile.json"]),
                    },
                }],
            }
            source = root / "source.json"
            output = root / "run" / "manifest.json"
            source.write_text(json.dumps(manifest), encoding="utf-8")
            files["teacher_training.json"].write_bytes(
                teacher_crlf.replace(b"\r\n", b"\n")
            )
            relocated = relocate_manifest(source, output, root)
            self.assertEqual(
                Path(relocated["circuits"][0]["circuit"]),
                files["test.bench"].resolve(),
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), relocated)

            files["test.bench"].write_text("changed", encoding="utf-8")
            output.unlink()
            with self.assertRaisesRegex(ValueError, "hash changed"):
                relocate_manifest(source, output, root)


class BenchmarkSummaryTests(unittest.TestCase):
    def test_benchmark_stages_an_immutable_circuit_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "test.bench"
            source.parent.mkdir()
            source.write_text("INPUT(a)\n", encoding="utf-8")
            item = {
                "name": "test",
                "circuit": str(source),
                "artifact_sha256": {"circuit": sha256(source)},
            }
            staged = _stage_circuit_copy(item, root / "results")
            self.assertEqual(staged.read_bytes(), source.read_bytes())
            staged.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Staged benchmark circuit"):
                _stage_circuit_copy(item, root / "results")

    def test_percentage_change(self):
        self.assertEqual(percentage_change(100, 80), -20.0)
        self.assertEqual(percentage_change(0, 0), 0.0)
        self.assertIsNone(percentage_change(0, 1))

    def test_summary_uses_timing_medians_and_reports_reduction(self):
        models = ("heuristic", "rl_best", "rl_final")
        records = []
        for model, backtracks, seconds in (
            ("heuristic", 100, (2.0, 4.0)),
            ("rl_best", 90, (1.5, 2.5)),
            ("rl_final", 80, (1.0, 2.0)),
        ):
            for repeat, atpg_seconds in enumerate(seconds, 1):
                records.append({
                    "repeat": repeat,
                    "circuit": "test",
                    "model": model,
                    "detected": 10,
                    "total_faults": 10,
                    "equivalent_detected": 5,
                    "equivalent_faults": 5,
                    "aborted": 0,
                    "redundant": 0,
                    "backtracks": backtracks,
                    "backtrace_steps": backtracks * 10,
                    "test_vectors": 2,
                    "atpg_seconds": atpg_seconds,
                    "native_total_seconds": atpg_seconds + 0.1,
                    "wall_seconds": atpg_seconds + 0.2,
                })
        _, totals, comparisons = _summarize(
            records, {"circuits": [{"name": "test"}]}, models, 2
        )
        self.assertEqual(totals["heuristic"]["atpg_seconds"], 3.0)
        self.assertEqual(
            comparisons["rl_final"]["backtracks"]["reduction_percent"], 20.0
        )


if __name__ == "__main__":
    unittest.main()
