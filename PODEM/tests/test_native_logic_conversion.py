import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeLogicConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            os.name == "nt"
            and shutil.which("cl") is None
            and not os.environ.get("VCINSTALLDIR")
        ):
            raise unittest.SkipTest("An MSVC developer environment is required")
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if os.name != "nt" and compiler is None:
            raise unittest.SkipTest("g++ (or a GCC-compatible CXX) is required")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.executable = Path(cls.directory.name) / (
            "logic_conversion.exe" if os.name == "nt" else "logic_conversion"
        )
        sources = [
            str(path) for path in sorted((ROOT / "src").glob("*.cpp"))
            if path.name not in {"main.cpp", "python_bindings.cpp"}
        ]
        harness = str(ROOT / "tests/native_logic_conversion.cpp")
        if os.name == "nt":
            from setuptools._distutils.ccompiler import new_compiler

            native_compiler = new_compiler(compiler="msvc")
            objects = native_compiler.compile(
                [harness, *sources], output_dir=cls.directory.name,
                include_dirs=[str(ROOT / "src")],
                extra_postargs=["/std:c++14", "/EHsc", "/O2", "/we4715", "/we4716"],
            )
            native_compiler.link_executable(
                objects, str(cls.executable.with_suffix("")),
                extra_postargs=["/MANIFEST:NO"],
            )
            return
        command = [
            compiler, "-std=c++11", "-O2", "-Werror=return-type",
            "-I", str(ROOT / "src"), harness, *sources,
            "-o", str(cls.executable),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(f"Native test build failed:\n{result.stderr}")

    def run_conversion(self, operation, value):
        return subprocess.run(
            [str(self.executable), operation, str(value)],
            capture_output=True, text=True, timeout=10,
        )

    def test_logic_values_round_trip(self):
        for value, character in [(0, "0"), (1, "1"), (2, "U")]:
            with self.subTest(value=value):
                encoded = self.run_conversion("itoc", value)
                self.assertEqual(encoded.returncode, 0, encoded.stderr)
                self.assertEqual(encoded.stdout, character)
                decoded = self.run_conversion("ctoi", character)
                self.assertEqual(decoded.returncode, 0, decoded.stderr)
                self.assertEqual(decoded.stdout, str(value))

    def test_legacy_unknown_character(self):
        result = self.run_conversion("ctoi", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "2")

    def test_invalid_integer_fails_explicitly(self):
        for value in [-1, 3, 4, 99]:
            with self.subTest(value=value):
                result = self.run_conversion("itoc", value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"invalid logic value: {value}", result.stderr)

    def test_invalid_character_fails_explicitly(self):
        for value in ["?", "9", "x"]:
            with self.subTest(value=value):
                result = self.run_conversion("ctoi", value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid logic character", result.stderr)

    def test_unknown_gate_fails_explicitly(self):
        result = self.run_conversion("gate", "invalid_gate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized gate type: invalid_gate", result.stderr)

    def test_known_gate_is_unchanged(self):
        result = self.run_conversion("gate", "AND")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "3")

    def test_scoap_control_selection_uses_target_cc_value(self):
        circuit = Path(self.directory.name) / "scoap_control.bench"
        circuit.write_text(
            "INPUT(a)\nINPUT(b)\nINPUT(c)\nINPUT(d)\n"
            "x=AND(a,b)\nz=OR(c,d)\ny=AND(x,z)\nOUTPUT(y)\n",
            encoding="ascii",
        )
        expected = {
            "scoap-control-easy-0": "x",
            "scoap-control-easy-1": "z",
            "scoap-control-hard-0": "z",
            "scoap-control-hard-1": "x",
        }
        for operation, selected in expected.items():
            with self.subTest(operation=operation):
                result = self.run_conversion(operation, circuit)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.endswith(selected), result.stdout)

    def test_scoap_propagation_selects_minimum_co_and_ties_are_stable(self):
        circuit = Path(self.directory.name) / "scoap_propagation.bench"
        circuit.write_text(
            "INPUT(a)\nINPUT(b)\np=AND(a,b)\nq=OR(a,b)\n"
            "r=buf(q)\nOUTPUT(p)\nOUTPUT(r)\n",
            encoding="ascii",
        )
        result = self.run_conversion("scoap-propagation", circuit)
        self.assertEqual(result.returncode, 0, result.stderr)
        selected, first_co, second_co = result.stdout.rsplit("\n", 1)[-1].split(":")
        self.assertEqual(selected, "p")
        self.assertLess(int(first_co), int(second_co))

        tie = Path(self.directory.name) / "scoap_propagation_tie.bench"
        tie.write_text(
            "INPUT(a)\nINPUT(b)\np=AND(a,b)\nq=OR(a,b)\n"
            "OUTPUT(p)\nOUTPUT(q)\n",
            encoding="ascii",
        )
        selections = []
        for _ in range(2):
            tied = self.run_conversion("scoap-propagation", tie)
            self.assertEqual(tied.returncode, 0, tied.stderr)
            selected, first_co, second_co = tied.stdout.rsplit("\n", 1)[-1].split(":")
            self.assertEqual(first_co, second_co)
            selections.append(selected)
        self.assertEqual(selections[0], selections[1])

    def test_undetected_fault_report_overwrites_previous_run(self):
        circuit = Path(self.directory.name) / "overwrite.bench"
        circuit.write_text(
            "INPUT(a)\nINPUT(b)\ny = AND(a,b)\nOUTPUT(y)\n",
            encoding="utf-8",
        )
        report = Path(str(circuit) + ".uf")

        first = self.run_conversion("undetected-output", circuit)
        self.assertEqual(first.returncode, 0, first.stderr)
        first_report = report.read_text(encoding="utf-8")
        self.assertTrue(first_report)

        second = self.run_conversion("undetected-output", circuit)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(report.read_text(encoding="utf-8"), first_report)


if __name__ == "__main__":
    unittest.main()
