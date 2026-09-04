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
        if os.name == "nt" and shutil.which("cl") is None:
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


if __name__ == "__main__":
    unittest.main()
