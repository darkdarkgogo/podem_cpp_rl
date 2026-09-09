import sys
import tempfile
import unittest
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

import cpp_podem


def native_path(path):
    resolved = Path(path).resolve()
    if os.name == "nt":
        try:
            relative = os.path.relpath(resolved, Path.cwd())
        except ValueError:
            pass
        else:
            if relative.isascii():
                return relative
    return str(resolved)


from convert_binary_bench import catalog_cpp_podem, convert_binary_bench


def profile_cpp_podem(circuit_path, backtrack_limit, seed, fault_map_path):
    return list(cpp_podem.profile_stuck_at(
        native_path(circuit_path), backtrack_limit, seed, native_path(fault_map_path)
    ))


def expanded_xor(extra_a_fanout: bool, extra_b_fanout: bool) -> str:
    extra_gates = []
    extra_outputs = []
    if extra_a_fanout:
        extra_gates.append("qa = NOT(a)")
        extra_outputs.append("OUTPUT(qa)")
    if extra_b_fanout:
        extra_gates.append("qb = NOT(b)")
        extra_outputs.append("OUTPUT(qb)")
    return "\n".join([
        "INPUT(a)",
        "INPUT(b)",
        "OUTPUT(G1)",
        *extra_outputs,
        "# G1 = XOR(a,b)",
        "W1 = NOT(a)",
        "Z1 = NOT(b)",
        "X1 = NAND(a,Z1)",
        "Y1 = NAND(b,W1)",
        "G1 = NAND(X1,Y1)",
        *extra_gates,
        "",
    ])


class ExpandedXorFaninFaultTests(unittest.TestCase):
    def convert(self, text: str):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        source = root / "source.bench"
        binary = root / "binary.bench"
        fault_map = root / "binary.faultmap"
        source.write_text(text, encoding="ascii")
        stats = convert_binary_bench(source, binary, fault_map)
        catalog = catalog_cpp_podem(binary, fault_map)
        return directory, binary, fault_map, stats, catalog

    def test_single_logical_load_keeps_inputs_collapsed(self):
        directory, _, fault_map, stats, catalog = self.convert(
            expanded_xor(False, False)
        )
        self.addCleanup(directory.cleanup)
        self.assertEqual(
            fault_map.read_text(encoding="utf-8").splitlines()[0],
            "SMARTATPG_FAULT_MAP_V2",
        )
        xor_input_faults = [
            fault for fault in catalog["faults"]
            if fault["fault_id"].startswith("G1:GI")
        ]
        self.assertEqual(xor_input_faults, [])
        self.assertEqual(stats["retained_xor_input_faults"], 0)

    def test_each_fanout_input_keeps_both_branch_faults(self):
        directory, binary, fault_map, stats, catalog = self.convert(
            expanded_xor(True, False)
        )
        self.addCleanup(directory.cleanup)
        xor_input_faults = {
            fault["fault_id"]: fault for fault in catalog["faults"]
            if fault["fault_id"].startswith("G1:GI")
        }
        self.assertEqual(
            set(xor_input_faults), {"G1:GI0:sa0", "G1:GI0:sa1"}
        )
        self.assertEqual(
            {fault["input_wire_name"] for fault in xor_input_faults.values()},
            {"a"},
        )
        self.assertEqual(stats["retained_xor_input_faults"], 2)
        profiles = {
            item["fault_id"]: item
            for item in profile_cpp_podem(binary, 97, 14, fault_map)
        }
        self.assertEqual(profiles["G1:GI0:sa0"]["outcome"], 1)
        self.assertEqual(profiles["G1:GI0:sa1"]["outcome"], 1)
        choose = lambda request: next(
            index for index, allowed in enumerate(request["action_mask"])
            if allowed
        )
        summary = cpp_podem.run_stuck_at(
            str(binary), choose, None, 97, 14, None, True,
            "backtrace_rl", str(fault_map),
        )
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["detected"], 3)
        self.assertEqual(summary["redundant"], 0)
        self.assertEqual(summary["aborted"], 0)

    def test_invalid_v3_logical_xor_markers_are_rejected(self):
        directory, binary, fault_map, _, _ = self.convert(
            expanded_xor(True, False)
        )
        self.addCleanup(directory.cleanup)
        original_lines = fault_map.read_text(encoding="utf-8").splitlines()
        logical_index = next(
            index for index, line in enumerate(original_lines)
            if line.startswith("fault G1:GI0:sa0 ")
        )
        regular_index = next(
            index for index, line in enumerate(original_lines)
            if line.startswith("fault ") and line.endswith(" 0 -1")
        )
        cases = (
            (logical_index, ("2", "0")),
            (logical_index, ("1", "2")),
            (regular_index, ("0", "0")),
        )
        for case, (line_index, marker) in enumerate(cases):
            with self.subTest(marker=marker):
                lines = list(original_lines)
                fields = lines[line_index].split()
                fields[-2:] = marker
                lines[line_index] = " ".join(fields)
                malformed = Path(directory.name) / f"malformed_{case}.faultmap"
                malformed.write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "Invalid V3 logical XOR marker"
                ):
                    catalog_cpp_podem(binary, malformed)

    def test_both_inputs_are_checked_independently(self):
        directory, _, _, stats, catalog = self.convert(
            expanded_xor(True, True)
        )
        self.addCleanup(directory.cleanup)
        ids = {
            fault["fault_id"] for fault in catalog["faults"]
            if fault["fault_id"].startswith("G1:GI")
        }
        self.assertEqual(ids, {
            "G1:GI0:sa0", "G1:GI0:sa1",
            "G1:GI1:sa0", "G1:GI1:sa1",
        })
        self.assertEqual(stats["retained_xor_input_faults"], 4)

    def test_catalog_matches_native_xor_collapsing_rule(self):
        for extra_a_fanout, extra_b_fanout in (
            (False, False), (True, False), (True, True)
        ):
            with self.subTest(
                extra_a_fanout=extra_a_fanout,
                extra_b_fanout=extra_b_fanout,
            ):
                directory, _, _, _, expanded_catalog = self.convert(
                    expanded_xor(extra_a_fanout, extra_b_fanout)
                )
                self.addCleanup(directory.cleanup)
                native = Path(directory.name) / "native.bench"
                extra_gates = []
                extra_outputs = []
                if extra_a_fanout:
                    extra_gates.append("qa = NOT(a)")
                    extra_outputs.append("OUTPUT(qa)")
                if extra_b_fanout:
                    extra_gates.append("qb = NOT(b)")
                    extra_outputs.append("OUTPUT(qb)")
                native.write_text("\n".join([
                    "INPUT(a)", "INPUT(b)", "OUTPUT(G1)",
                    *extra_outputs, "G1 = xor(a,b)", *extra_gates, "",
                ]), encoding="ascii")
                native_catalog = catalog_cpp_podem(native)

                def fault_signature(catalog):
                    return {
                        (fault["fault_id"], fault["eqv_fault_num"])
                        for fault in catalog["faults"]
                    }

                self.assertEqual(
                    fault_signature(expanded_catalog),
                    fault_signature(native_catalog),
                )
                self.assertEqual(
                    expanded_catalog["uncollapsed_total"],
                    native_catalog["uncollapsed_total"],
                )


if __name__ == "__main__":
    unittest.main()
