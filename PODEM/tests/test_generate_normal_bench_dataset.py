from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from generate_normal_bench_dataset import (
    BenchCircuit,
    Gate,
    SourceCircuit,
    extract_cone,
    normalize_bench,
    parse_blif,
    parse_bench,
    structural_hash,
    write_bench,
)


class NormalBenchDatasetTests(unittest.TestCase):
    def test_blif_names_lowering_uses_normal_logic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "logic.blif"
            source.write_text(
                ".model logic\n.inputs a b c\n.outputs y z\n"
                ".names a b y\n11 1\n.names c z\n0 1\n.end\n",
                encoding="utf-8",
            )
            circuit = normalize_bench(parse_blif(source))
        self.assertEqual(circuit.outputs, ("y", "z"))
        self.assertEqual(circuit.gates[0], Gate("y", "AND", ("a", "b")))
        self.assertEqual(circuit.gates[-1], Gate("z", "NOT", ("c",)))
        self.assertTrue(all(gate.kind in {"AND", "NAND", "OR", "NOR", "NOT"} for gate in circuit.gates))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "nor.blif"
            source.write_text(
                ".model nor\n.inputs a b\n.outputs y\n.names a b y\n00 1\n.end\n",
                encoding="utf-8",
            )
            circuit = normalize_bench(parse_blif(source))
        self.assertEqual(circuit.gates, (Gate("y", "NOR", ("a", "b")),))

    def test_normalization_scan_cuts_and_lowers_unsupported_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bench"
            source.write_text(
                "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(z)\n"
                "d = AND(a,b,c)\nq = DFF(d)\nx = XOR(q,c)\nz = EQV(x,a)\n",
                encoding="utf-8",
            )
            circuit = normalize_bench(parse_bench(source))
            destination = Path(directory) / "normal.bench"
            write_bench(circuit, destination)

        self.assertIn("q", circuit.inputs)
        self.assertIn("d", circuit.outputs)
        self.assertIn("z", circuit.outputs)
        self.assertTrue(all(gate.kind in {"AND", "NAND", "OR", "NOR", "NOT"} for gate in circuit.gates))
        self.assertTrue(all(len(gate.inputs) == (1 if gate.kind == "NOT" else 2) for gate in circuit.gates))

    def test_normalization_lowers_explicit_buf_without_changing_output_name(self):
        circuit = BenchCircuit(
            ("a",), ("y",), (Gate("y", "BUF", ("a",)),)
        )
        normalized = normalize_bench(circuit)
        self.assertEqual(normalized.outputs, ("y",))
        self.assertEqual([gate.kind for gate in normalized.gates], ["NOT", "NOT"])
        self.assertEqual(normalized.gates[-1].output, "y")

    def test_extraction_bounds_cut_boundary_and_deduplicate_structure(self):
        gates = []
        inputs = ["a", "b"]
        previous = "a"
        for index in range(100):
            output = f"g{index}"
            gates.append(Gate(output, "AND", (previous, "b")))
            previous = output
        source = SourceCircuit(
            "ISCAS89", "synthetic", Path("synthetic.bench"),
            BenchCircuit(tuple(inputs), (previous,), tuple(gates)), "0" * 64,
        )
        extracted = extract_cone(source, previous, min_nodes=20, max_nodes=50, max_levels=25)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertLessEqual(len(extracted.circuit.inputs) + len(extracted.circuit.gates), 50)
        self.assertGreaterEqual(len(extracted.circuit.inputs) + len(extracted.circuit.gates), 20)
        self.assertEqual(extracted.logic_depth, 25)
        self.assertTrue(set(extracted.circuit.inputs) & {"g74", "b"})

        repeated_pi = BenchCircuit(("a",), ("z",), (Gate("z", "AND", ("a", "a")),))
        separate_pi = BenchCircuit(("a", "b"), ("z",), (Gate("z", "AND", ("a", "b")),))
        self.assertNotEqual(structural_hash(repeated_pi), structural_hash(separate_pi))


if __name__ == "__main__":
    unittest.main()
