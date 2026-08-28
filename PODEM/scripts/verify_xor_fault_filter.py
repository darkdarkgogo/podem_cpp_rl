import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from convert_binary_bench import (  # noqa: E402
    SYNTHETIC_PREFIX,
    _expanded_xor_cells,
    _parse_bench,
    convert_binary_bench,
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _fault_map(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    header = {}
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] in {"count", "uncollapsed_total"}:
            header[fields[0]] = int(fields[1])
        elif fields[0] == "fault":
            records.append(
                {
                    "fault_id": fields[1],
                    "node_name": fields[2],
                    "input_wire_name": fields[3],
                    "io": int(fields[4]),
                    "input_occurrence": int(fields[5]),
                    "fault_type": int(fields[6]),
                    "eqv_fault_num": int(fields[7]),
                }
            )
    _check(header["count"] == len(records), f"Bad fault count in {path.name}.")
    _check(
        header["uncollapsed_total"]
        == sum(record["eqv_fault_num"] for record in records),
        f"Bad uncollapsed total in {path.name}.",
    )
    return header, records


def _verify_circuit(temp_dir, name, expected):
    source = ROOT / "sample_circuits" / f"{name}.bench"
    destination = temp_dir / f"{name}_binary.bench"
    fault_map = temp_dir / f"{name}_binary.faultmap"
    stats = convert_binary_bench(source, destination, fault_map)
    _check(
        destination.read_bytes()
        == (ROOT / "sample_circuits" / f"{name}_binary.bench").read_bytes(),
        f"{name} conversion unexpectedly changed the binary netlist.",
    )
    for key, value in expected.items():
        _check(stats[key] == value, f"{name} {key}: {stats[key]} != {value}.")

    _, ports, gates = _parse_bench(source)
    cells = _expanded_xor_cells(source, ports, gates)
    _, records = _fault_map(fault_map)
    _check(
        fault_map.read_bytes()
        == (ROOT / "sample_circuits" / f"{name}_binary.faultmap").read_bytes(),
        f"{name} checked-in fault map is stale.",
    )
    private_outputs = {
        output for cell in cells for output in cell.private_outputs
    }
    _check(
        not any(record["node_name"] in private_outputs for record in records),
        f"{name} retained a private expanded-XOR fault.",
    )
    _check(
        not any(
            (record["io"] == 1 and record["node_name"].startswith(SYNTHETIC_PREFIX))
            or record["input_wire_name"].startswith(SYNTHETIC_PREFIX)
            for record in records
        ),
        f"{name} assigned a fault to a binary-conversion internal wire.",
    )
    for cell in cells:
        output_faults = [
            record
            for record in records
            if record["node_name"] == cell.output and record["io"] == 1
        ]
        _check(
            len(output_faults) == 2
            and {record["fault_type"] for record in output_faults} == {0, 1}
            and all(record["eqv_fault_num"] == 1 for record in output_faults),
            f"{name} XOR output {cell.output} does not have exact SA0/SA1 faults.",
        )
    return fault_map


def _verify_malformed_expansion(temp_dir):
    malformed = temp_dir / "malformed_xor.bench"
    malformed.write_text(
        "INPUT(a)\n"
        "INPUT(b)\n"
        "OUTPUT(G1)\n"
        "# G1 = XOR(a,b)\n"
        "W1 = NOT(a)\n"
        "Z1 = NOT(b)\n"
        "X1 = NAND(a,Z1)\n"
        "Y1 = NAND(b,W1)\n"
        "G1 = NAND(X1,a)\n",
        encoding="ascii",
    )
    malformed_declaration = temp_dir / "malformed_xor_declaration.bench"
    malformed_declaration.write_text(
        "INPUT(a)\nOUTPUT(z)\n# G1 = XOR(a)\nz = BUF(a)\n",
        encoding="ascii",
    )
    trailing_comment = temp_dir / "trailing_xor_comment.bench"
    trailing_comment.write_text(
        "INPUT(a)\nOUTPUT(z)\n# G1 = XOR(a,a) # unexpected\nz = BUF(a)\n",
        encoding="ascii",
    )
    for path, expected_message in (
        (malformed, "does not match"),
        (malformed_declaration, "Malformed expanded XOR declaration"),
        (trailing_comment, "Malformed expanded XOR declaration"),
    ):
        _, ports, gates = _parse_bench(path)
        try:
            _expanded_xor_cells(path, ports, gates)
        except ValueError as error:
            _check(
                expected_message in str(error),
                f"Malformed XOR in {path.name} failed ambiguously.",
            )
        else:
            raise AssertionError(f"Malformed expanded XOR in {path.name} was accepted.")


def _expanded_xor_text(downstream):
    extra_input = "INPUT(c)\n" if "c" in downstream else ""
    return (
        "INPUT(a)\n"
        "INPUT(b)\n"
        f"{extra_input}"
        "OUTPUT(z)\n"
        "# G1 = XOR(a,b)\n"
        "W1 = NOT(a)\n"
        "Z1 = NOT(b)\n"
        "X1 = NAND(a,Z1)\n"
        "Y1 = NAND(b,W1)\n"
        "G1 = NAND(X1,Y1)\n"
        f"{downstream}\n"
    )


def _verify_collapsed_output_boundaries(temp_dir):
    cases = (
        ("xor_to_or", "z = OR(G1,c)", {0: 1, 1: 2}),
        ("xor_to_not", "z = NOT(G1)", {0: 1, 1: 1}),
    )
    for name, downstream, expected_downstream_weights in cases:
        source = temp_dir / f"{name}.bench"
        destination = temp_dir / f"{name}_binary.bench"
        fault_map = temp_dir / f"{name}_binary.faultmap"
        source.write_text(_expanded_xor_text(downstream), encoding="ascii")
        convert_binary_bench(source, destination, fault_map)
        _, records = _fault_map(fault_map)
        xor_faults = [
            record
            for record in records
            if record["node_name"] == "G1" and record["io"] == 1
        ]
        _check(
            len(xor_faults) == 2
            and {record["fault_type"] for record in xor_faults} == {0, 1}
            and all(record["eqv_fault_num"] == 1 for record in xor_faults),
            f"{name} failed to materialize exact XOR output faults.",
        )
        downstream_weights = {
            record["fault_type"]: record["eqv_fault_num"]
            for record in records
            if record["node_name"] == "z" and record["io"] == 1
        }
        _check(
            downstream_weights == expected_downstream_weights,
            f"{name} retained XOR-internal equivalent-fault weight: "
            f"{downstream_weights}.",
        )


def main():
    with tempfile.TemporaryDirectory(prefix="smartatpg_xor_faults_") as directory:
        temp_dir = Path(directory)
        _verify_circuit(
            temp_dir,
            "c432",
            {
                "expanded_xor_cells": 18,
                "removed_xor_faults": 108,
                "added_xor_output_faults": 9,
                "collapsed_faults": 461,
                "uncollapsed_faults": 792,
            },
        )
        _verify_circuit(
            temp_dir,
            "c499",
            {
                "expanded_xor_cells": 104,
                "removed_xor_faults": 624,
                "added_xor_output_faults": 0,
                "collapsed_faults": 534,
                "uncollapsed_faults": 774,
            },
        )
        c6288_map = _verify_circuit(
            temp_dir,
            "c6288",
            {
                "expanded_xor_cells": 0,
                "removed_xor_faults": 0,
                "added_xor_output_faults": 0,
                "collapsed_faults": 7744,
                "uncollapsed_faults": 12576,
            },
        )
        _check(
            c6288_map.read_bytes()
            == (ROOT / "sample_circuits" / "c6288_binary.faultmap").read_bytes(),
            "c6288 control fault map changed.",
        )
        _verify_malformed_expansion(temp_dir)
        _verify_collapsed_output_boundaries(temp_dir)
    print("XOR fault-filter verification passed.")


if __name__ == "__main__":
    main()
