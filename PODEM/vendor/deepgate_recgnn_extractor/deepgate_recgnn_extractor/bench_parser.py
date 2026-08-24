from dataclasses import dataclass


SUPPORTED_GATES = {"AND", "NAND", "OR", "NOR", "NOT", "XOR", "BUF", "BUFF", "XNOR"}
GATE_NAME_NORMALIZATION = {}


@dataclass
class ParsedBench:
    primary_inputs: list
    primary_outputs: list
    gates: list


def _clean_line(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_bench(bench_path: str) -> ParsedBench:
    primary_inputs = []
    primary_outputs = []
    gates = []

    with open(bench_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = _clean_line(raw_line)
            if not line:
                continue

            if line.startswith("INPUT(") and line.endswith(")"):
                primary_inputs.append(line[6:-1].strip())
                continue

            if line.startswith("OUTPUT(") and line.endswith(")"):
                primary_outputs.append(line[7:-1].strip())
                continue

            if "=" not in line:
                raise ValueError(f"Malformed .bench line {line_no}: {raw_line.rstrip()}")

            lhs, rhs = [part.strip() for part in line.split("=", 1)]
            if "(" not in rhs or not rhs.endswith(")"):
                raise ValueError(f"Malformed gate definition on line {line_no}: {raw_line.rstrip()}")

            gate_type = rhs.split("(", 1)[0].strip()
            gate_type = GATE_NAME_NORMALIZATION.get(gate_type, gate_type)
            if gate_type not in SUPPORTED_GATES:
                raise ValueError(f"Unsupported gate type '{gate_type}' on line {line_no}")

            fanins = [token.strip() for token in rhs.split("(", 1)[1][:-1].split(",") if token.strip()]
            gates.append(
                {
                    "name": lhs,
                    "gate_type": gate_type,
                    "fanins": fanins,
                    "line_no": line_no,
                }
            )

    return ParsedBench(
        primary_inputs=primary_inputs,
        primary_outputs=primary_outputs,
        gates=gates,
    )

