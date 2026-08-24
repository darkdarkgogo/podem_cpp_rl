# DeepGate RecGNN Extractor

Standalone `.bench` encoder extracted from `DeepGate-main`.

## What It Does

- parses `.bench` netlists
- constructs graph structure and topological order
- reproduces core DeepGate-style gate feature generation
- runs `recgnn` to produce node embeddings
- pools node embeddings into a graph embedding

## Supported Gates

- `INPUT`
- `AND`
- `NAND`
- `OR`
- `NOR`
- `NOT`
- `XOR`
- `BUF` / `BUFF`
- `XNOR`

## Quick Start

```python
from deepgate_recgnn_extractor import encode_bench

result = encode_bench("path/to/circuit.bench")
print(result["node_embeddings"].shape)
print(result["graph_embedding"].shape)
```

## CLI

```bash
python -m deepgate_recgnn_extractor.cli --bench path/to/circuit.bench
```

## Training

Train a DeepGate-style bidirectional GRU encoder directly on `.bench` circuits with:

- gate types aligned to the C++ PODEM netlist representation
- a 64-dim trainable gate-type embedding used as the initial hidden state
- explicit node features: `gate one-hot + normalized level + normalized fanout`
- forward and backward message passing repeated for multiple rounds
- supervision from simulated gate probabilities and pairwise truth-table distances

```bash
python -m deepgate_recgnn_extractor.train \
  --bench_dir path/to/bench_dir \
  --save_path path/to/model.pth \
  --epochs 30 \
  --num_rounds 10
```

## Notes

- If no checkpoint is provided, the model runs with randomly initialized weights.
- The gate vocabulary follows the C++ PODEM workflow and includes `input_pin` and `output_pin`.
- The package returns both raw gate attributes and the model input tensor so downstream code can choose what to consume.
- A practical environment for this package is Python `3.10` or `3.11` with `torch` and `torch-geometric` installed.
