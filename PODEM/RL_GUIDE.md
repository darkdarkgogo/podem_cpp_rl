# RL-Guided PODEM

The C++ PODEM implementation is the ATPG engine for both training and deployment. Python only runs DeepGate preprocessing and PPO optimization.

## Build the native executable

From `PODEM/src`, build with the existing Makefile or compile the sources including `rl_policy.cpp` and `rl_atpg.cpp`.

Run the original heuristic mode without RL options:

```text
atpg circuit.bench
```

Run native RL inference after exporting artifacts:

```text
atpg -fault-map circuit_binary.faultmap -rl-emb circuit_binary.emb -rl-actor actor_v2.txt -rl-mode backtrace_rl circuit_binary.bench
```

The embedding file is circuit-specific. The executable verifies its FNV-1a hash against the exact `.bench` file. `-rl-emb` and `-rl-actor` must be supplied together.
V2 actors support `backtrace_rl` only. They produce both input logits in one
forward pass. Native C++ stores embeddings by numeric gate ID and lazily computes
the two logits on the first use of each `(gate, objective value)` pair. Later
uses compare the cached values directly. The PODEM hot path still applies the
unknown-input filtering and honors the path lock. V1 actors remain loadable and
retain the legacy propagation modes.

The `.faultmap` maps selected source fault sites and equivalent-fault weights
onto the transformed binary netlist. For comment-declared XOR cells implemented
as private NAND/NOT networks, only the logical XOR output `GO-SA0/GO-SA1` faults
are retained; private `W/Z/X/Y` implementation faults are excluded. Its circuit
hash is verified before ATPG starts.

## Set up the Python training environment

Activate the `d2l` Conda environment and run all commands from this `PODEM`
directory:

```bat
call C:\ProgramData\Anaconda3\Scripts\activate.bat d2l
set DISTUTILS_USE_SDK=1
set MSSdk=1
python -m pip install -r python-requirements.txt
python -m pip install -e .
```

The editable install builds the `cpp_podem` pybind11 extension and installs the
`rl_podem` training package. It also keeps the vendored DeepGate extractor
discoverable under `vendor/deepgate_recgnn_extractor`, so no manual
`PYTHONPATH` configuration or sibling Python project is required.

On Windows, run the install command from an x64 Visual Studio Developer Command
Prompt. Microsoft Visual C++ 14 or newer is required. A POSIX-thread
MinGW toolchain can alternatively be selected with `--compiler=mingw32`; the
`win32`-thread MinGW variant cannot compile pybind11 because its standard
library lacks `std::mutex`. The two environment variables above make
setuptools use the compiler environment already initialized by the Developer
Command Prompt, including newer Visual Studio releases.

## Prepare and train the paper-reward V3 workflow

Convert any individual combinational benchmark without modifying its source:

```text
python scripts/convert_binary_bench.py sample_circuits/c432.bench
```

The converter builds balanced two-input AND/OR/NAND/NOR trees, filters expanded
XOR implementation-only faults, checks 256 random vectors for PO equivalence,
and verifies the mapped fault catalog through C++. Regenerate training manifests
and teacher profiles whenever a fault map changes.

Prepare the paper-style `c6288` and full-scan `s38417` data. This command
generates binary netlists, GPU DeepGate embeddings, profiles faults with a
500-backtrack limit, and deterministically splits the top 150 hard faults into
100 training faults and 50 validation faults per circuit:

```text
python scripts/prepare_paper_training.py artifacts/paper_v3 --count 100 --validation-count 50 --backtrack-limit 500 --deepgate-checkpoint artifacts/formal/deepgate_best.pth --device cuda:0
```

Train a fresh object-only PPO+RND model with the SmartATPG paper reward:

```text
python scripts/train_paper_rnd.py artifacts/paper_v3/training_manifest.json artifacts/paper_v3/training_state_v3.pth artifacts/paper_v3/actor_v2_best.txt --sweeps 5 --backtrack-limit 500
```

The extrinsic reward is `-0.1` per non-PI backtrace step,
`10 - 7.5 * exp(0.07 * (backtracks + PI visits))` for a non-terminal PI,
`+100` for detection, and `-100` for an undetected or aborted fault. RND remains
a separately scaled intrinsic exploration bonus.

Validation uses deterministic `argmax` without PPO or RND updates. The script
writes `actor_v2_best.txt` for deployment and `actor_v2_latest.txt` for resume
and diagnostics. Model selection first maximizes detected faults, then minimizes
aborted faults, backtracks, gate-to-input backtrace steps, and decisions. V3
training checkpoints intentionally reject V1/V2 checkpoints; the older actors
and checkpoints remain untouched. The V3 manifest binds the source circuit,
binary circuit, fault map, and embedding file by SHA256. Checkpoints also bind
all PPO/RND hyperparameters, while each training record stores separate
extrinsic, intrinsic, scaled-intrinsic, PPO-loss, and RND-loss metrics.

Run the automated reward, event, checkpoint, validation, and Python/C++ parity
checks after rebuilding the pybind extension:

```text
python scripts/verify_paper_v3.py
```
