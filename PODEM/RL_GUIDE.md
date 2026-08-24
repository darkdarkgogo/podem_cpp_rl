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
atpg -rl-emb circuit.emb -rl-actor actor.txt circuit.bench
```

The embedding file is circuit-specific. The executable verifies its FNV-1a hash against the exact `.bench` file. `-rl-emb` and `-rl-actor` must be supplied together.

## Build the Python training bridge

Install `python-requirements.txt`, then build from the `PODEM` directory:

```text
python setup.py build_ext --inplace
```

On Windows this requires Microsoft Visual C++ 14 or newer. A POSIX-thread MinGW toolchain can also be selected with `--compiler=mingw32`; the `win32`-thread MinGW variant cannot compile pybind11 because its standard library lacks `std::mutex`.

Add the `PODEM` directory containing the built `cpp_podem` module to `PYTHONPATH` before starting training.

## Export and train

Run these commands from `smartestATPG-main` after installing that package and its PyTorch/DeepGate dependencies:

```text
python scripts/export_cpp_embeddings.py CIRCUIT.bench DEEPGATE_CHECKPOINT CIRCUIT.emb
python scripts/train_cpp_podem.py CIRCUIT.bench CIRCUIT.emb PPO_CHECKPOINT ACTOR.txt --passes 1
```

Training calls the C++ PODEM engine through pybind11. At each multi-candidate backtrace or propagation decision, Python samples an action from PPO. C++ remains responsible for objective generation, implication, fault propagation, backtracking, and test detection.

To export an existing compatible PPO checkpoint without training:

```text
python scripts/export_cpp_actor.py PPO_CHECKPOINT ACTOR.txt
```

The native actor uses deterministic `argmax` selection. Python training uses categorical sampling.
