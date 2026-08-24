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

## Export and train

The Python RL tools are included in this project. Run them from `PODEM`:

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
