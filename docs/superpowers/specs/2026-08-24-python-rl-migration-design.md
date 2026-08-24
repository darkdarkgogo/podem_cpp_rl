# Python RL Migration Design

## Goal

Make `PoDemFan_N-detect_ATPG_Test_Compression/PODEM` self-contained for RL training and native inference. Training uses the existing C++ PODEM engine through pybind11 and Python/PyTorch PPO. Deployment continues to use the native C++ executable, exported embeddings, and exported actor parameters.

The source files in `smartestATPG-main` remain unchanged after migration.

## Scope

Copy only the Python components required by the C++ PODEM workflow:

- PPO actor, critic, rollout buffer, and optimizer.
- Python-to-C++ PODEM training bridge.
- DeepGate embedding export bridge.
- Training and actor/embedding export command-line scripts.
- The vendored `deepgate_recgnn_extractor` package.
- Python dependency and usage documentation.

Do not copy the Python PODEM implementation or its circuit, gate, and D-algebra classes. The C++ implementation remains the only PODEM engine in this workflow.

## Layout

The migrated files will use this structure:

```text
PODEM/
  python/rl_podem/
    __init__.py
    cpp_bridge.py
    deepgate_bridge.py
    ppo.py
  scripts/
    export_cpp_actor.py
    export_cpp_embeddings.py
    train_cpp_podem.py
  vendor/deepgate_recgnn_extractor/
  setup.py
  python-requirements.txt
  RL_GUIDE.md
```

Python module names will no longer reference `PodemQuest`, because the migrated package is specifically the training layer for the C++ PODEM implementation.

## Packaging

The existing `PODEM/setup.py` will continue to build the `cpp_podem` pybind11 extension and will additionally install the `rl_podem` Python package from `PODEM/python`.

The supported setup command will be:

```text
python -m pip install -e .
```

This makes both `cpp_podem` and `rl_podem` importable in the active Conda environment without manually setting `PYTHONPATH`. The vendored DeepGate extractor will be discovered relative to the installed source tree during editable development.

## Data Flow

Embedding export loads a trained DeepGate checkpoint and converts one `.bench` circuit into the existing versioned `.emb` format. Training loads that embedding table, calls C++ PODEM through `cpp_podem`, receives decision and episode callbacks, updates PPO in PyTorch, saves the PyTorch checkpoint, and exports the actor tensors for native C++ inference.

Native inference does not import Python or PyTorch. It loads the `.emb` and actor text artifacts through the existing C++ implementation.

## Error Handling

The migrated tools will keep the current checks for missing C++ extensions, mismatched circuit hashes, missing embeddings, malformed artifacts, and incompatible checkpoints. DeepGate discovery errors will report the expected vendor location inside `PODEM` rather than referring to a sibling project.

## Verification

Verification will cover:

- Importing `cpp_podem` and `rl_podem` from the `d2l` Conda environment.
- Running Python syntax/import checks for all migrated modules and scripts.
- Building the pybind11 extension with MSVC.
- Running the existing tiny C++/Python RL smoke circuit when the compiler is available.
- Confirming that exported actor and embedding formats remain accepted by native C++ tests.
- Confirming that `smartestATPG-main` has no deletions caused by this migration.
