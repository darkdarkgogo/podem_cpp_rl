"""PyTorch-independent helpers for invoking the native PODEM extension."""

import os
from pathlib import Path


def _native_circuit_path(path):
    resolved = Path(path).resolve()
    if os.name != "nt":
        return str(resolved)
    try:
        # The legacy C++ reader uses narrow paths. A relative ASCII path avoids
        # losing Unicode characters from the absolute Windows workspace path.
        relative = os.path.relpath(resolved, Path.cwd())
    except ValueError:
        return str(resolved)
    return relative if relative.isascii() else str(resolved)


def catalog_cpp_podem(circuit_path, fault_map_path=None):
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError(
            "Cannot import cpp_podem. Install this project in the active environment with "
            "'python -m pip install -e .'."
        ) from error
    return dict(cpp_podem.catalog_stuck_at(
        _native_circuit_path(circuit_path),
        _native_circuit_path(fault_map_path) if fault_map_path else "",
    ))
