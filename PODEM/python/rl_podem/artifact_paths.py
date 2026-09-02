"""Validate generated file namespaces before any checkpoint/export writes."""

import hashlib
from pathlib import Path


def matches_artifact_hash(path, expected):
    """Match exact bytes, allowing only LF/CRLF conversion for JSON files."""
    path = Path(path)
    if not path.is_file():
        return False
    data = path.read_bytes()
    expected = str(expected).lower()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    if path.suffix.lower() != ".json":
        return False

    lf_data = data.replace(b"\r\n", b"\n")
    crlf_data = lf_data.replace(b"\n", b"\r\n")
    return expected in {
        hashlib.sha256(lf_data).hexdigest(),
        hashlib.sha256(crlf_data).hexdigest(),
    }


def validate_output_paths(outputs, inputs=(), directories=()):
    resolved = [Path(path).resolve() for path in outputs]
    protected = {Path(path).resolve() for path in inputs}
    roots = [Path(path).resolve() for path in directories]
    if len(set(resolved)) != len(resolved) or set(resolved) & protected:
        raise ValueError("Output/checkpoint/sidecar paths collide with each other or an input artifact")
    for path in resolved:
        if any(other in path.parents for other in resolved if other != path):
            raise ValueError("An output file would also be used as an output directory")
    for root in roots:
        if any(path == root or root in path.parents or path in root.parents
               for path in resolved + list(protected)):
            raise ValueError("Snapshot output directory collides with a checkpoint or input artifact")


def training_output_paths(checkpoint, best, latest, backend, inputs=()):
    checkpoint, best, latest = map(Path, (checkpoint, best, latest))
    bc = checkpoint.with_suffix(checkpoint.suffix + ".bc")
    files = [checkpoint, bc, best, latest]
    directories = []
    if backend == "smartatpg":
        for actor in (best, latest):
            files.append(actor.with_suffix(actor.suffix + ".json"))
            directories.append(actor.parent / (actor.stem + "_snapshots"))
        if directories[0].resolve() == directories[1].resolve():
            raise ValueError("Best and latest snapshot directories must differ")
    all_outputs = files + [path.with_suffix(path.suffix + ".tmp") for path in files]
    validate_output_paths(all_outputs, inputs, directories)
