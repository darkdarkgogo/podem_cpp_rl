"""Prepare encoder-specific training data and shared validation data."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from smartatpg_portable import (
    ACTION_MASK_DIM,
    FEATURE_SCHEMA,
    GATE_EMBEDDING_DIM,
    GRAPH_CONFIG,
    GAT_GRU_GRAPH_CONFIG,
    load_graph,
)


LEGACY_MANIFEST_FORMAT = "SMARTATPG_DATA_SPLIT_MANIFEST_V6_TOP30_11D_CO_NO_BUF"
LAZY_VALIDATION_MANIFEST_FORMAT = (
    "SMARTATPG_DATA_SPLIT_MANIFEST_V7_LAZY_VALIDATION_CATALOG_11D_CO_NO_BUF"
)
MANIFEST_FORMAT = (
    "SMARTATPG_DATA_SPLIT_MANIFEST_V9_ENCODER_TRAIN_SPLIT_11D_CO_NO_BUF"
)
PREPARATION_STATE_FORMAT = "SMARTATPG_DATA_SPLIT_PREPARATION_V3"
PROFILE_FORMAT = "SMARTATPG_HEURISTIC_FAULT_PROFILE_V1"
FAULT_FILTER = "train_top30_hard_detected_validation_full_catalog"
MEAN_FAULT_FILTER = "train_top100_hard_detected_validation_full_catalog"
BACKTRACK_LIMIT = 100
LEGACY_TRAINING_ROUNDS = 5
NORMAL_TRAINING_ROUNDS = 2
FAULTS_PER_UPDATE = 8
TRAINING_HYPERPARAMETERS = {
    "level_gat_gru": {
        "k_epochs": 4,
        "actor_lr": 0.0003,
        "critic_lr": 0.001,
    },
    "fanin_mean": {
        "k_epochs": 1,
        "actor_lr": 0.001,
        "critic_lr": 0.01,
    },
}
TRAIN_FAULTS_PER_CIRCUIT = 30
MEAN_TRAIN_FAULTS_PER_CIRCUIT = 100
HEURISTIC = "scoap_heuristic"
TRAIN_CIRCUIT_COUNT = 1024
VALIDATION_NAMES = ("b12_C", "b15_C", "b17_C", "b20_C", "b21_C", "b22_C")
ENCODER_VARIANTS = ("level_gat_gru", "fanin_mean")
MEAN_REQUIRED_ASSETS = (
    "c6288.bench",
    "s38417.bench",
    "s38417_scan.bench",
    "s38417_scan_binary.bench",
    "s38417_scan_binary.faultmap",
)
ROOT = Path(__file__).resolve().parents[1]


def training_hyperparameters(encoder_variant):
    try:
        return dict(TRAINING_HYPERPARAMETERS[encoder_variant])
    except KeyError as error:
        raise ValueError(
            f"Unsupported SmartATPG encoder: {encoder_variant}"
        ) from error


def smartatpg_metadata(encoder_variant="level_gat_gru"):
    if encoder_variant not in ENCODER_VARIANTS:
        raise ValueError(f"Unsupported SmartATPG encoder: {encoder_variant}")
    if encoder_variant == "level_gat_gru":
        actor_input_dim = GATE_EMBEDDING_DIM + 1
        graph_config = {
            "input_dim": GATE_EMBEDDING_DIM,
            "hidden_dim": GATE_EMBEDDING_DIM,
            "attention_heads": 1,
            "schedule": "forward_levels_then_reverse_levels",
            "directions": "independent",
        }
        graph_config_id = GAT_GRU_GRAPH_CONFIG
    else:
        actor_input_dim = GATE_EMBEDDING_DIM
        graph_config = {
            "layers": 1,
            "input_dim": GATE_EMBEDDING_DIM,
            "output_dim": GATE_EMBEDDING_DIM,
            "aggregation": "fanin_mean",
        }
        graph_config_id = GRAPH_CONFIG
    return {
        "embedding_backend": "smartatpg",
        "encoder_variant": encoder_variant,
        "feature_schema": FEATURE_SCHEMA,
        "graph_config": graph_config,
        "graph_config_id": graph_config_id,
        "gate_embedding_dim": GATE_EMBEDDING_DIM,
        "actor_input_dim": actor_input_dim,
        "action_mask_dim": ACTION_MASK_DIM,
        "decision_state_dim": actor_input_dim + ACTION_MASK_DIM,
        "policy_state_dim": actor_input_dim + ACTION_MASK_DIM,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _relative_path(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def resolve_manifest_path(manifest_path, value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Data-split SmartATPG manifests must use relative paths")
    return (Path(manifest_path).resolve().parent / path).resolve()


def discover_dataset(
    dataset_root, *, expected_train_count=TRAIN_CIRCUIT_COUNT,
    expected_validation_names=VALIDATION_NAMES,
):
    dataset_root = Path(dataset_root).resolve()
    train_dir = dataset_root / "train"
    validation_dir = dataset_root / "validation"
    for split, directory in (("train", train_dir), ("validation", validation_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing SmartATPG {split} directory: {directory}")
    entries = {
        "train": tuple(train_dir.iterdir()),
        "validation": tuple(validation_dir.iterdir()),
    }
    train_paths = tuple(sorted(
        (path for path in entries["train"] if path.is_file() and path.suffix == ".bench"),
        key=lambda path: path.name,
    ))
    validation_paths = tuple(sorted(
        (
            path for path in entries["validation"]
            if path.is_file() and path.suffix == ".bench"
        ),
        key=lambda path: path.name,
    ))
    if len(train_paths) != expected_train_count:
        raise ValueError(
            f"SmartATPG train split must contain exactly {expected_train_count} "
            f"BENCH files, found {len(train_paths)}"
        )
    validation_names = tuple(path.stem for path in validation_paths)
    if validation_names != tuple(expected_validation_names):
        raise ValueError(
            "SmartATPG validation split must contain exactly: "
            + ", ".join(expected_validation_names)
        )
    all_paths = (*train_paths, *validation_paths)
    stems = [path.stem for path in all_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("SmartATPG dataset contains duplicate circuit names")
    for split, directory in (("train", train_dir), ("validation", validation_dir)):
        unexpected = sorted(
            path.name for path in entries[split]
            if not path.is_file() or path.suffix != ".bench"
        )
        if unexpected:
            raise ValueError(
                f"SmartATPG {split} directory contains non-BENCH files: "
                + ", ".join(unexpected)
            )
    return {"train": train_paths, "validation": validation_paths}


def discover_mean_dataset(
    dataset_root, *, expected_validation_names=VALIDATION_NAMES,
):
    dataset_root = Path(dataset_root).resolve()
    train_dir = dataset_root / "train_mean"
    validation_dir = dataset_root / "validation"
    for split, directory in (("train_mean", train_dir), ("validation", validation_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing SmartATPG {split} directory: {directory}")
    assets = tuple(train_dir / name for name in MEAN_REQUIRED_ASSETS)
    missing = [path.name for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing SmartATPG train_mean assets: " + ", ".join(missing)
        )
    validation_paths = tuple(sorted(
        (path for path in validation_dir.iterdir()
         if path.is_file() and path.suffix == ".bench"),
        key=lambda path: path.name,
    ))
    if tuple(path.stem for path in validation_paths) != tuple(expected_validation_names):
        raise ValueError(
            "SmartATPG validation split must contain exactly: "
            + ", ".join(expected_validation_names)
        )
    unexpected_validation = sorted(
        path.name for path in validation_dir.iterdir()
        if not path.is_file() or path.suffix != ".bench"
    )
    if unexpected_validation:
        raise ValueError(
            "SmartATPG validation directory contains non-BENCH files: "
            + ", ".join(unexpected_validation)
        )
    return {
        "train": (
            {"name": "c6288", "circuit": train_dir / "c6288.bench",
             "fault_map": None},
            {"name": "s38417", "circuit": train_dir / "s38417_scan_binary.bench",
             "fault_map": train_dir / "s38417_scan_binary.faultmap"},
        ),
        "validation": validation_paths,
        "assets": assets,
    }


def select_training_faults(
    profiles, *, limit=TRAIN_FAULTS_PER_CIRCUIT, circuit_name=None,
):
    if limit <= 0:
        raise ValueError("Training fault limit must be positive")
    detected = [dict(item) for item in profiles if int(item["outcome"]) == 1]
    if not detected:
        location = f" for train circuit {circuit_name}" if circuit_name else ""
        raise RuntimeError(
            f"No heuristic-detected faults are available for training{location}"
        )
    detected.sort(key=lambda item: (
        -int(item["backtracks"]),
        -int(item["backtrace_steps"]),
        str(item["fault_id"]),
    ))
    return detected[:limit]


def select_validation_faults(profiles):
    return [dict(item) for item in profiles]


def validation_fault_ids(catalog, circuit_path):
    faults = catalog.get("faults") if isinstance(catalog, dict) else None
    if not isinstance(faults, list) or not faults:
        raise ValueError(
            f"Validation circuit has an empty fault catalog: {circuit_path}"
        )
    fault_ids = []
    for entry in faults:
        fault_id = entry.get("fault_id") if isinstance(entry, dict) else None
        if not isinstance(fault_id, str) or not fault_id:
            raise ValueError(
                f"Validation circuit has an invalid fault ID: {circuit_path}"
            )
        fault_ids.append(fault_id)
    if len(fault_ids) != len(set(fault_ids)):
        raise ValueError(
            f"Validation circuit has duplicate fault IDs: {circuit_path}"
        )
    return fault_ids


def _validate_profiles(profiles, *, split, circuit_name):
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"{split} circuit {circuit_name} has an empty fault catalog")
    fault_ids = [str(item.get("fault_id", "")) for item in profiles]
    if any(not fault_id for fault_id in fault_ids):
        raise ValueError(f"{split} circuit {circuit_name} has an invalid fault ID")
    if len(fault_ids) != len(set(fault_ids)):
        raise ValueError(f"{split} circuit {circuit_name} has duplicate fault IDs")
    for item in profiles:
        if "outcome" not in item:
            raise ValueError(
                f"{split} circuit {circuit_name} profile is missing outcome"
            )
        try:
            outcome = int(item["outcome"])
            backtracks = int(item["backtracks"])
            backtrace_steps = int(item["backtrace_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{split} circuit {circuit_name} has invalid profile counters"
            ) from error
        if outcome not in (0, 1, 2) or backtracks < 0 or backtrace_steps < 0:
            raise ValueError(
                f"{split} circuit {circuit_name} has out-of-range profile data"
            )
    return profiles


def _training_contract(dataset_root, encoder_variant):
    if encoder_variant == "level_gat_gru":
        discovered = discover_dataset(dataset_root)
        return {
            "train": tuple({"name": path.stem, "circuit": path, "fault_map": None}
                           for path in discovered["train"]),
            "validation": discovered["validation"],
            "assets": discovered["train"],
            "training_split": "train",
            "fault_limit": TRAIN_FAULTS_PER_CIRCUIT,
            "fault_filter": FAULT_FILTER,
        }
    if encoder_variant == "fanin_mean":
        discovered = discover_mean_dataset(dataset_root)
        return {
            **discovered,
            "training_split": "train_mean",
            "fault_limit": MEAN_TRAIN_FAULTS_PER_CIRCUIT,
            "fault_filter": MEAN_FAULT_FILTER,
        }
    raise ValueError(f"Unsupported SmartATPG encoder: {encoder_variant}")


def _inventory(dataset_root, contract):
    return {
        "train_assets": [
            {
                "name": path.name,
                "dataset_path": _relative_path(path, dataset_root),
                "sha256": sha256_file(path),
            }
            for path in contract["assets"]
        ],
        "validation": [
            {
                "name": path.stem,
                "dataset_path": _relative_path(path, dataset_root),
                "sha256": sha256_file(path),
            }
            for path in contract["validation"]
        ],
    }


def _profile_payload(source, split, seed, graph_identity, *, name=None,
                     fault_map=None):
    from rl_podem.cpp_bridge import profile_cpp_podem

    profiles = profile_cpp_podem(
        source,
        backtrack_limit=BACKTRACK_LIMIT,
        seed=seed,
        fault_map_path=fault_map,
        use_scoap=True,
    )
    circuit_name = name or source.stem
    _validate_profiles(profiles, split=split, circuit_name=circuit_name)
    payload = {
        "format": PROFILE_FORMAT,
        "split": split,
        "name": circuit_name,
        "source_sha256": sha256_file(source),
        "circuit_hash": graph_identity[0],
        "gate_count": graph_identity[1],
        "backtrack_limit": BACKTRACK_LIMIT,
        "profile_seed": seed,
        "profiles": profiles,
    }
    if fault_map is not None:
        payload["fault_map_sha256"] = sha256_file(fault_map)
    return payload


def _load_reusable_profile(path, source, split, seed, graph_identity, *, name=None,
                           fault_map=None):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": PROFILE_FORMAT,
        "split": split,
        "name": name or source.stem,
        "source_sha256": sha256_file(source),
        "circuit_hash": graph_identity[0],
        "gate_count": graph_identity[1],
        "backtrack_limit": BACKTRACK_LIMIT,
        "profile_seed": seed,
    }
    if fault_map is not None:
        expected["fault_map_sha256"] = sha256_file(fault_map)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Cannot resume changed {split} profile: {source}")
    _validate_profiles(
        payload.get("profiles"), split=split, circuit_name=name or source.stem
    )
    return payload


def _record(manifest_path, profile_path, source, split, payload, *, name=None,
            fault_map=None, fault_limit=TRAIN_FAULTS_PER_CIRCUIT):
    profiles = payload["profiles"]
    selected = (
        select_training_faults(
            profiles, limit=fault_limit, circuit_name=name or source.stem
        )
        if split == "train"
        else select_validation_faults(profiles)
    )
    record = {
        "name": name or source.stem,
        "circuit": _relative_path(source, Path(manifest_path).parent),
        "profile": _relative_path(profile_path, Path(manifest_path).parent),
        "artifact_sha256": {
            "circuit": sha256_file(source),
            "profile": sha256_file(profile_path),
        },
        "circuit_hash": payload["circuit_hash"],
        "gate_count": int(payload["gate_count"]),
        "profiled_faults": len(profiles),
        "episode_faults": selected,
        "episode_fault_ids": [str(item["fault_id"]) for item in selected],
    }
    if fault_map is not None:
        record["fault_map"] = _relative_path(
            fault_map, Path(manifest_path).parent
        )
        record["artifact_sha256"]["fault_map"] = sha256_file(fault_map)
    return record


def _validation_record(manifest_path, source, graph_identity):
    return {
        "name": source.stem,
        "circuit": _relative_path(source, Path(manifest_path).parent),
        "artifact_sha256": {"circuit": sha256_file(source)},
        "circuit_hash": graph_identity[0],
        "gate_count": int(graph_identity[1]),
    }


def _validate_manifest(manifest, manifest_path):
    manifest_format = manifest.get("format")
    supported_formats = (MANIFEST_FORMAT,)
    if manifest_format not in supported_formats:
        raise ValueError("Existing data-split SmartATPG manifest configuration changed")
    current_format = manifest_format == MANIFEST_FORMAT
    profiled_validation = manifest_format == LEGACY_MANIFEST_FORMAT
    encoder_variant = manifest.get("encoder_variant", "level_gat_gru")
    if not current_format and encoder_variant != "level_gat_gru":
        raise ValueError("Legacy SmartATPG manifests require level_gat_gru")
    contract = _training_contract(
        resolve_manifest_path(manifest_path, manifest["dataset_root"]),
        encoder_variant,
    )
    expected_rounds = NORMAL_TRAINING_ROUNDS if current_format else LEGACY_TRAINING_ROUNDS
    expected_fault_limit = (
        contract["fault_limit"] if current_format else TRAIN_FAULTS_PER_CIRCUIT
    )
    expected_fault_filter = contract["fault_filter"] if current_format else FAULT_FILTER
    expected = {
        "fault_filter": expected_fault_filter,
        "train_faults_per_circuit": expected_fault_limit,
        "backtrack_limit": BACKTRACK_LIMIT,
        "normal_rounds": expected_rounds,
        "heuristic": HEURISTIC,
        **smartatpg_metadata(encoder_variant),
    }
    if current_format:
        expected["training_split"] = contract["training_split"]
        expected["dataset_inventory"] = _inventory(
            resolve_manifest_path(manifest_path, manifest["dataset_root"]),
            contract,
        )
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing data-split SmartATPG manifest configuration changed")
    expected_train_count = len(contract["train"])
    if (
        manifest.get("train_circuit_count") != expected_train_count
        or manifest.get("validation_circuit_count") != len(VALIDATION_NAMES)
    ):
        raise ValueError("SmartATPG manifest circuit counts are invalid")
    if Path(str(manifest.get("dataset_root", ""))).is_absolute():
        raise ValueError("SmartATPG manifest dataset root must be relative")
    expected_train = contract["train"]
    expected_validation = contract["validation"]
    all_names = []
    for split, key in (("train", "train_circuits"), ("validation", "validation_circuits")):
        circuits = manifest.get(key)
        if not isinstance(circuits, list) or not circuits:
            raise ValueError(f"SmartATPG manifest has no {split} circuits")
        names = [item.get("name") for item in circuits]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError(f"SmartATPG {split} circuit order or names are invalid")
        if split == "train" and len(circuits) != expected_train_count:
            raise ValueError("SmartATPG manifest must contain all training circuits")
        if split == "validation" and tuple(names) != VALIDATION_NAMES:
            raise ValueError("SmartATPG manifest validation circuits are invalid")
        expected_entries = (
            expected_train if split == "train"
            else tuple({"name": path.stem, "circuit": path, "fault_map": None}
                       for path in expected_validation)
        )
        if names != [entry["name"] for entry in expected_entries]:
            raise ValueError(f"SmartATPG {split} dataset inventory changed")
        all_names.extend(names)
        for item, expected_entry in zip(circuits, expected_entries):
            uses_profile = split == "train" or profiled_validation
            expected_artifacts = {"circuit", "profile"} if uses_profile else {"circuit"}
            if expected_entry.get("fault_map") is not None:
                expected_artifacts.add("fault_map")
            if set(item.get("artifact_sha256", {})) != expected_artifacts:
                raise ValueError(
                    f"Manifest {split} artifact list is incomplete: {item.get('name')}"
                )
            for artifact_key, expected_hash in item.get("artifact_sha256", {}).items():
                path = resolve_manifest_path(manifest_path, item[artifact_key])
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise ValueError(f"Manifest artifact changed: {path}")
            source_path = resolve_manifest_path(manifest_path, item["circuit"])
            if source_path != expected_entry["circuit"].resolve():
                raise ValueError(
                    f"Manifest {split} source path changed: {item['name']}"
                )
            expected_fault_map = expected_entry.get("fault_map")
            if expected_fault_map is None:
                if "fault_map" in item:
                    raise ValueError(
                        f"Manifest {split} has an unexpected fault map: {item['name']}"
                    )
            elif resolve_manifest_path(manifest_path, item.get("fault_map", "")) != expected_fault_map.resolve():
                raise ValueError(
                    f"Manifest {split} fault map changed: {item['name']}"
                )
            if not uses_profile:
                forbidden = {
                    "profile", "profiled_faults", "episode_faults",
                    "episode_fault_ids",
                }
                if forbidden.intersection(item):
                    raise ValueError(
                        f"Manifest validation record contains profile data: "
                        f"{item['name']}"
                    )
                try:
                    graph = load_graph(source_path)
                except Exception as error:
                    raise ValueError(
                        f"Manifest validation graph is invalid: {item['name']}"
                    ) from error
                if (
                    item.get("circuit_hash") != graph.circuit_hash
                    or item.get("gate_count") != len(graph.names)
                ):
                    raise ValueError(
                        f"Manifest validation graph identity changed: "
                        f"{item['name']}"
                    )
                continue
            profile_path = resolve_manifest_path(manifest_path, item["profile"])
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            profile_expected = {
                "format": PROFILE_FORMAT,
                "split": split,
                "name": item["name"],
                "source_sha256": sha256_file(source_path),
                "circuit_hash": item.get("circuit_hash"),
                "gate_count": item.get("gate_count"),
                "backtrack_limit": BACKTRACK_LIMIT,
                "profile_seed": manifest.get("profile_seed"),
            }
            if expected_fault_map is not None:
                profile_expected["fault_map_sha256"] = sha256_file(expected_fault_map)
            if any(payload.get(key) != value for key, value in profile_expected.items()):
                raise ValueError(
                    f"Manifest {split} profile metadata changed: {item['name']}"
                )
            profiles = _validate_profiles(
                payload.get("profiles"), split=split, circuit_name=item["name"]
            )
            selected = (
                select_training_faults(
                    profiles, limit=expected_fault_limit,
                    circuit_name=item["name"],
                )
                if split == "train"
                else select_validation_faults(profiles)
            )
            if item.get("episode_faults") != selected or item.get(
                "episode_fault_ids"
            ) != [str(row["fault_id"]) for row in selected]:
                raise ValueError(f"Manifest {split} fault list changed: {item['name']}")
            if item.get("profiled_faults") != len(profiles):
                raise ValueError(f"Manifest profile count changed: {item['name']}")
    if len(all_names) != len(set(all_names)):
        raise ValueError("Training and validation circuit names must be disjoint")
    expected_train_episodes = sum(
        len(item["episode_fault_ids"]) for item in manifest["train_circuits"]
    )
    if manifest.get("training_episode_count") != expected_train_episodes:
        raise ValueError("SmartATPG manifest episode counts are invalid")
    if profiled_validation:
        expected_validation_episodes = sum(
            len(item["episode_fault_ids"])
            for item in manifest["validation_circuits"]
        )
        if manifest.get("validation_episode_count") != expected_validation_episodes:
            raise ValueError("SmartATPG manifest episode counts are invalid")
    elif "validation_episode_count" in manifest:
        raise ValueError(
            "Lazy validation manifests must not persist an episode count"
        )
    if current_format:
        hyperparameters = training_hyperparameters(encoder_variant)
        if (
            manifest.get("faults_per_update") != FAULTS_PER_UPDATE
            or manifest.get("k_epochs") != hyperparameters["k_epochs"]
        ):
            raise ValueError("Current SmartATPG batching configuration changed")
    elif "faults_per_update" in manifest or "k_epochs" in manifest:
        raise ValueError("Legacy SmartATPG manifests must not define batching")
    return manifest


def prepare(
    dataset_root, output_dir, seed=14, resume=False,
    encoder_variant="level_gat_gru",
):
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "training_manifest.json"
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to use non-empty preparation directory: {output_dir}"
        )
    contract = _training_contract(dataset_root, encoder_variant)
    inventory = _inventory(dataset_root, contract)
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing manifest without --resume: {manifest_path}"
            )
        state_path = output_dir / "preparation_state.json"
        if not state_path.is_file():
            raise ValueError("Completed preparation is missing preparation_state.json")
        preparation_state = json.loads(state_path.read_text(encoding="utf-8"))
        state_expected = {
            "format": PREPARATION_STATE_FORMAT,
            "dataset_root": _relative_path(dataset_root, output_dir),
            "backtrack_limit": BACKTRACK_LIMIT,
            "profile_seed": seed,
            "encoder_variant": encoder_variant,
            "training_split": contract["training_split"],
            "inventory": inventory,
        }
        if any(
            preparation_state.get(key) != value
            for key, value in state_expected.items()
        ):
            raise ValueError("Dataset inventory changed since preparation completed")
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_completed = {
            f"train/{entry['name']}": sha256_file(
                output_dir / "profiles" / "train" / f"{entry['name']}.json"
            )
            for entry in contract["train"]
        }
        if preparation_state.get("completed") != expected_completed:
            raise ValueError("Completed preparation profile hashes changed")
        manifest = _validate_manifest(stored_manifest, manifest_path)
        if (
            manifest.get("profile_seed") != seed
            or manifest.get("encoder_variant") != encoder_variant
            or manifest.get("dataset_root")
            != _relative_path(dataset_root, manifest_path.parent)
        ):
            raise ValueError("Completed preparation invocation changed")
        print(f"MANIFEST_REUSED {manifest_path}", flush=True)
        return manifest

    graph_identities = {}
    graph_entries = {
        "train": contract["train"],
        "validation": tuple(
            {"name": path.stem, "circuit": path, "fault_map": None}
            for path in contract["validation"]
        ),
    }
    for split in ("train", "validation"):
        for index, entry in enumerate(graph_entries[split], 1):
            source = entry["circuit"]
            try:
                graph = load_graph(source)
            except Exception as error:
                raise ValueError(
                    f"Graph validation failed for {split} circuit {source}"
                ) from error
            graph_identities[(split, entry["name"])] = (
                graph.circuit_hash, len(graph.names)
            )
            print(
                f"GRAPH_VALIDATED split={split} index={index}/"
                f"{len(graph_entries[split])} circuit={entry['name']}",
                flush=True,
            )
    state_path = output_dir / "preparation_state.json"
    expected_state = {
        "format": PREPARATION_STATE_FORMAT,
        "dataset_root": _relative_path(dataset_root, output_dir),
        "backtrack_limit": BACKTRACK_LIMIT,
        "profile_seed": seed,
        "encoder_variant": encoder_variant,
        "training_split": contract["training_split"],
        "inventory": inventory,
    }
    if state_path.is_file():
        if not resume:
            raise FileExistsError(
                f"Preparation state already exists; use --resume: {state_path}"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise ValueError("Dataset inventory changed since preparation started")
    else:
        state = {**expected_state, "completed": {}}
        _atomic_json(state_path, state)

    records = {"train": [], "validation": []}
    completed = state.get("completed", {})
    if not isinstance(completed, dict):
        raise ValueError("Preparation state has an invalid completed-profile map")
    expected_keys = {f"train/{entry['name']}" for entry in contract["train"]}
    if not set(completed).issubset(expected_keys):
        raise ValueError("Preparation state contains unknown completed circuits")
    split = "train"
    profile_dir = output_dir / "profiles" / split
    for index, entry in enumerate(contract["train"], 1):
        source = entry["circuit"]
        fault_map = entry.get("fault_map")
        profile_path = profile_dir / f"{entry['name']}.json"
        key = f"{split}/{entry['name']}"
        if key in completed and (
            not profile_path.is_file()
            or sha256_file(profile_path) != completed[key]
        ):
            raise ValueError(f"Completed profile changed: {profile_path}")
        payload = _load_reusable_profile(
            profile_path, source, split, seed,
            graph_identities[(split, entry["name"])],
            name=entry["name"], fault_map=fault_map,
        )
        if payload is None:
            print(
                f"PROFILE split={split} index={index}/{len(contract['train'])} "
                f"circuit={entry['name']}",
                flush=True,
            )
            payload = _profile_payload(
                source, split, seed,
                graph_identities[(split, entry["name"])],
                name=entry["name"], fault_map=fault_map,
            )
            _atomic_json(profile_path, payload)
        profile_digest = sha256_file(profile_path)
        if key in completed and completed[key] != profile_digest:
            raise ValueError(f"Completed profile changed: {profile_path}")
        if key not in completed:
            completed[key] = profile_digest
            state["completed"] = dict(sorted(completed.items()))
            _atomic_json(state_path, state)
        records[split].append(
            _record(
                manifest_path, profile_path, source, split, payload,
                name=entry["name"], fault_map=fault_map,
                fault_limit=contract["fault_limit"],
            )
        )
    records["validation"] = [
        _validation_record(
            manifest_path, entry["circuit"],
            graph_identities[("validation", entry["name"])],
        )
        for entry in graph_entries["validation"]
    ]

    hyperparameters = training_hyperparameters(encoder_variant)
    manifest = {
        "format": MANIFEST_FORMAT,
        "fault_filter": contract["fault_filter"],
        "train_faults_per_circuit": contract["fault_limit"],
        "training_split": contract["training_split"],
        "dataset_inventory": inventory,
        **smartatpg_metadata(encoder_variant),
        "dataset_root": _relative_path(dataset_root, manifest_path.parent),
        "backtrack_limit": BACKTRACK_LIMIT,
        "profile_seed": seed,
        "heuristic": HEURISTIC,
        "normal_rounds": NORMAL_TRAINING_ROUNDS,
        "faults_per_update": FAULTS_PER_UPDATE,
        "k_epochs": hyperparameters["k_epochs"],
        "train_circuit_count": len(records["train"]),
        "validation_circuit_count": len(records["validation"]),
        "training_episode_count": sum(
            len(item["episode_fault_ids"]) for item in records["train"]
        ),
        "train_circuits": records["train"],
        "validation_circuits": records["validation"],
    }
    _validate_manifest(manifest, manifest_path)
    _atomic_json(manifest_path, manifest)
    print(f"MANIFEST {manifest_path}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--backtrack-limit", type=int, default=BACKTRACK_LIMIT)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--normal-rounds", type=int, default=NORMAL_TRAINING_ROUNDS)
    parser.add_argument(
        "--encoder", choices=ENCODER_VARIANTS, default="level_gat_gru"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.backtrack_limit != BACKTRACK_LIMIT:
        raise ValueError(f"SmartATPG preparation requires backtrack limit {BACKTRACK_LIMIT}")
    if args.normal_rounds != NORMAL_TRAINING_ROUNDS:
        raise ValueError(
            f"SmartATPG preparation requires exactly {NORMAL_TRAINING_ROUNDS} rounds"
        )
    prepare(
        args.dataset_root, args.output_dir, seed=args.seed, resume=args.resume,
        encoder_variant=args.encoder,
    )


if __name__ == "__main__":
    main()
