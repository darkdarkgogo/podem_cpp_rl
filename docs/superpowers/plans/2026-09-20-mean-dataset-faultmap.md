# Mean Independent Dataset and Fault Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route `fanin_mean` training through the two-circuit `data/train_mean` contract, preserve scan fault identities on the binary s38417 netlist, keep both models on the same validation split, and enforce a formal backtrack limit of 100.

**Architecture:** Extend the existing preparation pipeline with an explicit encoder-specific dataset contract and emit one manifest per encoder. Carry an optional fault-map artifact from preparation through manifest validation into each training episode. Change the dual launcher and validation comparison to consume separate training manifests while proving that both validation catalogs are identical.

**Tech Stack:** Python 3, PyTorch, pybind11 C++ bridge, `unittest`, Linux shell launchers, JSON manifests.

**Spec:** `docs/superpowers/specs/2026-09-20-mean-dataset-faultmap-design.md`

## Global Constraints

- `level_gat_gru` trains from `data/train`; `fanin_mean` trains from `data/train_mean`.
- Both encoders validate against the complete fault catalogs in `data/validation`.
- Mean uses `c6288.bench` directly and `s38417_scan_binary.bench` with `s38417_scan_binary.faultmap`.
- Mean selects only `outcome == 1` faults, ordered by descending backtracks, descending backtrace steps, and ascending fault ID, capped at 100 per circuit.
- A mean circuit with fewer than 100 detected faults uses all detected faults; a circuit with zero detected faults fails preparation.
- `.uf` files are ignored and never become protocol identity.
- Formal SmartATPG preparation, training, validation, model loading, and benchmark paths require `backtrack_limit=100`.
- Do not run full training or benchmark experiments during verification.

---

### Task 1: Add encoder-specific dataset discovery and fault selection

**Files:**
- Modify: `PODEM/scripts/prepare_smartatpg_training.py`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Produces: `discover_dataset(dataset_root, encoder_variant=...) -> {"train", "validation", "inventory"}` where training entries describe execution circuit and optional fault map.
- Produces: `select_training_faults(profiles, limit, circuit_name=None) -> list[dict]`.
- Produces: manifest records with optional `fault_map` and corresponding `artifact_sha256["fault_map"]`.

- [ ] **Step 1: Write dataset-contract tests**

Create temporary GAT and mean layouts. Assert that GAT retains the 1024/30 contract, mean emits exactly `c6288` and `s38417`, s38417 points at the binary BENCH plus fault map, and an existing `.uf` is absent from inventory and records.

```python
mean = discover_dataset(root, encoder_variant="fanin_mean")
self.assertEqual([item["name"] for item in mean["train"]], ["c6288", "s38417"])
self.assertIsNone(mean["train"][0]["fault_map"])
self.assertEqual(mean["train"][1]["circuit"].name, "s38417_scan_binary.bench")
self.assertEqual(mean["train"][1]["fault_map"].name, "s38417_scan_binary.faultmap")
self.assertNotIn("s38417_scan_binary.bench.uf", json.dumps(mean["inventory"]))
```

- [ ] **Step 2: Write capped-selection tests**

Verify 105 detected profiles produce the hardest 100, 17 detected profiles produce all 17, and zero detected profiles raises the existing explicit error.

```python
self.assertEqual(len(select_training_faults(profiles_105, limit=100)), 100)
self.assertEqual(len(select_training_faults(profiles_17, limit=100)), 17)
with self.assertRaisesRegex(RuntimeError, "No heuristic-detected faults"):
    select_training_faults(non_detected, limit=100, circuit_name="c6288")
```

- [ ] **Step 3: Run the new preparation tests and confirm failure**

Run: `python -m unittest tests.test_smartatpg_training.SmartATPGPreparationTests -v`

Expected: FAIL because discovery is path-only and the selection limit is fixed at 30.

- [ ] **Step 4: Implement explicit dataset contracts**

Add constants for the two encoder variants, GAT limit 30, mean limit 100, and the five required mean assets. Return structured training records such as:

```python
{
    "name": "s38417",
    "circuit": train_mean / "s38417_scan_binary.bench",
    "fault_map": train_mean / "s38417_scan_binary.faultmap",
}
```

Keep validation entries path-based or normalize them to the same record shape. Hash the five required mean assets in preparation inventory and intentionally ignore `.uf`.

- [ ] **Step 5: Parameterize selection and preparation metadata**

Make profile generation pass `fault_map_path` when present. Store `training_split`, `encoder_variant`, `train_faults_per_circuit`, actual episode count, optional relative fault-map path, and its hash. Bump the current manifest/preparation formats so old shared-manifest output cannot resume as the new split protocol.

- [ ] **Step 6: Run preparation tests**

Run: `python -m unittest tests.test_smartatpg_training.SmartATPGPreparationTests -v`

Expected: PASS.

### Task 2: Enforce manifest/encoder identity and carry fault maps into training

**Files:**
- Modify: `PODEM/scripts/prepare_smartatpg_training.py`
- Modify: `PODEM/scripts/train_smartatpg.py`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Consumes: manifest `encoder_variant`, `training_split`, and optional per-circuit `fault_map`.
- Produces: resolved circuit records containing absolute `fault_map` paths only when declared.
- Training call: `trainer.run(..., fault_map_path=item.get("fault_map"))`.

- [ ] **Step 1: Write manifest-validation tests**

Assert changed/missing fault-map hashes fail, mean manifest passed to GAT fails, GAT manifest passed to mean fails, and resolved mean records include the absolute map path.

```python
with self.assertRaisesRegex(ValueError, "encoder"):
    training.main([str(mean_manifest), str(output), "--encoder", "level_gat_gru"])
```

- [ ] **Step 2: Write trainer propagation test**

Patch the trainer and stop after one episode. Assert s38417 receives:

```python
trainer.run.assert_called_with(
    str(binary.resolve()),
    backtrack_limit=100,
    seed=expected_seed,
    fault_ids=[fault_id],
    fault_map_path=str(fault_map.resolve()),
    use_scoap=True,
)
```

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `python -m unittest tests.test_smartatpg_training -v`

Expected: new identity and propagation assertions fail.

- [ ] **Step 4: Validate optional artifacts and encoder identity**

Teach manifest validation to verify the expected artifact key set per record, resolve and hash optional fault maps, and recompute selections with the manifest-specific limit. Before constructing the agent, reject `args.encoder != manifest["encoder_variant"]` for the new format.

- [ ] **Step 5: Resolve and pass the fault map**

Resolve `raw["fault_map"]` alongside `circuit` and `profile`. Add `fault_map_path=item.get("fault_map")` to every training episode call. Do not pass a training fault map into validation catalog loading or native validation.

- [ ] **Step 6: Run training-state tests**

Run: `python -m unittest tests.test_smartatpg_training -v`

Expected: PASS.

### Task 3: Prepare and compare two independent manifests

**Files:**
- Modify: `PODEM/scripts/run_smartatpg_training_linux.py`
- Modify: `PODEM/scripts/run_dual_smartatpg_training_linux.py`
- Modify: `PODEM/scripts/compare_smartatpg_validation.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Single GAT preparation command includes `--encoder level_gat_gru`.
- Dual launcher produces `preparation/gat/training_manifest.json` and `preparation/mean/training_manifest.json`.
- Comparison CLI becomes `compare_smartatpg_validation.py GAT_MANIFEST MEAN_MANIFEST GAT_DIR MEAN_DIR OUTPUT_DIR`.

- [ ] **Step 1: Write launcher command tests**

Assert two preparation commands are run, each includes its encoder, each training command receives the matching manifest, and metadata contains separate hashes/counts/episode counts.

```python
self.assertEqual(len(prepare_commands), 2)
self.assertIn("level_gat_gru", prepare_commands[0])
self.assertIn("fanin_mean", prepare_commands[1])
```

- [ ] **Step 2: Write dual-manifest comparison tests**

Create two manifests with different hashes and training inventories but identical validation circuits/catalogs. Assert comparison succeeds. Change one validation circuit or fault list and assert it fails before producing a report.

- [ ] **Step 3: Run launcher/comparison tests and confirm failure**

Run: `python -m unittest tests.test_linux_smartatpg.DualTrainingLauncherTests tests.test_linux_smartatpg.ValidationComparisonTests -v`

Expected: FAIL because the launcher and comparison currently share one manifest.

- [ ] **Step 4: Split preparation and metadata**

Run the GAT and mean preparation commands sequentially before parallel training. Record a per-model protocol object:

```python
"training_protocols": {
    "smartatpg_gat_gru": {"manifest_hash": ..., "training_episode_count": ...},
    "smartatpg_mean": {"manifest_hash": ..., "training_episode_count": ...},
}
```

- [ ] **Step 5: Update comparison identity rules**

Remove `manifest_hash` from shared identity fields. Validate each run identity against its own manifest hash. Resolve validation circuits from both manifests, generate both runtime catalogs, and require names, fault IDs, and catalog hashes to be identical. Use the GAT validation records for the single SCOAP baseline only after this equality check.

- [ ] **Step 6: Run launcher/comparison tests**

Run: `python -m unittest tests.test_linux_smartatpg.DualTrainingLauncherTests tests.test_linux_smartatpg.ValidationComparisonTests -v`

Expected: PASS.

### Task 4: Update current documentation and audit stale protocol assumptions

**Files:**
- Modify: `PODEM/RL_GUIDE.md`
- Modify: `PODEM/docs/SMARTATPG_11D_使用说明.md`
- Modify: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Current documentation states GAT/mean train splits, shared validation, mean fault-map behavior, two rounds, batch size eight, and backtrack 100.

- [ ] **Step 1: Replace current-workflow stale descriptions**

Update active guides so they no longer say both encoders use `data/train`, five rounds, immediate per-episode update, or backtrack 200. Do not rewrite historical design/plan documents.

- [ ] **Step 2: Add static regression assertions**

Extend launcher inventory tests to assert active wrappers and guides contain `bt100`/`backtrack 100`, identify `data/train_mean`, and do not contain active-workflow `bt200` paths.

- [ ] **Step 3: Run the three targeted suites**

Run: `python -m unittest tests.test_smartatpg tests.test_smartatpg_training tests.test_linux_smartatpg -v`

Expected: PASS.

- [ ] **Step 4: Audit residual numeric defaults**

Run:

```text
rg -n -S "bt200|backtrack[^\n]*200|BACKTRACK_LIMIT = (200|500|2000)|data/train" \
  PODEM/scripts PODEM/python PODEM/src PODEM/*.sh PODEM/RL_GUIDE.md \
  PODEM/docs/SMARTATPG_11D_使用说明.md
```

Classify every remaining match. Formal SmartATPG workflow matches must be removed; generic ATPG defaults (97), explicit negative-test values, and unrelated/historical protocols may remain only when their context proves they are not the formal workflow.

- [ ] **Step 5: Run final repository checks**

Run: `git diff --check`

Run: `python -m unittest tests.test_smartatpg tests.test_smartatpg_training tests.test_linux_smartatpg -v`

Expected: no whitespace errors and all targeted tests pass without starting training or benchmark runs.
