# SmartATPG Train/Validation Separation Design

## Decision

Adopt the attached `SmartATPG_Train_Validation_Separation_Design_CN.pdf` with one required correction: every training round must persist an inference checkpoint containing the complete `policy_old` state, not only the native actor text. Independent validation needs the graph-encoder weights from that state to generate circuit embeddings after training.

## Public workflow

SmartATPG exposes exactly two experiment entry points under `PODEM/scripts/`:

- `train_smartatpg.py` prepares the training split, launches GAT-GRU and Mean training on separate GPUs, saves each round, and exits without reading or evaluating validation circuits.
- `validate_smartatpg.py` loads the saved round checkpoints, evaluates GAT-GRU and Mean rounds plus a fresh SCOAP baseline, selects the best round for each model, and writes the three-way comparison.

Reusable implementation lives in `PODEM/python/rl_podem/`. A package-internal training worker may be launched as a subprocess so the two models can use separate `CUDA_VISIBLE_DEVICES` values; it is not a user-facing script.

## Training artifacts

Each model directory contains:

- `training_state.pth` for training resume only.
- `inference_round_01.pth`, `inference_round_02.pth`, ... with CPU-cloned `policy_old`, manifest identity, encoder variant, reward scheme, and round number.
- `model_round_01.txt`, `model_round_02.txt`, ... native actor exports paired with the inference checkpoints.
- `inference_final.pth` and `model_final.txt` for the final trained state.

Training does not load validation graphs, generate validation embeddings, call native validation, compute validation scores, or write validation state/metrics/best-model files.

## Validation behavior

Validation is a fresh, read-only run. For each model and each round it:

1. Loads the round inference checkpoint.
2. Builds validation graphs and generates temporary/persisted embeddings outside the ATPG timer.
3. Runs one native batch per circuit over the shared ordered validation fault catalog.
4. Summarizes detected faults, backtracks, backtrace steps, reward, and `runtime_atpg_s = sum(atpg_seconds)`.
5. Applies the existing lexicographic selection rule `(-detected_faults, backtracks_total, backtrace_steps_total, -return_total, round_number)`.

After selecting GAT and Mean best rounds, validation runs SCOAP fresh once per circuit. It never reads an older SCOAP cache as input and provides no validation resume/checkpoint mode.

## Timing contract

`runtime_atpg_s` includes `ATPG::test()`, PODEM/backtrace/implication, and each actor forward executed during search. It excludes model/checkpoint loading, graph and embedding construction, circuit input, levelization, dummy-gate creation, fault-list generation/filtering, and result serialization. The existing native boundary (`policy->start_run()` immediately before `atpg.test()`) is retained and tested.

## Outputs

`validation/validation_three_way_comparison.csv` and its JSON equivalent contain one row for each of `b12_C`, `b15_C`, `b17_C`, `b20_C`, `b21_C`, `b22_C`, followed by `TOTAL`. Each row contains SCOAP/GAT/Mean backtracks, backtrace steps, ATPG runtime, and fault coverage. Derived reduction/delta columns are also emitted. `model_selection.json` records best rounds and scores; `detailed/*.jsonl` stores per-fault results but is never used for resume.

TOTAL counts and runtimes are sums. TOTAL fault coverage is computed from summed detected and attempted fault counts, not as an unweighted mean of circuit percentages.

## Cleanup boundary

SmartATPG orchestration, comparison, and preparation launchers superseded by the two public commands are removed after their reusable logic and tests move into the package. Generic conversion, dataset-generation, native-build, and plotting utilities are tools rather than SmartATPG experiment entry points and remain available; they are not deleted merely to make the directory shorter.

## Acceptance criteria

- Training logs never contain validation/SCOAP comparison events.
- Training succeeds using only the training split and writes every round inference/native actor artifact.
- Validation executes all three methods on the same ordered fault catalog, seed, and backtrack limit.
- Each circuit/method uses exactly one native circuit setup and one batch call.
- The comparison has six circuit rows plus TOTAL and uses only summed per-fault `atpg_seconds` for runtime.
- Validation does not create `validation_state.json`, resume state, or consume a prior SCOAP result.
- Documentation names only `train_smartatpg.py` and `validate_smartatpg.py` as formal SmartATPG experiment commands.

