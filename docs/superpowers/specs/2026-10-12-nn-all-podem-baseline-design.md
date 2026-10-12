# NN-All PODEM Baseline Design

Date: 2026-10-12

## Goal and provenance

Add one independent NN-All baseline to the SmartATPG stuck-at-fault comparison, implemented in this repository's PODEM engine. The name in output is `nn_all_podem` to distinguish it from the original FAN-based experimental program.

The paper and repository README define NN-All as neural-network fanin selection at every applicable backtrace level. However, the published `NN_All/fan.cpp` currently uses `cont0`/`cont1` selection and has no network inference call. The published `NN_Hyb/fan.cpp` contains the useful model interface and feature recipe, but applies it only when the current gate's minimum PI distance is 6. Therefore this implementation follows the stated NN-All method, using the project's own engine, rather than claiming bit-for-bit reproduction of the published `NN_All` executable. No Atalanta source or embedded weight arrays are copied.

References:

- https://github.com/lzzh97/NN-for-ATPG
- https://raw.githubusercontent.com/lzzh97/NN-for-ATPG/main/NN_All/fan.cpp
- https://raw.githubusercontent.com/lzzh97/NN-for-ATPG/main/NN_Hyb/fan.cpp
- Zhang and Davoodi, *Efficient and Effective Neural Networks for Automatic Test Pattern Generation*, MLCAD 2024.

## Scope

- Single stuck-at faults on the existing normal binary BENCH dataset and current PODEM search semantics.
- Add only NN-All. NN-Hyb, FAN algorithm changes, transition-delay ATPG and external Atalanta execution are outside this change.
- Keep GAT-GRU, Mean and SCOAP training and inference paths intact.
- Train only on the complete `data/nn_all_paper_train/c1908.bench` and `c2670.bench`; evaluate only on the discovered `data/validation` circuits.
- Validation uses the same ordered fault IDs, seed 2026 and backtrack limit 100 as the current three-way comparison.

## Decision policy

At each PODEM backtrace gate with more than one unresolved fanin, score every legal unresolved fanin and select the maximum score. Apply the network at every such gate, with no level-6 filter. Unary gates continue directly. Existing objective propagation, implication, backtracking and fault simulation remain in the PODEM engine. Equal scores use the existing deterministic fanin order. A model-loading or feature-integrity failure stops NN-All evaluation with an error; it cannot silently fall back to SCOAP.

The paper-defined input for each candidate is 12 values in fixed order: candidate LDPI (longest PI distance), SDPI (shortest PI distance), SCOAP CC0, SCOAP CC1, COP logic-1 probability, followed by a seven-position one-hot of the *current objective gate* type in the published inference-source order AND, NAND, OR, NOR, PI, BUF, NOT. Within each circuit and SDPI group, independently min-max normalize LDPI, CC0 and CC1 to [0,1], with constant groups mapped to zero. The paper specifies the group and target range but not the precise normalization formula, so min-max is this port's explicit choice. COP starts at 0.5 for each PI and follows the gate's independent-input probability recurrence for AND, NAND, OR, NOR and NOT. COP remains in [0,1] and SDPI remains the raw level. Do not apply `log1p` or dataset-wide mean/standard deviation transforms. The policy scores candidates individually, with one hidden sigmoid layer of width 60 and a sigmoid scalar output. Training and native inference use the same feature computation and serialized parameter order. The policy is independent of the existing 11D SmartATPG graph encoders, whose feature meaning differs. The old NN-All 11D model and feature-file versions must be rejected.

COP and depth are static circuit properties computed before fault search. A score lookup may be used because all 11 inputs are static. Its key must identify both the candidate and the current gate type, and its lifetime is one circuit evaluation. A cache hit and miss return the same score. Native ATPG timing includes score computation on misses and lookup on hits.

## Training data and artifacts

Create a separate training command and run directory for NN-All. Instrument the existing PODEM backtrace decisions on training faults to record the 12D candidate features, selected fanin, and the search outcome attributable to that path. A path that reaches a detecting test before that PI decision is reversed is a positive example; a path reversed by backtracking is a negative example. Unresolved paths ending at the backtrack limit or an interrupted run are excluded. Apply the same attribution rule to every recorded decision on a completed path. Collect data across every applicable level, retain circuit and fault IDs for audit, and prevent validation circuits from entering the collector.

Train the 12→60→1 sigmoid classifier with ordinary binary cross-entropy and seed 2026. Per the latest user instruction, use the complete ISCAS85 c1908 and c2670 circuits and their full stuck-at fault catalogs, and all collected paths for 30,000 epochs, with no internal circuit holdout or checkpoint selection; export the final weights. Repeated identical feature/label rows may be count weighted to compute the exact full-sample BCE. Export one versioned, self-describing native model file with feature schema, architecture, seed, training-manifest hash and weights. Training must emit counts of positive, negative and excluded paths, plus level distribution, so an empty or one-class dataset cannot be mistaken for a trained baseline. The validation command loads the saved model; it never trains on validation faults.

## Evaluation and output

Add an explicit NN-All run-directory option to the existing validation command. When supplied, evaluate NN-All fresh after the existing GAT-GRU, Mean and SCOAP runs. The four methods share the validation fault catalog and per-fault metric schema. Produce `validation_four_way_comparison.csv` and `.json`, a separate `detailed/nn_all_podem_records.jsonl`, and metadata giving the NN-All model identity and feature schema. Preserve the existing three-way output contract for callers that omit NN-All.

The four-way table includes each validation circuit and TOTAL, with detected, redundant and aborted counts, backtracks, backtrace steps, fault coverage, and ATPG runtime. `runtime_atpg_s` is the sum of per-fault `atpg_seconds`: PODEM search and NN inference or lookup are timed, whereas parsing, feature precomputation, model loading, training and result writing are not. Relative metrics use the same SCOAP row as denominator. The plotting utility accepts the four-way table without mislabeling NN-All as either GAT or Mean.

## Verification

- Unit checks for feature order, static COP/CC/depth values, candidate masking, equal-score tie order, all-level invocation, and lookup equivalence.
- A native test confirms that disabling or corrupting the model cannot silently execute SCOAP, and that NN-All and SCOAP share the same ordered fault IDs.
- A small end-to-end training/validation fixture confirms a nonempty two-class training set, model export and load, deterministic inference, complete four-way rows, TOTAL arithmetic and timing fields.
- Existing three-way validation and PODEM tests continue to pass.

## Known interpretation limit

This is a method-level PODEM port. The original paper used FAN and the published NN-All directory does not expose the documented NN scoring in its `fan.cpp`. Differences from the paper's numerical results must be reported as solver, training-data and implementation differences, not presented as an exact replication.
