# SmartATPG-Style Backend: One-Seed Experiment

Same ten circuits, fault splits, teachers, seed 2026, BC 20 epochs and GAE sweeps 2/2/3.
CPU single-thread training; this is a backend pipeline comparison, not exact paper reproduction.
SmartATPG best selection: medium_sweep_1; previous DeepGate best: behavior_cloning.
Best may be a BC fallback; final always denotes the last PPO policy.

## Interpretation

- The final SmartATPG policy detects 12 more faults on the uncollapsed count than the previous final DeepGate-GAE policy (133349 vs 133337 out of 135344), with 14 fewer reported aborted faults.
- Versus that previous final policy, total backtracks decrease 4.35%, backtrace steps 6.31%, native ATPG time 5.76%, and native process wall time 6.41% in this run.
- Versus the heuristic, the final policy detects 37 more faults; ATPG time decreases 14.82%, but whole-process wall time decreases only 1.62%. Do not equate stage speedup with end-to-end speedup.
- Improvements are not universal: c1355 backtracks increase from 4104 to 8569, and s38417_scan loses five detections versus the previous final policy. c3540, c5315 and c7552 gain seven, two and eight detections respectively.
- The selected SmartATPG best checkpoint scores 497/500 on validation but only 133275/135344 in full-circuit evaluation; final scores 495/500 yet 133349/135344. Current validation selection does not rank these policies consistently with the full ATPG workload. Both snapshots are preserved, and best has NOT been replaced using evaluation results.
- This is an exploratory one-seed result on the training circuits, not evidence of improved unseen-circuit generalization or a statistically established gain across training seeds.

## Native Full-Circuit Evaluation

All five models use the same executable, seed 14, bt=500, one warmup and 5 rotated measured runs per circuit.
Times sum per-circuit medians. ATPG is the native elapsed stage interval; wall time includes loading/inference/output, not offline encoding/export.
This scope includes training/validation faults, so it does not establish unseen-circuit generalization.

| Model | Detected / total | Aborted | Backtracks | Backtrace steps | ATPG s | Wall s |
|---|---:|---:|---:|---:|---:|---:|
| heuristic | 133312/135344 | 395 | 238872 | 3906240 | 16.370 | 17.659 |
| deepgate_gae_best | 133312/135344 | 395 | 238872 | 3906240 | 15.988 | 19.941 |
| deepgate_gae_final | 133337/135344 | 390 | 227855 | 3798786 | 14.796 | 18.562 |
| smartatpg_best | 133275/135344 | 394 | 234603 | 4101326 | 16.210 | 20.075 |
| smartatpg_final | 133349/135344 | 376 | 217950 | 3558930 | 13.944 | 17.372 |

## Per-Circuit Final Policies

| Circuit | Heuristic detected | DeepGate final detected | Smart final detected | Heuristic BT | DeepGate final BT | Smart final BT |
|---|---:|---:|---:|---:|---:|---:|
| c432 | 782 | 781 | 781 | 2036 | 3253 | 3865 |
| c499 | 751 | 766 | 766 | 6793 | 4013 | 4012 |
| c1355 | 2702 | 2702 | 2702 | 9497 | 4104 | 8569 |
| c1908 | 3560 | 3560 | 3560 | 38229 | 38706 | 38246 |
| c2670 | 5010 | 5010 | 5010 | 29167 | 28271 | 24190 |
| c3540 | 6201 | 6194 | 6201 | 45253 | 42640 | 42087 |
| c5315 | 10454 | 10451 | 10453 | 9870 | 10305 | 9186 |
| c7552 | 14914 | 14936 | 14944 | 65296 | 61840 | 54351 |
| c6288 | 12508 | 12508 | 12508 | 1609 | 1609 | 1036 |
| s38417_scan | 76430 | 76429 | 76424 | 31122 | 33114 | 32408 |

## Fixed 500-Fault Validation

| Stage | DeepGate detected | Smart detected |
|---|---:|---:|
| behavior_cloning | 496 | 496 |
| easy_sweep_1 | 496 | 481 |
| easy_sweep_2 | 494 | 490 |
| medium_sweep_1 | 494 | 497 |
| medium_sweep_2 | 494 | 495 |
| hard_sweep_1 | 494 | 495 |
| hard_sweep_2 | 494 | 494 |
| hard_sweep_3 | 494 | 495 |

## Audit

4200 episodes, 150 work units, 4001 PPO updates and 399075 decisions; all checked metrics/weights finite and reward counter mismatches zero.
All 30 historical heuristic/DeepGate rows match structural counters on the rebuilt executable; inference artifact hashes unchanged.
One training seed only; repeated benchmark timings are not independent training seeds.

## Cost And Reproduction

Training took 5614.3 seconds (93.6 minutes), versus 4028.5 seconds (67.1 minutes) for the previous run with the same episode/update budget. It used 399075 Actor decisions versus 358655 previously; this is not an equal-environment-step or equal-wall-time comparison.

Offline SmartATPG preprocessing was measured separately in a single CPU pass after training. Graph parsing/feature construction across the ten circuits took 0.832 seconds. Graph encoding plus descriptor export took 1.552 seconds for best and 1.331 seconds for final. These exclude Python startup/checkpoint loading; historical DeepGate preprocessing was not rerun, so these are not a preprocessing speed comparison.

The accepted benchmark is `benchmark_20260831_170700`: 50 warmups and 250 measured native runs, with complete raw logs, artifact hashes and timing samples. The earlier `benchmark_20260831_170532` contains export probes only: preflight stopped before any native measurement because the existing compiler output was named `atpg_rl_smartatpg.exe.exe`. The accepted run explicitly hashes and executes that actual file; no failed-preflight data is included.

Run from the PODEM directory with the existing d2l Python environment:

```text
python artifacts/paper_v8_smartatpg/run_training.py
python artifacts/paper_v8_smartatpg/evaluate_experiment.py
```

The first command resumes the existing completed checkpoint without retraining; a fresh run requires a separate artifact directory. TensorBoard integration and training algorithm changes were intentionally deferred.
