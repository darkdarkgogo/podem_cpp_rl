# SmartATPG RL 使用入口

## 拼接 fanin 之前的 GAT 消融（独立 worktree）

本 worktree 基于 `7d92ab8`：GAT Actor 接收 11 维 object gate embedding 和 1 维 objective value，输出两个 fanin 位置的 logits；没有拼接候选 fanin embedding。`no_object_value` 的 Actor/Critic 输入均为 11 维，PODEM 仍使用 objective value。`no_lock` 在训练和验证中关闭 backtrace 路径锁。`no_cache` 在原生验证时关闭 Actor logits 缓存，复用 `full` 同一 checkpoint 和 Actor，不单独训练。

```powershell
cd PODEM
python -m pip install -e .
python scripts/train_pre_fanin_ablation.py --full-run-dir 'PATH_TO_OLD_FULL_RUN' --gpu 0
python scripts/validate_pre_fanin_ablation.py
```

`--full-run-dir` 指向已有旧版完整训练与验证结果，需包含 `train_summary.json`、`training/gat`、`validation/detailed/gat_records.jsonl` 和原三方验证报告。不会重新训练或验证 `full`。训练和验证必须分别启动。训练复用完整组的 manifest、seed、奖励和两轮协议，只训练 `no_lock`、`no_object_value`，按顺序使用一张 CUDA GPU。输出位于 `artifacts/pre_fanin_ablation`，日志保存在 `logs`，中断后重复训练命令可恢复。验证在同一有序 fault catalog 上读取已有 `full` 结果并运行其余三组，写入逐故障 JSONL、逐电路与 TOTAL 的 CSV/JSON 对照表。当前开发机的 PyTorch 是 CPU 版，需要在有 CUDA 版 PyTorch 的训练环境执行正式实验。

如果只想先验证 `no_cache`，无需运行消融训练：

```bash
cd PODEM
python3 -m pip install -e .
python3 scripts/validate_pre_fanin_no_cache.py --full-run-dir artifacts/smartatpg_dual
```

此命令复用旧版 `full` 的最佳轮 checkpoint 和原验证结果，输出 `artifacts/pre_fanin_ablation/no_cache_validation/validation_no_cache.csv`。如果只有旧训练产物、没有原验证报告和 GAT 逐故障记录，需指定 `--round 1` 或 `--round 2`；此时只生成 `no_cache` 指标，不计算相对 `full` 的差值。旧训练产物须包括 `train_summary.json`、GAT manifest、所选轮的 `inference_round_XX.pth` 与 `model_round_XX.txt`，以及 validation 电路数据。

正式实验只有训练和验证两个入口。训练不会读取 validation fault catalog、生成 validation embedding、计算 validation score 或自动开始 comparison；训练完成后必须显式运行验证命令。

## 训练

```bash
cd PODEM
python3 scripts/train_smartatpg.py \
  --dataset-root data \
  --output-dir artifacts/smartatpg_dual \
  --gat-gpu 0 \
  --mean-gpu 1 \
  --seed 2026
```

GAT-GRU 与 Mean 分别在两张 GPU 上训练。训练 manifest 仅包含 train circuits，验证目录缺失也不影响训练准备。每轮都会同时保存完整的 `inference_round_XX.pth` 与 native `model_round_XX.txt`；前者包含稍后生成 validation embedding 所需的图编码器权重。`training_state.pth` 只用于训练恢复。

训练和验证命令都支持 `--backtrace-lock {auto,on,off}`。默认 `auto` 会为 GAT-GRU 开启 backtrace 路径锁、为 Mean 关闭路径锁；`on` 或 `off` 可在该次命令中强制两个模型使用相同行为。

正式 PPO 协议对两个 encoder 一致：每个 rollout 收集 4 个完整 fault，先在保持时序和 terminal 边界的条件下计算一次 GAE，再于每个 PPO epoch 独立打乱所有 transitions，并按 512 个 transition 切分 minibatch。GAT-GRU 与 Mean 都执行 4 个 epoch；最后不足 512 个 transition 的 minibatch 和 round 末不足 4 个 fault 的 rollout 都不会丢弃。

日志中的 PPO loss、entropy、ratio、KL 与 clip fraction 是全部 4 个 epoch、所有 minibatch 按 transition 数加权的均值，同时保留 `*_last_epoch` 指标。模型协议包含 `faults_per_update=4`、`minibatch_size=512` 与 `k_epochs=4`；旧 MINIBATCH128、Batch8 或 Mean-K1 checkpoint 不能恢复到当前训练。

## 验证

```bash
python3 scripts/validate_smartatpg.py
```

默认读取项目内的 `artifacts/smartatpg_dual`，并将结果写入其 `validation` 子目录。需要验证其他训练运行时，可用 `--run-dir PATH` 覆盖默认目录；`--dataset-root`、`--output-dir` 和 `--seed` 仍可按需指定。

验证运行时才独立发现 `data/validation` 中的全部 `.bench` 电路，并按文件名顺序评估 GAT-GRU 各轮、Mean 各轮和 fresh SCOAP baseline，分别选择两个模型的 best round，并生成：

目录中的电路必须是 SmartATPG 支持的 normal BENCH：`AND`、`NAND`、`OR`、`NOR` 均为二输入，`NOT` 为单输入，并且不包含 `BUF`。多输入 BENCH 可先用 `tools/convert_binary_bench.py` 转换。

- `validation_three_way_comparison.csv`
- `validation_three_way_comparison.json`
- `model_selection.json`
- `detailed/{scoap,gat,mean}_records.jsonl`

主表对每个验证电路并排给出三种方法的 backtracks、backtrace steps、ATPG runtime 和 fault coverage，最后一行为 TOTAL。验证不提供 resume，也不会读取旧的 SCOAP cache。

正式 runtime 只使用逐 fault `atpg_seconds` 的和：模型/embedding 加载、图构建、circuit input、levelize、fault-list generation 与写盘不计时；`ATPG::test()` 内的 Actor forward 计时。

## 内部模块边界

- `rl_podem.training` 只负责训练、恢复和每轮模型导出；
- `rl_podem.validation` 只负责 fresh validation 编排与结果写出；
- `rl_podem.validation_core` 负责 fault catalog、native batch、记录汇总和 best-round score，且不依赖 PyTorch；
- `rl_podem.artifact_io` 提供训练和验证共用的 checkpoint 格式标识、manifest hash 与原子 JSON 写入。

`validation` 不导入 `training`，`training` 也不导入 `validation_core`。
