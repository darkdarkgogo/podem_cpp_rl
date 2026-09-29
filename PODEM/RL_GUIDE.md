# SmartATPG RL 使用入口

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

正式 PPO 协议对两个 encoder 一致：每个 rollout 收集 4 个完整 fault，先在保持时序和 terminal 边界的条件下计算一次 GAE，再于每个 PPO epoch 独立打乱所有 transitions，并按 512 个 transition 切分 minibatch。GAT-GRU 与 Mean 都执行 4 个 epoch；最后不足 512 个 transition 的 minibatch 和 round 末不足 4 个 fault 的 rollout 都不会丢弃。

日志中的 PPO loss、entropy、ratio、KL 与 clip fraction 是全部 4 个 epoch、所有 minibatch 按 transition 数加权的均值，同时保留 `*_last_epoch` 指标。模型协议包含 `faults_per_update=4`、`minibatch_size=512` 与 `k_epochs=4`；旧 MINIBATCH128、Batch8 或 Mean-K1 checkpoint 不能恢复到当前训练。

## 验证

```bash
python3 scripts/validate_smartatpg.py \
  --run-dir artifacts/smartatpg_dual \
  --dataset-root data \
  --output-dir artifacts/smartatpg_dual/validation \
  --seed 2026
```

验证运行时才独立发现六个 validation circuits，再按顺序评估 GAT-GRU 各轮、Mean 各轮和 fresh SCOAP baseline，分别选择两个模型的 best round，并生成：

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
