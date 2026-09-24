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
