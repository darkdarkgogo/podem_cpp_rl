# SmartATPG 11D 训练与验证使用说明

SmartATPG 使用 11 维 gate embedding：6 维 gate 类型 one-hot（不含 BUF），以及 level、fanout、静态 SCOAP CC0、CC1、CO。GAT-GRU Actor/Critic 额外拼接 1 维目标值，Mean 保持 11 维输入。

正式实验固定使用 backtrack limit 100、训练 2 轮、每 8 个 fault 更新一次。GAT-GRU 每个 PPO batch 训练 4 epoch；Mean 训练 1 epoch。训练与验证是两个独立命令。

## 安装

```bash
cd PODEM
conda activate d2l
python -m pip install -r python-requirements.txt
python -m pip install -e .
```

## 训练

```bash
python3 scripts/train_smartatpg.py \
  --dataset-root data \
  --output-dir artifacts/smartatpg_dual \
  --gat-gpu 0 \
  --mean-gpu 1 \
  --seed 2026
```

训练入口为 GAT-GRU 与 Mean 分别准备训练 manifest，并在两张 GPU 上并行运行。训练 manifest 只记录 train circuits，不枚举、不哈希 validation 文件；每个 worker 也不会加载 validation graph、生成 validation embedding、运行 native validation、计算 validation score 或选择 best round。

每轮训练结束保存：

- `training/{gat,mean}/training_state.pth`：训练断点恢复状态；
- `training/{gat,mean}/inference_round_XX.pth`：完整 `policy_old`，包含稍后生成 embedding 所需的图编码器；
- `training/{gat,mean}/model_round_XX.txt`：与该轮 checkpoint 配对的 native Actor；
- `training/{gat,mean}/inference_final.pth` 与 `model_final.txt`：最终轮工件。

原任务中断后重跑相同命令即可恢复。若要从旧 checkpoint 开始新任务，分别传入 `--gat-continue-from` 与 `--mean-continue-from`，并使用新的输出目录。

## 验证

训练完成后显式执行：

```bash
python3 scripts/validate_smartatpg.py \
  --run-dir artifacts/smartatpg_dual \
  --dataset-root data \
  --output-dir artifacts/smartatpg_dual/validation \
  --seed 2026
```

验证脚本在启动时独立发现并校验 `data/validation` 中的六个电路，然后按顺序执行：

1. 对 GAT-GRU 每轮 checkpoint 生成 validation embedding，并按电路运行 native batch；
2. 对 Mean 每轮执行相同流程；
3. 使用现有 score 规则分别选择 best round；
4. fresh 运行一次共享 SCOAP baseline；
5. 写出三方主表和逐 fault 明细。

Best-round score 的字典序为：检出 fault 更多、总 backtracks 更少、总 backtrace steps 更少、总外在回报更高、轮次更早。

验证输出：

- `validation_three_way_comparison.csv`：6 个 validation circuits 加 TOTAL；
- `validation_three_way_comparison.json`：同内容机器可读版本；
- `model_selection.json`：GAT/Mean best round 与 score；
- `detailed/{scoap,gat,mean}_records.jsonl`：最终逐 fault 结果。

Validation 不提供 checkpoint/resume，也不读取旧 SCOAP cache。中断后重新运行会 fresh 计算。

## Runtime 定义

论文/主表的 `runtime_atpg_s` 是逐 fault `atpg_seconds` 的和。计时包含 `ATPG::test()`、PODEM/backtrace/implication 与搜索过程中的 Actor forward；不包含 checkpoint/model/embedding 加载、图构建、`ATPG::input()`、levelize、dummy gate、fault-list generation/filter 或结果写盘。

三种方法使用同一 ordered validation fault catalog、seed 和 backtrack limit，并按顺序执行以避免并发负载污染比较。

## 内部代码边界

`rl_podem.training` 只包含训练工作流；`rl_podem.validation` 负责独立验证编排；不依赖 PyTorch 的 catalog、native batch、summary 和 best-round score 集中在 `rl_podem.validation_core`。两个工作流只通过 `rl_podem.artifact_io` 共享 checkpoint 格式和原子文件写入约定，验证实现不再导入训练实现。

## 其他工具

通用 BENCH 转换、数据生成、绘图与 native build 工具位于 `PODEM/tools/`。它们不是 SmartATPG 正式训练/验证入口。
