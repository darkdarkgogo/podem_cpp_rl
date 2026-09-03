# SmartATPG 11维复现使用说明

## 模型结构

SmartATPG 不读取 DeepGate。每个 gate 的初始特征固定为11维：7维 gate 类型 one-hot，加上 level、fanout、CC0、CC1。支持 `PI、AND、NAND、OR、NOR、NOT、BUF`，不支持 XOR/XNOR，也不使用 SCOAP CO。

GraphSAGE 使用一次 fanin mean 聚合，将当前 gate 的11维特征和 fanin 均值11维拼成22维，再经过可训练的 `Linear(22,11)` 和 ReLU，输出该 gate 的11维 embedding。策略运行时动态拼接2维 action mask，形成13维 policy state。GraphSAGE、Actor 和 Critic 随 PPO 共同更新。

## 环境一：训练

训练环境需要 PyTorch、TensorBoard 和 `cpp_podem` Python 扩展。从 `PODEM` 目录执行：

```bash
conda activate d2l
python -m pip install -r python-requirements.txt
python -m pip install -e .

python scripts/run_smartatpg_training_linux.py \
  --output-dir artifacts/smartatpg_paper_11d
```

训练入口只完成以下工作：

1. 使用基础 PODEM 对 c6288 和 full-scan s38417 的完整 fault catalog 排序，各选择 backtrack 最困难的前100个 fault。
2. 训练20轮，每轮200个 episode，每个 episode 后立即更新 PPO、RND、Actor、Critic 和 GraphSAGE。
3. 每轮确定性评估并保存 backtrack 表现最好的完整参数。
4. 导出包含 GraphSAGE W/bias 和 Actor 参数的 `SMARTATPG_MODEL_V5`。
5. 准备16个评测电路和 faultmap，生成可复制的 `benchmark_bundle/`。

训练入口不会编译独立 C++ 可执行文件，也不会运行最终 heuristic/RL 比较。中断后重新执行同一命令，会从 `training_state.pth` 继续。

TensorBoard：

```bash
tensorboard \
  --logdir artifacts/smartatpg_paper_11d/tensorboard \
  --host 0.0.0.0 \
  --port 6006
```

主要输出：

- `training_state.pth`：最近训练状态；
- `best_training_state.pth`：最佳轮完整训练状态；
- `model_best.txt`：最佳 V5 模型，包含 GraphSAGE W/bias 和 Actor；
- `model_latest.txt`：最近 V5 模型；
- `round_metrics.json`：每轮确定性评估；
- `tensorboard/`：TensorBoard 日志；
- `benchmark_bundle/`：交给环境二的自包含目录。

## 环境二：编译与评测

把整个 `benchmark_bundle/` 复制到评测环境。该环境只需要 Python 3 标准库和支持 C++11 的 `g++`，不需要 PyTorch、CUDA、NumPy 或 `.pth` checkpoint。

```bash
python3 scripts/run_smartatpg_benchmark_linux.py \
  /path/to/benchmark_bundle \
  --output-dir benchmark_results \
  --repeats 5
```

评测入口会：

1. 校验 bundle 内模型、电路和 faultmap 的哈希；
2. 对每个电路重新计算11维原始特征；
3. 使用 V5 模型中的 GraphSAGE W/bias 重新计算该电路每个 gate 的11维 embedding；
4. 使用 `g++` 重新编译 C++ PODEM；
5. 通过同一可执行文件分别运行 heuristic 和 RL；
6. 输出 JSON、CSV、Markdown 汇总和每次原生运行日志。

新电路必须用同一个 V5 模型重新计算 embedding，不能复用其他电路的 embedding。mask 不写入 embedding，而由 C++ 在每次决策时动态加入。

## 时间口径

正式 runtime 比较只采用 C++ 输出的 ATPG 区间时间 `atpg_seconds`。以下时间全部不计入 heuristic/RL 提升比例：

- 电路解析和11维特征构建；
- GraphSAGE embedding 计算与文件导出；
- C++ 编译；
- Python 启动、进程启动和报告汇总；
- whole-process wall time。

embedding 计算时间单独写入 `preprocessing.json`，wall time 只保留在 `raw_results.json` 中用于排障。

## 兼容性

旧 SmartATPG 的14维特征、64/80维 descriptor、旧 checkpoint 以及缺少 GraphSAGE W/bias 的 V4 Actor 都不能作为当前评测模型。加载时会明确报错，不会静默解释成 V5。
