# SmartATPG 11维复现使用说明

## 模型结构

每个 gate 的初始特征固定为11维：7维 gate 类型 one-hot，加上 level、fanout、CC0、CC1。支持 `PI、AND、NAND、OR、NOR、NOT、BUF`，不支持 XOR/XNOR，也不使用 SCOAP CO。

基线编码器使用一次 fanin mean 聚合，将当前 gate 的11维特征和 fanin 均值11维拼成22维，再经过可训练的 `Linear(22,11)` 和 ReLU，输出11维 embedding。对比编码器保持11维，按 level 完成一次正向 fanin GAT+GRU，再用独立参数完成一次反向 fanout GAT+GRU。

Actor 只接收11维 gate embedding。两位 action mask 与 embedding 分开保存，逻辑决策状态可记为 `11D embedding + 2D mask`，但 mask 只在 Actor 输出 logits 后用于屏蔽动作。RL 第一次选择一路后由 `BacktraceLock.selected_wire` 锁定；若该 fanin 仿真值仍为 `U`，继续同一路而不再次调用 Actor；只有仿真确认 fanin 满足局部目标后才将该路屏蔽。回退撤销赋值后根据当前电路值重新计算 mask。

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
2. 使用完全相同的 episode 顺序和共享超参数，分别训练20轮 fanin-mean 与 GAT-GRU；每轮各200个 episode。
3. 每轮确定性评估并保存 backtrack 表现最好的完整参数。
4. 分别导出包含完整图编码器和 Actor 参数的 `SMARTATPG_MODEL_V6`。
5. 准备16个评测电路和 faultmap，生成同时包含两个 best model 的 `benchmark_bundle/`。

训练入口不会编译独立 C++ 可执行文件，也不会运行最终 heuristic/RL 比较。中断后重新执行同一命令，会从 `training_state.pth` 继续。

TensorBoard：

```bash
tensorboard \
  --logdir artifacts/smartatpg_paper_11d \
  --host 0.0.0.0 \
  --port 6006
```

主要输出：

- `smartatpg_mean/`：基线训练状态、V6 模型、每轮指标和 TensorBoard；
- `smartatpg_gat_gru/`：逐 level 双向 GAT-GRU 的对应训练输出；
- `preparation/`：两种模型共享且固定的训练 manifest、转换电路和 fault profile；
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
3. 分别使用两个 V6 模型重新计算该电路每个 gate 的11维 embedding；
4. 使用 `g++` 重新编译 C++ PODEM；
5. 通过同一可执行文件分别运行 heuristic、fanin-mean 和 GAT-GRU；
6. 输出 JSON、CSV、Markdown 汇总和每次原生运行日志。

新电路必须分别用两个 V6 模型重新计算 embedding，不能复用其他电路的 embedding。mask 不写入 embedding，也不送入 Actor，而由 C++ 在 logits 之后动态应用。

## 时间口径

正式 runtime 比较只采用 C++ 输出的 ATPG 区间时间 `atpg_seconds`。以下时间全部不计入 heuristic/RL 提升比例：

- 电路解析和11维特征构建；
- 两种图 embedding 的计算与文件导出；
- C++ 编译；
- Python 启动、进程启动和报告汇总；
- whole-process wall time。

embedding 计算时间单独写入 `preprocessing.json`，wall time 只保留在 `raw_results.json` 中用于排障。

## 兼容性

V6 model 与 V4 embedding 会校验 encoder variant、graph configuration、11维 Actor 输入、两位 mask、13维逻辑状态和 snapshot。旧 V5 model 与 V3 embedding 仍可读取，但只用于历史工件兼容，不参与新的公平对比；更早的14维特征、64/80维 descriptor 和缺少图权重的 V4 Actor 会被明确拒绝。
