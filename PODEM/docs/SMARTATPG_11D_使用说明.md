# SmartATPG 基线与 agentATPG 使用说明

## 模型结构

每个 gate 的初始特征固定为12维：7维 gate 类型 one-hot，加上 level、fanout、CC0、CC1、CO。支持 `PI、AND、NAND、OR、NOR、NOT、BUF`，不支持 XOR/XNOR。

CC0、CC1 从 PI 向 PO 计算，CO 从 PO（值为0）向 PI 反向计算。AND/NAND 的旁路输入使用 CC1，OR/NOR 使用 CC0，NOT/BUF 只增加一级代价；多扇出取最小可观测代价。CO 是静态结构特征，不随 PI 赋值更新，也不是可测性的证明。有限代价封顶为 `10**9`，无输出可达路径时原始 CO 为 `inf`；CO 和 CC0/CC1 一样使用按电路的 log1p 缩放，非有限值映射为1，避免网络输入出现 inf/NaN。

基线编码器使用一次 fanin mean 聚合，将当前 gate 的12维特征和 fanin 均值12维拼成24维，再经过可训练的 `Linear(24,12)` 和 ReLU，输出12维 embedding。对比编码器保持12维，按 level 完成一次正向 fanin GAT+GRU，再用独立参数完成一次反向 fanout GAT+GRU；两方向 GAT 投影矩阵均为 `12×12`。

SmartATPG 基线将12维 gate embedding 直接送入 Actor/Critic，不额外拼接目标值。agentATPG（代码变体 `level_gat_gru`）将12维 embedding 与当前1维目标值 `object_val`（0/1）拼接成13维，直接送入 Actor/Critic。两者均移除前置 `gate_encoder` 和 `objective_value_embedding`；Actor/Critic 自身保留原有32维隐藏层。没有在图编码器输出与 Actor/Critic 之间另做升维。本次不增加 propagation gate 预测头。

两位 action mask 不属于 embedding，不进入 Actor/Critic，只在 logits 后应用。RL 第一次选择一路后由 `BacktraceLock.selected_wire` 锁定；若该 fanin 仿真值仍为 `U`，继续同一路而不再次调用 Actor；只有仿真确认 fanin 满足局部目标后才将该路屏蔽。回退撤销赋值后根据当前电路值重新计算 mask。

这些是本项目明确选择的模型配置，不标为严格复现论文原模型。

## 环境一：训练

训练环境需要 PyTorch、TensorBoard 和 `cpp_podem` Python 扩展。从 `PODEM` 目录执行：

```bash
conda activate d2l
python -m pip install -r python-requirements.txt
python -m pip install -e .

python scripts/run_smartatpg_training_linux.py \
  --output-dir artifacts/smartatpg_12d_co
```

训练入口只完成以下工作：

1. 使用基础 PODEM（默认 backtrack 上限500）测试 c6288 和 full-scan s38417 的完整 fault catalog。只保留成功检出（`outcome == 1`）的 fault，再按 backtracks、backtrace_steps 降序及 fault_id 升序，各选择最困难的100个；不包含达到上限未解决或已判不可测的 fault。任一电路成功检出不足100个时直接报错，不用未检出故障补齐。
2. 使用完全相同的 episode 顺序和共享超参数，分别训练30轮 fanin-mean 与 GAT-GRU；每轮各200个 episode。
3. 每轮确定性评估并保存 backtrack 表现最好的完整参数。
4. 分别导出包含完整图编码器和 Actor 参数的 `SMARTATPG_MODEL_V8`。
5. 准备16个评测电路和 faultmap，生成同时包含两个 best model 的 `benchmark_bundle/`。

训练入口不会编译独立 C++ 可执行文件，也不会运行最终 heuristic/RL 比较。中断后重新执行同一命令，会从 `training_state.pth` 继续。

训练清单已升级为 `SMARTATPG_PAPER_TRAINING_V2`，两种模型共享同一份200个 hard-detected fault。旧清单可能混有未检出故障，不能直接复用；请使用新的 `--output-dir` 重新准备和训练，不覆盖旧结果。本次不改变 PPO 的失败更新规则；启发式能检出的故障不保证当前 RL 策略也能在预算内检出。

TensorBoard：

```bash
tensorboard \
  --logdir artifacts/smartatpg_12d_co \
  --host 0.0.0.0 \
  --port 6006
```

主要输出：

- `smartatpg_mean/`：基线训练状态、V8 模型、每轮指标和 TensorBoard；
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
2. 对每个电路重新计算包含 CO 的12维原始特征；
3. 分别使用两个 V8 模型重新计算该电路每个 gate 的12维 embedding；
4. 使用 `g++` 重新编译 C++ PODEM；
5. 通过同一可执行文件分别运行 heuristic、fanin-mean 和 GAT-GRU；
6. 输出 JSON、CSV、Markdown 汇总和每次原生运行日志。

新电路必须分别用两个 V8 模型重新计算 embedding，不能复用其他电路的 embedding。C++ 为 agentATPG 在推理时拼接目标值；mask 不写入 embedding，也不送入 Actor，而在 logits 之后动态应用。

## 时间口径

正式 runtime 比较只采用 C++ 输出的 ATPG 区间时间 `atpg_seconds`。以下时间全部不计入 heuristic/RL 提升比例：

- 电路解析和12维特征构建；
- 两种图 embedding 的计算与文件导出；
- C++ 编译；
- Python 启动、进程启动和报告汇总；
- whole-process wall time。

embedding 计算时间单独写入 `preprocessing.json`，wall time 只保留在 `raw_results.json` 中用于排障。

## 兼容性

新工件为 V8 model / V6 embedding，特征标识为 `SMARTATPG_FEATURES_V3_12D_CO`，benchmark bundle 为 V5，分别记录每种模型的 Actor 输入维度（12或13）。旧11维 checkpoint 不能作为新结构的断点继续训练；请使用新的训练输出目录。旧 V5/V3、V6/V4、V7/V5 推理工件保留兼容读取，并使用原有前11列特征，不参与新对比。更早的14维特征、64/80维 descriptor 和缺少图权重的 V4 Actor 会被明确拒绝。
