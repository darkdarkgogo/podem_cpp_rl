# SmartATPG 11D BUF-free 使用说明

## 模型结构

每个 gate 的初始特征固定为11维：6维 gate 类型 one-hot，加上 level、fanout、CC0、CC1、CO。支持 `PI、AND、NAND、OR、NOR、NOT`，不支持 BUF、XOR/XNOR。输入数据中的显式 BUF 必须先被规范化；fanout stem 仍由一条 wire 的多个消费者表示，与 BUF 类型无关。

CC0、CC1 从 PI 向 PO 计算，CO 从 PO（值为0）向 PI 反向计算。AND/NAND 的旁路输入使用 CC1，OR/NOR 使用 CC0，NOT 增加一级代价；多扇出取最小可观测代价。CO 是静态结构特征，不随 PI 赋值更新，也不是可测性的证明。有限代价封顶为 `10**9`，无输出可达路径时原始 CO 为 `inf`；CO 和 CC0/CC1 一样使用按电路的 log1p 缩放，非有限值映射为1，避免网络输入出现 inf/NaN。

基线编码器使用一次 fanin mean 聚合，将当前 gate 的11维特征和 fanin 均值11维拼成22维，再经过可训练的 `Linear(22,11)` 和 ReLU，输出11维 embedding。对比编码器保持11维，按 level 完成一次正向 fanin GAT+GRU，再用独立参数完成一次反向 fanout GAT+GRU；两方向 GAT 投影矩阵均为 `11×11`。

SmartATPG 基线将11维 gate embedding 直接送入 Actor/Critic，不额外拼接目标值。agentATPG（代码变体 `level_gat_gru`）将11维 embedding 与当前1维目标值 `object_val`（0/1）拼接成12维，直接送入 Actor/Critic。两者均移除前置 `gate_encoder` 和 `objective_value_embedding`；Actor/Critic 自身保留原有32维隐藏层。没有在图编码器输出与 Actor/Critic 之间另做升维。本次不增加 propagation gate 预测头。

两位 action mask 不属于 embedding，不进入 Actor/Critic，只在 logits 后应用。RL 第一次选择一路后由 `BacktraceLock.selected_wire` 锁定；若该 fanin 仿真值仍为 `U`，继续同一路而不再次调用 Actor；只有仿真确认 fanin 满足局部目标后才将该路屏蔽。回退撤销赋值后根据当前电路值重新计算 mask。

这些是本项目明确选择的模型配置，不标为严格复现论文原模型。

## 环境一：训练

训练环境需要 PyTorch、TensorBoard 和 `cpp_podem` Python 扩展。从 `PODEM` 目录执行：

```bash
conda activate d2l
python -m pip install -r python-requirements.txt
python -m pip install -e .

chmod +x train_smartatpg_linux.sh benchmark_smartatpg_linux.sh tensorboard_smartatpg_linux.sh
./train_smartatpg_linux.sh
```

脚本先完成16个电路的 fault 筛选，再在 `--gpu` 指定的物理 GPU 上训练 GAT-GRU。子进程内部通过 `CUDA_VISIBLE_DEVICES` 只看见该卡，因此代码中显示为 `cuda:0`；实际物理卡编号记录在 `training_run_metadata.json`。正常8轮结束后，GAT-GRU 在同一 GPU 上执行最多5轮失败 fault 强化训练。

默认且唯一支持的普通训练轮数为8轮，backtrack 上限为2000。输出目录默认为 `artifacts/smartatpg_11d_co_nobuf_all16_8rounds_bt2000`；不自动沿用旧的12D或其他旧格式目录。

训练入口只完成以下工作：

1. 使用基础 SCOAP PODEM（backtrack 上限2000）测试16个标准电路各自的完整 fault catalog。只保留成功检出（`outcome == 1`）的 fault，再按 backtracks、backtrace_steps 降序及 fault_id 升序，每个电路选择最困难的50个；不足50个时直接报错，不用未检出故障补齐。
2. 按固定电路顺序训练8轮 GAT-GRU；每轮共16×50=800个 episode。
3. 每轮确定性评估并保存 backtrack 表现最好的完整参数。
4. 从最佳 GAT-GRU checkpoint 恢复参数，找出全部800个训练 fault 中仍未检出的项；强化阶段每轮只训练当前仍失败的 fault，每个 fault 一次，最多执行5轮。每轮结束后仍评估完整训练集；若全部检出则提前结束。
5. 强化候选只有在完整800-fault 训练集上优于当前最佳模型时才成为新最佳，避免局部改善造成整体退化。
6. 导出包含 GAT-GRU 图编码器和 Actor 参数的 `SMARTATPG_MODEL_V11`。
7. 准备16个评测电路和 faultmap，生成只包含强化后 GAT-GRU best model 的 `benchmark_bundle/`。

训练入口不会编译独立 C++ 可执行文件，也不会运行最终 heuristic/RL 比较。普通训练中断后重新执行同一命令，会从 `training_state.pth` 继续；GAT 强化中断后会从 `reinforcement_state.pth` 的当前 fault 继续。

训练清单格式为 `SMARTATPG_ALL_CIRCUITS_TRAINING_V4_11D_CO_NO_BUF`，包含16×50个 hard-detected fault。旧清单和旧 checkpoint 不能复用；请使用新的 `--output-dir` 重新准备和训练。本次不改变 PPO 的失败更新规则；启发式能检出的故障不保证当前 RL 策略也能在预算内检出。

TensorBoard：

```bash
./tensorboard_smartatpg_linux.sh
```

主要输出：

- `smartatpg_gat_gru/`：逐 level 双向 GAT-GRU 的普通训练和强化训练输出；其中 `model_best.txt` 是强化前模型，`model_best_reinforced.txt` 是最终打包模型，`reinforcement_metrics.json` 和 `reinforcement_unresolved_faults.json` 记录逐轮结果；
- `preparation/`：固定的训练 manifest、转换电路和 fault profile；
- `benchmark_bundle/`：交给环境二的自包含目录。

## 环境二：编译与评测

把整个 `benchmark_bundle/` 复制到评测环境。该环境只需要 Python 3 标准库和支持 C++11 的 `g++`，不需要 PyTorch、CUDA、NumPy 或 `.pth` checkpoint。

```bash
./benchmark_smartatpg_linux.sh
```

评测入口会：

1. 校验 bundle 内模型、电路和 faultmap 的哈希；
2. 对每个电路重新计算包含 CO 的11维原始特征；
3. 使用 GAT-GRU V11 模型重新计算该电路每个 gate 的11维 embedding；
4. 使用 `g++` 重新编译 C++ PODEM；
5. 通过同一可执行文件分别运行 heuristic 和强化后的 GAT-GRU；
6. 输出 JSON、CSV、Markdown 汇总和每次原生运行日志。

新电路必须用对应的 V11 模型重新计算 embedding，不能复用其他电路的 embedding。C++ 为 agentATPG 在推理时拼接目标值；mask 不写入 embedding，也不送入 Actor，而在 logits 之后动态应用。

## 时间口径

正式 runtime 比较只采用 C++ 输出的 ATPG 区间时间 `atpg_seconds`。以下时间全部不计入 heuristic/RL 提升比例：

- 电路解析和11维特征构建；
- GAT-GRU 图 embedding 的计算与文件导出；
- C++ 编译；
- Python 启动、进程启动和报告汇总；
- whole-process wall time。

embedding 计算时间既单独写入 `preprocessing.json`，也按模型汇总到 `FINAL_RESULTS.md`；wall time 只保留在 `raw_results.json` 中用于排障。

原生日志会额外打印 RL backtrace 的策略选择次数、Actor 实际前向次数、策略选择总时间、Actor 前向总时间及各自平均时间。最终报告使用各电路重复测量的中位时间汇总，并列出完整模型参数量和在线 Actor 参数量。GAT-GRU 保留按 gate 和目标值索引的 logit 缓存。

当前 V11 Actor 的 `11→32→2` 和 `12→32→2` 前向在 C++ 中使用固定尺寸专用内核，第一层权重会在加载时转置成适合连续更新32个神经元的布局。Linux 原生程序及扩展使用 `-O3 -march=native` 编译，以便在评测机器上进行循环展开和 SIMD 自动向量化；没有启用会改变浮点结合顺序的 fast-math。

## 兼容性

唯一受支持的新工件为 V10/V11 model 与 V7 embedding，特征标识为 `SMARTATPG_FEATURES_V4_11D_CO_NO_BUF`，benchmark bundle 为 `SMARTATPG_BENCHMARK_BUNDLE_V8_11D_CO_NO_BUF`，分别记录 Mean 的11维和 GAT-GRU 的12维 Actor 输入。所有旧 checkpoint、model 和 embedding 均不兼容并会在加载阶段被拒绝；请重新生成 manifest、重新训练并重新导出。
