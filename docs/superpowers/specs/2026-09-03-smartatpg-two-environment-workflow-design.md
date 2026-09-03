# SmartATPG 双环境训练与评测流程设计

## 目标

将 SmartATPG 的训练与 C++ 编译评测彻底拆分，避免 PyTorch/CUDA 训练环境和 C++ 编译运行环境互相影响。

最终只向用户暴露两个 Linux 入口：

1. `scripts/run_smartatpg_training_linux.py`：在包含 PyTorch 的训练环境中准备数据、训练模型并导出可移植模型包。
2. `scripts/run_smartatpg_benchmark_linux.py`：在不包含 PyTorch 的评测环境中重新编译 C++、为每个待测电路重新计算 embedding，并完成 heuristic/RL 对比。

DeepGate 和 DeepGate2 源码不在本次修改范围内。

## 训练环境

训练入口执行以下步骤：

1. 使用基础 PODEM 对 c6288 和全扫描 s38417 的完整 fault catalog 排序，各选择前100个 fault。
2. 按既定参数训练 SmartATPG 20轮，每个 fault 是一个 episode，并在每个 episode 后更新 PPO、RND、Actor、Critic 和 GraphSAGE。
3. 保存最新 checkpoint、backtrack 最优 checkpoint、Actor、轮次指标和 TensorBoard 日志。
4. 从最优 checkpoint 导出可移植的 `SMARTATPG_MODEL_V5` 模型文件。

`SMARTATPG_MODEL_V5` 必须包含：

- backend、11维特征 schema、GraphSAGE 配置、11维 gate embedding、13维 policy state 和 snapshot 身份；
- GraphSAGE `Linear(22, 11)` 的 `weight[11,22]` 和 `bias[11]`；
- Actor 推理所需的 gate encoder、objective-value embedding 和 backtrace actor 参数；
- 最佳训练轮次及最佳评测指标。

训练导出只保存模型参数，不把某个固定电路的 gate embedding 当作模型参数。GraphSAGE 的 W 和 bias 是随强化学习共同训练后的最终参数。

## 评测环境

评测入口仅依赖 Python 标准库、C++ 编译器和项目源码，不导入 PyTorch、NumPy 或 CUDA。它执行以下步骤：

1. 校验 `SMARTATPG_MODEL_V5` 的格式、维度、有限数值和 snapshot。
2. 将待测组合电路转换成二输入形式；对 s 系列电路先执行 full-scan 转换。
3. 为每个新电路重新构造每个 gate 的11维原始特征：7维类型 one-hot、level、fanout、CC0、CC1。
4. 使用模型中的 GraphSAGE W 和 bias 执行一次 fanin mean 聚合：

   `h_v = ReLU(W * concat(x_v, mean(x_u for u in fanin(v))) + bias)`

5. 将该电路每个 gate 的11维 `h_v` 导出为带 schema、snapshot 和电路哈希的 embedding 文件。
6. 调用 `g++` 重新编译 C++ PODEM。
7. 使用同一可执行文件、同一电路、faultmap、seed 和 backtrack limit，分别运行 heuristic PODEM 与 RL PODEM。
8. RL 模式由 C++ 加载 Actor 和该电路刚计算的 embedding；两个动态 action mask 只在运行时加入13维 state，不写入 embedding。
9. 汇总 fault coverage、backtracks、backtrace steps 和 C++ 报告的 ATPG runtime，输出 JSON、CSV、Markdown 与原始日志。

性能比较只使用 C++ PODEM 内部计时区间报告的 ATPG runtime。GraphSAGE 特征构建、embedding 计算与导出、C++ 编译、进程启动和其他 Python 编排时间均不计入 heuristic/RL 时间提升比例。wall runtime 可以保留在原始记录中用于排障，但不进入正式汇总和性能比较。

评测输入默认覆盖16个电路，也允许用户传入新的 BENCH 电路。新电路必须重新计算 embedding，不能复用其他电路的 embedding。

## 环境交接

训练环境与评测环境通过可移植目录交接。目录内部只使用相对路径，并对模型、电路、faultmap 和 embedding 记录哈希，复制到另一台机器或另一套环境后仍可校验。

评测环境的 Python 代码直接读取 V5 文本模型并执行小规模矩阵计算。这样既保留“新电路必须重新计算 embedding”的语义，也避免在第二个环境安装 PyTorch。

## 脚本整理

保留以下通用或当前流程组件：

- `convert_binary_bench.py`
- `convert_full_scan_bench.py`
- `prepare_smartatpg_training.py`
- `train_smartatpg.py`
- `prepare_smartatpg_benchmark.py`
- `benchmark_smartatpg.py`
- `build_native.py`
- 新的两个 Linux 入口脚本

删除已经被当前 SmartATPG 流程替代的旧脚本：

- `export_cpp_actor.py`
- `export_cpp_embeddings.py`
- `prepare_curriculum_training.py`
- `prepare_paper_training.py`
- `run_smartatpg_linux.py`
- `select_hard_faults.py`
- `train_cpp_podem.py`
- `train_curriculum.py`
- `train_paper_rnd.py`
- `verify_curriculum_v4.py`
- `verify_full_fault_gae.py`
- `verify_paper_v3.py`
- `verify_smartatpg.py`
- `verify_xor_fault_filter.py`

文档和测试不得继续引用被删除的脚本。

## 错误处理

- 旧 SmartATPG 80维 descriptor、旧 V2/V3/V4 模型或维度不匹配时明确报错，不能静默解释成 V5。
- 模型 snapshot、Actor、GraphSAGE 参数和 embedding snapshot 不一致时拒绝运行。
- 新电路包含 XOR/XNOR、非二输入逻辑门、环、未定义驱动或非有限特征时停止并提示先转换。
- 已存在的输出与 manifest 哈希不一致时停止，避免把不同实验结果混合在同一目录。
- C++ 编译或任一次 benchmark 失败时保留日志并返回非零状态。

## 验证

测试至少覆盖：

1. V5 导出确实包含训练后的 GraphSAGE W、bias 和全部 Actor 参数。
2. 同一电路上，纯 Python 标准库重算的 embedding 与 PyTorch `MeanGraphEncoder` 输出在浮点容差内一致。
3. 修改 GraphSAGE W 后，新电路 embedding 随之变化。
4. 两个入口互不调用：训练入口不编译或跑最终 benchmark，评测入口不导入 Torch 或加载 `.pth`。
5. C++ 能加载 V5 Actor 与新生成的11维 embedding，并拒绝 snapshot 不匹配。
6. heuristic/RL 比较继续使用相同 fault、seed、backtrack limit、预热和重复次数，时间提升只比较 C++ ATPG runtime。
7. `scripts` 中只剩设计规定的当前脚本，文档和测试没有悬空引用。
