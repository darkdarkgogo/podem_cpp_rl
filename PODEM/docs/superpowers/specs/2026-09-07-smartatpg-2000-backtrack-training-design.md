# SmartATPG 2000 回溯上限训练设计

## 目标

将当前 SmartATPG 实验的故障筛选、mean 与 GAT-GRU 训练验证、portable/native
推理和最终 benchmark 的 backtrack limit 从 500 统一提高到 2000。新的实验重新
profile 两个训练电路，并在传统启发式于 2000 次回溯内检测到的故障中，每个电路
选择回溯代价最高的 100 个。

新的 mean 和 GAT-GRU 模型都从随机初始化开始训练。不读取、不转换，也不兼容
原有 500-limit 模型或训练 checkpoint。

## 方案

Linux 训练入口使用新的输出目录，先以 2000 次回溯上限重新生成训练 manifest，
再分别从零训练 mean 和 GAT-GRU。两个模型共用同一份新 manifest、随机种子和
训练参数，保证模型架构仍是唯一的实验变量。

旧实验目录保持不变，避免覆盖已有 checkpoint、TensorBoard 数据和 benchmark
结果。新训练不会增加旧模型加载参数，也不会尝试恢复旧输出目录中的训练状态。

## 回溯上限与数据流

Linux 训练入口默认使用 2000，并把同一个值传给准备脚本。准备脚本对 c6288 和
full-scan s38417 的全部折叠故障运行传统启发式 PODEM，禁用 fault dropping，
记录每个故障的 outcome、backtracks 和 backtrace steps。筛选仍只接受 detected
故障，并按 backtracks、backtrace steps 降序选择每个电路的前 100 个。

生成的 training_manifest.json 保存 backtrack_limit 为 2000。训练脚本只从
manifest 读取上限，使训练 episode 和每轮固定 200-fault 验证使用相同的 2000。

训练完成后，mean 与 GAT-GRU 的新最佳模型组成新的 benchmark bundle。Linux
benchmark 入口和底层 benchmark 脚本默认使用 2000，并将该值传给 heuristic、
mean 和 GAT-GRU 三种模式，确保最终推理比较采用同一搜索预算。

## 新旧实验隔离

2000-limit 实验使用独立的新输出目录。准备 manifest、训练 checkpoint、
TensorBoard、导出模型、bundle 和 benchmark 结果均不会与旧 500-limit 实验混写。

如果新输出目录已经存在兼容的 2000-limit checkpoint，现有恢复逻辑可以继续一次
被中断的新训练。如果目录包含 500-limit manifest 或 checkpoint，配置和 manifest
校验必须拒绝恢复。这里的恢复只支持新的 2000-limit 实验自身，不构成旧模型兼容。

训练 metadata 记录实际回溯上限；训练 checkpoint 的 config 继续包含 2000，
后续只能在相同 manifest 和配置下恢复。

## 随机种子

随机种子保持当前分工：profile seed 控制 C++ 随机填值，training seed 控制 Torch
随机初始化、随机状态和每轮 episode 顺序，benchmark seed 控制三种模式的最终
向量补全。默认 seed 不因回溯上限变化而改变。

## 错误处理

- 重新使用包含 500-limit manifest 的准备目录时，现有 resume 校验必须拒绝它。
- 新实验必须使用新目录重新 profiling，并从随机参数开始训练。
- benchmark 的回溯上限必须为正；生成的运行 metadata 必须记录实际值 2000。
- 训练、bundle 或 benchmark 的必要新工件缺失时，错误信息必须包含具体路径。

## 验证

自动测试覆盖：准备脚本、双模型训练启动器、benchmark 启动器及底层 benchmark
的默认回溯上限为 2000；准备命令和 benchmark 命令都收到 2000；训练从 manifest
读取 2000；500-limit manifest 不能作为新实验恢复。

端到端检查使用小型测试电路验证 profiling、训练调用和推理调用均收到同一个
2000。Linux 实际训练启动前确认新 manifest 重新生成且包含 backtrack_limit
为 2000。训练完成后，最终 benchmark metadata 和原生命令必须显示 -bt 2000。
