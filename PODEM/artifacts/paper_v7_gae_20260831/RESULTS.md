# GAE 实测结论：部分指标改善，但并非全面优于旧版

完整训练和评测均已完成，旧 V6 模型没有被覆盖。

## 实验设置

- 与旧 V6 使用相同十电路数据划分、seed=2026、20 轮 BC、课程训练轮数 2/2/3。
- 新流程：完整 fault rollout，GAE，gamma=0.99，lambda=0.97，reward / 100，Actor advantage 标准化，不标准化单 fault return。
- 实际完成 4,200 个 fault episode、4,001 次 PPO 更新、358,655 次 Actor 决策；CPU 单线程训练用时 67.1 分钟。
- 统一使用 `build/atpg_rl_v4.exe`、seed=14、回溯上限 500。四种策略交错运行，每电路每策略 1 次预热、5 次正式测量，共 240 次成功运行。
- 下面的时间是各电路 5 次测量中位数的总和。ATPG 阶段耗时不包含全部启动/加载成本，端到端耗时包含它们。

## 真正的最终 GAE 策略

| 指标，十电路合计 | 启发式 | 旧 V6 最佳 | 最终 GAE |
|---|---:|---:|---:|
| 加权检出故障数 | 133312 | 133307 | 133337 |
| 加权故障总数 | 135344 | 135344 | 135344 |
| 回溯超限故障数 | 395 | 378 | 390 |
| ATPG 阶段耗时，秒 | 15.311 | 15.281 | 14.734 |
| 端到端耗时，秒 | 16.464 | 18.494 | 18.144 |

检出数采用 uncollapsed 加权计数，超限数采用 collapsed fault 计数，二者不是同一计数口径。

相对旧 V6：最终 GAE 多检出 30 个加权故障，ATPG 耗时减少 3.6%，端到端耗时减少 1.9%，但超限故障增加 12 个。覆盖率从 98.4949% 变为 98.5171%，提升约 0.0222 个百分点。

相对启发式：多检出 25 个加权故障，ATPG 耗时减少 3.8%，但端到端耗时增加 10.2%。不能只看 ATPG 阶段就宣称整体运行更快。

## 电路间的取舍

- `c3540`：比旧 V6 多检出 30 个加权故障，ATPG 耗时减少 14.2%。
- `c1908` / `c2670`：分别多检出 8 / 4 个加权故障，ATPG 耗时也下降。
- `c7552`：少检出 6 个加权故障，超限数从 84 增至 112，ATPG 耗时从 1.750 秒增至 2.308 秒，增加 31.9%。
- `s38417_scan`：ATPG 耗时减少 7.5%，但少检出 2 个加权故障。
- 总体提速主要来自 `s38417_scan`；去掉它，另外九个电路的 ATPG 耗时合计由 4.213 秒升至 4.491 秒，增加 6.6%。
- 按加权检出数逐电路比较：3 个电路改善、4 个退化、3 个持平。

## 验证集与模型选择

旧 V6 的最佳验证结果是 497/500；新流程初始 BC 为 496/500，最终 GAE 为 494/500。所有后续 GAE/PPO checkpoint 都未超过 BC 的完整选择分数，因此自动选择的 `actor_v2_best.txt` 实际是 BC 回退模型，不是最终 GAE 模型。

上表测的是 [actor_v2_latest.txt](<E:/桌面/cpp podem/PoDemFan_N-detect_ATPG_Test_Compression/PODEM/artifacts/paper_v7_gae_20260831/actor_v2_latest.txt>)。BC 回退模型的完整电路加权检出数与启发式相同，ATPG 耗时为 16.364 秒。

验证集逐 fault 评测与完整电路 ATPG 的样本、计数及运行流程不同。本次结果表明，两种评测的排名并不完全一致，不能仅凭全电路结果反向把最终 GAE 宣称为验证最佳模型。

## 限制与校验

这只是一个训练 seed；完整电路评测包含训练/验证 fault，不是未见电路泛化实验。与旧版相比还同时改变了 gamma、Actor advantage 标准化和 Critic 初始化，不能把所有差异归因于 GAE 本身。

5 次重复用于检查计数确定性和计时波动，不等于 5 个独立训练 seed。按每次重复的十电路 ATPG 总耗时看，GAE 四次较快、一次持平；这里没有做统计显著性声明。

240 次成功评测的重复计数检查通过；旧启发式/V6 的 20 组历史结构性结果全部复现。旧模型文件和训练代码哈希未变，训练指标全部有限，奖励归因计数不一致为 0。

首次评测曾误用不支持 RL/fault-map 参数的 `src/atpg.exe`，该次失败日志已保留且未用于任何结果；正式评测统一切换至 RL 支持版，实际二进制路径与哈希已记录在实验配置和 protocol 中。

## 数据入口

- [逐电路最终 GAE 对比](<E:/桌面/cpp podem/PoDemFan_N-detect_ATPG_Test_Compression/PODEM/artifacts/paper_v7_gae_20260831/final_policy_comparison.md>)
- [自动选择模型与验证曲线](<E:/桌面/cpp podem/PoDemFan_N-detect_ATPG_Test_Compression/PODEM/artifacts/paper_v7_gae_20260831/comparison.md>)
- [四策略完整 CSV](<E:/桌面/cpp podem/PoDemFan_N-detect_ATPG_Test_Compression/PODEM/artifacts/paper_v7_gae_20260831/benchmark_20260831_111818/summary.csv>)
- [实验配置与校验记录](<E:/桌面/cpp podem/PoDemFan_N-detect_ATPG_Test_Compression/PODEM/artifacts/paper_v7_gae_20260831/experiment.json>)
