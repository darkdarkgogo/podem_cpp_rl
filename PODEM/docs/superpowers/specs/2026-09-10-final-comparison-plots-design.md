# 最终比较结果绘图设计

## 目标

新增一个 Python 命令，直接读取现有的 `final_comparison.csv`，不需要转换数据，
一次生成四张独立的分组柱状图。这些图按电路比较回溯搜索步骤数、回溯次数、
原生 ATPG 运行时间相对于 `smartatpg_mean` 的比值，以及三个模型的故障覆盖率。

## 输入格式

命令接收 `final_comparison.csv` 的路径作为位置参数。该 CSV 是
`scripts/benchmark_smartatpg.py` 生成的长表，每一行对应一个 `circuit` 与
`model` 的组合。绘图脚本需要以下字段：

- `circuit`
- `model`
- `backtrace_steps`
- `backtracks`
- `atpg_seconds`
- `fault_coverage`

必需的模型为 `heuristic`、`smartatpg_mean` 和 `smartatpg_gat_gru`。每个电路
必须分别包含这三个模型的一行数据。额外模型不参与这四张图。

## 命令与输出

实现文件为 `scripts/plot_final_comparison.py`，命令格式如下：

```text
python scripts/plot_final_comparison.py FINAL_COMPARISON_CSV [--output-dir DIR]
```

如果不指定 `--output-dir`，图片将保存在输入 CSV 所在目录。每次运行生成以下
四张高分辨率 PNG 图片：

- `backtrace_steps_by_circuit.png`
- `backtracks_by_circuit.png`
- `runtime_by_circuit.png`
- `fault_coverage_by_circuit.png`

如果输出目录中已经存在这些同名文件，脚本会覆盖它们，因为它们是根据输入 CSV
重新生成的结果。

## 绘图方式

每张图以电路名称作为横轴。回溯搜索步骤数、回溯次数和运行时间分别计算
`heuristic / smartatpg_mean` 与 `smartatpg_gat_gru / smartatpg_mean`，只绘制
这两根柱；`smartatpg_mean / smartatpg_mean = 1` 不绘制为柱，而用纵轴值为 1
的水平虚线表示基线。运行时间取自 `atpg_seconds`，它只计算原生 C++ ATPG
区间，不包含图嵌入和 Python 调度时间。如果某个电路的 `smartatpg_mean` 分母为
零，脚本会报告无法归一化，而不会生成无效比值。

故障覆盖率图不做相除，仍为每个电路绘制 `heuristic`、`smartatpg_mean` 和
`smartatpg_gat_gru` 三根柱。覆盖率在 CSV 中以 0 到 1 的比例保存，绘图时转换
成百分比，并将纵轴范围设为 0% 到 100%。

图片宽度随电路数量增加；必要时旋转横轴标签，并自动调整边距，防止标签被裁切。
四张图使用一致的标题、单位、图例、模型配色和网格样式。脚本使用 Python 标准库的
`csv` 模块读取输入，使用 Matplotlib 绘图，不引入 Pandas 依赖。

## 校验与错误处理

遇到以下情况时，命令会输出简洁、明确的错误信息并终止：输入文件不存在或为空、
缺少必需字段、指标包含非数值内容、同一个 `circuit` 与 `model` 组合重复，或者某个
电路缺少必需模型。输出目录不存在时，脚本会自动创建。

自动化测试使用一个符合实际基准测试表头的小型临时 CSV，验证四张 PNG 图片均能
生成、前三个指标能正确除以 `smartatpg_mean`、图中不包含 mean 柱、故障覆盖率
能正确转换成百分比，并检查格式错误和零分母时的提示。命令行冒烟测试使用非交互式
Matplotlib 后端，确认脚本可以从仓库根目录直接运行。
