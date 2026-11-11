# V11：独立实现的消融实验

当前任务（2026-11-04）：**confidence_only 的 seed 4 单种子重跑**，两个任务采用当前独立设置 MNIST batch=32、HWF batch=16；进度见 [重跑报告](../../../research/confidence_only_seed4_20261104_rerun/REPORT.md)。此次不改变用户现有参数；与旧 full 的 batch 不同，不能直接作为严格单模块配对比较。对应源码、协议回归检查和数据预检已通过；跨变体参数全等断言不适用于这次有意改变 batch 的重跑。

下文关于“本次未运行”的说明记录最初代码编写阶段；当前批次状态以运行报告为准。

版本为 `frt-abl-v11-independent-20261104`。当前独立实现保留；旧消融专用矩阵入口、运行数据、汇总及上批台账已按用户要求删除，原 full 检查点和其他实验保留。

## 独立代码布局

每个实验各有三个完整文件：

- `settings.py`：在代码中分别完整列出 MNIST-add2 和 HWF 参数，没有加载公共 JSON、导入 full 参数、继承配置或合并默认参数。
- `protocol.py`：该实验自己的图处理、阈值选择、软接纳、权重、课程和批次调度实现。
- `train.py`：该实验自己的完整训练循环及命令行入口，不调用 full 的训练函数，也没有 `--variant`、`resolve_policy` 或运行时模块开关。

目录为 `full/`、`confidence_only/`、`fixed_threshold/`、`no_curriculum/`、`no_weight/`、`no_boundary/`、`standard_abl/`、`anchor_only/`；另有独立的双因素实验 `fixed_no_curriculum/`。

`variant` 字段只用于实验身份校验、日志和输出命名，不能切换当前入口的实验行为。更改实验需要调用另一个目录里的入口。

CNN、候选搜索、图近似计算、校准器、交叉熵、EMA、数据文件校验和可选计时仍使用已有的底层函数。这些函数不提供共享实验参数，不负责选择消融，也不替代各实验自己的训练循环。图的参考库上限等原先隐式的数值已经在每个参数块里显式写出。

## 单模块差异

五个单模块消融保持 full 的非目标逻辑：找到支持候选就进入正域；未找到候选才进入边界；门控失败时仍使用 `alpha_start=0.95` 进行软缩放；所有适用实验保留较高阈值优先、90% 门控覆盖要求及失败回退；HWF 边界系数均明确写为 0.2，MNIST 为 0.1。

| 实验 | 唯一直接修改 | 保留的非目标行为 |
|---|---|---|
| `confidence_only` | 图近似替换为 `L=U=q` | full 阈值、软接纳、缩放、可靠性权重、课程、边界损失 |
| `fixed_threshold` | 阈值搜索集合固定为 `[0.95]` | 风险检验、校正族大小 20、覆盖规则、软接纳和所有损失 |
| `no_curriculum` | 课程权重全部为 1 | 可靠性权重、软缩放和前 10 轮的损失渐增 |
| `no_weight` | 去掉基础可靠性乘子 `L*q*(1-(U-L))` | 软接纳缩放及课程权重；最终权重并非统一为 1 |
| `no_boundary` | 训练代码中直接移除边界 KL 损失及对应前向 | 同样的候选划分、阈值、正域权重、课程、每轮更新数和 anchor 随机流 |

令 `s = f + (1-f)*min(1,max(0,L/alpha_ref))`。full 的基础伪标签权重为 `L*q*(1-(U-L))*s`；`no_weight` 的权重为 `s`。两者之后都使用同一形式的课程权重。`f` 在 MNIST 为 0、HWF 为 0.5；它是缩放公式中的参数，不是最终训练权重的绝对下限。交叉熵仍按 batch 内权重和归一化。

V10 的 `full_gate_coverage_floor`、`full_soft_scale_floor`、`full_boundary_weight` 在每个独立参数块中分别写为 `gate_coverage_floor`、`soft_scale_floor`、`boundary_weight`，数值与 full 对应，不再受实验名称条件判断控制。旧的未使用诊断参数 `alpha_decay_step`、`alpha_min`、`train_fraction` 不进入新配置。

`fixed_no_curriculum` 同时改变阈值集合和课程权重，属于双因素实验。`standard_abl` 是移除多项机制的组合对照，`anchor_only` 是仅有监督 anchor 的对照；二者不作为单模块消融解释。anchor-only 的完整循环不执行候选搜索、校准、伪标签更新或边界更新，但保留相同的每轮优化步数预算和 anchor 调度。

## 参数与结果隔离

各参数文件独立写入相同的非目标数值，并用静态检查防止抄写时产生偏差：MNIST batch=64、30 轮；HWF batch=32、50 轮；预热均为 5 epoch、Adam 学习率 0.001、EMA=0.99，课程和损失渐增均为 10 轮；其余数值见各自 `settings.py`。

每个入口分别接受 `--task` 和 `--seed`。同一参数函数每次构造全新的字典，不跨实验或跨调用共享可变对象。更改学习率、设备等实验参数应编辑对应实验的独立参数块，不从公共默认配置覆盖。

未来运行的结果根目录为 `methods/frt_abl/runs-v11-independent/`，每个 run 名称包含新版本、任务、实验名和种子。代码会保存该实验的实际参数、实现说明、整套 V11 源码及底层依赖源码哈希。所有实验使用相同范围的源码清单，以便配对汇总，同时与 V10 隔离。此轮未创建该结果目录。

## 入口与后续验证

下列命令仅展示计划，不会训练；本次也未执行这些计划入口：

```powershell
python -m methods.frt_abl.ablation_v11.full.train --task hwf --seed 4
python -m methods.frt_abl.ablation_v11.no_curriculum.train --task hwf --seed 4
python -m methods.frt_abl.ablation_v11.plan
```

`plan.py` 默认列出 2 任务 × 8 实验 × 5 种子 = 80 条独立入口命令；`--include-factorial` 可额外列出双因素实验。规划器没有执行功能。单个训练入口只有显式传入 `--execute-training` 才会构造模型和开始训练。旧 `methods.frt_abl.run_matrix` 已删除。

`check_source.py` 只解析源码、检查字面量参数、编译语法和比较 AST，不导入实验、不读数据、不创建模型。它检查独立配置、禁止公共配置与旧模块开关、核对非目标参数，以及每个单模块实现相对 full 的函数级差异。

`tests/test_independent_protocols.py` 已编写纯合成回归用例，覆盖门控关闭时的软接纳、固定阈值保留风险设置、阈值回退、去权重仍保留软缩放、课程样本集合及分层权重质量、设置对象隔离等。**用例尚未执行**；运行正确性和性能仍需后续验证，不能由静态检查替代。
