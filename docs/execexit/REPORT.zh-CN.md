<a id="execexit-testing-public-execution-signals-for-stopping-code-reasoning"></a>

# ExecExit：检验公开执行信号能否决定代码推理的停止时机

[English](REPORT.md) · 简体中文

这是 README 背后的完整报告。README 只讲主线，这里把方法细节、每个数字的出处、局限和复现方式都摊开。

[结果](#results) · [CPU 示例与复现](#reproduction-and-appendices)

<a id="problem"></a>

## 问题

我最初注意到两件事：模型写出的中间草稿可能通过公开样例，却过不了完整测试；而模型明明已经写出能用的程序，还会继续想很久。于是我把问题拆成四步：更早的正确草稿是否存在？信号能否认出它？按信号停下是否安全？算上探测成本后，能否更快交付？

四个问题的简要回答：

- **有可省的空间。** 知道答案的 oracle 在已记录的草稿中可节省 **14.79%**（Qwen）和 **13.11%**（DS）的 token。
- **信号能认出正确草稿。** 三个模型上，PUB 相对元数据基线的 AUROC 提升 **+0.065 到 +0.152**，区间都在零以上；模型置信度（CONF-50）则比基线低 **0.14–0.15**。
- **按信号停下未通过安全检验。** Qwen3-8B 上 HYB-B 节省 **7.91%** 的 token，通过率显著下降 **1.88** 个百分点 [−3.39, −0.63]（区间不含 0），下界越过 −3 个百分点的预设非劣效容限。
- **时延没有可确认的改善。** 服务实验中异步 HYB-B / N 的时延比为 **0.983 [0.870, 1.101]**，区间包含 1；异步 HYB-B 实际生成的 token 总数反而比 N 多 **3.72%**。

合起来看，这个项目最重要的结果是把“能认出正确草稿”和“可以安全停下”分开了：前者在三个模型上都成立，后者没有成立，两者之间的缺口主要来自公开测试放过的错误草稿。

<a id="related-work-and-scope"></a>

### 相关工作与范围

| 工作 | 相关先例 | 我的做法 |
|---|---|---|
| [interwhen](https://arxiv.org/html/2602.11202v3#A3.SS1.SSS5) | 探测中间 Lean 代码，并根据编译成功提前停止。 | 在公开样例上检查 Python 草稿，直接交付被接受的草稿；失败的检查结果不反馈给原生推理。 |
| [RLEF](https://arxiv.org/html/2410.02089v2) | 用执行反馈训练迭代式代码修复；通过公开测试可以结束交互。 | 在固定模型的原生推理轨迹上评估停止，不做修复训练。 |
| [Certaindex / Dynasor](https://arxiv.org/html/2412.20993v2) | 在服务中使用中间答案的一致性，并过滤犹豫词。 | 比较公开执行信号与非执行信号，计入访问过的探测及回退成本。犹豫词过滤不是我的新贡献。 |
| [STP / OTBench](https://arxiv.org/html/2610.13570v1) | 沿固定轨迹研究多个前缀处的停止机会，包括代码任务。 | 在策略回放中计入访问过但被拒绝的探测及交付答案的成本，另外单独测量服务时延。 |
| [DEER](https://arxiv.org/abs/2504.15895) | 使用转折词探测与置信度提前退出。 | CONF-50 是组件级的比较对象，不是 DEER 的完整复现。 |

这些工作和我的问题很近。我在此基础上做的是受控比较：把几类信号放进同一套成本核算、同一批题目里。

<a id="setup-and-evidence-status"></a>

## 实验设置

主比较用 Qwen3-8B，DeepSeek-R1-Distill-Qwen-7B（下文简称 DS 或 DS-Qwen）是事先登记的次要模型。每个模型跑 **199 道题、650 个样本**：LiveCodeBench v6 留出集（H-v6）中的 AtCoder **63 × 4**、LeetCode **63 × 4**，以及 LiveCodeBench Pro（下文简称 LCB Pro）**73 × 2**。统计时先在每题内平均，再让各题等权，这样采样次数多的题不会占更大权重。[勘误结果清单](../../runs/qwen_erratum_comparison/summary.json)

原生思考预算是 **28,000 token**，用完就强制作答。草稿用贪心解码生成，最长 **2,048 token**，全部保存下来，之后我按 v1 注释中止规则离线回放各种停止规则。成本核算上我尽量不给提前停止占便宜：被拒的草稿、交付的草稿和回退后的原生生成都计费，交付的草稿只计一次；工具生成的 token 在首次需要时计费，CPU 成本另报。这种固定轨迹上的 token 核算和 C4 实测的服务成本是两回事，我分开报告。

N 是原生生成；BF@4096 在第 4,096 个 token 处尝试作答，无效就回退。FIRST、CONF-50、CONV-TEXT 和 EG 用不同的信号决定是否接受草稿。**43 个配置**我全部保留；CONF-full 只用于信号诊断。[全部 Qwen 区间](../../runs/qwen_erratum/secondary_intervals43/FULL_TABLES.md)

候选位置按已生成的原生 token 计数，不含提示词，由转折词事件与固定位置共同确定。

| 信号或策略 | 操作定义 | 决策与信息边界 |
|---|---|---|
| **CONF-50** | 对前**最多 50 个实际生成的草稿 token** 计算 `exp(mean(rank-one log probabilities))`，即 top-1 概率的几何平均值。正常结束但较短的草稿用已有的 token；不足 50 个 token 且尚未结束的前缀没有可用分数。 | 分数**严格 > 0.95** 才允许继续生成完整草稿，否则拒绝。接受仍需要有效的完整草稿，但不要求 PUB PASS。 |
| **HYB-B** | 固定 **k=4, m=0**：按位置排列候选，选取第 **1、5、9、…** 个，访问其中位于 **4,096** 之前的候选，接受第一份有效且通过 PUB 的草稿。如果都没有被接受，再单独检查一次恰好位于 **4,096** 的候选，不受上面稀疏日程的限制。 | 边界草稿同样必须有效且 PUB PASS，否则继续 N，不再做后续探测。如果生成在到达边界前就结束了，不凭空补造候选。 |
| **PUB** | 在固定的题目执行与比较规则下，取草稿**第一次在公开样例上的执行结果**。PASS 要求全部公开样例通过；UNKNOWN 不能触发接受。 | 停止策略看不到隐藏测试结果，也看不到经重复评估、多数裁决后的标签。离线正确性是另一个独立的评估目标：原始完整标签要求完成交付，并同时通过公开样例与隐藏测试；C2 的 PRIV-IND 标签先去掉与公开输入完全相同的测试，再用剩下的隐藏测试结果。 |

有效草稿须正常结束、能按规则提取出无歧义且可解析的程序，并且没有触发 v1 注释中止。[候选选择](../../exec_exit/replay/policies_v1.py)、[HYB-B 实现](../../exec_exit/replay/hybrid_v1.py)

策略比较用按数据集×难度分层、按题配对的 **10,000 次 bootstrap 重抽样**。平均通过率可能掩盖个别题受到的损害，所以我另设了一个更严格的判据：只要某题有一个配对样本在 N 下通过、在 HYB-B 下失败，这道题就算受损；受损题比例的单侧 **97.5%** Clopper–Pearson（CP）上界必须低于 **3%**。信号的 AUROC 比较按题分组做交叉验证；这类区间和后文的相关系数区间都来自 **10,000 次**按题 bootstrap 重抽样，涉及交叉验证时固定折外预测、不重新拟合。[冻结统计输出](../../runs/qwen_erratum/statistics/summary.json)

第三个模型 DeepSeek-R1-Distill-Llama-8B（下文简称 Llama）是后来补的，用同样的 **199 道题、650 个样本**。Qwen/DS 独立标签的补全过程和剩余缺失放在补充材料里；本报告的图表都直接读取保存的统计输出。[Llama 结果与审计](../../runs/llama_results/summary.json)

<a id="evidence-status"></a>

### 证据状态

我在看到留出结果之前写下了假设和分析计划，但并不是每项分析都是这样，所以每个结果开头都按“时间 · 角色”标明来源，例如 **预登记 · 主检验**。

时间指这项分析是什么时候写下的：

- **预登记：** 在留出结果出现前写下，包括 Qwen 的主检验和事先指定的次要分析；DS-R1-Qwen 当时就登记为次要模型。
- **补充登记：** 之后追加、但在相应结果出来之前写下，包括 Llama 的 C2 复现（README 中的“登记复现”）、ACD 假设和隐藏状态探针。
- **事后：** 看到相关结果后才提出或修改的分析，例如 Qwen/DS 的独立标签补全和公开测试强度分析；文中的“探索性”分析都属于这一类。

角色指这项结果在研究中的作用：

- **主检验：** Qwen 上的 C3 主比较，整个研究只有这一项。
- **次要：** 登记时列为次要结果的比较。设了判据的按判据判定，没设判据的只报告区间。
- **描述性：** 只作描述的数值，例如计数、点估计或冻结代码顺带输出的区间；没有事先设定的判据。

事后分析不再细分角色。同一段里各模型身份不同时，我在段首分别写明，例如“Qwen/DS：事后；Llama：补充登记 · 次要”。次要区间都是逐项的名义区间，没有做多重比较校正。

<a id="results"></a>

## 结果

<a id="main-results-figure"></a>

![三组区间：公开执行改善信号区分能力，但未确立策略安全性或服务时延改善](figures/main_results.svg)

*图 1。左：PUB 相对元数据基线的 AUROC 提升；中：HYB-B 相对 N 的通过率差；右：异步 HYB-B 与 N 的平均时延比。三栏刻度各自独立，误差线为名义 95% 区间；中栏 −3 个百分点的虚线只对 Qwen 和 DS 有判据意义。[绘图程序](render_results.py)。*

<a id="c1-earlier-passing-drafts-create-a-bounded-opportunity"></a>

### C1：更早通过的草稿最多可省 13–15% 的 token

**预登记 · 次要。** 先看空间有多大。假设有一个知道答案的 oracle，在已记录的草稿里挑最便宜的正确答案：Qwen 能节省 **14.79% [11.73, 18.19]** 的 token；**304/650** 条轨迹里出现过比最终答案更早的通过草稿，位置余量的中位数是 **4,301.5 token**。DS 对应的节省是 **13.11% [9.81, 16.80]**。[Qwen 来源](../../runs/queue_rebuild/c1c2/result/c1_aggregate_rebuilt.json)、[DS 来源](../../runs/ds_c1c2_erratum/summary.json)

这是理想上限：oracle 知道最终正确性，草稿成本只算原生前缀加草稿自身生成的 token，未被选中的草稿和所有检查都不计成本，没有可通过的选项时保留 N。位置余量按第一份通过的草稿计算，同样不含探测和检查成本。[Qwen C1](figures/c1_qwen.png)、[DS C1](figures/c1_ds.png)

<a id="c2-confidence-execution-and-the-metadata-baseline"></a>

<a id="c2-public-execution-carries-information-beyond-the-metadata-baseline"></a>

### C2：公开执行提供元数据基线之外的信息

**Qwen/DS：事后；Llama：补充登记 · 次要（R1）。** C2 问的是：在只知道难度、已用 token 数和探测序号的元数据基线之外，公开执行还能多给多少信息。标签用 PRIV-IND（只用与公开输入不重合的隐藏测试），并在这个标签上重新拟合基线。Qwen/DS 算事后，是因为补全标签的决定是在看到第一次结果之后才做的；Llama 的 R1 在结果出来前就登记了。[补全定义与状态](../../runs/c2_label_completion/final_counts.json)

| 模型 | PUB AUROC，PRIV-IND | 重拟合基线 AUROC | PUB − 基线 [95% 置信区间] | 元数据＋PUB AUROC | 已知／符合条件的草稿；题数 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 0.8694 | 0.8044 | +0.0649 [0.0189, 0.1153] | 0.9313；Δ +0.1268 [0.0960, 0.1620] | 4,203/4,204；197 |
| DS-R1-Qwen-7B | 0.9376 | 0.8157 | +0.1219 [0.0780, 0.1728] | 0.9590；Δ +0.1433 [0.1039, 0.1887] | 7,023/7,128；199 |
| DS-R1-Llama-8B | 0.9235 | 0.7720 | +0.1515 [0.0995, 0.2105] | 0.9349；Δ +0.1628 [0.1188, 0.2127] | 6,094/6,099；199 |

*来源：[Qwen](../../runs/c2_label_completion/raw/qwen_completed/main/summary.json)、[DS-Qwen](../../runs/c2_label_completion/raw/ds_completed/main/summary.json)、[Llama](../../runs/llama_execution_repair1/blind/analysis/results/private_c2.json)。三个模型分别还有 **1、105 和 5** 份草稿的标签为 UNKNOWN，我把它们排除在比较之外，没有当作 FAIL。*

“元数据＋PUB AUROC”一列回答的是另一个问题：已经知道元数据之后，再加上执行信号能提升多少；其中 Δ 是这个联合模型相对重拟合基线的提升。三个模型的提升分别为 **+0.127、+0.143、+0.163**，区间都在零以上：执行结果提供了元数据之外的区分信息。这一列标为 Qwen/DS：事后；Llama：补充登记 · 描述性（冻结代码顺带的输出）。

CONF-50 沿用原始完整标签（交付的程序须同时通过公开样例与隐藏测试），和 PRIV-IND 是不同的目标。原始标签下 PUB 比较的循环依赖问题见[补充表](#original-complete-label-c2-results)。

| 模型 | CONF-50 AUROC，原始完整标签 | 元数据基线 AUROC | CONF-50 − 基线 [95% 置信区间] |
|---|---:|---:|---:|
| Qwen3-8B | 0.680 | 0.817 | −0.138 [−0.190, −0.081] |
| DS-R1-Qwen-7B | 0.676 | 0.824 | −0.148 [−0.200, −0.095] |
| DS-R1-Llama-8B | 0.6551 | 0.7960 | −0.1409 [−0.2099, −0.0727] |

*Qwen/DS：预登记 · 次要；Llama：补充登记 · 次要（R2 复现）。[Qwen 数据行](../../runs/qwen_c1c2_erratum/raw/signal_metrics.jsonl)、[DS-Qwen 数据行](../../runs/ds_c1c2_erratum/raw/signal_metrics.jsonl)、[Llama 数据行](../../runs/llama_execution_repair1/blind/analysis/results/raw/signal_metrics.jsonl)。*

Llama 的两项登记复现都按判据成立（R1 区间下界 > 0，R2 区间上界 < 0），方向与前两个模型一致。Qwen/DS 的 PUB 总体优势在超时敏感性分析下仍然成立，但并不均匀：Qwen 的 AtCoder、LeetCode 区间和 DS 的 LeetCode 敏感性区间都包含零。能认出正确草稿，不等于按它停下就安全，这一点由 C3 检验。

<a id="policy-outcomes"></a>

<a id="c3-the-stopping-policy-did-not-establish-the-primary-claim"></a>

### C3：停止规则节省了 token，但未通过非劣效检验

**预登记 · 主检验（Qwen）。** 在 Qwen3-8B 上，HYB-B 节省 **7.91%** 的 token，通过率下降 **1.88** 个百分点（95% 置信区间 −3.39 到 −0.63）。区间不含 0，说明下降是真实的；下界越过 −3 个百分点的预设容限，名义非劣效未成立；更严格的受损判据同样未满足。下表同时列出勘误前的结果，结论以勘误版为准（勘误原因见[后文](#lcb-pro-public-input-defect-and-correction)）。[原版与勘误版比较表](../../runs/qwen_erratum_comparison/raw/comparisons.json)

| 版本 / 比较 | 通过率差，百分点（95% 置信区间） | token 节省，%（95% 置信区间） | 登记判定 |
|---|---:|---:|---|
| 原版：HYB-B − N | −1.13 [−2.76, 0.38] | 7.53 [4.61, 10.77] | 仅名义非劣效成立；已被勘误版取代 |
| 勘误版：HYB-B − N | −1.88 [−3.39, −0.63] | 7.91 [4.97, 11.15] | 名义与严格非劣效均未成立 |
| 原版：HYB-B − BF@4096 | 1.63 [−0.50, 3.77] | −10.22 [−14.88, −6.01] | 顺序检验未成立 |
| 勘误版：HYB-B − BF@4096 | 1.63 [−0.50, 3.77] | −9.77 [−14.36, −5.56] | 顺序检验未成立 |

<details>
<summary>Qwen3 主结果：原冻结版与勘误版</summary>

| 版本 | N 题均通过率 | BF@4096 题均通过率 | HYB-B 题均通过率 | HYB-B 平均 token 数 |
|---|---:|---:|---:|---:|
| 原冻结版 | 36.68% | 33.92% | 35.55% | 16,007.06 |
| 勘误版 | 40.95% | 37.44% | 39.07% | 15,941.62 |

| 集合 | HYB-B − N，百分点（原冻结版 → 勘误版） | token 节省 %（原冻结版 → 勘误版） | 至少一次受损的题 / 题数 | 单侧 97.5% CP 上界 |
|---|---:|---:|---:|---:|
| 总体 | −1.13 → −1.88 | 7.53 → 7.91 | 11 / 199 | 9.68% |
| AtCoder | 0.40（不变） | 9.73（不变） | 0 / 63 | 5.69% |
| LeetCode | −6.35（不变） | 21.35（不变） | 11 / 63 | 29.10% |
| LCB Pro | 2.05 → 0.00 | −1.66 → −0.81 | 0 / 73 | 4.93% |

按集合看，变化只出现在 LCB Pro；各集合的受损题数和上界两版相同。只看 N 至少通过一次的题，原冻结版受损 **11/91**（上界 20.60%），勘误版 **11/102**（上界 18.48%）。

[完整比较字段与来源哈希](../../runs/qwen_erratum_comparison/summary.json) · [逐集合比较、N 至少通过一次的子集、救回／受损明细](../../runs/qwen_erratum_comparison/raw/comparisons.json) · [43 个配置的次要区间](../../runs/qwen_erratum/secondary_intervals43/FULL_TABLES.md)

</details>

按严格判据，勘误后有 **11/199** 道题受损，CP 上界 **9.68%**，远高于 3%。N 从未解出的题不可能受损，所以我还预登记了一项次要结果：只看 N 至少解出一次的题目，受损为 **11/102**，上界 **18.48%**。[来源](../../runs/qwen_erratum_comparison/raw/comparisons.json)

**预登记 · 次要（DS）。** HYB-B − N 为 **+0.75 个百分点 [−1.13, 2.76]**，token 节省 **13.39% [9.38, 17.51]**。名义非劣效通过，严格判据仍未通过（**10/199** 道受损；上界 **9.05%**）。[来源](../../runs/queue_rebuild/ds_checkpoint_recovery1/result/statistics/summary.json)

**补充登记 · 次要（Llama，无预设判据）。** HYB-B − N 为 **−0.63 个百分点 [−2.39, 0.88]**，token 节省 **10.72% [6.68, 15.09]**；BF@4096 − N 为 **−4.15 个百分点 [−6.41, −2.14]**，token 节省 **38.18% [34.12, 42.30]**。[43 配置输出](../../runs/llama_execution_repair1/blind/analysis/results/intervals43.json)

**补充登记 · 描述性（Llama）。** 按 C3 的受损定义，Llama 的受损题为 **11/199**。[保存的统计](../../runs/llama_execution_repair1/blind/analysis/results/statistics.json)

名义非劣效要求 HYB-B − N 的 95% 区间下界**严格高于 −3 个百分点**。这个判据只用于 Qwen 主比较和 DS 次要比较。[判据实现：summarize](../../exec_exit/analysis/stats.py)

![全部已保存配置的勘误后结果](figures/pareto_43.png)

*三个模型全部 43 个配置的通过率差与 token 节省，误差线为逐项名义 95% 区间，ORACLE 为离线参照。−3 个百分点的虚线只适用于 Qwen 与 DS 的 HYB-B − N 比较。[来源与筛选条件](../../runs/report_figures/sources.json)*

<a id="policy-harm-and-public-false-acceptance"></a>

#### 策略损害与公开样例的误放行

策略为什么会让题目受损？下表把受损题按题集拆开，并给出被公开样例检查放行的草稿里还有多少是错的。

左栏沿用 C3 的受损定义，标为 Qwen/DS：预登记 · 次要；Llama：补充登记 · 描述性。[Qwen 损害](../../runs/qwen_erratum_comparison/raw/comparisons.json)、[DS 损害](../../runs/queue_rebuild/ds_checkpoint_recovery1/result/statistics/summary.json)、[Llama 损害](../../runs/llama_execution_repair1/blind/analysis/results/statistics.json)

右栏是 **PRIV-IND** 标签下放行草稿中的错误比例，即 1 − 加权精确率，各题等权。分号后的“原始计数”是未加权的 `fp/(tp+fp)` 草稿数；Qwen/DS 这六格的计数和原完整标签下相同。右栏标为 Qwen/DS：事后；Llama：补充登记 · 描述性。[Qwen PRIV-IND](../../runs/c2_label_completion/raw/qwen_completed/main/summary.json)、[DS PRIV-IND](../../runs/c2_label_completion/raw/ds_completed/main/summary.json)、[Llama PRIV-IND](../../runs/llama_execution_repair1/blind/analysis/results/private_c2.json)

| 数据集 | Qwen 受损 / 题数 | DS 受损 / 题数 | Llama 受损 / 题数 | Qwen 放行错误比例 [95% 区间]；原始计数 | DS 放行错误比例 [95% 区间]；原始计数 | Llama 放行错误比例 [95% 区间]；原始计数 |
|---|---:|---:|---:|---:|---:|---:|
| AtCoder | 0/63 | 2/63 | 3/63 | 25.6% [14.7, 38.1]；262/706 | 21.5% [11.6, 34.5]；221/563 | 15.2% [6.9, 26.6]；149/526 |
| LeetCode | 11/63 | 7/63 | 7/63 | 35.0% [24.6, 46.2]；504/1,022 | 27.6% [17.6, 40.0]；306/672 | 39.9% [27.2, 54.6]；450/752 |
| LCB Pro | 0/73 | 1/73 | 1/73 | 34.2% [20.6, 49.7]；96/278 | 44.9% [28.2, 63.8]；120/218 | 37.5% [22.1, 56.4]；100/215 |

![按题集的 HYB-B 受损题与 PRIV-IND 下放行草稿中的错误比例](figures/safety_by_collection.png)

受损题主要集中在 LeetCode（Qwen 11/11、DS 7/10、Llama 7/11）。右栏统计的是所有通过公开样例检查的草稿，它们不一定被策略选中，所以右栏不是实际交付的错误率。逐题的分析见[后文机制部分](#mechanism-public-test-strength-exploratory)。

**Qwen/DS：预登记 · 次要；Llama：补充登记 · 次要。** 按置信度停止的代价更明显。在同一个冻结的 CONF-50 `k1,m0` 配置下，相对 N 的通过率差为：Qwen **−12.44 个百分点 [−15.95, −9.17]**（节省 **45.71%**），DS **−1.38 个百分点 [−3.27, 0.38]**（节省 **51.59%**），Llama **−6.16 个百分点 [−8.54, −3.89]**（节省 **44.31%**）。损害有多大因模型而异。[Qwen 区间](../../runs/qwen_erratum/secondary_intervals43/raw/comparisons.jsonl)、[DS 区间](../../runs/queue_rebuild/ds_checkpoint_recovery1/result/secondary_intervals43/raw/comparisons.jsonl)、[Llama 区间](../../runs/llama_execution_repair1/blind/analysis/results/intervals43.json)

**预登记 · 次要（消融）。** 我也试过在公开样例之外多加检查：自生成的压力测试（STRESS）和与前一份草稿行为一致（AGREE）。在 AtCoder `k4,m0` 子集上，HYB-B、EG-2 和 EG-S 都是 **121/252** 通过，EG-A 和 EG-SA 是 **120/252**，平均计费 token 数依次为 **15,058.08、17,198.52、17,869.50、17,832.68 和 18,618.42**。多加的检查没有比“等第二份通过公开样例的草稿”更好，反而更贵。其余登记网格保留在[消融图](figures/atcoder_ablation.png)与[源表](../../runs/qwen_atcoder_ablation/heldout/comparison_tables.json)中。

<a id="service-costs"></a>

<a id="c4-token-savings-did-not-establish-a-service-latency-improvement"></a>

### C4：token 节省未确立服务时延改善

**预登记 · 次要。** 离线省下的 token 能不能变成真实的时间？我用 Qwen3-8B 在 **40 道题**（18 道 AtCoder、22 道 LCB Pro，均为 stdin/stdout 接口）上实际跑了服务实验，每道题在每个实验组和负载下各跑一次。串行主比较中，异步 HYB-B / N 的平均端到端时延比为 **0.983 [0.870, 1.101]**，同步版本为 **0.992 [0.891, 1.092]**，区间都包含 1。区间为按题配对 bootstrap 重抽样的 95% 区间。同一批题上，异步 HYB-B 实际生成的 token 总数反而比 N 多 **3.72%**（见下文成本部分）。[保存的时延输出](../../runs/c4_recovery1/latency_analysis/summary.json)

**事后（敏感性分析）。** 实验中途出过中断。我在原冻结之后、看结果之前声明排除中断附近的索引 **28–30**（`2121A`、`2121B`、`2101E`），剩下 **37 道题**，异步 / N 为 **0.950 [0.838, 1.071]**。这不是完全盲态：之前为了估计运行时长，我看过计时。两个区间都包含“没有变化”。[声明与输出](../../runs/c4_recovery1/sensitivity/summary.json)

![C4 主分析与敏感性估计](figures/c4_latency.png)

**预登记 · 描述性。** H 是暂停后直接恢复、不生成草稿的对照组。下表的正确性来自对实际交付程序的独立完整测试。

| 串行实验组 | 平均端到端时延，秒 | 已交付程序通过数 / 计划数 |
|---|---:|---:|
| N | 227.219 | 16/40 (40.0%) |
| BF@4096 | 212.671 | 15/40 (37.5%) |
| CONF-50 | 151.651 | 12/40 (30.0%) |
| HYB-B-SYNC | 225.497 | 15/40 (37.5%) |
| HYB-B-ASYNC | 223.258 | 14/40 (35.0%) |
| H | 223.200 | 15/40 (37.5%) |

*来源：[串行时延](../../runs/c4_recovery1/latency_analysis/summary.json)、[实际交付程序评分](../../runs/c4_recovery1/delivered_evaluation/summary.json)。通过数是每个实验组和负载下 AtCoder 与 LCB Pro 两部分之和。*

**预登记 · 描述性（负载实验）。** 同样的 40 道题也在两种泊松到达负载下运行。冻结的到达速率为 `lambda = c / S`，其中 `S` 是 N 的串行平均时延；**c=4 和 c=8 是负载参数，不是固定的实际并发数**。每种负载下 N 和异步 HYB-B 各运行一次，题目顺序与到达时间序列相同。c=4 时先跑 N 后跑异步，c=8 时先跑异步后跑 N。

| 泊松负载 | 平均时延比 | p50 时延比 | p90 时延比 | N 通过数 / 计划数 | 异步 HYB-B 通过数 / 计划数 |
|---|---:|---:|---:|---:|---:|
| c=4 | 1.158 | 1.424 | 1.116 | 15/40 (37.5%) | 16/40 (40.0%) |
| c=8 | 0.886 | 0.959 | 0.998 | 16/40 (40.0%) | 16/40 (40.0%) |

*各比值均为对应时延统计量的异步 HYB-B / N，小于 1 表示更快。这些是保存的点估计，没有置信区间。来源：[泊松负载时延](../../runs/c4_recovery1/latency_analysis/summary.json)、[交付程序评分](../../runs/c4_recovery1/delivered_evaluation/summary.json)。*

c=4 时平均时延变高，c=8 时变低，但 p90 几乎没变。每种负载只跑了一次，看不出可重复的加速，也看不出交付质量的差别。

**预登记 · 描述性（成本与拒绝开销）。** 实际服务中，串行异步 HYB-B 一共生成了 **812,296 个 token**，反而比 N 的 **783,162** 个多 **3.72%**（含可见的中止尾部）。对于至少探测过一次、但草稿全部被拒的题目，异步时延平均多花 **4.77%**（**34 题**），同步多花 **4.25%**（**35 题**），都在登记的 5% 描述性检查以内。[成本输出](../../runs/c4_recovery1/latency_analysis/summary.json)

全部 **400 个计划逻辑请求**（240 个串行、160 个泊松）都恢复出了时延与 PASS/FAIL 标签。三次中断留下的未落盘尾部和取消开销观测不到，重启的影响也没有单独识别。[恢复与计时来源](../../runs/c4_recovery1/sensitivity/summary.json)

<a id="hidden-state-probe-addendum"></a>

<a id="hidden-state-probes"></a>

<a id="hidden-state-probes-no-established-advantage-over-pub-conditional-gain-only-for-llama"></a>

## 隐藏状态探针：未确立优于 PUB；仅 Llama 的条件比较有增益

**补充登记 · 次要。** 除了运行草稿，模型内部是否已经“知道”草稿对不对？我在已知 C2 结果之后、做开发集选择和留出集拟合之前登记了这项后续研究（不读标签的特征提取开始得更早）。做法是在完整草稿末端的隐藏状态上训练单层逻辑回归，标签用补全后的 PRIV-IND；层和正则化参数在开发集 Dev121 上选好后锁定。H-P1 问探针能否胜过 PUB；H-P2 问在已通过 PUB 的草稿里，探针能否胜过重新拟合的元数据基线。题集仍是原来那批，不是新的盲测。[完成记录](../../runs/probe_execution/FULL_TABLES.md)

| 模型 | H-P1：探针 − PUB 的 AUROC 差 [95% 置信区间] | H-P2：PUB 通过子集内探针 − 基线的 AUROC 差 [95% 置信区间] | 已知草稿／题目；PUB 通过草稿／题目 |
|---|---:|---:|---:|
| Qwen3-8B | +0.0148 [−0.0202, +0.0487] | +0.0456 [−0.0111, +0.1024] | 4,203/197；2,006/148 |
| DS-R1-Qwen-7B | −0.0074 [−0.0346, +0.0165] | +0.0615 [−0.0115, +0.1341] | 7,023/199；1,453/133 |
| DS-R1-Llama-8B | −0.0187 [−0.0538, +0.0133] | +0.1721 [+0.0930, +0.2508] | 6,094/199；1,493/133 |

*数值及各模型的判定都转录在[完整表格](../../runs/probe_execution/FULL_TABLES.md)中。[Qwen 输出](../../runs/probe_execution/fit/qwen/heldout/summary.json)、[DS-Qwen 输出](../../runs/probe_execution/fit/ds/heldout/summary.json)、[Llama 输出](../../runs/probe_execution/fit/llama/heldout/summary.json)。*

![PUB 通过子集内元数据、CONF-50 与草稿末端探针的绝对 AUROC](figures/signals_after_public_pass.png)

*PUB 通过子集采用 PRIV-IND 标签。元数据与 CONF-50 画出了保存的 95% 区间；探针的绝对 AUROC 没有保存区间，所以只画点。*

**Qwen/DS：事后；Llama：补充登记 · 次要。** 在已通过公开样例的草稿里，CONF-50 − 重拟合元数据基线的 AUROC 差，Qwen、DS-Qwen、Llama 分别为 **−0.124 [−0.206, −0.041]**、**−0.070 [−0.151, 0.014]**、**−0.062 [−0.158, 0.035]**，只有 Qwen 的区间排除零。[Qwen](../../runs/c2_label_completion/raw/qwen_completed/main/summary.json)、[DS-Qwen](../../runs/c2_label_completion/raw/ds_completed/main/summary.json)、[Llama](../../runs/llama_execution_repair1/blind/analysis/results/private_c2.json)

H-P2 的三个差值为 **+0.046 / +0.062 / +0.172**，只有 Llama 的区间排除零。拆开看，探针的绝对 AUROC 是 **0.782 / 0.796 / 0.836**，元数据基线是 **0.737 / 0.735 / 0.664**：Llama 的差值较大，一部分来自它的基线偏低，一部分来自探针略高。[Qwen 探针](../../runs/probe_execution/fit/qwen/heldout/summary.json)、[DS-Qwen 探针](../../runs/probe_execution/fit/ds/heldout/summary.json)、[Llama 探针](../../runs/probe_execution/fit/llama/heldout/summary.json)

**结论：H-P1 在三个模型上都不成立，H-P2 只在 Llama 上成立。** 探针还需要先生成完整草稿、再多做一次前向；它能否变成更好的停止规则、能否省时间，我还没有测。[次要结果、方法限制、审计与资源](#probe-methods-and-resources)

<a id="mechanism-public-test-strength-exploratory"></a>

<a id="exploratory-public-test-strength-miss-rates-are-associated-with-policy-harm"></a>

## 公开测试强度：漏检率与策略损害相关（探索性）

**事后。** C3 的受损题集中在 LeetCode，我想知道是不是那里的公开样例太弱。计划是在已知 C2 和损害汇总之后、逐题计算之前写下的。我把每道题的**漏检率**定义为：错误草稿中通过 PUB 的比例（分母和前面“放行草稿中的错误比例”不同）。[保存的估计](../../runs/public_test_strength/summary.json)

**M1：漏检率是不是题目的属性？** 如果是，不同模型在同一道题上的漏检率应当相近。结果确实正相关（每对模型只纳入两边各至少有三份错误草稿的题目）。不过这既可能是因为题目的测试弱，也可能是因为模型犯了相同的错，这里分不开。

| 模型对 | 符合条件的题目 | Spearman ρ [95% 置信区间] |
|---|---:|---:|
| Qwen3 / DS-R1-Qwen | 149 | 0.644 [0.520, 0.750] |
| Qwen3 / DS-R1-Llama | 149 | 0.581 [0.448, 0.697] |
| DS-R1-Qwen / DS-R1-Llama | 170 | 0.695 [0.585, 0.789] |

*来源：[保存的 M1 相关系数](../../runs/public_test_strength/summary.json)。*

**M2：是不是公开样例太少？** 看起来不是。对跨模型合计至少有三份错误草稿的题目，样例数与合并漏检率的相关系数，总体为 **+0.14 [0.0009, 0.29]**（**181** 道题），AtCoder 为 **−0.14 [−0.39, 0.12]**（**56**），LeetCode 为 **−0.08 [−0.33, 0.19]**（**53**），LCB Pro 为 **−0.20 [−0.30, −0.11]**（**72**）。总体为正、各题集内部却为负，说明这个关系取决于怎么汇总，不能说样例数本身在起作用。LCB Pro 的样例数几乎不变，**443/10,000** 次重抽样的相关系数无定义，区间用其余的重抽样计算。[M1/M2 来源](../../runs/public_test_strength/summary.json)

**M3：能不能用别的模型预测受损？** 对每个目标模型，我用另外两个模型的合并漏检率给题目排序，看它能否区分 HYB-B 是否降低了该模型的**逐题平均通过比例**。纳入的题目要求目标模型下 N 至少通过一次，且另外两个模型至少有一份错误草稿。

| 目标模型 | 符合条件的题目 | 按该定义受损的题目 | 其他模型漏检率的 AUROC [95% 置信区间] |
|---|---:|---:|---:|
| Qwen3 | 91 | 10 | 0.91 [0.82, 0.98] |
| DS-R1-Qwen | 53 | 9 | 0.76 [0.57, 0.93] |
| DS-R1-Llama | 53 | 8 | 0.92 [0.82, 0.99] |

这里的 Qwen 10/91 和主检验的 11/199 不是一回事：主检验看任一配对样本是否 N 通过而 HYB-B 失败，这里要求逐题平均通过比例下降，且题目要能从其他模型算出分数。Llama 的区间去掉了 **3** 次结果无定义的重抽样。[M3 来源](../../runs/public_test_strength/summary.json)、[主要损害来源](../../runs/qwen_erratum_comparison/raw/comparisons.json)

受损题主要在 LeetCode，所以一个自然的质疑是：漏检率会不会只是题集身份的代理？这项检查是我**看到 M3 之后**才加的。只用题集身份打分，Qwen/DS-Qwen/Llama 的 AUROC 为 **0.84/0.64/0.76**。只看 LeetCode 内部，其他模型漏检率的 AUROC 仍有：Qwen **0.91 [0.79, 0.98]**（**10/36** 道受损），DS-Qwen **0.75 [0.46, 0.98]**（**6/22**），Llama **0.89 [0.69, 1.00]**（**6/18**）。DS 的区间包含 0.5，样本也小（三个模型分别有 **0/6/10** 次结果无定义的重抽样）。[LeetCode 内部检查](../../runs/public_test_strength/posthoc_within_leetcode.json)

**M4：描述性汇总。** AtCoder/LeetCode/LCB Pro 的公开样例数中位数分别为 **3/2/1**。按题等权的平均漏检率，Qwen 为 **21%/37%/13%**，DS-Qwen 为 **10%/19%/7%**，Llama 为 **6%/21%/8%**。一个意外是：Qwen 的漏检率在三个题集上都最高，而它也是解题最多的模型，原因这里分不开。[描述性计数与符合条件的样本](../../runs/public_test_strength/summary.json)

**小结。** 在符合条件的已有题目上，其他模型的漏检率能区分 HYB-B 降低了目标模型逐题平均通过比例的题目（M3，AUROC **0.76–0.92**），为跨模型风险分层提供了离线证据。这些是相关性结果，漏检率的因果来源以及对新题的泛化，还需要干预实验和新题验证。

<a id="lcb-pro-public-input-defect-and-correction"></a>

## LCB Pro 公开输入缺陷与勘误

这是改变我主结论的那个问题。LCB Pro 抽取的公开输入里混进了空行，会让本来正确、按行读取输入的程序出错，也就是说，正确的草稿被冤枉成没通过公开样例检查。我对照 Codeforces 官方 HTML 核对了全部 **73 道题、78 个样例**。修正后的全部 **78 个输入**与结构化 HTML 解析结果逐字节一致；**70 个输出**逐字节一致，其余 **8 个**只有行尾空白不同，我保持原样。[验证收据](../../runs/lcb_public_erratum/official_html_v3/verification_receipt_v3.json)、[通过审计的勘误](../../runs/qwen_erratum_comparison/raw/comparisons.json)

修正统一应用到所有策略的公开样例检查、最终标签、回放和统计上，我没有重新生成任何模型输出；隐藏测试结果只在缓存身份完全一致时复用。修正之后，主比较的名义非劣效从成立变成了不成立。

<a id="limitations-and-interpretation"></a>

## 局限与解释

- **模型：** 三个 7–8B 模型、两个基座家族。它们的规模和训练历史都不同，我分不开基座和训练各自的影响，也不知道结论能否推广到更大的模型。
- **污染：** H-v6 的题目早于 Qwen3 发布；LCB Pro 题目较新，但同样不能排除污染。[审计结果](../../runs/llama_results/summary.json)
- **题目与轨迹固定：** 所有结论都建立在这批题目和这批采样轨迹上。LCB Pro 每题只采样两次，更难观察到受损；分层的异质性也不完全满足 CP 公式要求的独立同概率假设。[统计局限](../../runs/qwen_erratum_comparison/raw/comparisons.json)
- **信号指标的范围：** 只针对符合条件的草稿，不能代表完整的生成过程。
- **服务实验：** 有中断，也有观测不到的取消开销，限制了时延结论。[C4 局限](../../runs/c4_recovery1/sensitivity/summary.json)

<a id="supplementary-analyses-and-historical-diagnostics"></a>

## 补充分析与历史诊断

<details>
<summary>原始完整标签 C2 表与当前三模型信号图</summary>

<a id="original-complete-label-c2-results"></a>

原始标签表覆盖 Qwen **197 题、4,204 份草稿**，DS **199 题、7,128 份草稿**。基线、按题交叉验证和区间方法与 C2 相同；其他口径见[附录](../../runs/report_figures/C2_APPENDIX.md)。[Qwen 来源](../../runs/qwen_c1c2_erratum/raw/signal_metrics.jsonl)、[DS 来源](../../runs/ds_c1c2_erratum/raw/signal_metrics.jsonl)

| 原始完整标签（包含 PUB），总体可交付草稿 | Qwen AUROC | 与基线之差（95% 置信区间） | DS AUROC | 与基线之差（95% 置信区间） |
|---|---:|---:|---:|---:|
| 元数据基线 | 0.817 | — | 0.824 | — |
| CONF-50 | 0.680 | −0.138 [−0.190, −0.081] | 0.676 | −0.148 [−0.200, −0.095] |
| CONF-full | 0.772 | −0.045 [−0.086, −0.002] | 0.814 | −0.010 [−0.054, 0.037] |
| PUB | 0.879 | +0.062 [0.016, 0.112] | 0.947 | +0.123 [0.078, 0.175] |

*CONF 两行：预登记 · 次要；PUB 行：事后（原协议登记的是 PUB 精确率／召回率）。精确的筛选条件见[绘图数据](../../runs/report_figures/raw/plot_data.json)。*

**预登记 · 次要。** PUB 的加权精确率，Qwen 为 **0.685 [0.611, 0.753]**，DS 为 **0.717 [0.636, 0.788]**；匹配的基线精确率分别为 **0.701** 和 **0.675**，两个配对精确率差的区间都跨过零。在原始完整标签下，PUB 的召回率在两个模型上都是 **1.000**，但这是由定义决定的（见下一段）。所以我登记的精确率优势没有确立。[信号表](../../runs/report_figures/C2_APPENDIX.md)

**标签循环依赖。** 完整标签要求同时通过公开样例与隐藏测试，PUB 因而是目标的一部分，完美召回不能当作独立的预测证据。超时多数裁决也可能让第一次的 PUB 信号与最终标签不一致。[裁决实现](../../exec_exit/execute/evaluate_dev.py)、[C2 输入构造](../../runs/ds_c1c2_erratum/source/heldout_c1c2.py)

![三模型 AUROC 对比：独立标签下的公开执行与原始标签下的模型置信度](figures/signal_quality_3models.png)

*三个模型总体可交付草稿的 95% 区间。左：PRIV-IND 下 PUB − 重拟合基线；右：原始完整标签下 CONF-50 − 基线。[数值与来源](../../runs/readme_figure/values.json)。*

</details>

<details>
<summary>补全 PRIV-IND：标签补全、超时敏感性与剩余 UNKNOWN</summary>

<a id="completed-priv-ind-post-hoc-completion-and-timeout-sensitivity"></a>

### 补全 PRIV-IND：事后补全与超时敏感性

**事后。** 补全规则是我在查看选择性缺失诊断之后修订、在新一轮执行之前写下的。补全时保留 LeetCode 的调用顺序和同一个 `Solution` 实例；与公开输入完全相同的调用不计入标签，它们失败时继续往下执行，计入标签的隐藏测试调用失败时仍然停止。CPU 超时复测和多数裁决只用计入标签的隐藏测试结果。我只补此前为 UNKNOWN 的标签，已知标签、PUB、题目折分和估计设置都不变。[执行与合并记录](../../runs/c2_label_completion/final_counts.json)

我一共完成了 **2,867** 个唯一的题目／程序作业，对应 **2,980** 处原先为 UNKNOWN 的草稿引用。下表按草稿计数：同一个程序可能被多次引用，完成的作业也可能仍留下 UNKNOWN。[最终计数](../../runs/c2_label_completion/final_counts.json)

| 模型 | 此前 UNKNOWN | 新增 PASS | 新增 FAIL | 主分析仍为 UNKNOWN | 敏感性分析 UNKNOWN |
|---|---:|---:|---:|---:|---:|
| Qwen | 843 | 6 | 836 | 1 | 38 |
| DS | 2,137 | 13 | 2,019 | 105 | 215 |

剩余的 UNKNOWN 为 Qwen **1/4,204**、DS **105/7,128**，都在 LeetCode；计划的作业没有缺失。在 **104** 个仍为 UNKNOWN 的唯一程序中，**15** 个的结果数与计时数对不上，其中 **10** 个第一个保留用例就带 `MemoryError`；由于缺少原始数组和调用栈，我无法判断是执行、计时还是错误格式化出了问题。超时敏感性分析把新补全、且只因超时而失败的草稿（Qwen **37** 份、DS **110** 份）改回 UNKNOWN；规则同时涵盖 CPU 超时和墙钟超时，这次实际都是 CPU 超时。此前已知的标签保持不变。[Qwen 合并](../../runs/c2_label_completion/raw/qwen_completed/summary.json)、[DS 合并](../../runs/c2_label_completion/raw/ds_completed/summary.json)、[失败类型与剩余 UNKNOWN](../../runs/c2_label_completion/final_counts.json)、[一致性检查](../../exec_exit/execute/private_sequence.py)

Llama R1 的 **6,099** 份草稿中有 **6,094** 份标签已知，覆盖 **199 题**。剩下的 **5** 份都是没通过 PUB 的 LeetCode 草稿，执行记录对齐失败，其中 **3** 份带 `MemoryError` 元数据，我把它们保留为 UNKNOWN。R2 以及策略／ACD 使用的完整标签没有 UNKNOWN。[未知项详情](../../runs/llama_results/raw/unknown_details.json)

下表是 PUB − 重拟合基线的 AUROC 差及 **95%** 区间。主分析覆盖 Qwen **197** 题、DS **199** 题；UNKNOWN 留在记录里，但不进入比较。[Qwen 主分析](../../runs/c2_label_completion/raw/qwen_completed/main/summary.json)、[Qwen 敏感性分析](../../runs/c2_label_completion/raw/qwen_completed/timeout_sensitivity/summary.json)、[DS 主分析](../../runs/c2_label_completion/raw/ds_completed/main/summary.json)、[DS 敏感性分析](../../runs/c2_label_completion/raw/ds_completed/timeout_sensitivity/summary.json)

| 模型 | 样本范围 | 主分析 ΔAUROC [95% 置信区间] | 超时敏感性 ΔAUROC [95% 置信区间] | 已知草稿，主分析 / 敏感性分析 |
|---|---|---:|---:|---:|
| Qwen | 总体 | +0.0649 [0.0189, 0.1153] | +0.0641 [0.0179, 0.1145] | 4,203 / 4,166 |
| Qwen | AtCoder | −0.0027 [−0.0768, 0.0795] | −0.0027 [−0.0768, 0.0795] | 1,469 / 1,469 |
| Qwen | LeetCode | +0.0203 [−0.0730, 0.1245] | +0.0175 [−0.0767, 0.1240] | 1,842 / 1,805 |
| Qwen | LCB Pro | +0.1397 [0.0710, 0.2212] | +0.1397 [0.0710, 0.2212] | 892 / 892 |
| DS | 总体 | +0.1219 [0.0780, 0.1728] | +0.1205 [0.0762, 0.1719] | 7,023 / 6,913 |
| DS | AtCoder | +0.0596 [0.0002, 0.1310] | +0.0596 [0.0002, 0.1310] | 2,559 / 2,559 |
| DS | LeetCode | +0.0672 [0.0059, 0.1487] | +0.0602 [−0.0032, 0.1433] | 2,630 / 2,520 |
| DS | LCB Pro | +0.1838 [0.1231, 0.2535] | +0.1838 [0.1231, 0.2535] | 1,834 / 1,834 |

两种补全都没有改变 PUB 通过子集的数据行或估计，所以[下文的子集表](#historical-priv-ind)沿用原值；原始 CONF 结果与 R2 也不变。[Qwen 合并检查](../../runs/c2_label_completion/raw/qwen_completed/summary.json)、[DS 合并检查](../../runs/c2_label_completion/raw/ds_completed/summary.json)

**测量限制。** 新补全在 Gold 6430 上运行，旧标签来自 Platinum 8470Q；超时敏感性分析既不能分离 CPU 差异的影响，也不能消除选择偏差。LeetCode 的公开调用仍可能通过实例或全局状态影响后续测试；去掉输入完全相同的测试，也不能消除语义重合、覆盖有限或数据污染的问题。[补全来源与缺失情况](../../runs/c2_label_completion/final_counts.json)

</details>

<a id="historical-priv-ind"></a>
<details>
<summary>独立标签定义、输入审计与公开通过子集</summary>

<a id="priv-ind-post-hoc-calculation-under-a-definition-written-before-calculation"></a>

**历史背景。** 最初计算时，Qwen 的总体区间包含零，而且标签存在选择性缺失。我随后修订了补充登记 §3，补全那些被遇错即停截断的标签，再重新计算 R1。[当前状态](../../runs/c2_label_completion/final_counts.json)

PRIV-IND 先把 CRLF 转成 LF、恰好去掉一个末尾 LF，然后剔除与公开输入相同的隐藏测试用例。用例保持原始顺序，超时裁决只用保留下来的结果，没有观察到的用例记为 UNKNOWN。C2 的函数、折分、重抽样种子和权重都不变；[补充登记 §3 原文](../../runs/c2_label_independence/label_definition.zh-CN.md)写于新增标签与估计之前。

输入审计从隐藏测试中去掉了 AtCoder **63/2,288**、LeetCode **10/2,367**、LCB Pro **5/4,243** 个用例，没有哪道题因此失去全部隐藏测试。如果再去掉行尾空格／制表符和全部末尾空行，匹配数**一个也没有增加**；这条描述性规则没有改变 PRIV-IND。[输入审计](../../runs/c2_label_independence/raw/overlap_summary.json)

最初的标签缺失并不随机：Qwen **843/4,204**、DS **2,137/7,128** 份草稿的 PRIV-IND 标签为 UNKNOWN，而且全都是 PUB 失败的草稿，其中 LeetCode 分别占 **821** 和 **2,063**。原因是旧版测试框架遇错即停：它在同一个 `Solution` 实例中先运行公开样例、再运行隐藏测试，只要公开部分有一个失败，后面的结果就观察不到。补全后的合并记录保留了这些旧计数，并核对过此前已知的标签没有变化。[Qwen 旧计数与合并](../../runs/c2_label_completion/raw/qwen_completed/summary.json)、[DS 旧计数与合并](../../runs/c2_label_completion/raw/ds_completed/summary.json)

**事后。** PUB 通过子集的这项分析使用原始折分，在子集内重新拟合元数据基线。下表数值为 AUROC [95% 置信区间]；样本数为草稿数/题目数。

| 样本范围 | Qwen 基线 / CONF-50 | Qwen 样本数 | DS 基线 / CONF-50 | DS 样本数 |
|---|---|---:|---|---:|
| 总体 | 0.7368 [0.6686, 0.8002] / 0.6124 [0.5571, 0.6654] | 2,006/148 | 0.7349 [0.6590, 0.8054] / 0.6649 [0.6046, 0.7233] | 1,453/133 |
| AtCoder | 0.7265 [0.5951, 0.8436] / 0.6779 [0.5962, 0.7522] | 706/49 | 0.8253 [0.7181, 0.9113] / 0.6901 [0.5804, 0.7897] | 563/45 |
| LeetCode | 0.7652 [0.6590, 0.8554] / 0.6002 [0.5196, 0.6793] | 1,022/59 | 0.7794 [0.6727, 0.8692] / 0.6993 [0.6155, 0.7795] | 672/55 |
| LCB Pro | 0.6500 [0.4798, 0.8102] / 0.5650 [0.4363, 0.6911] | 278/40 | 0.5240 [0.3558, 0.6863] / 0.5614 [0.4083, 0.7067] | 218/33 |

*来源：[Qwen](../../runs/c2_label_completion/raw/qwen_completed/main/summary.json)、[DS](../../runs/c2_label_completion/raw/ds_completed/main/summary.json)。每个有效折的训练与验证部分都含两类标签；UNKNOWN 折数为 **0**。全部区间都基于 **10,000** 次有效的 bootstrap 重抽样。*

</details>

<details>
<summary>被放弃的正确草稿（ACD）：原生失败样本中 5.6–11.0% 曾有可通过的强制作答草稿；两项 Llama 假设未得到支持</summary>

<a id="abandoned-correct-drafts-the-two-llama-hypotheses-were-not-supported"></a>

**Qwen/DS：事后；Llama：补充登记 · 次要。** ACD 指的是：完整的强制作答草稿通过了公开样例与隐藏测试，最终的原生答案却失败；稳定 ACD 还要求相邻两份草稿都通过。每个模型的分母是 **650** 个样本，区间按整题配对 bootstrap 重抽样。三个模型都存在 ACD：占全部样本的 **3.5–7.9%**、N 失败样本的 **5.6–11.0%**。也就是说，相当一部分原生失败，在中间某个位置强制作答本可以得到正确程序。

| 模型 | ACD / 样本数；% [95% 置信区间] | 稳定 ACD / 样本数；% [95% 置信区间] | ACD / N 失败样本数；% [95% 置信区间] |
|---|---|---|---|
| Qwen3 | 23/650；3.54 [1.89, 5.39] | 10/650；1.54 [0.59, 2.79] | 23/359；6.41 [3.42, 9.83] |
| DS-R1-Qwen | 51/650；7.85 [5.25, 10.70] | 28/650；4.31 [2.38, 6.50] | 51/464；10.99 [7.35, 15.00] |
| DS-R1-Llama | 25/650；3.85 [2.34, 5.59] | 12/650；1.85 [0.81, 3.03] | 25/450；5.56 [3.36, 8.16] |

*来源：[冻结代码的正式输出](../../runs/llama_results/acd/summary.json)。全部已登记的数据集／难度、探测次数、稳定 ACD／N 失败、位置、触及上限的轨迹、程序差异和救回／受损汇总，都保留在[描述性表格](../../runs/llama_results/ACD_DESCRIPTIVE.md)中。*

**H-ACD 未得到支持：** Llama 减 Qwen 的 ACD 率为 **+0.31 个百分点 [−1.98, 2.49]**。稳定 ACD 的次要对比为 **+0.31 个百分点 [−1.25, 1.86]**。描述性的 Llama 减 DS 对比为 **−4.00 个百分点 [−6.84, −1.38]**。ACD 指标没有 UNKNOWN。[正式输出](../../runs/llama_results/acd/summary.json)、[描述性 DS 对比](../../runs/llama_results/raw/acd_llama_vs_ds.json)

**H-EG 未得到支持：** EG `k4,m0` 减 N 为 **+0.25 个百分点 [−1.63, 2.01]**，token 节省 **10.02% [5.75, 14.51]**。这项检验沿用 C3 的 **199 道题等权**策略估计和数据集×难度分层，不是上面按样本计数的 ACD 估计方法。H-ACD 与 H-EG 各自按名义 95% 判定，不做多重比较校正。[冻结的 43 配置区间](../../runs/llama_execution_repair1/blind/analysis/results/intervals43.json)

强制作答分支、候选频率、重复草稿、采样和测试覆盖都限制了对 ACD 的解释。

</details>

<a id="probe-methods-and-resources"></a>

<details>
<summary>隐藏状态探针：次要结果、方法、审计与资源</summary>

**补充登记 · 次要。** 我用字典序排序：先把所有 PUB 通过的草稿排在失败草稿之前，再在每个 PUB 分组内按探针的折外分数排序。相对 PUB 的总体 AUROC 增益，Qwen、DS-Qwen 与 Llama 分别为 **+0.0759 [0.0580, 0.0956]**、**+0.0424 [0.0318, 0.0542]**、**+0.0460 [0.0331, 0.0600]**。只要分数在 PUB 组内的排序好于随机打破平局，就会抬高二值 PUB 的 AUROC。元数据＋PUB 联合模型已经达到 **0.931 / 0.959 / 0.935**，探针＋PUB 为 **0.945 / 0.980 / 0.969**；但前者是逻辑回归、后者是字典序，也没有配对区间，不能直接比较（联合模型数值见 [C2](#c2-public-execution-carries-information-beyond-the-metadata-baseline)）。在次要候选位置，也就是强制生成草稿之前，Qwen 的探针减 PUB 区间包含零，另两个模型的区间为负；候选位置上每个 PUB 通过子集的比较都包含零。全部数据集／位置的结果都保留在[完整表格](../../runs/probe_execution/FULL_TABLES.md)中。

开发集上锁定的层（草稿末端／候选位置）：Qwen 为 **36/36**，DS-Qwen 为 **28/21**，Llama 为 **32/32**，全部使用 **C = 0.01**。预处理与拟合只使用训练折。

没有缺失特征、新增 UNKNOWN 或不收敛的拟合。我没有验证 Hugging Face 与原 vLLM 的前向状态是否逐位一致；标签的限制与 PRIV-IND 相同。

</details>

<a id="reproduction-and-appendices"></a>

<a id="reproduction"></a>

## 复现

<a id="a-cpu-example-with-no-dependencies"></a>

### 无需额外依赖的 CPU 示例

在仓库根目录用 Python 3.12 运行：

```bash
python -m scripts.demo_execexit
```

示例用两条合成轨迹调用 HYB-B，只依赖标准库，不写文件。计费只算实际访问过的探测：

| 合成案例 | 决策 | 生成 token 计费 |
|---|---|---|
| 早期访问时 PUB 失败，在 4,096 处通过 | 交付边界草稿 | 4,096 主轨迹 + 80 被拒草稿 + 120 被接受草稿 = **4,296** |
| 两份访问过的草稿都没通过 PUB | 回退到原生生成 | 6,000 原生 + 80 被拒草稿 + 120 被拒草稿 = **6,200** |

注入的控制 token 另计 **12 / 16**，不含题面和已有前缀。公开样例结果是合成输入，示例只演示决策与计费。

下列规则检查需要 Linux，只用标准库：

```bash
python -m unittest exec_exit.tests.test_policies exec_exit.tests.test_policies_v1 exec_exit.tests.test_hybrid_v1 exec_exit.tests.test_execution_signals exec_exit.tests.test_heldout_service_template exec_exit.tests.test_early_abort
```

<a id="cpu-checks-and-figure-rendering"></a>

### CPU 检查与绘图

[CPU 工作流](../../.github/workflows/cpu.yml)列出了固定依赖与检查命令。安装后设置 `PYTHONPATH=src:.`、`CUDA_VISIBLE_DEVICES=""`；`python -m pytest` 默认收集 `tests/`，ExecExit 的检查由工作流显式指定。绘图只需要 Python 和 Matplotlib，读取的是已保存的估计：

```bash
python docs/execexit/render_results.py --output-dir artifacts/reproductions/main_results
```

<a id="review-or-recompute-the-recorded-study"></a>

### 查阅或重新计算已记录的研究

查阅结果不需要重跑实验：文中各处的来源链接都指向保存的输出。当前结论采用勘误结果；历史复现使用对应版本绑定的源码、模型、输入与标签。

研究入口的路径配置如下，需在启动进程前设置：

| 配置 | 用途 |
|---|---|
| `EXECEXIT_MODELS` | 模型根目录，默认是本仓库的 `models/`；其下使用 `Qwen3-8B/`、`DeepSeek-R1-Distill-Qwen-7B/` 子目录 |
| `EXECEXIT_VLLM_SOURCE` | 可选的 vLLM 源码目录，默认取当前 Python 环境的 `site-packages/vllm`，用于源码审计 |
| `EXECEXIT_SANDBOX_PYTHON` | 执行候选程序时必填，指定隔离执行器的 Python 绝对路径 |
| `EXECEXIT_GPU_LOCK` | 服务实验和隐藏状态提取时必填，使用锁文件的绝对路径；共用 GPU 的进程须指向同一个锁文件 |

评分入口显式传入 `--model`，轨迹注释与回放入口显式传入 `--tokenizer`。GPU 子进程沿用启动命令的 Python 环境。历史研究入口还需要对应运行绑定的源码、执行器和语料；保存的运行记录保持原有路径与身份。

主图最初在 **Python 3.12.3 / Matplotlib 3.11.0** 下检查；当前 CPU 工作流记录它自己的依赖版本。

隐藏状态探针在独立的 **scikit-learn 1.7.2** 环境中拟合，这个依赖不在当前 CPU 工作流里；在拟合环境中运行 `python -m unittest exec_exit.tests.test_probe_fit` 可以执行合成测试。

公开版抽取和拟合入口使用运行收据及源码、输入文件的哈希；新运行需要绑定当前源码版本。

工程包名为 `execexit`，基础模块保留 `cs336_basics`；研究环境以各运行记录为准。大体积的特征、语料、模型、缓存和运行时没有随仓库分发。

[来源映射](../../runs/report_figures/sources.json)、[绘图数据](../../runs/report_figures/raw/plot_data.json)和 [C2 附录](../../runs/report_figures/C2_APPENDIX.md)提供数值与筛选依据。题面和测试按原始许可证获取。
