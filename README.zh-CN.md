# ExecExit

[English](README.md) · 简体中文

**在公开样例上运行草稿程序，能否告诉推理模型何时停止？**

自主开展的研究项目 · 3 个开源推理模型 · 199 道留出编程题

[研究发现](#what-i-found) · [停止为何失败](#why-a-good-signal-still-stops-unsafely--exploratory) · [两个转折点](#two-moments-that-changed-my-conclusions) · [我构建了什么](#what-i-built) · [技术报告](docs/execexit/REPORT.zh-CN.md)

<a id="tldr"></a>

## 概要

推理模型常常在已经写出可用程序之后还继续思考。我研究的问题是：把每份中间草稿放到题目的公开样例上运行，能否告诉模型何时该停止。

- **信号有信息量。** 在三个开源推理模型、199 道留出题上，公开样例检查识别正确草稿的能力在每个模型上都优于只知道题目难度、已用 token 数和草稿位置的元数据基线：AUROC 提升 +0.06 到 +0.15（AUROC 衡量把正确草稿排在错误草稿前面的能力，0.5 为随机）。模型自身的置信度反而低于基线（−0.14 到 −0.15）。
- **按它停止并未被证明安全。** 我预登记的规则节省了 7.9% 的生成 token。准确率显著下降 1.9 个百分点（95% 置信区间 −3.4 到 −0.6，不含 0），下界越过预设的 −3 个百分点容限，非劣效未成立。时延改善也未能确立。
- **失败模式（探索性）。** 检查几乎不会拒绝正确草稿（放行 97–98%），但它放行的草稿中约 30% 是错的，停止规则会把这些错误程序直接交付。此外，用*其他*两个模型在同一道题上的漏检率（错误草稿通过公开样例的比例）给题目排序，能区分出提前停止让目标模型在这道题上平均通过率下降的题目（AUROC 0.76–0.92）。

我的贡献是一项受控比较：把"认出正确草稿""按它安全地停止""节省时间"三件事分开检验，并发现、修正了[两处会影响结论的标签与数据问题](#two-moments-that-changed-my-conclusions)。工程上，我为本研究搭建了草稿抽取与沙箱执行、离线策略回放和配对 bootstrap 统计流水线；引出这个问题的前期工作是从零实现的 BPE 分词器、Transformer 训练和带回滚 KV cache 的推理引擎（[我构建了什么](#what-i-built)）。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/execexit/figures/signal_quality_3models_dark.png">
  <img alt="两栏区间图。左：在公开样例上运行草稿，三个模型都优于元数据基线。右：模型自身的置信度，三个模型都低于基线。" src="docs/execexit/figures/signal_quality_3models.png">
</picture>

**图 1.** 相对元数据基线（题目难度、已用 token 数、草稿位置）的 AUROC 差，点为估计值，线为 95% 置信区间。左栏用独立的隐藏测试标签、右栏用原始标签，各自与自己的基线比较。

[数值与来源](runs/readme_figure/values.json) · [绘图程序](docs/execexit/render_results.py)

<a id="the-question"></a>

## 研究问题

推理模型在给出最终答案前会写一段很长的思维链。对编程题，我可以在检查点暂停它，让它写出一个完整程序，我称之为*草稿*。如果草稿已经正确，之后的每个 token 都是浪费。代码提供了大多数任务没有的检验手段：题面里的公开样例可以真正运行。

我把"能否按公开样例检查来停止"拆成四个递进的问题，下面四项发现依次回答：

1. **有没有可省的？** 推理结束前是否已经写出正确草稿。
2. **能不能认出来？** 公开样例检查是否比模型置信度和简单元数据更能识别正确草稿。
3. **按它停止是否安全？** 提前交付的程序是否仍能通过隐藏测试。
4. **能不能更快？** token 节省能否变成更短的端到端交付时间。

<a id="how-i-evaluated-it"></a>

## 评估方法

| | 做法 |
|---|---|
| **什么算成功** | 交付给客户端的完整程序通过隐藏测试；每次被拒的探测和回退都计入成本 |
| **规则何时确定** | 假设、3 个百分点容限和全部 43 个策略配置都在留出结果出现前确定；没有因为表现差而删掉任何配置 |
| **不确定性** | 配对的按题 bootstrap（10,000 次重抽样，每道题权重相同），外加一个针对受损题数的严格判据 |
| **标签独立性** | 隐藏测试标签，排除任何与公开样例输入相同的测试 |
| **速度** | 单独的端到端服务实验 |

**实验设置。**
- **模型：** Qwen3-8B（主模型）、DeepSeek-R1-Distill-Qwen-7B（登记的次要模型）和 DeepSeek-R1-Distill-Llama-8B（登记复现）。
- **题目：** 199 道留出题：126 道此前未使用的 LiveCodeBench v6 题（63 道 AtCoder、63 道 LeetCode）和 73 道来自 Codeforces 的 LiveCodeBench Pro 题。
- **样本：** 每个模型 650 次采样推理运行。

**证据等级。**
- **预登记：** 在留出结果出现前确定。
- **登记复现：** 在之后的补充登记中确定，且早于该模型的结果。
- **事后／探索性：** 在相关结果已知后才计划。

<a id="what-i-found"></a>

## 研究发现

### 1. 确实有可节省的空间 · *预登记*

在 Qwen3-8B 的 650 次运行中，有 304 次在推理结束前写出的某份草稿就已能通过测试。一个知道测试结果、在已记录的草稿和原生答案中选取最便宜可通过选项的 oracle，可以节省 14.8% 的生成 token（95% 置信区间 11.7–18.2），DeepSeek-R1-Distill-Qwen-7B 为 13.1%（9.8–16.8）。这是不计未使用的探测和所有检查开销的上界。[详情](docs/execexit/REPORT.zh-CN.md#c1-earlier-passing-drafts-create-a-bounded-opportunity)

### 2. 执行信号优于元数据基线，置信度不如基线 · *部分事后*

如果执行结果有信息量，它就应当胜过一个只知道难度、已用 token 数和草稿位置的基线。相对这个基线，公开样例检查在三个模型上分别提升 +0.06、+0.12 和 +0.15 AUROC；置信度分别下降 0.14、0.15 和 0.14（图 1）。这一优势并不均匀：Qwen3-8B 在 AtCoder 和 LeetCode 上的区间包含 0，提升在 LiveCodeBench Pro 上最大。公开样例检查在前两个模型上为事后分析，在第三个模型上为登记复现（[原因](#two-moments-that-changed-my-conclusions)）。[详情](docs/execexit/REPORT.zh-CN.md#c2-confidence-execution-and-the-metadata-baseline)

### 3. 更好的信号没有带来安全的停止规则 · *预登记主检验*

登记的规则（HYB-B）在前 4,096 个 token 内按固定日程检查草稿，交付第一份通过公开样例的完整草稿；都没有通过就继续原生推理。预设容限是：相对原生生成，准确率损失不超过 3 个百分点（非劣效检验）。

- **Qwen3-8B 上的结果：**
  - 节省了 7.9% 的 token（5.0–11.1），准确率下降 1.9 个百分点（−3.4 到 −0.6）。区间不含 0，下降是真实的；区间越过了 −3，因此非劣效未能成立。
  - 199 道题中有 11 道受损，全部是 LeetCode 题。受损的定义是：这道题至少有一次采样运行，原生生成通过，而同一次运行中提前停止交付的程序失败。严格判据要求受损题比例的单侧 97.5% 置信上界低于 3%，实际上界为 9.7%。
  - 在第二项登记比较中，即与"在 4,096 token 处直接强制作答"相比，它高出 +1.6 个百分点（−0.5 到 +3.8），但多用了 9.8% 的 token。这一优势同样未能成立。
- **在另外两个模型上**（次要结果）：DeepSeek-R1-Distill-Qwen-7B 上为 +0.8 个百分点（−1.1 到 +2.8），节省 13.4%，名义非劣效成立，但未通过更严格的受损判据（199 道题中 10 道受损）；DeepSeek-R1-Distill-Llama-8B（无预设判据）上为 −0.6 个百分点（−2.4 到 +0.9），节省 10.7%。

[详情](docs/execexit/REPORT.zh-CN.md#c3-the-stopping-policy-did-not-establish-the-primary-claim)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/execexit/figures/pareto_43_dark.png">
  <img alt="三张散点图：全部 43 个冻结配置的准确率变化与 token 节省，分别为 Qwen3-8B、DeepSeek-R1-Distill-Qwen-7B 和 DeepSeek-R1-Distill-Llama-8B" src="docs/execexit/figures/pareto_43.png">
</picture>

**图 2.** 三个模型上的全部 43 个冻结配置，均相对于原生生成（N）：越靠右越省，越靠上越准。灰点是其余配置，误差线是两个方向上的名义 95% 置信区间。HYB-B 是登记的规则；BF@4096 在 4,096 token 处强制作答；CONF-50 按模型置信度停止；EG 接受第一份通过公开样例的草稿，不设 4,096 截止；ORACLE 是知道隐藏测试结果、无法实际部署的离线参照。k4 表示每 4 个候选检查一个，m0 表示从推理开始就检查。虚线是 HYB-B 相对 N 的 −3 个百分点非劣效容限。

更便宜的信号节省更多，代价也更大：按模型置信度停止节省约 46% 的 token，却在 Qwen3-8B 上损失 12.4 个百分点（−16.0 到 −9.2）。[数据](runs/qwen_erratum/secondary_intervals43/raw/comparisons.jsonl)

### 4. 节省 token 未能带来更快的交付 · *预登记*

在一个单独的 40 题串行服务实验中，门控系统相对原生生成的时延比为 0.983（0.870–1.101），区间包含"没有变化"。实际服务中，门控系统生成的 token 总数反而比原生生成多 3.7%（含被中止的尾部）。[详情](docs/execexit/REPORT.zh-CN.md#c4-token-savings-did-not-establish-a-service-latency-improvement)

<a id="why-a-good-signal-still-stops-unsafely--exploratory"></a>

## 好信号为何仍会导致不安全的停止 · *探索性*

平均而言把草稿排得好，并不等于从不接受错误的草稿。检查的错误几乎都是放过错误草稿：在独立的隐藏测试标签上，它放行了 98%、98% 和 97% 的正确草稿，但它放行的草稿中分别有 31%、29% 和 31% 是错的（按题加权；Qwen3、DeepSeek-Qwen、DeepSeek-Llama）。这些比例按草稿统计，不是策略交付答案的错误率。停止规则一旦接受了被公开样例放过的错误草稿，就会交付错误答案。[Qwen3](runs/c2_label_completion/raw/qwen_completed/main/summary.json) · [DeepSeek-Qwen](runs/c2_label_completion/raw/ds_completed/main/summary.json) · [DeepSeek-Llama](runs/llama_execution_repair1/blind/analysis/results/private_c2.json)

我对每道题测量了**漏检率**：错误草稿中通过公开样例的比例。

- **它跟着题目走，而不只跟着模型走。** 不同模型在同一道题上的漏检率正相关（三对模型的 Spearman ρ 为 0.58–0.70）。这可能是因为题目的公开样例偏弱，也可能是因为模型犯了相同的错，目前分不开。
- **它不是公开样例数量的反映。** 样例数量和漏检率之间没有稳定的关系。
- **跨模型漏检率能在已有题目上区分平均通过率下降的题目。** 对每个目标模型，我用另外两个模型的合并漏检率给题目排序，计算它区分"提前停止是否降低了目标模型在这道题上的平均通过率"的 AUROC（0.5 为随机）。只纳入目标模型原生生成至少通过一次、且另外两个模型至少有一份错误草稿的题目。

这里的"受损"指逐题平均通过率下降，与主检验口径不同，所以 Qwen3-8B 是 10/91 而不是 11/199；图 3 右栏中"仅 LeetCode"和"只用题集身份"是看到总体结果后补做的检查。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/execexit/figures/public_check_risk_dark.png">
  <img alt="两栏区间图。左：公开样例检查放行了约 97–98% 的正确草稿，但被放行的草稿中约 30% 是错的。右：另外两个模型的漏检率区分目标模型平均通过率下降题目的 AUROC，全部纳入题目与仅 LeetCode 两种口径。" src="docs/execexit/figures/public_check_risk.png">
</picture>

**图 3.** 左：正确草稿被放行的比例，以及被放行草稿中错误的比例（按题加权）。右：用另外两个模型的漏检率区分目标模型平均通过率下降题目的 AUROC，分别在全部纳入题目和仅 LeetCode 上计算；菱形是只用题集身份的参照。

在符合条件的已有题目上，其他模型的漏检率能区分目标模型平均通过率下降的题目（AUROC 0.76–0.92），这是跨模型风险分层的离线证据；它的因果来源和对新题的泛化还有待验证，也就是下文的第一个后续问题。[详情](docs/execexit/REPORT.zh-CN.md#mechanism-public-test-strength-exploratory) · [已保存的估计](runs/public_test_strength/summary.json)

<a id="two-moments-that-changed-my-conclusions"></a>

## 改变我结论的两个转折点

**1. 我的正确性标签里包含了我正在检验的信号。**

按我最初登记的评估规则，一份草稿只有同时通过公开样例和隐藏测试才算正确。这让公开样例检查成了它自己目标的一部分：它永远不会漏掉正确草稿，它相对基线的一部分优势是由定义决定的。所以我做了三件事：

- **定义了一个独立标签。** 这个标签只用隐藏测试，并去掉任何输入与公开样例相同的测试。我在计算前就把定义写了下来。
- **补全了缺失的标签。** 首次计算时，很多标签是未知的，主要因为原来的 LeetCode 测试框架遇到第一个失败的公开样例就停止了。在这些不完整的标签上，Qwen3-8B 尚未确立总体优势。缺失标签不是随机的：每一个都属于没通过公开样例的草稿。所以我没有丢弃它们，而是补全了它们，重新执行了 2,867 次程序，并做了超时敏感性检查。在补全后的标签上，两个模型的优势都成立。
- **为第三个模型登记了同一检验。** 补全标签的决定是在看到第一次结果之后做的，所以我把前两个模型的估计视为事后分析。随后，我在第三个模型的任何结果出现之前登记了同一检验。结果复现：+0.15（0.10–0.21）。

[补全记录](runs/c2_label_completion/final_counts.json) · [登记复现](runs/llama_execution_repair1/blind/analysis/results/private_c2.json)

**2. 一次数据修正翻转了我的主要结果。**

为 LiveCodeBench Pro 抽取的公开样例输入中含有空行，可能让原本正确、逐行读取输入的程序出错。我对照 Codeforces 官方页面核对了 73 道题的全部 78 个样例并做了修正。修正统一应用于所有策略，没有重新生成任何模型输出。修正前，主比较通过了名义非劣效检验（−1.13 个百分点，−2.76 到 +0.38）；修正后，比较未通过（−1.88，−3.39 到 −0.63）。原因是原始标签要求通过公开样例：缺陷让 3 次本已正确的原生运行被判为失败，看起来就像被停止规则救回。

同样的现象，即程序没通过公开样例却通过了全部隐藏测试，也出现在 AtCoder 上（19 个程序）。逐个核查后，这 19 个程序都是真的错了，只是隐藏测试没有发现，所以我没有把修正推广过去。[案例分析](docs/case_studies.zh-CN.md#a-data-defect-that-flipped-the-primary-result) · [AtCoder 核查](runs/atcoder_public_discordance/summary.json) · [报告](docs/execexit/REPORT.zh-CN.md#lcb-pro-public-input-defect-and-correction)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/execexit/figures/turning_points_dark.png">
  <img alt="两栏区间图。左：补全独立标签后，两个模型的公开样例检查优势都变大，Qwen3-8B 的区间离开了 0。右：数据勘误后，主检验的区间越过了 −3 个百分点的非劣界。" src="docs/execexit/figures/turning_points.png">
</picture>

**图 4.** 两个转折点前后的估计（空心为修正前，实心为修正后）。左：独立标签上公开样例检查相对元数据基线的 AUROC 差，修正前只用当时已知的标签；右：主检验的通过率差，虚线为 −3 个百分点的非劣界。[数值与来源](runs/readme_figure/values.json)

<a id="things-i-tried-that-didnt-work"></a>

## 尝试过但没有成功的方法

- **置信度门控的提前退出（引出本研究的预实验）。** 在 36 道已见过的开发题上，DEER 式置信度门控把平均交付时间从 110 秒降到了 90 秒，但正确交付的程序从 19 个降到 14 个：救回 1 道题，损害 6 道题。这促使我转向外部信号。[预实验](docs/case_studies.zh-CN.md#evaluating-faster-code-generation) · [已保存结果](runs/early_exit_pilot/summary.json)
- **在公开样例之外增加更多检查。** 我尝试加入自生成的压力测试，或要求与前一份草稿行为一致。两者都没有胜过直接等待第二份通过公开样例的草稿：在 252 次 AtCoder 运行中分别通过 121 次和 120 次，对照为 121 次，而且 token 成本更高。[详情](docs/execexit/REPORT.zh-CN.md#policy-harm-and-public-false-acceptance)

<a id="what-i-built"></a>

## 我构建了什么

本研究结合了从零实现的基础和为这个问题搭建的评估流水线。研究实验用 vLLM 运行开源模型。

| 组件 | 说明了什么 | 代码 |
|---|---|---|
| 从零实现的语言模型 | 基于 Stanford CS336 作业实现的 BPE 分词器、Transformer、AdamW 和训练循环。流式编码在每个分块边界都与整段编码一致（[案例](docs/case_studies.zh-CN.md#streaming-tokenization-must-preserve-the-token-sequence)）；检查点同时保存优化器和随机数状态：只存权重的话，AdamW 的一阶/二阶矩和步数会归零，训练批次的抽样也会从头开始，恢复后的训练就会偏离原轨迹。测试验证了中断后恢复的训练与不中断的训练轨迹一致（[案例](docs/case_studies.zh-CN.md#restoring-training-state-not-only-weights)） | [tokenizer.py](src/cs336_basics/tokenizer.py)、[transformer.py](src/cs336_basics/transformer.py)、[train.py](src/cs336_basics/train.py)、[恢复测试](tests/test_training_reproducibility.py) |
| 推理引擎 | 带位置检查、模型调用失败后可回滚的 KV cache，分块 prefill | [kv_cache.py](src/cs336_basics/kv_cache.py)、[chunked_prefill.py](src/cs336_basics/chunked_prefill.py) |
| 一个答案可手算的正确性测试 | 缓存注意力必须把新查询与绝对位置对齐。已有三个缓存位置、再加入两个新查询，q、k 全为零、值为 1–5 时，注意力对可见位置取平均：正确的掩码给出 2.5 和 3.0，形状合法但错误的掩码给出 1.0 和 1.5 | [test_attention_backends.py](tests/test_attention_backends.py#L29-L41) |
| 程序执行 | 抽取每份草稿，在隔离环境中用公开测试和隐藏测试运行 | [extract.py](exec_exit/execute/extract.py)、[sandbox.py](exec_exit/execute/sandbox.py) |
| 离线策略回放 | 用已记录的探测机会和信号评估停止策略，无需 GPU，并计入每次访问过的探测和回退成本；用于全部 43 个配置 | [policies_v1.py](exec_exit/replay/policies_v1.py)、[hybrid_v1.py](exec_exit/replay/hybrid_v1.py) |
| 统计与标签 | 配对的按题 bootstrap；独立的隐藏测试标签；隐藏状态探针 | [stats.py](exec_exit/analysis/stats.py)、[private_labels.py](exec_exit/analysis/private_labels.py)、[extract_hidden_states.py](exec_exit/analysis/extract_hidden_states.py)、[probe_fit.py](exec_exit/analysis/probe_fit.py) |
| 服务实验 | 在提前停止参与的情况下测量端到端交付时间 | [service_experiment_v2.py](exec_exit/serve/service_experiment_v2.py) |

三个"通过了一致性检查却仍然是错的"实例，即上面的缓存掩码、流式分词和第二个转折点背后的数据缺陷，连同训练状态恢复的检查，写在[看似正确，实则有误](docs/case_studies.zh-CN.md)中。

<a id="next-questions"></a>

## 后续问题

- **提前估计公开测试的强度。** 能否在部署前估计一道题公开测试的强度，例如利用其他模型的草稿或自动生成的输入？这样就可以只在安全的题目上使用执行门控的停止。
- **在通过公开样例的草稿里再筛一次。** 被放行的错误草稿都在这一组里。在每份草稿结尾的隐藏状态上训练的逻辑回归探针，在三个模型中都没有胜过公开样例检查（登记的后续分析）；在这一组中，它只在 DeepSeek-R1-Distill-Llama-8B 上确立了优于元数据的区分能力（AUROC +0.17）。加入这样的第二道筛选、并计入额外一次前向计算的成本后，停止规则能否减少受损？[详情](docs/execexit/REPORT.zh-CN.md#hidden-state-probes)

<a id="limitations"></a>

## 局限

- 三个 7–8B 模型，来自两个基座家族；结果不一定能推广到更大的模型。
- 只有一组固定的 199 道题。LiveCodeBench v6 的题目早于 Qwen3 发布，因此不能排除数据污染。
- 隐藏测试数量有限。独立标签只去掉了输入完全相同的测试，没有去掉语义上的重合。
- 服务实验规模小：40 道题，每种设置各运行一次，且经历过中断恢复。
- 次要区间为逐项区间，没有做多重比较校正。

完整列表见[报告](docs/execexit/REPORT.zh-CN.md#limitations-and-interpretation)。

<a id="running-it-and-repository-map"></a>

## 运行方式与仓库结构

上文每个数字都直接链接到、或经由[技术报告](docs/execexit/REPORT.zh-CN.md)链接到已保存的输出。以下命令在仓库根目录、Linux、Python 3.12 下运行，只需 CPU：

```bash
# 只需标准库
python -m scripts.demo_execexit        # 在两条合成轨迹上演示登记的停止规则
python -m unittest exec_exit.tests.test_policies exec_exit.tests.test_policies_v1 \
  exec_exit.tests.test_hybrid_v1 exec_exit.tests.test_execution_signals \
  exec_exit.tests.test_heldout_service_template exec_exit.tests.test_early_abort \
  exec_exit.tests.test_service_experiment_v2   # 停止规则与成本核算

# 安装 CPU 依赖（与 CI 相同的固定版本）
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.4 matplotlib==3.11.1 pytest==9.0.2 pytest-timeout==2.4.0 regex==2026.3.32 tiktoken==0.12.0 tqdm==4.67.3
export PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=""
python -m pytest tests/ exec_exit/analysis/test_stats.py   # 分词器、注意力、KV cache、训练恢复与统计
python docs/execexit/render_results.py --output-dir artifacts/reproductions/main_results   # 用已保存的估计重绘图 1–4
```

同一组检查在 [CPU 工作流](.github/workflows/cpu.yml)中运行。重新生成推理轨迹需要 GPU、vLLM、模型权重和基准测试数据，这些不在本仓库中分发；重跑研究入口所需的路径配置见[报告](docs/execexit/REPORT.zh-CN.md#review-or-recompute-the-recorded-study)。

| 路径 | 内容 |
|---|---|
| [docs/](docs/) | 技术报告、图和案例分析 |
| [exec_exit/](exec_exit/) | 研究代码：生成、执行、回放、分析、服务 |
| [runs/](runs/) | 每个数字背后的已保存结果 |
| [src/cs336_basics/](src/cs336_basics/)、[tests/](tests/) | 从零实现的语言模型和推理引擎 |

<a id="related-work-and-acknowledgements"></a>

## 相关工作与致谢

- **[DEER](https://arxiv.org/abs/2504.15895)：** 基于置信度的提前退出。我预实验中的置信度门控是 DEER 式的，不是完整复现。
- **[Dynasor / Certaindex](https://arxiv.org/abs/2412.20993)：** 探测中间答案，并在服务中利用其一致性。
- **[interwhen](https://arxiv.org/abs/2602.11202)：** 中间生成的 Lean 代码一旦编译通过就提前停止。
- **[RLEF](https://arxiv.org/abs/2410.02089)：** 把执行反馈用于训练和修复。
- **[STP / OTBench](https://arxiv.org/abs/2610.13570)：** 研究固定推理轨迹上的理想停止点。

从零实现部分以 [Stanford CS336](https://github.com/stanford-cs336/assignment1-basics) 课程材料为起点（MIT 许可证，保留在 [LICENSE](LICENSE) 中）。实验使用了 [vLLM](https://github.com/vllm-project/vllm)、[PyTorch](https://pytorch.org) 和 [Hugging Face](https://huggingface.co) 上的模型。题目和测试来自 [LiveCodeBench](https://livecodebench.github.io) 和 LiveCodeBench Pro，遵循其各自的许可证。
