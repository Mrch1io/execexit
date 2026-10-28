# Hidden-state probes: all registered results

English · [简体中文](FULL_TABLES.zh-CN.md)

A read-only transcription of the frozen analysis JSON; nothing is refit and no intervals are recomputed.

All intervals are per-item nominal 95%, with no multiplicity correction. Within each collection the fits are redone on the original folds, and a favorable collection cannot stand in for the pooled primary test. The primary specification is PRIV-IND / virtual_v1_pipeline / complete_deliverable; draft_end is the primary position and candidate is the secondary one.

Comparison labels: `probe_minus_pub` is probe − PUB; `probe_minus_baseline_pub_pass` is probe − refit difficulty baseline within the PUB-pass subset; `probe_pub_minus_pub` is the fixed lexicographic combination − PUB.

## Qwen3-8B

[Raw results](fit/qwen/heldout/summary.json)

| Population | Position | Comparison | Probe/combination AUROC | Reference AUROC | Difference [95% CI] | Drafts/problems | Verdict |
|---|---|---|---:|---:|---|---:|---|
| Pooled | draft_end | probe_minus_pub | 0.8842 | 0.8694 | +0.0148 [-0.0202, +0.0487] | 4203/197 | Not supported (includes 0) |
| Pooled | draft_end | probe_pub_minus_pub | 0.9453 | 0.8694 | +0.0759 [+0.0580, +0.0956] | 4203/197 | Supported |
| Pooled | draft_end | probe_minus_baseline_pub_pass | 0.7825 | 0.7368 | +0.0456 [-0.0111, +0.1024] | 2006/148 | Not supported (includes 0) |
| Pooled | candidate | probe_minus_pub | 0.8565 | 0.8694 | -0.0129 [-0.0521, +0.0238] | 4203/197 | Not supported (includes 0) |
| Pooled | candidate | probe_pub_minus_pub | 0.9400 | 0.8694 | +0.0706 [+0.0533, +0.0898] | 4203/197 | Supported |
| Pooled | candidate | probe_minus_baseline_pub_pass | 0.7452 | 0.7368 | +0.0084 [-0.0457, +0.0620] | 2006/148 | Not supported (includes 0) |
| Collection: hv6_atcoder | draft_end | probe_minus_pub | 0.8870 | 0.8589 | +0.0281 [-0.0334, +0.0864] | 1469/63 | Not supported (includes 0) |
| Collection: hv6_atcoder | draft_end | probe_pub_minus_pub | 0.9398 | 0.8589 | +0.0810 [+0.0466, +0.1207] | 1469/63 | Supported |
| Collection: hv6_atcoder | draft_end | probe_minus_baseline_pub_pass | 0.7941 | 0.7265 | +0.0677 [-0.0409, +0.2016] | 706/49 | Not supported (includes 0) |
| Collection: hv6_atcoder | candidate | probe_minus_pub | 0.8629 | 0.8589 | +0.0040 [-0.0677, +0.0687] | 1469/63 | Not supported (includes 0) |
| Collection: hv6_atcoder | candidate | probe_pub_minus_pub | 0.9384 | 0.8589 | +0.0795 [+0.0473, +0.1176] | 1469/63 | Supported |
| Collection: hv6_atcoder | candidate | probe_minus_baseline_pub_pass | 0.7305 | 0.7265 | +0.0040 [-0.0975, +0.1375] | 706/49 | Not supported (includes 0) |
| Collection: hv6_leetcode | draft_end | probe_minus_pub | 0.8730 | 0.7959 | +0.0770 [+0.0093, +0.1380] | 1842/63 | Supported |
| Collection: hv6_leetcode | draft_end | probe_pub_minus_pub | 0.9278 | 0.7959 | +0.1319 [+0.0926, +0.1721] | 1842/63 | Supported |
| Collection: hv6_leetcode | draft_end | probe_minus_baseline_pub_pass | 0.7840 | 0.7652 | +0.0188 [-0.0706, +0.1114] | 1022/59 | Not supported (includes 0) |
| Collection: hv6_leetcode | candidate | probe_minus_pub | 0.8496 | 0.7959 | +0.0537 [-0.0241, +0.1217] | 1842/63 | Not supported (includes 0) |
| Collection: hv6_leetcode | candidate | probe_pub_minus_pub | 0.9259 | 0.7959 | +0.1300 [+0.0895, +0.1702] | 1842/63 | Supported |
| Collection: hv6_leetcode | candidate | probe_minus_baseline_pub_pass | 0.7792 | 0.7652 | +0.0140 [-0.0718, +0.1050] | 1022/59 | Not supported (includes 0) |
| Collection: lcbpro_2025 | draft_end | probe_minus_pub | 0.8750 | 0.9291 | -0.0541 [-0.1129, -0.0033] | 892/71 | Reversed |
| Collection: lcbpro_2025 | draft_end | probe_pub_minus_pub | 0.9639 | 0.9291 | +0.0348 [+0.0140, +0.0569] | 892/71 | Supported |
| Collection: lcbpro_2025 | draft_end | probe_minus_baseline_pub_pass | 0.7636 | 0.6500 | +0.1136 [-0.0567, +0.2891] | 278/40 | Not supported (includes 0) |
| Collection: lcbpro_2025 | candidate | probe_minus_pub | 0.8336 | 0.9291 | -0.0955 [-0.1604, -0.0420] | 892/71 | Reversed |
| Collection: lcbpro_2025 | candidate | probe_pub_minus_pub | 0.9599 | 0.9291 | +0.0308 [+0.0115, +0.0518] | 892/71 | Supported |
| Collection: lcbpro_2025 | candidate | probe_minus_baseline_pub_pass | 0.6510 | 0.6500 | +0.0010 [-0.1586, +0.1659] | 278/40 | Not supported (includes 0) |

## DS-R1-Qwen-7B

[Raw results](fit/ds/heldout/summary.json)

| Population | Position | Comparison | Probe/combination AUROC | Reference AUROC | Difference [95% CI] | Drafts/problems | Verdict |
|---|---|---|---:|---:|---|---:|---|
| Pooled | draft_end | probe_minus_pub | 0.9302 | 0.9376 | -0.0074 [-0.0346, +0.0165] | 7023/199 | Not supported (includes 0) |
| Pooled | draft_end | probe_pub_minus_pub | 0.9801 | 0.9376 | +0.0424 [+0.0318, +0.0542] | 7023/199 | Supported |
| Pooled | draft_end | probe_minus_baseline_pub_pass | 0.7964 | 0.7349 | +0.0615 [-0.0115, +0.1341] | 1453/133 | Not supported (includes 0) |
| Pooled | candidate | probe_minus_pub | 0.8968 | 0.9376 | -0.0408 [-0.0727, -0.0131] | 7023/199 | Reversed |
| Pooled | candidate | probe_pub_minus_pub | 0.9746 | 0.9376 | +0.0369 [+0.0274, +0.0470] | 7023/199 | Supported |
| Pooled | candidate | probe_minus_baseline_pub_pass | 0.7539 | 0.7349 | +0.0190 [-0.0521, +0.0883] | 1453/133 | Not supported (includes 0) |
| Collection: hv6_atcoder | draft_end | probe_minus_pub | 0.9347 | 0.9279 | +0.0068 [-0.0340, +0.0432] | 2559/63 | Not supported (includes 0) |
| Collection: hv6_atcoder | draft_end | probe_pub_minus_pub | 0.9738 | 0.9279 | +0.0459 [+0.0261, +0.0683] | 2559/63 | Supported |
| Collection: hv6_atcoder | draft_end | probe_minus_baseline_pub_pass | 0.7901 | 0.8253 | -0.0352 [-0.1422, +0.0635] | 563/45 | Not supported (includes 0) |
| Collection: hv6_atcoder | candidate | probe_minus_pub | 0.8812 | 0.9279 | -0.0467 [-0.1062, +0.0015] | 2559/63 | Not supported (includes 0) |
| Collection: hv6_atcoder | candidate | probe_pub_minus_pub | 0.9680 | 0.9279 | +0.0401 [+0.0229, +0.0595] | 2559/63 | Supported |
| Collection: hv6_atcoder | candidate | probe_minus_baseline_pub_pass | 0.7937 | 0.8253 | -0.0317 [-0.1222, +0.0623] | 563/45 | Not supported (includes 0) |
| Collection: hv6_leetcode | draft_end | probe_minus_pub | 0.9211 | 0.9171 | +0.0040 [-0.0430, +0.0417] | 2630/63 | Not supported (includes 0) |
| Collection: hv6_leetcode | draft_end | probe_pub_minus_pub | 0.9768 | 0.9171 | +0.0597 [+0.0400, +0.0817] | 2630/63 | Supported |
| Collection: hv6_leetcode | draft_end | probe_minus_baseline_pub_pass | 0.8157 | 0.7794 | +0.0362 [-0.0648, +0.1344] | 672/55 | Not supported (includes 0) |
| Collection: hv6_leetcode | candidate | probe_minus_pub | 0.8888 | 0.9171 | -0.0283 [-0.0826, +0.0146] | 2630/63 | Not supported (includes 0) |
| Collection: hv6_leetcode | candidate | probe_pub_minus_pub | 0.9675 | 0.9171 | +0.0504 [+0.0322, +0.0698] | 2630/63 | Supported |
| Collection: hv6_leetcode | candidate | probe_minus_baseline_pub_pass | 0.7862 | 0.7794 | +0.0067 [-0.0862, +0.1061] | 672/55 | Not supported (includes 0) |
| Collection: lcbpro_2025 | draft_end | probe_minus_pub | 0.9118 | 0.9634 | -0.0516 [-0.1154, -0.0024] | 1834/73 | Reversed |
| Collection: lcbpro_2025 | draft_end | probe_pub_minus_pub | 0.9861 | 0.9634 | +0.0227 [+0.0098, +0.0381] | 1834/73 | Supported |
| Collection: lcbpro_2025 | draft_end | probe_minus_baseline_pub_pass | 0.7456 | 0.5240 | +0.2217 [+0.0004, +0.4350] | 218/33 | Supported |
| Collection: lcbpro_2025 | candidate | probe_minus_pub | 0.8308 | 0.9634 | -0.1326 [-0.2040, -0.0769] | 1834/73 | Reversed |
| Collection: lcbpro_2025 | candidate | probe_pub_minus_pub | 0.9798 | 0.9634 | +0.0164 [+0.0070, +0.0275] | 1834/73 | Supported |
| Collection: lcbpro_2025 | candidate | probe_minus_baseline_pub_pass | 0.4886 | 0.5240 | -0.0354 [-0.2608, +0.1825] | 218/33 | Not supported (includes 0) |

## DS-R1-Llama-8B

[Raw results](fit/llama/heldout/summary.json)

| Population | Position | Comparison | Probe/combination AUROC | Reference AUROC | Difference [95% CI] | Drafts/problems | Verdict |
|---|---|---|---:|---:|---|---:|---|
| Pooled | draft_end | probe_minus_pub | 0.9048 | 0.9235 | -0.0187 [-0.0538, +0.0133] | 6094/199 | Not supported (includes 0) |
| Pooled | draft_end | probe_pub_minus_pub | 0.9695 | 0.9235 | +0.0460 [+0.0331, +0.0600] | 6094/199 | Supported |
| Pooled | draft_end | probe_minus_baseline_pub_pass | 0.8362 | 0.6641 | +0.1721 [+0.0930, +0.2508] | 1493/133 | Supported |
| Pooled | candidate | probe_minus_pub | 0.8613 | 0.9235 | -0.0622 [-0.1057, -0.0247] | 6094/199 | Reversed |
| Pooled | candidate | probe_pub_minus_pub | 0.9628 | 0.9235 | +0.0393 [+0.0283, +0.0511] | 6094/199 | Supported |
| Pooled | candidate | probe_minus_baseline_pub_pass | 0.7446 | 0.6641 | +0.0805 [-0.0006, +0.1622] | 1493/133 | Not supported (includes 0) |
| Collection: hv6_atcoder | draft_end | probe_minus_pub | 0.8875 | 0.9245 | -0.0371 [-0.0935, +0.0106] | 2204/63 | Not supported (includes 0) |
| Collection: hv6_atcoder | draft_end | probe_pub_minus_pub | 0.9533 | 0.9245 | +0.0287 [+0.0035, +0.0537] | 2204/63 | Supported |
| Collection: hv6_atcoder | draft_end | probe_minus_baseline_pub_pass | 0.7614 | 0.6275 | +0.1339 [-0.0242, +0.3110] | 526/42 | Not supported (includes 0) |
| Collection: hv6_atcoder | candidate | probe_minus_pub | 0.8440 | 0.9245 | -0.0806 [-0.1545, -0.0206] | 2204/63 | Reversed |
| Collection: hv6_atcoder | candidate | probe_pub_minus_pub | 0.9525 | 0.9245 | +0.0280 [+0.0133, +0.0449] | 2204/63 | Supported |
| Collection: hv6_atcoder | candidate | probe_minus_baseline_pub_pass | 0.6896 | 0.6275 | +0.0621 [-0.0563, +0.1904] | 526/42 | Not supported (includes 0) |
| Collection: hv6_leetcode | draft_end | probe_minus_pub | 0.9158 | 0.8871 | +0.0287 [-0.0299, +0.0774] | 2502/63 | Not supported (includes 0) |
| Collection: hv6_leetcode | draft_end | probe_pub_minus_pub | 0.9715 | 0.8871 | +0.0844 [+0.0558, +0.1128] | 2502/63 | Supported |
| Collection: hv6_leetcode | draft_end | probe_minus_baseline_pub_pass | 0.8673 | 0.7644 | +0.1029 [+0.0102, +0.2033] | 752/59 | Supported |
| Collection: hv6_leetcode | candidate | probe_minus_pub | 0.8942 | 0.8871 | +0.0071 [-0.0483, +0.0535] | 2502/63 | Not supported (includes 0) |
| Collection: hv6_leetcode | candidate | probe_pub_minus_pub | 0.9656 | 0.8871 | +0.0785 [+0.0524, +0.1053] | 2502/63 | Supported |
| Collection: hv6_leetcode | candidate | probe_minus_baseline_pub_pass | 0.7657 | 0.7644 | +0.0013 [-0.1092, +0.1175] | 752/59 | Not supported (includes 0) |
| Collection: lcbpro_2025 | draft_end | probe_minus_pub | 0.8850 | 0.9648 | -0.0798 [-0.1362, -0.0281] | 1388/73 | Reversed |
| Collection: lcbpro_2025 | draft_end | probe_pub_minus_pub | 0.9829 | 0.9648 | +0.0181 [+0.0052, +0.0342] | 1388/73 | Supported |
| Collection: lcbpro_2025 | draft_end | probe_minus_baseline_pub_pass | 0.7467 | 0.5518 | +0.1948 [-0.0024, +0.3891] | 215/32 | Not supported (includes 0) |
| Collection: lcbpro_2025 | candidate | probe_minus_pub | 0.8271 | 0.9648 | -0.1377 [-0.2213, -0.0626] | 1388/73 | Reversed |
| Collection: lcbpro_2025 | candidate | probe_pub_minus_pub | 0.9830 | 0.9648 | +0.0182 [+0.0070, +0.0332] | 1388/73 | Supported |
| Collection: lcbpro_2025 | candidate | probe_minus_baseline_pub_pass | 0.6498 | 0.5518 | +0.0979 [-0.0768, +0.2765] | 215/32 | Not supported (includes 0) |

Fitting and statistics were actually run in `/root/probe_fit_env`: NumPy 2.3.5, scikit-learn 1.7.2, SciPy 1.16.3.
