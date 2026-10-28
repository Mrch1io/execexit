# ACD: All descriptive secondary results

English · [简体中文](ACD_DESCRIPTIVE.zh-CN.md)

Llama is a supplementary registration frozen before the results; Qwen3/DS-R1 remain exploratory and post hoc. All tables use the full PUB + hidden-test label, not PRIV-IND. Percentage intervals are per-item nominal 95% intervals, with no guarantee of simultaneous coverage.

Source: [official output of the frozen code](acd/summary.json).

## By collection and difficulty, with native-failure denominators

Rates use all samples in the row as the denominator; ACD/N-pass and ACD/N-fail are count ratios in their own columns. An UNKNOWN N result is not treated as a failure. Each interval resamples the full set of problems for that row, keeping every sample within a problem; it does not first filter down to problems where N failed.

| Model | Collection/difficulty | Samples / problems | N pass / fail | ACD count; rate % [CI] | Stable count; rate % [CI] | ACD/N-pass % [CI] | Stable/N-pass % [CI] | ACD/N-fail % [CI] | Stable/N-fail % [CI] |
|---|---|---:|---:|---|---|---|---|---|---|
| qwen | Pooled/all | 650 / 199 | 291 / 359 | 23; 3.54 [1.89, 5.39] | 10; 1.54 [0.59, 2.79] | 7.90 [4.18, 12.64] | 3.44 [1.20, 6.38] | 6.41 [3.42, 9.83] | 2.79 [0.97, 5.07] |
| qwen | Pooled/easy | 196 / 70 | 141 / 55 | 3; 1.53 [0.00, 4.21] | 2; 1.02 [0.00, 2.66] | 2.13 [0.00, 6.06] | 1.42 [0.00, 3.91] | 5.45 [0.00, 14.81] | 3.64 [0.00, 9.52] |
| qwen | Pooled/medium | 200 / 63 | 87 / 113 | 6; 3.00 [0.53, 5.88] | 3; 1.50 [0.00, 3.96] | 6.90 [1.41, 14.15] | 3.45 [0.00, 9.47] | 5.31 [0.93, 10.91] | 2.65 [0.00, 7.34] |
| qwen | Pooled/hard | 254 / 66 | 63 / 191 | 14; 5.51 [2.36, 9.27] | 5; 1.97 [0.39, 4.30] | 22.22 [9.09, 43.08] | 7.94 [1.37, 18.75] | 7.33 [3.06, 12.57] | 2.62 [0.50, 5.70] |
| qwen | hv6_atcoder/all | 252 / 63 | 120 / 132 | 12; 4.76 [1.59, 8.33] | 5; 1.98 [0.40, 4.37] | 10.00 [3.38, 19.51] | 4.17 [0.73, 9.52] | 9.09 [3.27, 16.44] | 3.79 [0.68, 8.26] |
| qwen | hv6_atcoder/easy | 44 / 11 | 44 / 0 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | UNKNOWN | UNKNOWN |
| qwen | hv6_atcoder/medium | 44 / 11 | 31 / 13 | 2; 4.55 [0.00, 11.36] | 1; 2.27 [0.00, 6.82] | 6.45 [0.00, 17.39] | 3.23 [0.00, 11.11] | 15.38 [0.00, 60.00] | 7.69 [0.00, 38.88] |
| qwen | hv6_atcoder/hard | 164 / 41 | 45 / 119 | 10; 6.10 [1.83, 11.59] | 4; 2.44 [0.00, 5.49] | 22.22 [5.88, 53.12] | 8.89 [0.00, 23.68] | 8.40 [2.38, 16.19] | 3.36 [0.00, 7.96] |
| qwen | hv6_leetcode/all | 252 / 63 | 136 / 116 | 8; 3.17 [1.19, 5.95] | 3; 1.19 [0.00, 3.17] | 5.88 [2.03, 11.03] | 2.21 [0.00, 6.06] | 6.90 [2.42, 13.22] | 2.59 [0.00, 7.27] |
| qwen | hv6_leetcode/easy | 68 / 17 | 64 / 4 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| qwen | hv6_leetcode/medium | 104 / 26 | 54 / 50 | 4; 3.85 [0.00, 8.65] | 2; 1.92 [0.00, 5.77] | 7.41 [0.00, 18.18] | 3.70 [0.00, 13.33] | 8.00 [0.00, 20.41] | 4.00 [0.00, 14.29] |
| qwen | hv6_leetcode/hard | 80 / 20 | 18 / 62 | 4; 5.00 [1.25, 10.00] | 1; 1.25 [0.00, 3.75] | 22.22 [5.56, 45.45] | 5.56 [0.00, 20.00] | 6.45 [1.43, 14.04] | 1.61 [0.00, 5.56] |
| qwen | lcbpro_2025/all | 146 / 73 | 35 / 111 | 3; 2.05 [0.00, 5.48] | 2; 1.37 [0.00, 3.42] | 8.57 [0.00, 25.93] | 5.71 [0.00, 16.13] | 2.70 [0.00, 7.29] | 1.80 [0.00, 4.63] |
| qwen | lcbpro_2025/easy | 84 / 42 | 33 / 51 | 3; 3.57 [0.00, 9.52] | 2; 2.38 [0.00, 5.95] | 9.09 [0.00, 27.59] | 6.06 [0.00, 17.24] | 5.88 [0.00, 15.91] | 3.92 [0.00, 10.20] |
| qwen | lcbpro_2025/medium | 52 / 26 | 2 / 50 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| qwen | lcbpro_2025/hard | 10 / 5 | 0 / 10 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | UNKNOWN | UNKNOWN | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| ds | Pooled/all | 650 / 199 | 186 / 464 | 51; 7.85 [5.25, 10.70] | 28; 4.31 [2.38, 6.50] | 27.42 [17.29, 41.76] | 15.05 [8.20, 24.72] | 10.99 [7.35, 15.00] | 6.03 [3.34, 9.13] |
| ds | Pooled/easy | 196 / 70 | 115 / 81 | 15; 7.65 [3.65, 12.37] | 10; 5.10 [2.11, 8.42] | 13.04 [5.97, 23.40] | 8.70 [3.55, 15.65] | 18.52 [9.30, 29.58] | 12.35 [5.26, 20.78] |
| ds | Pooled/medium | 200 / 63 | 48 / 152 | 17; 8.50 [4.08, 13.37] | 8; 4.00 [1.46, 7.14] | 35.42 [15.00, 74.19] | 16.67 [5.26, 37.93] | 11.18 [5.41, 17.65] | 5.26 [1.86, 9.32] |
| ds | Pooled/hard | 254 / 66 | 23 / 231 | 19; 7.48 [3.08, 12.60] | 10; 3.94 [0.39, 8.47] | 82.61 [29.03, 260.00] | 43.48 [4.55, 153.36] | 8.23 [3.29, 13.91] | 4.33 [0.43, 9.40] |
| ds | hv6_atcoder/all | 252 / 63 | 76 / 176 | 25; 9.92 [5.16, 15.48] | 15; 5.95 [1.98, 11.11] | 32.89 [15.38, 64.71] | 19.74 [6.25, 42.86] | 14.20 [7.37, 22.35] | 8.52 [2.82, 15.83] |
| ds | hv6_atcoder/easy | 44 / 11 | 42 / 2 | 2; 4.55 [0.00, 11.36] | 2; 4.55 [0.00, 11.36] | 4.76 [0.00, 12.82] | 4.76 [0.00, 12.82] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| ds | hv6_atcoder/medium | 44 / 11 | 18 / 26 | 6; 13.64 [2.27, 27.27] | 3; 6.82 [0.00, 18.18] | 33.33 [5.88, 127.27] | 16.67 [0.00, 75.00] | 23.08 [5.88, 41.67] | 11.54 [0.00, 27.59] |
| ds | hv6_atcoder/hard | 164 / 41 | 16 / 148 | 17; 10.37 [3.66, 18.29] | 10; 6.10 [0.61, 12.80] | 106.25 [30.77, 450.00] | 62.50 [7.14, 275.00] | 11.49 [4.23, 20.14] | 6.76 [0.68, 14.58] |
| ds | hv6_leetcode/all | 252 / 63 | 90 / 162 | 18; 7.14 [3.57, 11.51] | 9; 3.57 [1.59, 6.35] | 20.00 [8.99, 36.36] | 10.00 [3.70, 19.40] | 11.11 [5.26, 18.18] | 5.56 [2.14, 9.93] |
| ds | hv6_leetcode/easy | 68 / 17 | 53 / 15 | 5; 7.35 [0.00, 17.65] | 4; 5.88 [0.00, 13.24] | 9.43 [0.00, 25.58] | 7.55 [0.00, 19.05] | 33.33 [0.00, 100.00] | 26.67 [0.00, 75.00] |
| ds | hv6_leetcode/medium | 104 / 26 | 30 / 74 | 11; 10.58 [3.85, 18.27] | 5; 4.81 [0.96, 8.65] | 36.67 [11.90, 88.89] | 16.67 [4.17, 41.18] | 14.86 [5.68, 25.42] | 6.76 [1.59, 12.50] |
| ds | hv6_leetcode/hard | 80 / 20 | 7 / 73 | 2; 2.50 [0.00, 6.25] | 0; 0.00 [0.00, 0.00] | 28.57 [0.00, 100.00] | 0.00 [0.00, 0.00] | 2.74 [0.00, 7.25] | 0.00 [0.00, 0.00] |
| ds | lcbpro_2025/all | 146 / 73 | 20 / 126 | 8; 5.48 [2.05, 9.59] | 4; 2.74 [0.68, 5.48] | 40.00 [12.00, 100.00] | 20.00 [4.00, 54.55] | 6.35 [2.31, 11.38] | 3.17 [0.76, 6.50] |
| ds | lcbpro_2025/easy | 84 / 42 | 20 / 64 | 8; 9.52 [3.57, 16.67] | 4; 4.76 [1.19, 9.52] | 40.00 [12.00, 100.00] | 20.00 [4.00, 53.85] | 12.50 [4.55, 21.67] | 6.25 [1.47, 12.50] |
| ds | lcbpro_2025/medium | 52 / 26 | 0 / 52 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | UNKNOWN | UNKNOWN | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| ds | lcbpro_2025/hard | 10 / 5 | 0 / 10 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | UNKNOWN | UNKNOWN | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| llama | Pooled/all | 650 / 199 | 200 / 450 | 25; 3.85 [2.34, 5.59] | 12; 1.85 [0.81, 3.03] | 12.50 [7.39, 19.28] | 6.00 [2.63, 10.40] | 5.56 [3.36, 8.16] | 2.67 [1.20, 4.38] |
| llama | Pooled/easy | 196 / 70 | 119 / 77 | 6; 3.06 [0.98, 5.61] | 5; 2.55 [0.51, 4.95] | 5.04 [1.53, 9.92] | 4.20 [0.85, 8.86] | 7.79 [2.44, 14.67] | 6.49 [1.35, 12.86] |
| llama | Pooled/medium | 200 / 63 | 51 / 149 | 10; 5.00 [1.94, 9.05] | 4; 2.00 [0.00, 4.64] | 19.61 [6.82, 44.24] | 7.84 [0.00, 22.86] | 6.71 [2.52, 12.05] | 2.68 [0.00, 6.21] |
| llama | Pooled/hard | 254 / 66 | 30 / 224 | 9; 3.54 [1.22, 6.20] | 3; 1.18 [0.00, 2.73] | 30.00 [12.00, 62.96] | 10.00 [0.00, 22.73] | 4.02 [1.40, 7.21] | 1.34 [0.00, 3.15] |
| llama | hv6_atcoder/all | 252 / 63 | 89 / 163 | 13; 5.16 [2.38, 8.73] | 4; 1.59 [0.00, 3.57] | 14.61 [6.25, 27.59] | 4.49 [0.00, 11.39] | 7.98 [3.41, 14.20] | 2.45 [0.00, 5.95] |
| llama | hv6_atcoder/easy | 44 / 11 | 43 / 1 | 1; 2.27 [0.00, 6.82] | 0; 0.00 [0.00, 0.00] | 2.33 [0.00, 7.32] | 0.00 [0.00, 0.00] | 100.00 [100.00, 100.00] | 0.00 [0.00, 0.00] |
| llama | hv6_atcoder/medium | 44 / 11 | 20 / 24 | 5; 11.36 [0.00, 25.00] | 2; 4.55 [0.00, 13.64] | 25.00 [0.00, 85.71] | 10.00 [0.00, 50.00] | 20.83 [0.00, 50.00] | 8.33 [0.00, 25.00] |
| llama | hv6_atcoder/hard | 164 / 41 | 26 / 138 | 7; 4.27 [1.22, 7.93] | 2; 1.22 [0.00, 3.05] | 26.92 [8.82, 58.33] | 7.69 [0.00, 18.18] | 5.07 [1.40, 9.70] | 1.45 [0.00, 3.88] |
| llama | hv6_leetcode/all | 252 / 63 | 91 / 161 | 7; 2.78 [1.19, 4.76] | 3; 1.19 [0.00, 2.78] | 7.69 [2.63, 15.79] | 3.30 [0.00, 8.33] | 4.35 [1.68, 7.59] | 1.86 [0.00, 4.09] |
| llama | hv6_leetcode/easy | 68 / 17 | 56 / 12 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| llama | hv6_leetcode/medium | 104 / 26 | 31 / 73 | 5; 4.81 [0.96, 8.65] | 2; 1.92 [0.00, 4.81] | 16.13 [3.57, 42.11] | 6.45 [0.00, 22.22] | 6.85 [1.54, 12.66] | 2.74 [0.00, 6.67] |
| llama | hv6_leetcode/hard | 80 / 20 | 4 / 76 | 2; 2.50 [0.00, 6.25] | 1; 1.25 [0.00, 3.75] | 50.00 [0.00, 300.00] | 25.00 [0.00, 100.00] | 2.63 [0.00, 6.67] | 1.32 [0.00, 4.23] |
| llama | lcbpro_2025/all | 146 / 73 | 20 / 126 | 5; 3.42 [0.68, 6.85] | 5; 3.42 [0.68, 6.85] | 25.00 [5.88, 58.33] | 25.00 [5.88, 58.33] | 3.97 [0.79, 7.75] | 3.97 [0.79, 7.75] |
| llama | lcbpro_2025/easy | 84 / 42 | 20 / 64 | 5; 5.95 [1.19, 10.71] | 5; 5.95 [1.19, 10.71] | 25.00 [6.67, 60.00] | 25.00 [6.67, 60.00] | 7.81 [1.67, 15.25] | 7.81 [1.67, 15.25] |
| llama | lcbpro_2025/medium | 52 / 26 | 0 / 52 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | UNKNOWN | UNKNOWN | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| llama | lcbpro_2025/hard | 10 / 5 | 0 / 10 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | UNKNOWN | UNKNOWN | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

## Probe counts and isolated correct drafts

| Model | Probe-count stratum | Samples | ACD count; rate % [CI] | Stable count; rate % [CI] | ACDs correct only once | ACDs correct more than once, isolated, none adjacent |
|---|---|---:|---|---|---:|---:|
| qwen | 0 | 1 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| qwen | 1 | 33 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| qwen | 2-4 | 48 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| qwen | 5-8 | 86 | 3; 3.49 [0.00, 8.00] | 2; 2.33 [0.00, 6.33] | 1 | 0 |
| qwen | 9-16 | 240 | 8; 3.33 [1.27, 5.73] | 4; 1.67 [0.39, 3.46] | 3 | 1 |
| qwen | 17+ | 242 | 12; 4.96 [2.02, 8.48] | 4; 1.65 [0.39, 3.40] | 6 | 2 |
| ds | 0 | 2 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| ds | 1 | 34 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| ds | 2-4 | 60 | 1; 1.67 [0.00, 5.26] | 1; 1.67 [0.00, 5.26] | 0 | 0 |
| ds | 5-8 | 77 | 4; 5.19 [0.00, 12.00] | 1; 1.30 [0.00, 4.17] | 3 | 0 |
| ds | 9-16 | 295 | 26; 8.81 [5.30, 12.75] | 12; 4.07 [1.71, 6.80] | 12 | 2 |
| ds | 17+ | 182 | 20; 10.99 [5.26, 17.82] | 14; 7.69 [2.79, 13.71] | 0 | 6 |
| llama | 0 | 1 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| llama | 1 | 35 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| llama | 2-4 | 55 | 0; 0.00 [0.00, 0.00] | 0; 0.00 [0.00, 0.00] | 0 | 0 |
| llama | 5-8 | 56 | 1; 1.79 [0.00, 5.88] | 0; 0.00 [0.00, 0.00] | 1 | 0 |
| llama | 9-16 | 341 | 19; 5.57 [3.15, 8.26] | 9; 2.64 [1.14, 4.42] | 8 | 2 |
| llama | 17+ | 162 | 5; 3.09 [0.63, 6.04] | 3; 1.85 [0.00, 4.17] | 1 | 1 |

## Position, later failures, and structural change

Quantiles are shown as min / 25th percentile / median / 75th percentile / max. The position margin does not deduct probe or check costs; AST distance describes structure only and cannot show that a draft was "edited into a broken state" or that the model "switched approaches".

| Model | ACD / 28k-capped ACD | First-correct position / native end position, quantiles | Position margin in tokens, quantiles | Uncapped margin, quantiles | AST distance, quantiles |
|---|---:|---|---|---|---|
| qwen | 23 / 3 | 0.0135 / 0.3087 / 0.4494 / 0.6249 / 0.8070 | 5237.0000 / 7236.5000 / 9247.0000 / 11855.5000 / 17512.0000 | 5237.0000 / 7239.2500 / 9504.5000 / 11270.7500 / 17512.0000 | 0.0098 / 0.2184 / 0.2996 / 0.3975 / 0.7158 |
| ds | 51 / 0 | 0.0067 / 0.0920 / 0.1744 / 0.3925 / 0.9260 | 1113.0000 / 6036.5000 / 10759.0000 / 14143.0000 / 20521.0000 | 1113.0000 / 6036.5000 / 10759.0000 / 14143.0000 / 20521.0000 | 0.0785 / 0.2908 / 0.3747 / 0.4584 / 0.7374 |
| llama | 25 / 3 | 0.0059 / 0.1095 / 0.2561 / 0.4119 / 0.8437 | 2312.0000 / 5589.0000 / 9242.0000 / 12764.0000 / 27834.0000 | 2312.0000 / 5370.0000 / 8560.0000 / 11999.2500 / 16653.0000 | 0.0938 / 0.3523 / 0.4162 / 0.4555 / 0.6707 |

| Model | Later failing probes: category → count | ACDs with no later failing probe | AST bins: category → count |
|---|---|---:|---|
| qwen | {"WRONG_ANSWER": 72, "EXTRACTION_OR_DELIVERY_FAILURE": 14, "RUNTIME_ERROR": 6, "TIMEOUT_CPU": 13} | 2 | {"MEDIUM_AST_EDIT": 21, "LARGE_AST_REWRITE": 1, "SMALL_AST_EDIT": 1} |
| ds | {"WRONG_ANSWER": 294, "EXTRACTION_OR_DELIVERY_FAILURE": 19, "RUNTIME_ERROR": 71, "TIMEOUT_CPU": 7} | 2 | {"LARGE_AST_REWRITE": 9, "MEDIUM_AST_EDIT": 40, "UNPARSEABLE": 1, "SMALL_AST_EDIT": 1} |
| llama | {"WRONG_ANSWER": 139, "EXTRACTION_OR_DELIVERY_FAILURE": 26, "RUNTIME_ERROR": 14, "TIMEOUT_CPU": 6} | 0 | {"MEDIUM_AST_EDIT": 20, "LARGE_AST_REWRITE": 3, "UNPARSEABLE": 1, "SMALL_AST_EDIT": 1} |

### Llama's 25 ACD samples (28k-capped runs flagged in their own column)

| Collection | Problem ID | Sample | 28k-capped | First-correct position | Native thinking length | Ratio | Margin | Final failure category | AST category / distance |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| hv6_atcoder | atcoder:abc389_f | 0 | False | 9913 | 15502 | 0.6395 | 5589 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.3970545769563962 |
| hv6_atcoder | atcoder:abc397_f | 0 | False | 12483 | 14795 | 0.8437 | 2312 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.3709375 |
| hv6_atcoder | atcoder:abc397_b | 0 | False | 3388 | 10446 | 0.3243 | 7058 | RUNTIME_ERROR | LARGE_AST_REWRITE / 0.6706896551724137 |
| hv6_atcoder | atcoder:abc397_b | 1 | False | 2048 | 6918 | 0.2960 | 4870 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.142018779342723 |
| hv6_atcoder | atcoder:abc399_c | 1 | False | 77 | 12294 | 0.0063 | 12217 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.3964830789648308 |
| hv6_atcoder | atcoder:abc400_e | 1 | False | 2048 | 14812 | 0.1383 | 12764 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.47805082965763496 |
| hv6_atcoder | atcoder:arc195_a | 1 | False | 1459 | 12088 | 0.1207 | 10629 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.45359477124183006 |
| hv6_atcoder | atcoder:abc399_f | 2 | False | 293 | 10011 | 0.0293 | 9718 | WRONG_ANSWER | LARGE_AST_REWRITE / 0.5227087299428882 |
| hv6_atcoder | atcoder:abc399_b | 2 | False | 243 | 4808 | 0.0505 | 4565 | EXTRACTION_OR_DELIVERY_FAILURE | UNPARSEABLE / UNKNOWN |
| hv6_atcoder | atcoder:abc400_e | 2 | False | 1150 | 13349 | 0.0861 | 12199 | WRONG_ANSWER | LARGE_AST_REWRITE / 0.5145693243675953 |
| hv6_atcoder | atcoder:abc395_f | 3 | False | 2586 | 7883 | 0.3280 | 5297 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.3194400254533885 |
| hv6_atcoder | atcoder:abc397_b | 3 | False | 4501 | 17573 | 0.2561 | 13072 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.26897689768976896 |
| hv6_atcoder | atcoder:arc191_d | 3 | True | 166 | 28000 | 0.0059 | 27834 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.4347037484885127 |
| hv6_leetcode | leetcode:3677 | 0 | False | 4096 | 13338 | 0.3071 | 9242 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.3632326820603908 |
| hv6_leetcode | leetcode:3763 | 0 | False | 6856 | 11489 | 0.5967 | 4633 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.26598639455782314 |
| hv6_leetcode | leetcode:3786 | 1 | False | 2048 | 13448 | 0.1523 | 11400 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.44945164506480556 |
| hv6_leetcode | leetcode:3770 | 2 | False | 10422 | 16673 | 0.6251 | 6251 | WRONG_ANSWER | SMALL_AST_EDIT / 0.09382422802850356 |
| hv6_leetcode | leetcode:3720 | 3 | False | 560 | 17002 | 0.0329 | 16442 | RUNTIME_ERROR | MEDIUM_AST_EDIT / 0.39993726474278546 |
| hv6_leetcode | leetcode:3744 | 3 | True | 8192 | 28000 | 0.2926 | 19808 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.24845147969717826 |
| hv6_leetcode | leetcode:3743 | 3 | False | 2287 | 10171 | 0.2249 | 7884 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.4456304202801868 |
| lcbpro_2025 | codeforces:2114C | 0 | True | 3602 | 28000 | 0.1286 | 24398 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.4179357021996616 |
| lcbpro_2025 | codeforces:2120A | 0 | False | 6861 | 11525 | 0.5953 | 4664 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.4144193725921849 |
| lcbpro_2025 | codeforces:2121A | 0 | False | 6469 | 15705 | 0.4119 | 9236 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.49588815789473684 |
| lcbpro_2025 | codeforces:2109A | 1 | False | 7327 | 14088 | 0.5201 | 6761 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.4289493575207861 |
| lcbpro_2025 | codeforces:2120B | 1 | False | 2048 | 18701 | 0.1095 | 16653 | WRONG_ANSWER | MEDIUM_AST_EDIT / 0.4612973989345033 |

## Paired differences between models

| Comparison | ACD difference (pp) [CI] | Stable ACD difference (pp) [CI] | Status |
|---|---|---|---|
| ds minus qwen | +4.31 [1.36, 7.50] | +2.77 [0.47, 5.25] | Descriptive only; no new hypothesis |
| llama minus qwen | +0.31 [-1.98, 2.49] | +0.31 [-1.25, 1.86] | H-ACD; stable version is secondary |
| llama minus ds | -4.00 [-6.84, -1.38] | -2.46 [-4.71, -0.46] | Descriptive only; no new hypothesis |

## Rescued and harmed

| Model | Group | Policy | Samples | Rescued | Harmed | UNKNOWN |
|---|---|---|---:|---:|---:|---:|
| qwen | ACD | EG:k4:m0 | 23 | 9 | 0 | 0 |
| qwen | ACD | HYB-B | 23 | 2 | 0 | 0 |
| qwen | non_ACD | EG:k4:m0 | 627 | 0 | 24 | 0 |
| qwen | non_ACD | HYB-B | 627 | 0 | 17 | 0 |
| ds | ACD | EG:k4:m0 | 51 | 24 | 0 | 0 |
| ds | ACD | HYB-B | 51 | 18 | 0 | 0 |
| ds | non_ACD | EG:k4:m0 | 599 | 0 | 15 | 0 |
| ds | non_ACD | HYB-B | 599 | 0 | 15 | 0 |
| llama | ACD | EG:k4:m0 | 25 | 14 | 0 | 0 |
| llama | ACD | HYB-B | 25 | 10 | 0 | 0 |
| llama | non_ACD | EG:k4:m0 | 625 | 0 | 16 | 0 |
| llama | non_ACD | HYB-B | 625 | 0 | 16 | 0 |

Zero harmed in the ACD group, and zero rescued in the non-ACD group over the same full candidate set, follow from the definitions. Rescued/harmed are sample counts; H-EG weights the 199 problems equally, so it cannot be replaced by the net number of rescued samples divided by 650.

## Reconciling intervals and unknowns

- qwen: ACD / stable ACD UNKNOWN = 0 / 0; N UNKNOWN = 0.
- qwen/hv6_atcoder/easy/acd: 10000 resamples had an N-fail denominator of 0 and are undefined.
- qwen/hv6_atcoder/easy/stable_acd: 10000 resamples had an N-fail denominator of 0 and are undefined.
- qwen/hv6_atcoder/medium/acd: 10 resamples had an N-fail denominator of 0 and are undefined.
- qwen/hv6_atcoder/medium/stable_acd: 10 resamples had an N-fail denominator of 0 and are undefined.
- qwen/hv6_leetcode/easy/acd: 1174 resamples had an N-fail denominator of 0 and are undefined.
- qwen/hv6_leetcode/easy/stable_acd: 1174 resamples had an N-fail denominator of 0 and are undefined.
- ds: ACD / stable ACD UNKNOWN = 0 / 0; N UNKNOWN = 0.
- ds/hv6_atcoder/easy/acd: 1162 resamples had an N-fail denominator of 0 and are undefined.
- ds/hv6_atcoder/easy/stable_acd: 1162 resamples had an N-fail denominator of 0 and are undefined.
- ds/hv6_leetcode/easy/acd: 7 resamples had an N-fail denominator of 0 and are undefined.
- ds/hv6_leetcode/easy/stable_acd: 7 resamples had an N-fail denominator of 0 and are undefined.
- llama: ACD / stable ACD UNKNOWN = 0 / 0; N UNKNOWN = 0.
- llama/hv6_atcoder/easy/acd: 3525 resamples had an N-fail denominator of 0 and are undefined.
- llama/hv6_atcoder/easy/stable_acd: 3525 resamples had an N-fail denominator of 0 and are undefined.
- llama/hv6_leetcode/easy/acd: 101 resamples had an N-fail denominator of 0 and are undefined.
- llama/hv6_leetcode/easy/stable_acd: 101 resamples had an N-fail denominator of 0 and are undefined.

The number of valid and undefined bootstrap resamples for each cell is saved in summary.json; a zero denominator must not be filled in as 0.

For the paired Llama−Qwen comparison and its stable version, see the [official summary](acd/summary.json); Llama−DS is descriptive, see the [standalone ACD check](raw/acd_llama_vs_ds.json).

## Limits of interpretation and frozen output metadata

ACDs come from forced-answer branches at intermediate prefixes; they do not show that the native reasoning stream ever wrote out the correct program as is. A test PASS does not mean mathematically correct; probe count is correlated with difficulty and length, and stratifying does not remove opportunity bias. Adjacent PASSes can come from the same program and do not mean the untested stretch in between was correct throughout. Sampling noise, execution timeouts, and limited test coverage have not been ruled out.
