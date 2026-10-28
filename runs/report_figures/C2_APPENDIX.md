<a id="c2-saved-estimates-for-all-four-corpusscope-definitions"></a>

# C2: Saved estimates for all four corpus/scope definitions

English · [简体中文](C2_APPENDIX.zh-CN.md)

The main report uses virtual_v1_pipeline + complete_deliverable. The other three definitions are included as appendix checks.

The metadata baseline is matched to each signal separately. PUB's AUROC and its differences come from the output of the frozen analysis code; the pre-registered comparison metrics for PUB are precision/recall.

| Model | Population | Corpus | Scope | Signal | AUROC [95% CI] | Baseline AUROC | Precision | Recall | False positives / probes positive on PUB or the signal |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | Pooled | virtual_v1_pipeline | pipeline | CONF-50 | 0.7559 [0.7133, 0.7942] | 0.7638 | 0.2482 | 1.0000 | 7452 / 8596 |
| Qwen3-8B | Pooled | virtual_v1_pipeline | pipeline | CONF-full | 0.7723 [0.7329, 0.8077] | 0.8174 | 0.3447 | 1.0000 | 3060 / 4204 |
| Qwen3-8B | Pooled | virtual_v1_pipeline | pipeline | PUB | 0.8792 [0.8531, 0.9029] | 0.8174 | 0.6852 | 1.0000 | 862 / 2006 |
| Qwen3-8B | Pooled | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.6795 [0.6339, 0.7229] | 0.8174 | 0.3447 | 1.0000 | 3060 / 4204 |
| Qwen3-8B | Pooled | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.7723 [0.7329, 0.8077] | 0.8174 | 0.3447 | 1.0000 | 3060 / 4204 |
| Qwen3-8B | Pooled | virtual_v1_pipeline | complete_deliverable | PUB | 0.8792 [0.8531, 0.9029] | 0.8174 | 0.6852 | 1.0000 | 862 / 2006 |
| Qwen3-8B | Pooled | complete_v0_auxiliary | pipeline | CONF-50 | 0.7469 [0.7041, 0.7862] | 0.7595 | 0.2631 | 1.0000 | 7364 / 8607 |
| Qwen3-8B | Pooled | complete_v0_auxiliary | pipeline | CONF-full | 0.7618 [0.7207, 0.7996] | 0.8178 | 0.3369 | 1.0000 | 3500 / 4743 |
| Qwen3-8B | Pooled | complete_v0_auxiliary | pipeline | PUB | 0.8812 [0.8573, 0.9047] | 0.8178 | 0.6813 | 1.0000 | 946 / 2189 |
| Qwen3-8B | Pooled | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.6819 [0.6359, 0.7253] | 0.8178 | 0.3369 | 1.0000 | 3500 / 4743 |
| Qwen3-8B | Pooled | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.7618 [0.7207, 0.7996] | 0.8178 | 0.3369 | 1.0000 | 3500 / 4743 |
| Qwen3-8B | Pooled | complete_v0_auxiliary | complete_deliverable | PUB | 0.8812 [0.8573, 0.9047] | 0.8178 | 0.6813 | 1.0000 | 946 / 2189 |
| Qwen3-8B | Collection: hv6_atcoder | virtual_v1_pipeline | pipeline | CONF-50 | 0.7786 [0.7195, 0.8285] | 0.8677 | 0.3114 | 1.0000 | 2924 / 3368 |
| Qwen3-8B | Collection: hv6_atcoder | virtual_v1_pipeline | pipeline | CONF-full | 0.8226 [0.7560, 0.8781] | 0.8881 | 0.4120 | 1.0000 | 1025 / 1469 |
| Qwen3-8B | Collection: hv6_atcoder | virtual_v1_pipeline | pipeline | PUB | 0.8796 [0.8299, 0.9259] | 0.8881 | 0.7443 | 1.0000 | 262 / 706 |
| Qwen3-8B | Collection: hv6_atcoder | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.7390 [0.6793, 0.7942] | 0.8881 | 0.4120 | 1.0000 | 1025 / 1469 |
| Qwen3-8B | Collection: hv6_atcoder | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.8226 [0.7560, 0.8781] | 0.8881 | 0.4120 | 1.0000 | 1025 / 1469 |
| Qwen3-8B | Collection: hv6_atcoder | virtual_v1_pipeline | complete_deliverable | PUB | 0.8796 [0.8299, 0.9259] | 0.8881 | 0.7443 | 1.0000 | 262 / 706 |
| Qwen3-8B | Collection: hv6_atcoder | complete_v0_auxiliary | pipeline | CONF-50 | 0.7737 [0.7147, 0.8244] | 0.8492 | 0.3306 | 1.0000 | 2874 / 3368 |
| Qwen3-8B | Collection: hv6_atcoder | complete_v0_auxiliary | pipeline | CONF-full | 0.8072 [0.7391, 0.8647] | 0.8816 | 0.4056 | 1.0000 | 1229 / 1723 |
| Qwen3-8B | Collection: hv6_atcoder | complete_v0_auxiliary | pipeline | PUB | 0.8847 [0.8377, 0.9282] | 0.8816 | 0.7474 | 1.0000 | 305 / 799 |
| Qwen3-8B | Collection: hv6_atcoder | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.7426 [0.6832, 0.7960] | 0.8816 | 0.4056 | 1.0000 | 1229 / 1723 |
| Qwen3-8B | Collection: hv6_atcoder | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.8072 [0.7391, 0.8647] | 0.8816 | 0.4056 | 1.0000 | 1229 / 1723 |
| Qwen3-8B | Collection: hv6_atcoder | complete_v0_auxiliary | complete_deliverable | PUB | 0.8847 [0.8377, 0.9282] | 0.8816 | 0.7474 | 1.0000 | 305 / 799 |
| Qwen3-8B | Collection: hv6_leetcode | virtual_v1_pipeline | pipeline | CONF-50 | 0.7633 [0.6942, 0.8234] | 0.8052 | 0.3566 | 1.0000 | 2313 / 2831 |
| Qwen3-8B | Collection: hv6_leetcode | virtual_v1_pipeline | pipeline | CONF-full | 0.7563 [0.6917, 0.8136] | 0.7839 | 0.4244 | 1.0000 | 1325 / 1843 |
| Qwen3-8B | Collection: hv6_leetcode | virtual_v1_pipeline | pipeline | PUB | 0.8019 [0.7558, 0.8480] | 0.7839 | 0.6505 | 1.0000 | 504 / 1022 |
| Qwen3-8B | Collection: hv6_leetcode | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.6945 [0.6186, 0.7647] | 0.7839 | 0.4244 | 1.0000 | 1325 / 1843 |
| Qwen3-8B | Collection: hv6_leetcode | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.7563 [0.6917, 0.8136] | 0.7839 | 0.4244 | 1.0000 | 1325 / 1843 |
| Qwen3-8B | Collection: hv6_leetcode | virtual_v1_pipeline | complete_deliverable | PUB | 0.8019 [0.7558, 0.8480] | 0.7839 | 0.6505 | 1.0000 | 504 / 1022 |
| Qwen3-8B | Collection: hv6_leetcode | complete_v0_auxiliary | pipeline | CONF-50 | 0.7539 [0.6837, 0.8156] | 0.8017 | 0.3673 | 1.0000 | 2294 / 2834 |
| Qwen3-8B | Collection: hv6_leetcode | complete_v0_auxiliary | pipeline | CONF-full | 0.7589 [0.6952, 0.8161] | 0.7892 | 0.4185 | 1.0000 | 1445 / 1985 |
| Qwen3-8B | Collection: hv6_leetcode | complete_v0_auxiliary | pipeline | PUB | 0.8075 [0.7628, 0.8523] | 0.7892 | 0.6515 | 1.0000 | 531 / 1071 |
| Qwen3-8B | Collection: hv6_leetcode | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.6948 [0.6186, 0.7652] | 0.7892 | 0.4185 | 1.0000 | 1445 / 1985 |
| Qwen3-8B | Collection: hv6_leetcode | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.7589 [0.6952, 0.8161] | 0.7892 | 0.4185 | 1.0000 | 1445 / 1985 |
| Qwen3-8B | Collection: hv6_leetcode | complete_v0_auxiliary | complete_deliverable | PUB | 0.8075 [0.7628, 0.8523] | 0.7892 | 0.6515 | 1.0000 | 531 / 1071 |
| Qwen3-8B | Collection: lcbpro_2025 | virtual_v1_pipeline | pipeline | CONF-50 | 0.6624 [0.5926, 0.7317] | 0.7647 | 0.1002 | 1.0000 | 2215 / 2397 |
| Qwen3-8B | Collection: lcbpro_2025 | virtual_v1_pipeline | pipeline | CONF-full | 0.7088 [0.6368, 0.7775] | 0.7893 | 0.2143 | 1.0000 | 710 / 892 |
| Qwen3-8B | Collection: lcbpro_2025 | virtual_v1_pipeline | pipeline | PUB | 0.9291 [0.8960, 0.9581] | 0.7893 | 0.6578 | 1.0000 | 96 / 278 |
| Qwen3-8B | Collection: lcbpro_2025 | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.5636 [0.4730, 0.6569] | 0.7893 | 0.2143 | 1.0000 | 710 / 892 |
| Qwen3-8B | Collection: lcbpro_2025 | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.7088 [0.6368, 0.7775] | 0.7893 | 0.2143 | 1.0000 | 710 / 892 |
| Qwen3-8B | Collection: lcbpro_2025 | virtual_v1_pipeline | complete_deliverable | PUB | 0.9291 [0.8960, 0.9581] | 0.7893 | 0.6578 | 1.0000 | 96 / 278 |
| Qwen3-8B | Collection: lcbpro_2025 | complete_v0_auxiliary | pipeline | CONF-50 | 0.6538 [0.5736, 0.7306] | 0.7516 | 0.1150 | 1.0000 | 2196 / 2405 |
| Qwen3-8B | Collection: lcbpro_2025 | complete_v0_auxiliary | pipeline | CONF-full | 0.6769 [0.5934, 0.7569] | 0.7913 | 0.2054 | 1.0000 | 826 / 1035 |
| Qwen3-8B | Collection: lcbpro_2025 | complete_v0_auxiliary | pipeline | PUB | 0.9260 [0.8930, 0.9546] | 0.7913 | 0.6358 | 1.0000 | 110 / 319 |
| Qwen3-8B | Collection: lcbpro_2025 | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.5675 [0.4700, 0.6686] | 0.7913 | 0.2054 | 1.0000 | 826 / 1035 |
| Qwen3-8B | Collection: lcbpro_2025 | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.6769 [0.5934, 0.7569] | 0.7913 | 0.2054 | 1.0000 | 826 / 1035 |
| Qwen3-8B | Collection: lcbpro_2025 | complete_v0_auxiliary | complete_deliverable | PUB | 0.9260 [0.8930, 0.9546] | 0.7913 | 0.6358 | 1.0000 | 110 / 319 |
| DS-R1-Qwen-7B | Pooled | virtual_v1_pipeline | pipeline | CONF-50 | 0.6877 [0.6400, 0.7330] | 0.8088 | 0.2056 | 1.0000 | 7468 / 8274 |
| DS-R1-Qwen-7B | Pooled | virtual_v1_pipeline | pipeline | CONF-full | 0.8141 [0.7780, 0.8477] | 0.8238 | 0.2132 | 1.0000 | 6322 / 7128 |
| DS-R1-Qwen-7B | Pooled | virtual_v1_pipeline | pipeline | PUB | 0.9466 [0.9326, 0.9596] | 0.8238 | 0.7173 | 1.0000 | 647 / 1453 |
| DS-R1-Qwen-7B | Pooled | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.6760 [0.6282, 0.7213] | 0.8238 | 0.2132 | 1.0000 | 6322 / 7128 |
| DS-R1-Qwen-7B | Pooled | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.8141 [0.7780, 0.8477] | 0.8238 | 0.2132 | 1.0000 | 6322 / 7128 |
| DS-R1-Qwen-7B | Pooled | virtual_v1_pipeline | complete_deliverable | PUB | 0.9466 [0.9326, 0.9596] | 0.8238 | 0.7173 | 1.0000 | 647 / 1453 |
| DS-R1-Qwen-7B | Pooled | complete_v0_auxiliary | pipeline | CONF-50 | 0.6852 [0.6374, 0.7305] | 0.8095 | 0.2068 | 1.0000 | 7461 / 8274 |
| DS-R1-Qwen-7B | Pooled | complete_v0_auxiliary | pipeline | CONF-full | 0.8180 [0.7826, 0.8505] | 0.8181 | 0.2119 | 1.0000 | 6606 / 7419 |
| DS-R1-Qwen-7B | Pooled | complete_v0_auxiliary | pipeline | PUB | 0.9470 [0.9330, 0.9598] | 0.8181 | 0.7170 | 1.0000 | 661 / 1474 |
| DS-R1-Qwen-7B | Pooled | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.6772 [0.6294, 0.7226] | 0.8181 | 0.2119 | 1.0000 | 6606 / 7419 |
| DS-R1-Qwen-7B | Pooled | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.8180 [0.7826, 0.8505] | 0.8181 | 0.2119 | 1.0000 | 6606 / 7419 |
| DS-R1-Qwen-7B | Pooled | complete_v0_auxiliary | complete_deliverable | PUB | 0.9470 [0.9330, 0.9598] | 0.8181 | 0.7170 | 1.0000 | 661 / 1474 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | virtual_v1_pipeline | pipeline | CONF-50 | 0.7062 [0.6187, 0.7863] | 0.8761 | 0.2696 | 1.0000 | 2797 / 3139 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | virtual_v1_pipeline | pipeline | CONF-full | 0.8419 [0.7819, 0.8914] | 0.8793 | 0.2757 | 1.0000 | 2217 / 2559 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | virtual_v1_pipeline | pipeline | PUB | 0.9479 [0.9216, 0.9705] | 0.8793 | 0.7850 | 1.0000 | 221 / 563 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.6905 [0.6017, 0.7719] | 0.8793 | 0.2757 | 1.0000 | 2217 / 2559 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.8419 [0.7819, 0.8914] | 0.8793 | 0.2757 | 1.0000 | 2217 / 2559 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | virtual_v1_pipeline | complete_deliverable | PUB | 0.9479 [0.9216, 0.9705] | 0.8793 | 0.7850 | 1.0000 | 221 / 563 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | complete_v0_auxiliary | pipeline | CONF-50 | 0.7050 [0.6176, 0.7846] | 0.8754 | 0.2705 | 1.0000 | 2794 / 3139 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | complete_v0_auxiliary | pipeline | CONF-full | 0.8474 [0.7882, 0.8952] | 0.8767 | 0.2747 | 1.0000 | 2353 / 2698 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | complete_v0_auxiliary | pipeline | PUB | 0.9484 [0.9223, 0.9707] | 0.8767 | 0.7858 | 1.0000 | 230 / 575 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.6945 [0.6058, 0.7750] | 0.8767 | 0.2747 | 1.0000 | 2353 / 2698 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.8474 [0.7882, 0.8952] | 0.8767 | 0.2747 | 1.0000 | 2353 / 2698 |
| DS-R1-Qwen-7B | Collection: hv6_atcoder | complete_v0_auxiliary | complete_deliverable | PUB | 0.9484 [0.9223, 0.9707] | 0.8767 | 0.7858 | 1.0000 | 230 / 575 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | virtual_v1_pipeline | pipeline | CONF-50 | 0.7250 [0.6588, 0.7855] | 0.8519 | 0.3002 | 1.0000 | 2534 / 2900 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | virtual_v1_pipeline | pipeline | CONF-full | 0.7860 [0.7303, 0.8370] | 0.8525 | 0.3024 | 1.0000 | 2369 / 2735 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | virtual_v1_pipeline | pipeline | PUB | 0.9197 [0.8878, 0.9475] | 0.8525 | 0.7296 | 1.0000 | 306 / 672 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.7210 [0.6544, 0.7815] | 0.8525 | 0.3024 | 1.0000 | 2369 / 2735 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.7860 [0.7303, 0.8370] | 0.8525 | 0.3024 | 1.0000 | 2369 / 2735 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | virtual_v1_pipeline | complete_deliverable | PUB | 0.9197 [0.8878, 0.9475] | 0.8525 | 0.7296 | 1.0000 | 306 / 672 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | complete_v0_auxiliary | pipeline | CONF-50 | 0.7250 [0.6588, 0.7855] | 0.8519 | 0.3002 | 1.0000 | 2534 / 2900 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | complete_v0_auxiliary | pipeline | CONF-full | 0.7890 [0.7337, 0.8397] | 0.8511 | 0.3014 | 1.0000 | 2413 / 2779 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | complete_v0_auxiliary | pipeline | PUB | 0.9198 [0.8880, 0.9477] | 0.8511 | 0.7291 | 1.0000 | 309 / 675 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.7218 [0.6556, 0.7823] | 0.8511 | 0.3014 | 1.0000 | 2413 / 2779 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.7890 [0.7337, 0.8397] | 0.8511 | 0.3014 | 1.0000 | 2413 / 2779 |
| DS-R1-Qwen-7B | Collection: hv6_leetcode | complete_v0_auxiliary | complete_deliverable | PUB | 0.9198 [0.8880, 0.9477] | 0.8511 | 0.7291 | 1.0000 | 309 / 675 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | virtual_v1_pipeline | pipeline | CONF-50 | 0.5964 [0.4877, 0.6916] | 0.7524 | 0.0688 | 1.0000 | 2137 / 2235 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | virtual_v1_pipeline | pipeline | CONF-full | 0.7748 [0.6909, 0.8695] | 0.7796 | 0.0823 | 1.0000 | 1736 / 1834 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | virtual_v1_pipeline | pipeline | PUB | 0.9634 [0.9448, 0.9792] | 0.7796 | 0.5506 | 1.0000 | 120 / 218 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | virtual_v1_pipeline | complete_deliverable | CONF-50 | 0.5802 [0.4750, 0.6741] | 0.7796 | 0.0823 | 1.0000 | 1736 / 1834 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | virtual_v1_pipeline | complete_deliverable | CONF-full | 0.7748 [0.6909, 0.8695] | 0.7796 | 0.0823 | 1.0000 | 1736 / 1834 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | virtual_v1_pipeline | complete_deliverable | PUB | 0.9634 [0.9448, 0.9792] | 0.7796 | 0.5506 | 1.0000 | 120 / 218 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | complete_v0_auxiliary | pipeline | CONF-50 | 0.5821 [0.4740, 0.6770] | 0.7688 | 0.0713 | 1.0000 | 2133 / 2235 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | complete_v0_auxiliary | pipeline | CONF-full | 0.7685 [0.6911, 0.8569] | 0.7791 | 0.0804 | 1.0000 | 1840 / 1942 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | complete_v0_auxiliary | pipeline | PUB | 0.9637 [0.9452, 0.9795] | 0.7791 | 0.5467 | 1.0000 | 122 / 224 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | complete_v0_auxiliary | complete_deliverable | CONF-50 | 0.5731 [0.4676, 0.6655] | 0.7791 | 0.0804 | 1.0000 | 1840 / 1942 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | complete_v0_auxiliary | complete_deliverable | CONF-full | 0.7685 [0.6911, 0.8569] | 0.7791 | 0.0804 | 1.0000 | 1840 / 1942 |
| DS-R1-Qwen-7B | Collection: lcbpro_2025 | complete_v0_auxiliary | complete_deliverable | PUB | 0.9637 [0.9452, 0.9795] | 0.7791 | 0.5467 | 1.0000 | 122 / 224 |
