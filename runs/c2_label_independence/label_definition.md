## 3. Pre-specified criteria

English · [简体中文](label_definition.zh-CN.md)

**Primary specification:** `corpus=virtual_v1_pipeline`, `scope=complete_deliverable`, `population=pooled`. Intervals are 95% intervals from problem-level bootstrap resampling.

**Independent label `PRIV-IND` (used by R1):**
- For each deliverable draft, only a subset of the problem's hidden test cases is considered: test cases whose input is identical to any of the problem's public example inputs are removed. The comparison follows the existing example-verification rule: first convert CRLF to LF, then ignore exactly one trailing LF; all remaining bytes must be identical.
- If all of these remaining test cases pass, the label is PASS; if any test case fails, times out, or errors, it is FAIL; if the outcome cannot be determined, it stays UNKNOWN.
- Failures, timeouts, and errors are judged by the original frozen scoring rules, including two retests after a CPU timeout and a majority vote. Retests and majority votes are run only on the deduplicated hidden test cases, without mixing in public-example results. PRIV-IND changes only "which test cases count," not "how a single result is judged."
- If a problem has no test cases left after deduplication, its `PRIV-IND` is UNKNOWN, it is excluded from the denominator, and the number of such problems is reported.
- The metadata baseline is refit on `PRIV-IND` using the original frozen functions, grouped cross-validation folds, and bootstrap random seeds; only the label is replaced.
- If the existing records lack per-test-case results and programs need to be re-executed, they are re-executed under the same definition. All three models use the same implementation.
- **LeetCode execution order.** Under the original frozen scoring, LeetCode's public examples and hidden test cases run in sequence in the same `Solution` instance. `PRIV-IND` takes the per-case results of the deduplicated hidden test cases directly from this original execution sequence, rather than removing the public calls and re-executing; Llama is likewise scored on the original sequence and then filtered. This removes the circularity of "the label containing public-example results," but the public calls may still affect later test cases through instance or global state, and this is reported as a limitation. Standard-input problems start a separate process for each test case and are not affected.

| Criterion | Quantity | Replicated | Not replicated | Reversed |
|---|---|---|---|---|
| R1 The public-example check predicts the independent hidden tests better than the metadata baseline | On `PRIV-IND`, PUB AUROC − baseline AUROC | Interval lower bound > 0 | Interval includes 0 | Interval upper bound < 0 |
| R2 Confidence is worse than the metadata baseline | On the original complete label, CONF-50 AUROC − baseline AUROC | Interval upper bound < 0 | Interval includes 0 | Interval lower bound > 0 |

- R2 keeps the original complete label: confidence is not part of the label, so there is no circularity, and this also allows a direct comparison with the two models already reported.
- The R1 values for Qwen3 and DS-R1 are computed post hoc, but their definition was fixed before the computation; the report labels this accordingly.
- The two are each judged at nominal 95%, with no multiplicity correction; the report states this.
- **Secondary outcomes** (all per-comparison nominal intervals, with no claim of simultaneous coverage):
  - PUB AUROC − baseline AUROC on the original complete label, noted as "the label includes PUB, so this is partly determined by definition";
  - Among drafts that pass PUB, the AUROC of the baseline and of CONF-50 against `PRIV-IND` (once PUB lets a draft through, is there still a signal that can identify errors?). The baseline first filters to drafts that pass PUB and is then refit within this subset, keeping the original problem-level cross-validation folds; a fold with only one label class is recorded as UNKNOWN under the original rule;
  - The overlap rate between public examples and hidden test cases for each problem;
  - The gain from adding PUB to the baseline;
  - CONF-full;
  - PUB precision and recall;
  - Separate results for the three collections;
  - The other three specifications;
  - Intervals for the 43 configurations;
  - The C1 upper bound.
- Results are reported whatever they turn out to be. No samples are added, and no specification or model is switched, because of the results.
