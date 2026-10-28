<a id="correct-looking-but-wrong"></a>

# Correct-looking but wrong

English · [简体中文](case_studies.zh-CN.md)

A cached attention mask can have the right shape and still mean the wrong thing. An audited result can reproduce the recorded computation and still rest on defective input data. These cases expand on [What I built](../README.md#what-i-built) and [Two turning points that changed my conclusions](../README.md#two-moments-that-changed-my-conclusions) in the README. The development trial at the end adds background to [Things I tried that didn't work](../README.md#things-i-tried-that-didnt-work).

<a id="checking-cached-attention"></a>

## Checking cached attention

<a id="the-problem"></a>

### The problem

A KV cache stores earlier keys and values so that later generation can reuse them. When a new block of input arrives, the attention mask must reflect the block's absolute position in the accumulated context.

Suppose three positions are already cached and two new queries are added. The first query is at position 3 and may attend to the keys at positions 0 through 3; the second is at position 4 and may attend to all five keys. A generic top-left-aligned triangular mask has the right shape but allows the wrong positions: it treats the two queries as if they started at position 0, letting them see only the first 1 and the first 2 keys, so cached positions that should be visible are masked out.

<a id="the-implementation-and-check"></a>

### The implementation and check

The attention wrapper I implemented distinguishes three kinds of input: the initial input, single-token decoding, and multi-token blocks appended to an existing context. The implementation uses absolute RoPE positions, preallocated KV storage, and request-level transactions, and the cache checks positions and capacity. If some layers have already appended data and the model call then fails, the model first restores the previously visible cache length before allowing the next call.

The hand-checkable test sets all queries and keys to zero, so the attention weights are spread evenly over the allowed positions. With values `[1, 2, 3, 4, 5]`, the expected outputs are **2.5 and 3.0**; with the wrong mask, the outputs are **1.0 and 1.5**. This exposes alignment errors directly, without depending on the model's random behavior.

Other tests cover input blocks of unequal length and faults injected into later attention layers. The rollback tests check both that the cache length is restored and that the next call is correct.

<a id="what-this-establishes"></a>

### Code and tests

| Evidence | Location |
|---|---|
| Attention wrapper | [attention.py](../src/cs336_basics/attention.py) |
| Cache storage and checks | [kv_cache.py](../src/cs336_basics/kv_cache.py) |
| Model integration and rollback | [transformer.py](../src/cs336_basics/transformer.py) |
| Hand-computed example | [test_attention_backends.py, lines 29–41](../tests/test_attention_backends.py#L29-L41) |
| Unequal-length chunks and fault recovery | [test_attention_backends.py, lines 71–132](../tests/test_attention_backends.py#L71-L132), [cache tests](../tests/test_kv_cache.py), [chunked prefill tests](../tests/test_chunked_prefill.py) |

<a id="streaming-tokenization-must-preserve-the-token-sequence"></a>

## Streaming tokenization must preserve the token sequence

Chunking is only a way of delivering the input; it must not change the token IDs. With the GPT-2 vocabulary and merges used in the tests, `encode("we're")` returns `[732, 821]`. An early streaming implementation, however, produced `[732, 6, 260]` for `["we'r", "e"]`, even though the decoded text was the same. The [tokenizer](../src/cs336_basics/tokenizer.py) emits only the prefix whose boundaries are already settled, and holds back the trailing pre-token that might still merge with the next chunk so it is processed together with that chunk. The [regression tests](../tests/test_tokenizer.py) compare whole-text encoding against every two-way split and against character-by-character input, covering contractions, whitespace, Unicode, and overlapping special tokens.

<a id="restoring-training-state-not-only-weights"></a>

## Restoring training state, not only weights

Training and validation each use their own independent NumPy random number generator. Version 2 checkpoints save the optimizer and RNG state, and the training entry point verifies the data contents and the effective configuration before resuming. The [existing tests](../tests/test_training_reproducibility.py) use a real small Transformer and optimizer to compare the trajectory of continuous training with that of training resumed after an interruption, and check that changing the evaluation frequency does not change the training samples. The [serialization tests](../tests/test_serialization.py) cover the checkpoint interface.

<a id="a-data-defect-that-flipped-the-primary-result"></a>

## A data defect that flipped the primary result

**Symptom.** Some saved programs passed every recorded hidden test yet hit runtime errors on the public examples. In Qwen3's LiveCodeBench Pro (LCB Pro below) records, the affected ones were **N 17/146** and **probe drafts 58/2,405**. A discrepancy like this calls for investigation; it is not grounds for simply re-judging the programs as correct.

**Localization.** The blank lines were already present in the `problem_statement` field of the upstream Hugging Face parquet; they were not added during local extraction. All **78/78** public inputs show exactly the same serialization pattern: one LF at the start, two LFs between non-empty data lines, and two LFs at the end. This pins the defect to the upstream field. [Provenance trace](../runs/lcb_public_erratum/summary.json)

**Correction.** The fix reverses only this exact pattern: it removes the leading LF and restores a single LF between data lines and at the end, and changes nothing else. [Candidate rule](../runs/lcb_public_erratum/summary.json), [verified correction and its boundaries](../runs/lcb_public_erratum/official_html_v3/verification_receipt_v3.json)

**Verification.** The evidence accumulated in stages:

- The author test packages could verify only **3 problems and 5 examples**; the remaining **70 problems and 73 examples** were **UNKNOWN** at the time. [Author test package check](../runs/lcb_public_erratum/summary.json)
- I then checked against the official Codeforces HTML. The first version of the parser judged **every example UNKNOWN**: it did not implement the HTML5 rule "ignore the first LF after a `pre` start tag," and it treated trailing spaces in the outputs as conflicts.
- With the parser fixed, `html5lib` and an independent `HTMLParser` implementation agreed across **73/73 problems and 78/78 examples**. The corrected inputs of all **78/78** examples are byte-for-byte identical to the official parse, and all **78/78** original outputs agree field by field after splitting on whitespace, with **8 examples** differing only in trailing whitespace. [Official HTML check](../runs/lcb_public_erratum/official_html_v3/verification_receipt_v3.json)

**Contrast.** The same "fails the public examples, passes the hidden tests" pattern turned out to be a different story on AtCoder. All **19 records (2 N, 17 probe drafts)** are genuinely wrong answers that the hidden tests simply failed to catch. Take `abc398_d`: on a counterexample that can be checked by hand, the correct output is `0001101011`, but the wrong programs output `0000001000`; they track only the initial puff of smoke and miss the smoke generated afterward. The public examples themselves are fine, so I did not extend the LCB Pro correction to AtCoder.

<details>
<summary>AtCoder: an independent check of the 19 records that fail PUB but pass the hidden tests</summary>

| Problem | N / probe drafts | Distinct programs | Actual PUB failure | Cause |
|---|---:|---:|---|---|
| `abc398_d` | 2 / 13 | 15 | WRONG_ANSWER | Misses the smoke newly generated at the origin later on; 4 of the probe drafts also treat the reversed suffix trajectory as the answer for each time step |
| `arc195_a` | 0 / 4 | 4 | WRONG_ANSWER | Mistakes an element's occurrence at another position for a second complete valid subsequence, without verifying the remaining suffix |
| Total | **2 / 17** | **19** | **19 wrong answers, 0 runtime errors, 0 timeouts** | 2 problems, samples 0/1/2; sample 3 has no such records |

**Source check.** Across the four samples, the inputs/outputs of all 63 problems and 165 public examples are character-for-character identical to the fields in the original `test6.jsonl` (with only the `testtype` metadata removed). The 6 examples of the two problems above were also checked against the official HTML examples: after display-layer handling of only HTML escapes, CRLF, and trailing newlines, the inputs and outputs match character for character, with no leading or interline blank lines. Official sources: [abc398_d](https://atcoder.jp/contests/abc398/tasks/abc398_d?lang=en), [arc195_a](https://atcoder.jp/contests/arc195/tasks/arc195_a?lang=en).

**Checkable counterexamples.** The correct output for `abc398_d` example 2 is `0001101011`; the existing cached output of the two N programs and 9 other probe drafts is `0000001000`. They output 1 only when the initial puff of smoke reaches the target, ignoring smoke generated later: for example, at step 4, the smoke newly generated at time 1 has already reached the target, while the initial puff has not. The other 4 probe drafts, which reverse the order, get even example 1 wrong. `arc195_a` example 2 is `A=[1,2,1]`, `B=[1,2]`: there is only one match, at positions `(1,2)`, so the output should be `No`; all 4 programs output `Yes`, because they wrongly treat the trailing `1` as a usable substitute even though no `2` comes after it. All 19 of these failures in the cache exited normally with empty standard error, so they are not runtime, permission, timeout, or comparator-whitespace problems.

**Why the hidden tests still pass.** In the current dataset each problem has only 40 hidden tests, and none of these 6 public examples is among the corresponding hidden-test inputs. Of the 40 expected outputs for `abc398_d`, 38 consist entirely of 0s, and the other 2, which contain a 1, both have inputs of length 1, so they don't cover the continuous-smoke error the examples reveal. The hidden tests for `arc195_a` have expected outputs of 37 No and 3 Yes, and the existing 4 wrong programs pass all 40.

</details>

**Uniform re-evaluation.** I did not regenerate model outputs, and the prompts are the original ones. The public-example checks, final labels, policy replay, and statistics were updated uniformly for every policy. Hidden-test observations were reused by exact cache identity rather than re-executing every hidden test ([re-evaluation scope](execexit/REPORT.md#lcb-pro-public-input-defect-and-correction)). The runtime errors on the public examples for the original **17 N** programs and **58 probe drafts** all disappeared, and these programs all passed the final evaluation. No new harmed problems and no new UNKNOWN results appeared.

**Why the conclusion flipped.**

| Metric | Original | Corrected | Source |
|---|---:|---:|---|
| N problem-averaged pass rate | 36.68% | 40.95% | [Saved primary comparison](execexit/REPORT.md#policy-outcomes) |
| HYB-B problem-averaged pass rate | 35.55% | 39.07% | [Saved primary comparison](execexit/REPORT.md#policy-outcomes) |
| LCB Pro HYB-B − N, pp | +2.05 | 0.00 | [Saved per-dataset comparison](execexit/REPORT.md#policy-outcomes) |
| Overall HYB-B − N, pp (95% CI) | −1.13 [−2.76, 0.38] | −1.88 [−3.39, −0.63] | [Saved primary comparison](execexit/REPORT.md#policy-outcomes) |
| Nominal non-inferiority | Established | Not established | [Saved verdict](execexit/REPORT.md#policy-outcomes) |

The AtCoder and LeetCode results did not change, as verified in the [comparison summary](../runs/qwen_erratum_comparison/summary.json). After the correction, the interval [−3.39, −0.63] excludes 0, so the drop is real; its lower bound crosses the −3 margin, so nominal non-inferiority does not hold.

The LCB Pro rescue/harm details point to a specific mechanism. The original comparison listed **three rescued sample pairs: `2113A s0`, `2117D s0`, and `2121G s1`**, and no harmed pairs; the corrected comparison has neither rescues nor harms ([sample-pair lists for the original and corrected versions](../runs/qwen_erratum_comparison/raw/comparisons.json)). The erratum records show that once the public inputs were fixed, N went from failing to passing on all three pairs. The data defect made native generation's successes look like rescues by HYB-B.

**Lesson.** The original audit verified that the records and the computations were consistent, but not that the upstream inputs were faithful.

<a id="evaluating-faster-code-generation"></a>

## Evaluating faster code generation

<a id="the-question-and-comparison"></a>

### The question and comparison

Can short probes identify when a reasoning model should stop thinking, so that it answers earlier while still delivering programs successfully?

On an RTX 5090, I ran Qwen3-8B with unmodified vLLM 0.26.0 and compared three methods on the same 36 previously seen development problems, drawn from 18 contests:

- **Native reasoning (N):** the reference path, with no early-exit probing.
- **Ungated early exit (S):** accepts the probe result directly at the first eligible opportunity.
- **Confidence-gated early exit (D):** accepts only if the same probe's score exceeds a frozen threshold; otherwise it resumes reasoning and does not look for a second probing opportunity.

All methods share a total budget of 16,384 tokens and the same output protocol. Discarded probe tokens and control tokens count toward the budget. The native method takes the ordinary execution path and bears no extra probing cost. This component borrows DEER-style probing and does not implement the full DEER system.

<a id="measuring-the-actual-outcome"></a>

### Measuring the actual outcome

For a run to count as a success, the selected final response must end normally with the model's end-of-sequence token, and a correct program must reach the client and pass the frozen public examples and hidden tests. Running out of budget does not count as ending normally. All planned runs stay in the denominator.

Delivery time runs from request admission until a separate local client process receives the result, and includes post-admission probing, resumption, processing, and delivery. Shared setup work such as model loading, cache resets before admission, and client process startup is recorded separately. Repeated A/A checks were used to estimate run-to-run variation. This is a serial comparison.

<a id="results-and-decision"></a>

### Results and decision

| Method | Successful deliveries | Mean delivery time | 95th percentile delivery time |
|---|---:|---:|---:|
| Native reasoning (N) | 19 / 36 | 110.08 s | 182.53 s |
| Ungated early exit (S) | 13 / 36 | 65.83 s | 182.17 s |
| Confidence-gated early exit (D) | 14 / 36 | 90.11 s | 182.57 s |

Relative to native reasoning, confidence gating cut mean time by 18.14% but had 5 fewer successes in total; paired by problem, it rescued 1 problem and harmed 6. Relative to ungated early exit, gating added 1 success while increasing mean time by 36.89%.

The confidence rule rejected 17 probes. Of those problems, 13 failed under both probing methods, 3 passed under both, and 1 was rescued by the gate. Relative to ungated early exit, these rejection decisions cost a total of 874.01 extra seconds across the corresponding paired runs. An offline analysis shows that the single extra success relative to ungated early exit is sensitive to the output extraction protocol, so it cannot be credited to the confidence gate.

Neither early-exit policy met the frozen acceptance criteria. I kept native reasoning and ended my investigation of this component. These results show that the complete outcome and the actual cost of control decisions must be measured.
