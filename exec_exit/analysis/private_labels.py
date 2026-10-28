"""从已保存的逐用例结果生成PRIV-IND标签，不重新执行候选程序。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from exec_exit.execute.evaluate_dev import adjudicate

FAILURES = {"WRONG_ANSWER", "TIMEOUT_CPU", "TIMEOUT_WALL", "RUNTIME_ERROR", "OUTPUT_LIMIT"}
MISSING = {"UNKNOWN", "NOT_EVALUATED_AFTER_FAILURE"}


def subset_result(suite, indices, expected_total):
    #1.保留索引必须对应原执行顺序，不能把缺失后缀当成完整记录
    if indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < expected_total for i in indices):
        raise ValueError("Retained indices must be unique, ordered, and in range")
    if suite["total"] != expected_total:
        raise ValueError(f"Suite length differs from bound inputs: {suite['total']} != {expected_total}")
    sequence = suite.get("case_status_sequence", [])
    if len(sequence) not in (0, expected_total):
        raise ValueError("Truncated per-case sequence without explicit missing suffix")
    selected = [sequence[i] for i in indices] if sequence else []
    unexpected = set(selected) - FAILURES - MISSING - {"PASS"}
    if unexpected:
        raise ValueError(f"Unrecognized recorded case statuses: {sorted(unexpected)}")
    #2.已观察到失败即可判失败；空集合或缺失观察仍保留为UNKNOWN
    counts = Counter(selected)
    if any(counts[s] for s in FAILURES):
        passed = False
    elif indices and len(selected) == len(indices) and counts["PASS"] == len(indices):
        passed = True
    else:
        passed = None
    return {
        "all_passed": passed, "case_status_counts": dict(counts),
        "status": "PASS" if passed is True else "FAIL" if passed is False else "UNKNOWN",
        "retained_cases": len(indices), "observed_sequence_length": len(sequence),
        "cache_key": suite.get("cache_key"), "attempt_index": suite.get("attempt_index"),
    }


def independent_label(record, overlap):
    #用排除公开输入重叠后的隐藏测试评价信号，避免把公开测试同时当作特征和正确性标签。
    #1.只读取已保存的重复执行，首次CPU超时才要求三次裁决
    indices = overlap["execution_retained_indices"]
    saved = record.get("timeout_robustness", {}).get("attempts", [])
    if not saved:
        return {"outcome": "UNKNOWN", "reason": "NO_SAVED_ATTEMPTS", "attempts": [],
                "first_timeout": False, "timeout_flip": False}
    first = subset_result(saved[0]["private_result"], indices, overlap["execution_total"])
    first_timeout = first["case_status_counts"].get("TIMEOUT_CPU", 0) > 0
    required = 3 if first_timeout else 1
    selected = []
    for index, attempt in enumerate(saved[:required]):
        if attempt["attempt_index"] != index or attempt["private_result"]["attempt_index"] != index:
            raise ValueError("Required attempt indices are not consecutive from zero")
        selected.append(first if index == 0 else subset_result(attempt["private_result"], indices, overlap["execution_total"]))
    #2.裁决器固定接收两组结果；公开槽位用恒通过值，使标签仅由隐藏测试决定
    neutral = {"all_passed": True, "case_status_counts": {}}
    verdict = adjudicate([
        {"public_result": neutral, "private_result": r, "whole_program_result": r["status"]}
        for r in selected
    ])
    outcome = verdict["final_outcome"]
    reason = "EMPTY_REMAINING_SUITE" if not indices else "INCOMPLETE_REPEATS" if len(selected) < required else (
        "UNOBSERVED_RETAINED_CASES" if outcome == "UNKNOWN" else "OBSERVED_PRIVATE_SUBSET")
    return {
        "outcome": outcome, "reason": reason, "first_timeout": first_timeout,
        "timeout_flip": first_timeout and outcome == "PASS",
        "required_attempts": required, "used_attempts": len(selected),
        "ignored_extra_attempts": max(0, len(saved) - required), "attempts": selected,
        "first_outcome": first["status"],
        **{name: verdict[name] for name in ("unstable", "pass_fail_unstable", "status_type_unstable")},
    }


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_rows(signals_path, identities_path, overlap_path, output):
    #1.按题目和草稿身份对齐，重复身份不能覆盖原记录
    with Path(overlap_path).open() as stream:
        overlap_rows = [json.loads(line) for line in stream]
    overlap = {r["case_id"]: r for r in overlap_rows}
    if len(overlap) != len(overlap_rows):
        raise ValueError("Duplicate overlap question identity")
    with Path(signals_path).open() as stream:
        rows = [r for line in stream if (r := json.loads(line))["corpus"] == "virtual_v1_pipeline" and r["deliverable"]]
    expected = {(r["collection"], r["sample"], r["subject_id"]): r for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate eligible signal identity")
    evaluations = {}
    sources = []
    #2.评分来源必须与绑定文件一致，并覆盖全部有效草稿
    identities = json.loads(Path(identities_path).read_text())
    for job, identity in identities.items():
        sample = job.rsplit("_s", 1)[1]
        collection = next(c for c in ("hv6_atcoder", "hv6_leetcode", "lcbpro_2025") if c in job)
        source = identity["probe_evaluation"]
        path = Path(source["path"])
        if sha(path) != source["sha256"]:
            raise ValueError(f"Evaluation identity changed: {path}")
        sources.append(source)
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                key = (collection, sample, record["subject_id"])
                if key in expected:
                    if key in evaluations:
                        raise ValueError(f"Duplicate evaluation: {key}")
                    evaluations[key] = (record, str(path), line_number)
    if set(expected) != set(evaluations):
        raise ValueError("Missing eligible evaluation rows")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    by_collection = {}
    #3.保留原标签和来源位置，独立标签中的UNKNOWN不丢弃
    with (output / "rows.jsonl").open("x") as row_file, (output / "labels.jsonl").open("x") as label_file:
        for row in rows:
            key = (row["collection"], row["sample"], row["subject_id"])
            record, path, line_number = evaluations[key]
            if record["code_sha256"] != row["code_sha256"]:
                raise ValueError("Signal/evaluation code identity differs")
            if record["case_id"] != row["case_id"] or record["final_outcome"] != row["outcome"]:
                raise ValueError("Signal/evaluation question or original label differs")
            if overlap[row["case_id"]]["collection"] != row["collection"]:
                raise ValueError("Overlap collection binding differs")
            result = independent_label(record, overlap[row["case_id"]])
            outcome = result["outcome"]
            stats = {"rows": 1, outcome: 1, "first_timeout": int(result["first_timeout"]),
                     "first_timeout_flipped_to_pass": int(result["timeout_flip"]),
                     "different_from_original_joint": int(outcome != row["outcome"]),
                     "known_label_changes": int(outcome in ("PASS", "FAIL") and outcome != row["outcome"]),
                     "became_unknown": int(outcome == "UNKNOWN" and row["outcome"] != "UNKNOWN")}
            if outcome == "UNKNOWN":
                stats["unknown_" + result["reason"]] = 1
            counts.update(stats)
            by_collection.setdefault(row["collection"], Counter()).update(stats)
            label = {"case_id": row["case_id"], "sample": row["sample"], "subject_id": row["subject_id"],
                     "collection": row["collection"], "public_status": row["public_status"],
                     "original_joint_outcome": row["outcome"], "evaluation_path": path,
                     "evaluation_line": line_number, **result}
            label_file.write(json.dumps(label, ensure_ascii=False) + "\n")
            row_file.write(json.dumps({**row, "outcome": outcome}, ensure_ascii=False) + "\n")
    summary = {"status": "COMPLETE_RECORDED_LABELS", "counts": dict(counts),
               "by_collection": {k: dict(v) for k, v in by_collection.items()}, "sources": sources,
               "signals": {"path": str(signals_path), "sha256": sha(signals_path)},
               "identities": {"path": str(identities_path), "sha256": sha(identities_path)},
               "overlap": {"path": str(overlap_path), "sha256": sha(overlap_path)},
               "label_scope": "Original execution order, filtered private case observations; no new execution; UNKNOWN retained"}
    (output / "label_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--overlap", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_rows(args.signals, args.identities, args.overlap, args.out)["counts"]))


if __name__ == "__main__":
    main()
