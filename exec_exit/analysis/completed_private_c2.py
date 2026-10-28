"""Merge completed PRIV-IND labels; retain the original C2 estimator and rows."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from exec_exit.analysis import private_c2
from exec_exit.execute.evaluate_dev import adjudicate

FAILURES = {"WRONG_ANSWER", "TIMEOUT_CPU", "TIMEOUT_WALL", "RUNTIME_ERROR", "OUTPUT_LIMIT",
            "SIGNAL", "SANDBOX_VIOLATION"}
TIMEOUTS = {"TIMEOUT_CPU", "TIMEOUT_WALL"}
STATES = FAILURES | {"PASS", "UNKNOWN", "NOT_EVALUATED_AFTER_FAILURE"}
MODELS = {"qwen": "qwen3_8b", "ds": "dsr1_distill_qwen7b"}
ADAPTER_SHA = "056ca250f030cc7a1d12bd2da350256a92e1ec78feacec1a4944cf69da9d5239"
ADJUDICATE_SHA = "6ac88ce839a46ce0c0d02520d7047cefaadb2bc3c22699779a92e2d0e029cb7d"
FOLDS_SHA = "501b3af52d7c2afc0e872c7bbd0dac41a8fb8856c5e5bdfcff00907c5e56feb6"


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=private_c2.digest(path), bytes=path.stat().st_size)


def checked(ref):
    path = Path(ref["path"])
    actual = identity(path)
    if actual["sha256"] != ref["sha256"] or ("bytes" in ref and actual["bytes"] != ref["bytes"]):
        raise ValueError("Changed source: " + str(path))
    return path


def key(row):
    return row["collection"], row["case_id"], row["sample"], row["subject_id"]


def verified_receipt(job, receipt, binding_sha, overlap):
    for field in ("job_id", "case_id", "collection", "code_sha256", "references"):
        if receipt[field] != job[field]:
            raise ValueError("Job receipt identity differs: " + field)
    if receipt["binding_sha256"] != binding_sha or overlap["collection"] != job["collection"]:
        raise ValueError("Wrong execution binding or collection")
    functional = job["collection"] == "hv6_leetcode"
    indices = overlap["execution_retained_indices"] if functional else list(range(len(overlap["private_keep_indices"])))
    total = overlap["execution_total"] if functional else len(indices)
    if not indices or indices != sorted(set(indices)) or min(indices) < 0 or max(indices) >= total:
        raise ValueError("Invalid retained-private mask")
    attempts = []
    for number, item in enumerate(receipt["attempts"]):
        saved = read(checked(item["receipt"]))
        if {k: v for k, v in item.items() if k != "receipt"} != saved:
            raise ValueError("Embedded attempt differs from immutable receipt")
        if (saved["job_id"], saved["case_id"], saved["attempt"], saved["binding_sha256"]) != (
                job["job_id"], job["case_id"], number, binding_sha):
            raise ValueError("Attempt identity or order differs")
        cached = read(checked(saved["cache"]))
        stored, task = cached["result"], cached["result"]["task"]
        if cached["key"] != saved["cache_key"] or stored["job_id"] != job["job_id"] or stored["code_sha256"] != job["code_sha256"]:
            raise ValueError("Execution cache identity differs")
        expected = dict(task_id=job["case_id"], protocol="priv_ind_completion_preserved_sequence_v1",
                        attempt=number, stop_on_failure="retained_only", retained_indices=indices,
                        binding_sha256=binding_sha)
        if any(task.get(k) != v for k, v in expected.items()):
            raise ValueError("Cached execution mask or protocol differs")
        result = stored["result"]
        cases = {case["test_index"]: case for case in result["cases"]}
        if result["total"] != total or len(cases) != len(result["cases"]) or any(
                type(i) is not int or not 0 <= i < total for i in cases):
            raise ValueError("Execution cases are duplicate, out of range, or wrong size")
        if any(case["status"] not in STATES for case in cases.values()):
            raise ValueError("Unrecognized execution status")
        if functional and any(case.get("retained") != (i in indices) for i, case in cases.items()):
            raise ValueError("LC retained flags differ from the original sequence mask")
        counts = dict(Counter(cases.get(i, {}).get("status", "UNKNOWN") for i in indices))
        passed = False if set(counts) & FAILURES else True if set(counts) == {"PASS"} else None
        retained = dict(all_passed=passed, status={True: "PASS", False: "FAIL", None: "UNKNOWN"}[passed], case_status_counts=counts)
        if retained != saved["retained"]:
            raise ValueError("Retained result disagrees with saved per-case observations")
        attempts.append(dict(public_result=dict(all_passed=True, case_status_counts={}),
                             private_result=retained, whole_program_result=retained["status"]))
    if not attempts:
        raise ValueError("Completed job has no attempts")
    decision = adjudicate(attempts)
    if len(attempts) != decision["required_attempts"]:
        raise ValueError("Partial attempts must not be a completed job receipt")
    failures = sorted({status for attempt in attempts for status, count in attempt["private_result"]["case_status_counts"].items()
                       if count and status in FAILURES})
    expected = dict(outcome=decision["final_outcome"], required_attempts=decision["required_attempts"],
                    completed_attempts=len(attempts), first_timeout=bool(attempts[0]["private_result"]["case_status_counts"].get("TIMEOUT_CPU")),
                    timeout_only=decision["final_outcome"] == "FAIL" and bool(failures) and set(failures) <= TIMEOUTS,
                    failure_types=failures, adjudication=decision)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Job verdict differs from independently reduced retained cases")
    return expected


def merge(rows, labels, jobs, receipts, model):
    original = {key(row): row for row in rows}
    label_map = {key(row): row for row in labels}
    if len(original) != len(rows) or len(label_map) != len(labels) or set(original) != set(label_map):
        raise ValueError("Duplicate or mismatched original rows and labels")
    for k, row in original.items():
        if row["outcome"] != label_map[k]["outcome"] or row["public_status"] != label_map[k]["public_status"]:
            raise ValueError("Original label disagrees with signal row")
    references = {}
    for job in jobs:
        if hashlib.sha256(job["code_text"].encode()).hexdigest() != job["code_sha256"]:
            raise ValueError("Planned program bytes differ")
        for ref in job["references"]:
            if ref["model"] != model:
                continue
            k = job["collection"], job["case_id"], ref["sample"], ref["subject_id"]
            if k in references or k not in original:
                raise ValueError("Duplicate or unknown job reference")
            row, label = original[k], label_map[k]
            if row["outcome"] != "UNKNOWN" or row["code_sha256"] != job["code_sha256"]:
                raise ValueError("Job references an old known label or different code")
            if any(ref[name] != label[name] for name in ("evaluation_path", "evaluation_line")):
                raise ValueError("Original evaluation binding differs")
            references[k] = job["job_id"]
    if set(references) != {k for k, row in original.items() if row["outcome"] == "UNKNOWN"}:
        raise ValueError("Planned references do not exactly cover original UNKNOWN labels")
    main, sensitivity, changes = [], [], []
    for row in rows:
        k = key(row)
        current, alternate = dict(row), dict(row)
        if k in references:
            job_id = references[k]
            result = receipts.get(job_id)
            if result is not None:
                current["outcome"] = result["outcome"]
                alternate["outcome"] = "UNKNOWN" if result["timeout_only"] else result["outcome"]
            changes.append(dict(collection=row["collection"], case_id=row["case_id"], sample=row["sample"],
                                subject_id=row["subject_id"], job_id=job_id, old_outcome="UNKNOWN",
                                outcome=current["outcome"], sensitivity_outcome=alternate["outcome"],
                                reason="COMPLETED_RECEIPT" if result is not None else "MISSING_COMPLETION_RECEIPT",
                                timeout_only=result["timeout_only"] if result is not None else False))
        main.append(current)
        sensitivity.append(alternate)
    pub = [row for row in rows if row["public_status"] == "PASS"]
    same_pub = all([row for row in variant if row["public_status"] == "PASS"] == pub for variant in (main, sensitivity))
    summary = dict(model=MODELS[model], old_known_preserved=True, public_pass_rows_unchanged=same_pub,
                   public_pass_drafts=len(pub), public_pass_questions=len({row["case_id"] for row in pub}),
                   planned_unknown_rows=len(references), missing_receipt_rows=sum(r["reason"] == "MISSING_COMPLETION_RECEIPT" for r in changes),
                   newly_timeout_only_fail_rows=sum(r["timeout_only"] for r in changes), counts={})
    for name, variant in (("old", rows), ("main", main), ("timeout_sensitivity", sensitivity)):
        summary["counts"][name] = {population: dict(Counter(row["outcome"] for row in variant if population == "pooled" or row["collection"] == population))
                                   for population in ("pooled", *private_c2.COLLECTIONS)}
    summary["status"] = "COMPLETE_WITH_UNKNOWN" if any(row["outcome"] == "UNKNOWN" for row in main) else "COMPLETE"
    return main, sensitivity, changes, summary


def write_lines(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rows", "labels", "jobs", "job-dir", "binding", "folds", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--statistics", action="store_true", help="Run the unchanged private_c2 estimator for both variants")
    args = parser.parse_args()
    if private_c2.digest(private_c2.__file__) != ADAPTER_SHA or private_c2.digest(private_c2.grouped_signals.__file__) != private_c2.SOURCE_SHA:
        raise ValueError("Original C2 implementation changed")
    if private_c2.digest(Path(__file__).parents[1] / "execute/evaluate_dev.py") != ADJUDICATE_SHA:
        raise ValueError("Original majority adjudicator changed")
    if private_c2.digest(args.folds) != FOLDS_SHA:
        raise ValueError("Original question folds changed")
    binding = read(args.binding)
    config = read(checked(binding["config"]))
    bound = {str(Path(ref["path"]).resolve()): ref for ref in config["source_identities"]}
    for path in (args.rows, args.labels, args.jobs):
        checked(bound[str(path.resolve())])
    if identity(args.jobs) != binding["jobs"]:
        raise ValueError("Jobs do not match the immutable execution binding")
    overlap = {row["case_id"]: row for row in lines(checked(binding["overlap"]))}
    rows, labels, jobs = lines(args.rows), lines(args.labels), lines(args.jobs)
    folds = read(args.folds)
    private_c2.validate(rows, folds)
    binding_sha = private_c2.digest(args.binding)
    receipts, sources = {}, []
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate planned job ID")
    for job in jobs:
        if not any(ref["model"] == args.model for ref in job["references"]):
            continue
        path = args.job_dir / (job["job_id"] + ".json")
        if path.exists():
            receipts[job["job_id"]] = verified_receipt(job, read(path), binding_sha, overlap[job["case_id"]])
            sources.append(identity(path))
    primary, sensitive, changes, summary = merge(rows, labels, jobs, receipts, args.model)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, variant in (("main", primary), ("timeout_sensitivity", sensitive)):
        folder = args.out / name
        folder.mkdir()
        write_lines(folder / "rows.jsonl", variant)
        if args.statistics:
            stats, predictions, models = private_c2.analyse(variant, folds, MODELS[args.model])
            (folder / "summary.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
            write_lines(folder / "oof_predictions.jsonl", predictions)
            write_lines(folder / "fold_models.jsonl", models)
    write_lines(args.out / "label_changes.jsonl", changes)
    summary["sources"] = {name: identity(getattr(args, name)) for name in ("rows", "labels", "jobs", "binding", "folds")}
    summary["job_receipts"] = sources
    summary["implementation"] = identity(__file__)
    summary["statistics"] = dict(executed=args.statistics, adapter_sha256=ADAPTER_SHA,
                                estimator_sha256=private_c2.SOURCE_SHA, seed=private_c2.SEED, replicates=private_c2.REPLICATES)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
