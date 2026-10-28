"""Offline Dev36 evaluation with exact-byte cache and isolated execution.

Code bytes are hashed as received. TEXT/AST representations must never enter
this cache key. Callers supply frozen task, test, limit, and runtime identities.
"""
from __future__ import annotations

import hashlib
import argparse
from collections import Counter
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import time
import resource
import subprocess

from exec_exit.data.load_lcb import load_legacy_dev36, load_offline_dev36_tests
from exec_exit.execute.extract import CodeTokenMapper, extract_record, RULE
from exec_exit.execute.sandbox import (execution_identity, frozen_limits, make_sandbox, sha, evaluate_suite)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def execution_cache_key(
    code: bytes, *, task: dict, tests: list[dict], resource_limits: dict,
    interpreter: dict, comparator: dict, executor: dict, machine: dict,
) -> str:
    """Digest exact code bytes plus every dependency that can change execution.

    Test list order is significant (including for fail-fast). Mapping key order
    is not. Runtime identities should include interpreter binary/version and
    executor/comparator source identities; no implicit host-dependent defaults.
    """
    if not isinstance(code, bytes):
        raise TypeError("Pass original code bytes; implicit encoding/normalization is forbidden")
    for label, value in {"task": task, "resource_limits": resource_limits,
                         "interpreter": interpreter, "comparator": comparator,
                         "executor": executor, "machine": machine}.items():
        if not isinstance(value, dict) or not value:
            raise ValueError("An explicit nonempty identity is required for " + label)
    if not isinstance(tests, list):
        raise TypeError("tests must be an ordered list containing the actual test content")
    description = {"schema": "exec_exit.execution.v2", "task": task, "tests": tests,
                   "resource_limits": resource_limits, "interpreter": interpreter,
                   "comparator": comparator, "executor": executor, "machine": machine}
    header = _canonical(description)
    digest = hashlib.sha256(b"exec_exit.execution.v2\0")
    for part in (code, header):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


class ExecutionCache:
    """Atomic, immutable records keyed by execution_cache_key; cache misses None.

    A conflicting value is an error rather than silent overwrite. This class
    stores supplied observations; it does not convert missing results to PASS.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Execution cache key must be a SHA-256 hex digest")
        return self.root / key[:2] / (key + ".json")

    def get(self, key: str):
        path = self._path(key)
        if not path.exists():
            return None
        record = json.loads(path.read_bytes())
        if record.get("key") != key:
            raise ValueError("Execution cache record key mismatch")
        return record["result"]

    def put(self, key: str, result: dict):
        if not isinstance(result, dict):
            raise TypeError("Execution result must be an explicit observation mapping")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical({"key": key, "result": result}) + b"\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError("Conflicting result for existing execution cache key")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def run_tests(code: bytes, *, task_id, tests, visibility, limits, sandbox, cache):
    """Evaluate one ordered suite, fail-fast for both PUB and Dev private tests."""
    if visibility not in ("public", "dev_private"):
        raise ValueError("Only public and already-seen Dev private tests are admitted")
    identity = execution_identity(sandbox)
    descriptor = dict(task_id=task_id, protocol="stdio_exact_bytes_v1", visibility=visibility,
                      stop_on_failure=True)
    key = execution_cache_key(code, task=descriptor, tests=tests, resource_limits=limits, **identity)
    started = time.perf_counter_ns()
    stored = cache.get(key)
    reused = stored is not None
    if not reused:
        decoded = code.decode("utf-8")
        if decoded.encode("utf-8") != code:
            raise ValueError("Source bytes did not round-trip losslessly")
        result = evaluate_suite(sandbox, code, tests, limits, visibility)
        ended = time.perf_counter_ns()
        stored = dict(result=result, task=descriptor, code_sha256=hashlib.sha256(code).hexdigest(),
                      observed_offline_started_ns=started, observed_offline_finished_ns=ended,
                      online_visible_at="UNKNOWN_NOT_EXECUTED_ONLINE", identity=identity)
        cache.put(key, stored)
    observed = stored["result"]
    return dict(status=observed["status"], all_passed=observed["all_passed"],
                total=observed["total"], evaluated=observed["evaluated"],
                not_evaluated_after_failure=observed["not_evaluated_after_failure"],
                case_status_counts=dict(Counter(row["status"] for row in observed["cases"])),
                original_execution_wall_seconds=observed["wall_seconds"],
                original_execution_cpu_seconds=observed["cpu_seconds"],
                child_cpu_seconds=observed["child_cpu_seconds"],
                child_wall_seconds=observed["child_wall_seconds"],
                cache_key=key, cache_reused=reused,
                lookup_and_current_execution_wall_seconds=(time.perf_counter_ns()-started)/1e9,
                observed_offline_started_ns=stored["observed_offline_started_ns"],
                observed_offline_finished_ns=stored["observed_offline_finished_ns"],
                online_visible_at="UNKNOWN_NOT_EXECUTED_ONLINE")


_registry = _sandbox = _cache = _limits = None


def _initialize_worker(runtime_root, limits):
    global _sandbox, _cache, _limits
    root = Path(runtime_root)
    _sandbox = make_sandbox(root/"workers"/str(os.getpid()), template_dir=root/"template")
    _cache, _limits = ExecutionCache(root.parent/"execution_cache"), limits


def _execute_job(job):
    task_id, code = job
    public = run_tests(code, task_id=task_id, tests=_registry[task_id]["public"], visibility="public",
                       limits=_limits, sandbox=_sandbox, cache=_cache)
    private = run_tests(code, task_id=task_id, tests=_registry[task_id]["private"], visibility="dev_private",
                        limits=_limits, sandbox=_sandbox, cache=_cache)
    return task_id, hashlib.sha256(code).hexdigest(), public, private


def _stub(status, total):
    return dict(status=status, all_passed=None, total=total, evaluated=0,
                not_evaluated_after_failure=0, cache_key=None, cache_reused=False,
                original_execution_wall_seconds=None, original_execution_cpu_seconds=None,
                online_visible_at="UNKNOWN_NOT_EXECUTED_ONLINE")


def _write(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _write_lines(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed raw trajectories.jsonl or probes.jsonl")
    parser.add_argument("--source-kind", choices=("N", "probe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 20:
        raise ValueError("Use 1–20 CPU workers; default frozen plan is 20")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("CPU evaluation must wait for GPU measurement to drain")
    output, source = args.output.resolve(), args.input.resolve()
    source_summary_path = source.parent.parent/"summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    if (source_summary.get("completed") != source_summary.get("planned")
            or not isinstance(source_summary.get("planned"), int)
            or source_summary["planned"] <= 0):
        raise RuntimeError("Evaluate only a fully finalized source population, never a partial input file")
    if output.exists():
        raise FileExistsError("Use a new output run; do not overwrite an earlier evaluation")
    (output/"raw").mkdir(parents=True)
    (output/"runtime/workers").mkdir(parents=True)
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model/"tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    mapper = CodeTokenMapper(args.model/"tokenizer.json") if args.source_kind == "probe" else None
    tasks = load_legacy_dev36()
    private = load_offline_dev36_tests()
    global _registry
    _registry = {task["task_id"]: dict(public=[{k: row[k] for k in ("input", "output")} for row in task["public_test_cases"]],
                 private=[{k: row[k] for k in ("input", "output")} for row in private[task["task_id"]]]) for task in tasks}
    source_rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    if len(source_rows) != source_summary["planned"] or any(row.get("status") != "COMPLETE" for row in source_rows):
        raise ValueError("Source raw rows do not match the complete planned population")
    if any(row["case_id"] not in _registry for row in source_rows):
        raise ValueError("Evaluation source contains a task outside the original Dev36")
    if args.source_kind == "N" and [row["case_id"] for row in source_rows] != [task["task_id"] for task in tasks]:
        raise ValueError("Native source must preserve the exact original 36-question order")
    limits = frozen_limits()
    repo = Path(__file__).resolve().parents[2]
    (output/"commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()+"\n")
    sources = [Path(__file__), Path(__file__).with_name("sandbox.py"), Path(__file__).with_name("extract.py")]
    (output/"source").mkdir()
    for path in sources:
        (output/"source"/path.name).write_bytes(path.read_bytes())
    config = dict(schema="exec_exit.offline_dev_evaluation.v1", input=str(source), input_sha256=sha(source),
                  source_summary_sha256=sha(source_summary_path), source_process_status=source_summary.get("status"),
                  source_kind=args.source_kind, workers=args.workers, process_start_method="fork",
                  public_fail_fast=True, dev_private_fail_fast=True,
                  model=str(args.model), tokenizer_sha256=sha(args.model/"tokenizer.json"), limits=limits,
                  extraction_rule=RULE, source_files={str(path): sha(path) for path in sources},
                  heldout_read=False, dev_private_tests_read=True, gpu_used=False,
                  visibility="Only signals_public.jsonl is eligible for stopping-policy inputs; Dev labels are offline only")
    _write(output/"config.yaml", config)
    evaluations, jobs = [], {}
    for row in source_rows:
        case = row["case_id"]
        subject = "N:"+case if args.source_kind == "N" else row["probe_id"]
        extraction = extract_record(row, args.source_kind, tokenizer)
        if mapper is not None:
            extraction["code_token_mapping"] = mapper.probe_code_tokens(row, extraction)
        item = dict(subject_id=subject, case_id=case, source_kind=args.source_kind,
                    position=None if args.source_kind == "N" else row["position_generated_tokens"],
                    extraction=extraction, code_sha256=extraction["code_sha256"],
                    generation_valid=extraction["generation_valid"],
                    public_result=_stub("NOT_EVALUATED_INVALID_DELIVERY", len(_registry[case]["public"])),
                    private_result=_stub("NOT_EVALUATED_INVALID_DELIVERY", len(_registry[case]["private"])))
        if extraction["deliverable"]:
            code = extraction["code_text"].encode("utf-8")
            jobs[(case, extraction["code_sha256"])] = (case, code)
        evaluations.append(item)
    if len({row["subject_id"] for row in evaluations}) != len(evaluations):
        raise ValueError("Duplicate source subjects")
    _write(output/"summary.json", dict(status="RUNNING", planned=len(evaluations), unique_programs=len(jobs), completed=0))
    _write_lines(output/"raw/extractions.jsonl", evaluations)
    result_map, error = {}, None
    try:
        template = make_sandbox(output/"runtime/template")
        config["execution_identity"] = execution_identity(template)
        _write(output/"config.yaml", config)
        # CPU-only parent, no CUDA imports. Fork shares the immutable Dev suites
        # rather than duplicating their large strings in each worker initializer.
        context = multiprocessing.get_context("fork")
        with context.Pool(args.workers, initializer=_initialize_worker,
                          initargs=(str(output/"runtime"), limits)) as pool:
            for case, code_hash, public, private_result in pool.imap_unordered(_execute_job, jobs.values(), chunksize=1):
                result_map[(case, code_hash)] = (public, private_result)
                _write(output/"progress.json", dict(completed_unique_programs=len(result_map), planned_unique_programs=len(jobs)))
    except BaseException as failure:
        error = repr(failure)
    for row in evaluations:
        result = result_map.get((row["case_id"], row["code_sha256"]))
        if result is not None:
            row["public_result"], row["private_result"] = result
        elif row["extraction"]["deliverable"]:
            row["public_result"]["status"] = row["private_result"]["status"] = "UNKNOWN_EXECUTION_NOT_COMPLETED"
        verdicts = [row["public_result"]["all_passed"], row["private_result"]["all_passed"]]
        passed = (False if not row["extraction"]["deliverable"] or False in verdicts
                  else True if all(v is True for v in verdicts) else None)
        row["dev_result"] = dict(status="PASS" if passed is True else "FAIL" if passed is False else "UNKNOWN",
                                 all_passed=passed, definition="Deliverable program passes PUB and Dev private tests")
    _write_lines(output/"raw/evaluations.jsonl", evaluations)
    public_rows, labels = [], []
    for row in evaluations:
        public_rows.append({key: row[key] for key in ("subject_id", "case_id", "source_kind", "position", "generation_valid",
                                                      "extraction", "code_sha256", "public_result")})
        labels.append({key: row[key] for key in ("subject_id", "case_id", "dev_result", "private_result")})
    _write_lines(output/"raw/signals_public.jsonl", public_rows)
    _write_lines(output/"raw/dev_labels.jsonl", labels)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    unique_suites = {suite["cache_key"]: suite for suites in result_map.values() for suite in suites}
    new_suites = [suite for suite in unique_suites.values() if not suite["cache_reused"]]
    summary = dict(status="COMPLETE" if error is None and len(result_map) == len(jobs) else "FAILED", error=error,
                   planned=len(evaluations), unique_programs=len(jobs), completed_unique_programs=len(result_map),
                   dev_outcomes=dict(Counter(row["dev_result"]["status"] for row in evaluations)),
                   public_outcomes=dict(Counter(row["public_result"]["status"] for row in evaluations)),
                   invalid_deliveries=sum(not row["extraction"]["deliverable"] for row in evaluations),
                   extraction_status_counts=dict(Counter(row["extraction"]["status"] for row in evaluations)),
                   parsed_programs=sum(row["extraction"]["parsed"] for row in evaluations),
                   unique_suites_referenced=len(unique_suites), suites_actually_executed=len(new_suites),
                   unique_execution_cpu_seconds=sum(row["original_execution_cpu_seconds"] for row in new_suites),
                   unique_execution_child_cpu_seconds=sum(row["child_cpu_seconds"] for row in new_suites),
                   unique_execution_child_wall_seconds=sum(row["child_wall_seconds"] for row in new_suites),
                   unique_execution_suite_wall_seconds=sum(row["original_execution_wall_seconds"] for row in new_suites),
                   wall_seconds=time.perf_counter()-started, parent_cpu_user_seconds=usage.ru_utime-usage0.ru_utime,
                   parent_cpu_system_seconds=usage.ru_stime-usage0.ru_stime, gpu_used=False,
                   correctness_scope="Observed acceptance under frozen resources; not formal correctness or native official-runner equivalence",
                   online_visibility="UNKNOWN; these are offline executions")
    _write(output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
