"""Machine-bound LeetCode development evaluation and predeclared CPU-timeout replication.

First PUB signals remain first observations. Full-program majority labels are
offline only. Every timeout-triggered program receives exactly two additional
whole-program evaluations (PUB and Dev-private suites, each fail-fast).
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

from exec_exit.data.leetcode import load_runtime
DATASET=Path(__file__).resolve().parents[2] / "runs/leetcode_dev62_projection"
from exec_exit.execute.extract import CodeTokenMapper, extract_record, RULE
from exec_exit.execute.run_tests import ExecutionCache, execution_cache_key, _stub, _write, _write_lines
from exec_exit.execute.sandbox import frozen_limits, LEGACY_EXECUTOR
from exec_exit.execute.leetcode_call import execution_identity, make_sandbox, evaluate_suite
from exec_exit.data.load_lcb import official_evaluator_path
from exec_exit.gen.host_environment import collect_host_environment

ROOT = Path(__file__).resolve().parents[2]
_registry = _identity = _sandbox = _cache = _limits = _output = None
_functions = {}
_capacity_violations = {}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def limits_for_run(input_capacity_mib=8):
    """Copy historical limits; an explicit run option changes only the input cap."""
    if input_capacity_mib not in (8, 64):
        raise ValueError("Only explicit 8 or 64 MiB input capacities are supported")
    limits = dict(frozen_limits())
    limits.update(input_bytes=input_capacity_mib * 1024**2)
    return limits


def capacity_preflight(registry, selected_ids, limits):
    """Check every selected PUB/private test before any candidate is executed.

    Expected-output length is a capacity check, never a candidate correctness
    signal. Report only lengths and indices, not private test contents.
    """
    violations, test_sizes, counts = [], [], Counter()
    maximum = {"input_bytes": 0, "expected_output_bytes": 0}
    for task_id in selected_ids:
        for visibility in ("public", "dev_private"):
            for index, test in enumerate(registry[task_id][visibility]):
                counts[visibility] += 1
                sizes = {"input_bytes": len(test["input"].encode("utf-8")),
                         "expected_output_bytes": len(test["output"].encode("utf-8"))}
                test_sizes.append(dict(task_id=task_id, visibility=visibility, test_index=index, **sizes))
                for field, size in sizes.items():
                    maximum[field] = max(maximum[field], size)
                exceeded = [field for field, cap in (("input_bytes", "input_bytes"),
                                                     ("expected_output_bytes", "output_bytes"))
                            if sizes[field] > limits[cap]]
                if exceeded:
                    violations.append(dict(task_id=task_id, visibility=visibility, test_index=index,
                                           **sizes, exceeded_fields=exceeded))
    return dict(schema="exec_exit.test_io_capacity.v1", status="EXCEEDED" if violations else "PASS",
                selected_tasks=len(selected_ids), tests_checked=dict(counts), maxima=maximum,
                test_sizes=test_sizes,
                limits={k: limits[k] for k in ("input_bytes", "output_bytes")}, violations=violations,
                affected_tasks=sorted({v["task_id"] for v in violations}),
                affected_suites=len({(v["task_id"], v["visibility"]) for v in violations}),
                interpretation="Host execution capacity only; no test is declared invalid or removed")


def capacity_unknown_suite(tests, violations, attempt_index):
    """No execution/cache observation is fabricated for an unsupported suite."""
    return _stub("UNKNOWN_IO_CAPACITY", len(tests)) | dict(
        case_status_counts={}, case_status_sequence=[], attempt_index=attempt_index,
        not_evaluated_capacity=len(tests), capacity_violations=violations,
        execution_started=False, reason="Whole suite unavailable under chosen I/O capacity; no tests skipped to score it")


def blocking_capacity_violations(preflight, policy):
    if policy == "execute-output-limit":
        return [v for v in preflight["violations"] if "input_bytes" in v["exceeded_fields"]]
    if policy == "unknown":
        return []
    if policy == "error":
        return preflight["violations"]
    raise ValueError("Unknown capacity policy")


def program_result(public, private):
    values = [public.get("all_passed"), private.get("all_passed")]
    return "FAIL" if False in values else "PASS" if all(v is True for v in values) else "UNKNOWN"


def cpu_timeout(suite):
    return suite.get("case_status_counts", {}).get("TIMEOUT_CPU", 0) > 0


def adjudicate(attempts):
    if not attempts:
        return {"final_outcome": "UNKNOWN", "unstable": False, "required_attempts": 1, "passes": 0}
    trigger = any(cpu_timeout(attempts[0][name]) for name in ("public_result", "private_result"))
    required = 3 if trigger else 1
    if len(attempts) > required:
        raise ValueError("More attempts than the predeclared timeout rule permits")
    outcomes = [a["whole_program_result"] for a in attempts]
    signatures = [json.dumps([a[name].get("case_status_counts", {}) for name in ("public_result", "private_result")], sort_keys=True) for a in attempts]
    passfail_unstable = len(set(outcomes)) > 1
    unstable = passfail_unstable or len(set(signatures)) > 1
    if len(attempts) != required or "UNKNOWN" in outcomes:
        result = "UNKNOWN"
    elif trigger:
        result = "PASS" if outcomes.count("PASS") >= 2 else "FAIL"
    else:
        result = outcomes[0]
    return {"final_outcome": result, "unstable": unstable, "triggered": trigger,
            "pass_fail_unstable": passfail_unstable, "status_type_unstable": len(set(signatures)) > 1,
            "required_attempts": required, "completed_attempts": len(attempts), "passes": outcomes.count("PASS"),
            "rule": "First full program with any CPU timeout triggers exactly two more full PUB+Dev attempts; >=2 full passes required",
            "unstable_definition": "Whole-program PASS/FAIL/UNKNOWN or observed per-suite status-count signatures differ; durations excluded"}


def extract_for_evaluation(row, kind, tokenizer):
    result = extract_record(row, kind, tokenizer)
    rejected = kind == "probe" and (row.get("protocol_rejected") is True or row.get("call", {}).get("protocol_rejected") is True)
    result["protocol_rejected"] = rejected
    if rejected:
        result.update(generation_valid=False, deliverable=False, generation_reason="v1_comment_rejection_even_if_native_finished")
    return result


def _initialize_worker(output, cache_root, limits, identity):
    global _sandbox, _cache, _limits, _output, _identity
    _output, _limits, _identity = Path(output), limits, identity
    _sandbox = make_sandbox(_output / "runtime/workers" / str(os.getpid()), template=_output / "runtime/template")
    _cache = ExecutionCache(cache_root)


def observe_suite(code, task_id, visibility, attempt_index):
    tests = _registry[task_id][visibility]
    violations = _capacity_violations.get((task_id, visibility))
    if violations:
        return capacity_unknown_suite(tests, violations, attempt_index)
    descriptor = dict(task_id=task_id, protocol="official_call_based_same_solution_instance_v1", func_name=_functions[task_id], visibility=visibility, stop_on_failure=True, suite_semantics="PUB-only signal" if visibility=="public" else "PUB then private in same Solution instance")
    if attempt_index:
        descriptor |= dict(repetition_protocol="whole_program_cpu_timeout_two_extra_v1", repetition_index=attempt_index)
    key = execution_cache_key(code, task=descriptor, tests=tests, resource_limits=_limits, **_identity)
    started = time.perf_counter_ns()
    stored = _cache.get(key)
    reused = stored is not None
    if stored is None:
        intent_path = _cache.root / "execution_intents" / (key + ".json")
        intent_path.parent.mkdir(exist_ok=True, parents=True)
        intent = {"key": key, "run_dir": str(_output), "task_id": task_id, "visibility": visibility,
                  "attempt_index": attempt_index, "pid": os.getpid(), "started_ns": started,
                  "started_utc": datetime.now(timezone.utc).isoformat(), "status": "STARTED"}
        # Never silently retry a previously started suite lacking a completed
        # immutable observation. A crashed child leaves a decisive audit record.
        with intent_path.open("x") as f:
            json.dump(intent, f)
            f.flush()
            os.fsync(f.fileno())
        result = evaluate_suite(_sandbox, code, tests, _limits, "public" if visibility=="public" else "dev_full", _functions[task_id])
        stored = {"result": result, "task": descriptor, "code_sha256": hashlib.sha256(code).hexdigest(),
                  "observed_offline_started_ns": started, "observed_offline_finished_ns": time.perf_counter_ns(),
                  "online_visible_at": "UNKNOWN_NOT_EXECUTED_ONLINE", "identity": _identity,
                  "observation_run_dir": str(_output), "attempt_index": attempt_index}
        _cache.put(key, stored)
        _write(intent_path, intent | {"status": "COMPLETE", "finished_ns": stored["observed_offline_finished_ns"]})
    result = stored["result"]
    return dict(status=result["status"], all_passed=result["all_passed"], total=result["total"], evaluated=result["evaluated"],
                not_evaluated_after_failure=result["not_evaluated_after_failure"],
                case_status_counts=dict(Counter(r["status"] for r in result["cases"])),
                case_status_sequence=[r["status"] for r in result["cases"]],
                original_execution_wall_seconds=result["wall_seconds"], original_execution_cpu_seconds=result["cpu_seconds"],
                child_cpu_seconds=result["child_cpu_seconds"], child_wall_seconds=result["child_wall_seconds"],
                cache_key=key, cache_reused=reused, attempt_index=attempt_index,
                observation_run_dir=stored.get("observation_run_dir", "UNKNOWN_OLD_CACHE"),
                observed_offline_started_ns=stored["observed_offline_started_ns"],
                observed_offline_finished_ns=stored["observed_offline_finished_ns"],
                lookup_and_current_execution_wall_seconds=(time.perf_counter_ns()-started)/1e9,
                online_visible_at="UNKNOWN_NOT_EXECUTED_ONLINE")


def _execute_job(job):
    task_id, code = job
    code_hash = hashlib.sha256(code).hexdigest()
    job_id = task_id.replace(":", "_") + "." + code_hash
    checkpoint = _output / "checkpoints" / (job_id + ".json")
    attempts = []
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        if prior["task_id"] != task_id or prior["code_sha256"] != code_hash:
            raise ValueError("Program checkpoint identity mismatch")
        attempts = prior["attempts"]
        if prior["status"] == "COMPLETE":
            return prior
    required = 3 if attempts and any(cpu_timeout(attempts[0][k]) for k in ("public_result", "private_result")) else 1
    while len(attempts) < required:
        index = len(attempts)
        public = observe_suite(code, task_id, "public", index)
        private = observe_suite(code, task_id, "dev_private", index)
        attempts.append({"attempt_index": index, "public_result": public, "private_result": private,
                         "whole_program_result": program_result(public, private)})
        if index == 0 and (cpu_timeout(public) or cpu_timeout(private)):
            required = 3
        partial = dict(task_id=task_id, code_sha256=code_hash, job_id=job_id, attempts=attempts,
                       status="COMPLETE" if len(attempts) == required else "REPEATS_PENDING")
        _write(checkpoint, partial | {"timeout_robustness": adjudicate(attempts)})
    return json.loads(checkpoint.read_text())


def gpu_idle():
    result = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if result:
        raise RuntimeError("CPU performance evaluation waits for all GPU computation to drain")


def source_population(path, kind, tasks, cohort):
    summary = json.loads((path.parent.parent / "summary.json").read_text())
    if summary.get("status") != "COMPLETE" or summary.get("completed") != summary.get("planned"):
        raise ValueError("Require a complete source run")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if len(rows) != summary["planned"] or any(r.get("status") != "COMPLETE" for r in rows):
        raise ValueError("Source population differs from finalized summary")
    selected = [t for t in tasks if cohort == "all" or t["source_cohort"] == cohort]
    valid_ids = {t["task_id"] for t in selected}
    if any(r["case_id"] not in valid_ids for r in rows):
        raise ValueError("Source row outside explicit evaluation cohort")
    if kind == "N" and [r["case_id"] for r in rows] != [t["task_id"] for t in selected]:
        raise ValueError("Native source must exactly cover the selected cohort")
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-kind", choices=("N", "probe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-projection", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cohort", choices=("all",), default="all")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--input-capacity-mib", type=int, choices=(64,), default=64,
                        help="Explicit input capacity only; output and all other frozen limits stay unchanged")
    parser.add_argument("--capacity-policy", choices=("error", "unknown", "execute-output-limit"), default="error",
                        help="error: stop on preflight overflow; unknown: unsupported suites UNKNOWN; execute-output-limit: input overflow blocks, expected output diagnostic only, actual sandbox output limit applies")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 20:
        raise ValueError("Use the planned 1..20 CPU workers")
    gpu_idle()
    output, source = args.output.resolve(), args.input.resolve()
    started = time.perf_counter()
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    tasks = load_runtime(args.dataset / "data/runtime_tasks.json")
    if any(t["split"]!="development" for t in tasks): raise ValueError("No heldout evaluation before freeze")
    private_path=args.private_projection / "data/tests.json"
    private_config=json.loads((args.private_projection / "config.yaml").read_text())
    if private_config["tests_sha256"]!=sha(private_path) or private_config["runtime_tasks_sha256"]!=sha(args.dataset / "data/runtime_tasks.json"):raise ValueError("Development tests identity mismatch")
    private_data=json.loads(private_path.read_text())
    if private_data["split"]!="development":raise ValueError("Only development labels")
    private=private_data["tasks"]
    if set(private)!={t["task_id"] for t in tasks}:raise ValueError("Complete development test population required")
    global _registry,_functions
    _functions={t["task_id"]:t["func_name"] for t in tasks}
    _registry = {t["task_id"]: {"public": private[t["task_id"]]["public"], "dev_private": private[t["task_id"]]["public"]+private[t["task_id"]]["dev_private"]} for t in tasks}
    rows, source_summary = source_population(source, args.source_kind, tasks, args.cohort)
    if output.exists() and not args.resume:
        raise FileExistsError("New evaluation directory required; explicit --resume only for this exact run")
    output.mkdir(exist_ok=args.resume, parents=True)
    for name in ("raw", "source", "checkpoints", "runtime/workers"):
        (output / name).mkdir(parents=True, exist_ok=args.resume)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    mapper = CodeTokenMapper(args.model / "tokenizer.json") if args.source_kind == "probe" else None
    template = make_sandbox(output / "runtime/template")
    identity = execution_identity(template)
    generation_config = json.loads((source.parent.parent / "config.yaml").read_text())
    generation_host = generation_config.get("host_environment", {})
    current_host = collect_host_environment()
    same_generation_host = (generation_host.get("boot_id") == current_host.get("boot_id")
                            and generation_host.get("hostname") == current_host.get("hostname")
                            and generation_host.get("cpu_quota") == current_host.get("cpu_quota"))
    if not same_generation_host:
        raise ValueError("Generation/evaluation host identity differs or is missing; migration relabeling requires an explicit separate run")
    cache_root = args.cache_root.resolve()
    if "v2" not in cache_root.name:
        raise ValueError("Use an explicitly named v2 machine-aware cache namespace")
    limits = limits_for_run(args.input_capacity_mib)
    selected_ids = [t["task_id"] for t in tasks if args.cohort == "all" or t["source_cohort"] == args.cohort]
    preflight = capacity_preflight(_registry, selected_ids, limits)
    preflight_path = output / "raw/capacity_preflight.json"
    if args.resume:
        if json.loads(preflight_path.read_text()) != preflight:
            raise ValueError("Resume test capacity preflight changed")
    else:
        _write(preflight_path, preflight)
    global _capacity_violations
    _capacity_violations = {}
    if args.capacity_policy == "unknown":
        for violation in preflight["violations"]:
            _capacity_violations.setdefault((violation["task_id"], violation["visibility"]), []).append(violation)
    sources = [Path(__file__), ROOT / "exec_exit/data/leetcode.py", ROOT / "exec_exit/execute/leetcode_call.py", official_evaluator_path(), ROOT / "exec_exit/execute/run_tests.py", LEGACY_EXECUTOR,
               ROOT / "exec_exit/execute/extract.py", ROOT / "exec_exit/execute/sandbox.py", ROOT / "exec_exit/execute/machine_identity.py"]
    config = dict(schema="exec_exit.offline_leetcode_evaluation.v1", input=str(source), input_sha256=sha(source),
                  source_summary_sha256=sha(source.parent.parent / "summary.json"), source_kind=args.source_kind,
                  source_generation_config_sha256=sha(source.parent.parent / "config.yaml"), cohort=args.cohort,
                  private_projection=str(args.private_projection.resolve()), private_projection_sha256=sha(args.private_projection / "data/tests.json"),
                  dataset=str(args.dataset.resolve()), public_projection_sha256=sha(args.dataset / "data/runtime_tasks.json"),
                  workers=args.workers, cache_root=str(cache_root), execution_identity=identity,
                  source_generation_host=generation_host, source_generation_gpu_uuid=generation_config.get("expected_gpu_uuid"),
                  generation_and_evaluation_host_match=same_generation_host,
                  tokenizer_sha256=sha(args.model / "tokenizer.json"), limits=limits, extraction_rule=RULE,
                  input_capacity_mib=args.input_capacity_mib, output_capacity_mib=limits["output_bytes"] // 1024**2,
                  capacity_policy=args.capacity_policy,
                  capacity_preflight_sha256=sha(output / "raw/capacity_preflight.json"),
                  capacity_basis="input 64 MiB; CPU, memory, output and other limits unchanged; uniformly re-evaluate all methods",
                  output_scoring_semantics="Expected-output size is diagnostic only under execute-output-limit; actually execute native sandbox; observed OUTPUT_LIMIT is a resource failure, not an algorithmic wrong-answer diagnosis",
                  original_output_limit_evidence={"path": str(LEGACY_EXECUTOR), "sha256": sha(LEGACY_EXECUTOR),
                                                 "lines": "305-307", "rule": "stdout+stderr byte accumulation exceeding output_bytes sets OUTPUT_LIMIT; existing sandbox treats any non-PASS as failed execution"},
                  source_files={str(p): sha(p) for p in sources}, public_fail_fast=True, dev_private_fail_fast=True,
                  timeout_rule="Any first-program CPU timeout triggers exactly 2 extra complete PUB+Dev attempts; >=2 passes wins",
                  functional_semantics="Independent PUB-only signal; full label runs PUB then private in one Solution instance; official comparator unchanged; per-call CPU60/wall72/memory1GiB adaptation; outer aggregate guard separate",
                  signal_separation="signals_public contains first PUB only; final_outcome and retries offline only",
                  gpu_used=False, heldout_or_reserved_read=False, dev_private_tests_read=True)
    if args.resume:
        old = json.loads((output / "config.yaml").read_text())
        if {k: v for k, v in old.items() if k != "host_environment"} != config:
            raise ValueError("Resume source, environment or evaluation protocol changed")
    else:
        config["host_environment"] = current_host
        _write(output / "config.yaml", config)
        (output / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
        for p in sources:
            shutil.copyfile(p, output / "source" / p.name)
    blocked_violations = blocking_capacity_violations(preflight, args.capacity_policy)
    if blocked_violations:
        blocked = dict(status="BLOCKED_IO_CAPACITY", planned=len(rows), unique_programs=None,
                       completed_unique_programs=0, suites_actually_executed=0, gpu_used=False,
                       capacity_preflight=preflight, error="Selected test I/O exceeds this run's limits; no candidate executed")
        _write(output / "manifest.json", blocked)
        _write(output / "summary.json", blocked)
        print(json.dumps(blocked, indent=2))
        return 2
    evaluations, jobs = [], {}
    for row in rows:
        task_id = row["case_id"]
        subject = "N:" + task_id if args.source_kind == "N" else row["probe_id"]
        extraction = extract_for_evaluation(row, args.source_kind, tokenizer)
        if mapper is not None:
            extraction["code_token_mapping"] = mapper.probe_code_tokens(row, extraction)
        item = dict(subject_id=subject, case_id=task_id, source_kind=args.source_kind,
                    position=None if args.source_kind == "N" else row["position_generated_tokens"],
                    extraction=extraction, code_sha256=extraction["code_sha256"], generation_valid=extraction["generation_valid"],
                    public_result=_stub("NOT_EVALUATED_INVALID_DELIVERY", len(_registry[task_id]["public"])),
                    private_result=_stub("NOT_EVALUATED_INVALID_DELIVERY", len(_registry[task_id]["dev_private"])))
        if extraction["deliverable"]:
            jobs[(task_id, extraction["code_sha256"])] = (task_id, extraction["code_text"].encode())
        evaluations.append(item)
    if len({r["subject_id"] for r in evaluations}) != len(evaluations):
        raise ValueError("Duplicate source subjects")
    _write_lines(output / "raw/extractions.jsonl", evaluations)
    manifest = {"schema": "exec_exit.program_evaluation_manifest.v2", "status": "RUNNING", "planned_subjects": len(evaluations),
                "unique_jobs": [{"task_id": t, "code_sha256": h, "status": "PENDING"} for t, h in jobs],
                "resume_rule": "Completed checkpoints/cache reused; previously started suite without cache blocks rather than silently reruns"}
    _write(output / "manifest.json", manifest)
    result_map, error = {}, None
    _write(output / "summary.json", dict(status="RUNNING", planned=len(evaluations), unique_programs=len(jobs), completed_unique_programs=0))
    try:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(args.workers, initializer=_initialize_worker,
                      initargs=(str(output), str(cache_root), limits, identity)) as pool:
            for result in pool.imap_unordered(_execute_job, jobs.values(), chunksize=1):
                key = result["task_id"], result["code_sha256"]
                result_map[key] = result
                for job in manifest["unique_jobs"]:
                    if (job["task_id"], job["code_sha256"]) == key:
                        job.update(status="COMPLETE", checkpoint="checkpoints/" + result["job_id"] + ".json")
                        break
                _write(output / "manifest.json", manifest)
                _write(output / "progress.json", dict(completed_unique_programs=len(result_map), planned_unique_programs=len(jobs),
                       timeout_programs=sum(r["timeout_robustness"].get("triggered", False) for r in result_map.values()), elapsed_seconds=time.perf_counter()-started))
    except BaseException as exc:
        error = repr(exc)
    for item in evaluations:
        result = result_map.get((item["case_id"], item["code_sha256"]))
        if result is not None:
            first = result["attempts"][0]
            item["public_result"], item["private_result"] = first["public_result"], first["private_result"]
            item["timeout_robustness"] = result["timeout_robustness"] | {"attempts": result["attempts"]}
            item["final_outcome"], item["unstable"] = result["timeout_robustness"]["final_outcome"], result["timeout_robustness"]["unstable"]
            first_outcome = first["whole_program_result"]
        elif not item["extraction"]["deliverable"]:
            first_outcome = "FAIL"
            item.update(final_outcome="FAIL", unstable=False, timeout_robustness={"triggered": False, "attempts": [], "reason": "Invalid or protocol-rejected delivery"})
        else:
            first_outcome = "UNKNOWN"
            item["public_result"]["status"] = item["private_result"]["status"] = "UNKNOWN_EXECUTION_NOT_COMPLETED"
            item.update(final_outcome="UNKNOWN", unstable=False, timeout_robustness={"triggered": None, "attempts": [], "reason": "Execution incomplete"})
        item["dev_result"] = {"status": first_outcome, "all_passed": {"PASS": True, "FAIL": False, "UNKNOWN": None}[first_outcome],
                              "definition": "First complete program attempt; final_outcome is the independent offline adjudication"}
    _write_lines(output / "raw/evaluations.jsonl", evaluations)
    _write_lines(output / "raw/signals_public.jsonl", [{k: r[k] for k in ("subject_id", "case_id", "source_kind", "position", "generation_valid", "extraction", "code_sha256", "public_result")} for r in evaluations])
    _write_lines(output / "raw/dev_labels.jsonl", [{k: r[k] for k in ("subject_id", "case_id", "dev_result", "private_result", "final_outcome", "unstable", "timeout_robustness")} for r in evaluations])
    all_suites = {s["cache_key"]: s for result in result_map.values() for a in result["attempts"]
                  for s in (a["public_result"], a["private_result"]) if s.get("cache_key") is not None}
    new = [s for s in all_suites.values() if not s["cache_reused"]]
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = dict(status="COMPLETE" if error is None and len(result_map) == len(jobs) else "FAILED", error=error,
                   planned=len(evaluations), unique_programs=len(jobs), completed_unique_programs=len(result_map),
                   dev_outcomes=dict(Counter(r["final_outcome"] for r in evaluations)), first_dev_outcomes=dict(Counter(r["dev_result"]["status"] for r in evaluations)),
                   public_outcomes=dict(Counter(r["public_result"]["status"] for r in evaluations)),
                   invalid_deliveries=sum(not r["extraction"]["deliverable"] for r in evaluations),
                   protocol_rejected=sum(r["extraction"]["protocol_rejected"] for r in evaluations),
                   parsed_programs=sum(r["extraction"]["parsed"] for r in evaluations), extraction_status_counts=dict(Counter(r["extraction"]["status"] for r in evaluations)),
                   timeout_unique_programs=sum(r["timeout_robustness"].get("triggered", False) for r in result_map.values()),
                   unstable_unique_programs=sum(r["timeout_robustness"]["unstable"] for r in result_map.values()),
                   unstable_subjects=sum(r["unstable"] for r in evaluations), final_vs_first_label_changes=sum(r["final_outcome"] != r["dev_result"]["status"] for r in evaluations),
                   output_limit_first_subjects=sum(any(r[name].get("case_status_counts", {}).get("OUTPUT_LIMIT", 0) > 0
                                                       for name in ("public_result", "private_result")) for r in evaluations),
                   output_limit_unique_suites=sum(s.get("case_status_counts", {}).get("OUTPUT_LIMIT", 0) > 0 for s in all_suites.values()),
                   complete_program_attempts=sum(len(r["attempts"]) for r in result_map.values()), unique_suites_referenced=len(all_suites),
                   capacity_policy=args.capacity_policy, input_capacity_mib=args.input_capacity_mib,
                   output_capacity_mib=limits["output_bytes"] // 1024**2,
                   capacity_unknown_suites=sum(s.get("status") == "UNKNOWN_IO_CAPACITY" for result in result_map.values()
                                               for a in result["attempts"] for s in (a["public_result"], a["private_result"])),
                   capacity_affected_tasks=preflight["affected_tasks"],
                   suites_actually_executed=len(new), cache_suites_reused=sum(s["cache_reused"] for s in all_suites.values()),
                   unique_execution_cpu_seconds=sum(s["original_execution_cpu_seconds"] for s in new),
                   unique_execution_child_cpu_seconds=sum(s["child_cpu_seconds"] for s in new),
                   unique_execution_child_wall_seconds=sum(s["child_wall_seconds"] for s in new),
                   unique_execution_suite_wall_seconds=sum(s["original_execution_wall_seconds"] for s in new),
                   wall_seconds=time.perf_counter()-started, parent_cpu_user_seconds=usage.ru_utime-usage0.ru_utime,
                   parent_cpu_system_seconds=usage.ru_stime-usage0.ru_stime, gpu_used=False,
                   correctness_scope="Observed tests under frozen resources with predeclared timeout robustness; not formal correctness",
                   online_visibility="UNKNOWN; first PUB retained, private labels and majority votes offline only")
    manifest["status"] = summary["status"]
    _write(output / "manifest.json", manifest)
    _write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
