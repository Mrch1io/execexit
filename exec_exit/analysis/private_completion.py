"""Resume missing PRIV-IND labels without changing the frozen execution limits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import resource
import shutil
import signal
import tempfile
import time
import traceback
import uuid

from exec_exit.analysis.private_overlap import identity, checked, strict_bytes
from exec_exit.execute import leetcode_call as lc, sandbox as stdio
from exec_exit.execute import private_sequence
from exec_exit.execute.evaluate_dev import adjudicate
from exec_exit.execute.machine_identity import current_machine_identity
from exec_exit.execute.run_tests import ExecutionCache, execution_cache_key
from exec_exit.pipeline.postfreeze_evaluate import decoded_tests

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "runs/c2_cost_estimate/raw/jobs.jsonl"
OVERLAP = ROOT / "runs/c2_label_independence/raw/overlap_tasks.jsonl"
FAILURES = {"WRONG_ANSWER", "TIMEOUT_CPU", "TIMEOUT_WALL", "RUNTIME_ERROR", "OUTPUT_LIMIT",
            "SANDBOX_VIOLATION", "SIGNAL"}
MISSING = {"UNKNOWN", "NOT_EVALUATED_AFTER_FAILURE"}
TIMEOUTS = {"TIMEOUT_CPU", "TIMEOUT_WALL"}
_tasks = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError("Immutable receipt conflict: " + str(path))
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_inputs(jobs):
    """Decode only selected H-v6 rows in RAM; never materialize LCB or tests."""
    frozen = read(ROOT / "FROZEN.json")
    mapping = {r["case_id"]: r for r in map(json.loads, OVERLAP.open())}
    overlap_summary = read(OVERLAP.with_name("overlap_summary.json"))
    if overlap_summary["status"] != "COMPLETE" or identity(OVERLAP)["sha256"] != overlap_summary["rows"]["sha256"]:
        raise ValueError("Incomplete or changed overlap audit")
    selected = {j["case_id"]: j["collection"] for j in jobs}
    if set(selected.values()) - {"hv6_atcoder", "hv6_leetcode"}:
        raise ValueError("Only H-v6 missing labels are admitted; do not restore LCB")
    components = frozen["included_components"]
    ref = components["qwen3_8b::hv6_atcoder"]["dataset_source"]
    if ref != components["qwen3_8b::hv6_leetcode"]["dataset_source"]:
        raise ValueError("H-v6 source differs between collections")
    source = checked(ref)
    tasks = {}
    with source.open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            cid = row["platform"] + ":" + row["question_id"]
            if cid not in selected:
                continue
            if cid in tasks or cid not in components["qwen3_8b::" + selected[cid]]["case_ids"]:
                raise ValueError("Duplicate or unregistered task")
            overlap = mapping[cid]
            if overlap["collection"] != selected[cid] or overlap["source"]["source_row_sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("Source row differs from overlap audit")
            functional = selected[cid] == "hv6_leetcode"
            kind = "functional" if functional else "stdin"
            public = decoded_tests(row["public_test_cases"], kind)
            private = decoded_tests(row["private_test_cases"], kind)
            if len(public) != overlap["public_count"] or len(private) != overlap["private_count"]:
                raise ValueError("Test cardinality changed")
            for test, original in zip(private, overlap["private_cases"]):
                data = strict_bytes(test["input"].encode())
                if original["input_identity"]["strict"] != {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}:
                    raise ValueError("Private input identity changed")
            tests = public + private if functional else [private[i] for i in overlap["private_keep_indices"]]
            retained = overlap["execution_retained_indices"] if functional else list(range(len(tests)))
            metadata = row["metadata"]
            metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
            tasks[cid] = dict(tests=tests, retained_indices=retained, func_name=metadata.get("func_name"),
                              collection=selected[cid], original_private_indices=overlap["private_keep_indices"])
    if set(tasks) != set(selected) or any(not t["retained_indices"] for t in tasks.values()):
        raise ValueError("Missing task or empty retained-private suite")
    return tasks, ref


def retained_summary(result, indices):
    cases = {r["test_index"]: r for r in result["cases"]}
    states = [cases.get(i, {}).get("status", "UNKNOWN") for i in indices]
    unexpected = set(states) - FAILURES - MISSING - {"PASS"}
    if unexpected:
        raise ValueError("Unexpected execution status: " + repr(sorted(unexpected)))
    passed = False if set(states) & FAILURES else True if states and all(s == "PASS" for s in states) else None
    return dict(all_passed=passed, status="PASS" if passed is True else "FAIL" if passed is False else "UNKNOWN",
                case_status_counts=dict(Counter(states)))


def verdict(attempts):
    neutral = dict(all_passed=True, case_status_counts={})
    decision = adjudicate([dict(public_result=neutral, private_result=a["retained"],
                                whole_program_result=a["retained"]["status"]) for a in attempts])
    failed = sorted({status for a in attempts for status, count in a["retained"]["case_status_counts"].items()
                     if count and status in FAILURES})
    return dict(outcome=decision["final_outcome"], required_attempts=decision["required_attempts"],
                completed_attempts=len(attempts), first_timeout=bool(attempts and attempts[0]["retained"]["case_status_counts"].get("TIMEOUT_CPU")),
                timeout_only=decision["final_outcome"] == "FAIL" and bool(failed) and set(failed) <= TIMEOUTS,
                failure_types=failed, adjudication=decision)


def cgroup_cpu():
    fields = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    return int(fields["usage_usec"]) / 1e6


def process_start(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def guard_reason(cpu, free):
    if cpu >= 70 * 3600:
        return "STOPPED_CPU_CAP"
    if free < 20:
        return "STOPPED_DISK_CAPACITY"
    return None


def process_cpu():
    """Live descendants plus waited-child counters; cgroup guard covers exit races."""
    values = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            data = path.read_text().rsplit(")", 1)[1].split()
            values[int(path.parent.name)] = (int(data[1]), sum(int(data[i]) for i in (11, 12, 13, 14)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    included = {os.getpid()}
    while True:
        enlarged = included | {pid for pid, (parent, _) in values.items() if parent in included}
        if enlarged == included:
            break
        included = enlarged
    return sum(values[pid][1] for pid in included if pid in values) / os.sysconf("SC_CLK_TCK")


def worker(index, session, run, pending, events, stop, binding, limits):
    os.setsid()
    write(run / "raw/workers" / f"{session}.{index}.json",
          dict(pid=os.getpid(), start_ticks=process_start(os.getpid()), session=session,
               process_group=os.getpgrp(), binding_sha256=binding), immutable=True)
    runtime = run / "runtime" / f"worker_{index}"
    ac_path, lc_path = runtime / "stdio", runtime / "lc"
    ac_box = stdio.make_sandbox(ac_path, template_dir=None if ac_path.exists() else run / "runtime/template_stdio")
    lc_box = lc.make_sandbox(lc_path, None if lc_path.exists() else run / "runtime/template_lc")
    identities = {"hv6_atcoder": stdio.execution_identity(ac_box), "hv6_leetcode": lc.execution_identity(lc_box)}
    identities["hv6_leetcode"]["executor"]["private_sequence_sha256"] = identity(private_sequence.__file__)["sha256"]
    cache = ExecutionCache(run / "execution_cache")
    while not stop.is_set():
        job = pending.get()
        if job is None:
            return
        job_id, cid = job["job_id"], job["case_id"]
        task = _tasks[cid]
        try:
            attempts = []
            for number in range(3):
                if stop.is_set():
                    return
                path = run / "raw/attempts" / f"{job_id}.{number}.json"
                descriptor = dict(task_id=cid, protocol="priv_ind_completion_preserved_sequence_v1", attempt=number,
                                  stop_on_failure="retained_only", retained_indices=task["retained_indices"],
                                  func_name=task["func_name"], binding_sha256=binding)
                key = execution_cache_key(job["code_text"].encode(), task=descriptor, tests=task["tests"],
                                          resource_limits=limits, **identities[task["collection"]])
                if path.exists():
                    receipt = read(path)
                    if receipt["cache_key"] != key or receipt["binding_sha256"] != binding:
                        raise ValueError("Completed attempt identity differs")
                    checked(receipt["cache"])
                else:
                    started = run / "raw/attempts" / f"{job_id}.{number}.{session}.started.json"
                    previous = sorted(str(p) for p in started.parent.glob(f"{job_id}.{number}.*.started.json"))
                    write(started, dict(job_id=job_id, attempt=number, created_utc=now(), session=session,
                                        pid=os.getpid(), cache_key=key, binding_sha256=binding,
                                        incomplete_prior_starts=previous), immutable=True)
                    stored = cache.get(key)
                    reused = stored is not None
                    if stored is None:
                        code = job["code_text"].encode()
                        if task["collection"] == "hv6_leetcode":
                            result = private_sequence.evaluate_suite(lc_box, code, task["tests"], limits, "dev_full",
                                                                     task["func_name"], task["retained_indices"])
                        else:
                            result = stdio.evaluate_suite(ac_box, code, task["tests"], limits, "dev_private")
                        if stop.is_set():
                            return
                        stored = dict(result=result, identity=identities[task["collection"]], task=descriptor,
                                      job_id=job_id, code_sha256=job["code_sha256"], completed_utc=now())
                        cache.put(key, stored)
                    result = stored["result"]
                    receipt = dict(job_id=job_id, case_id=cid, attempt=number, cache_key=key,
                                   cache=identity(cache._path(key)), binding_sha256=binding,
                                   retained=retained_summary(result, task["retained_indices"]),
                                   child_cpu_seconds=result["child_cpu_seconds"],
                                   child_wall_seconds=result["child_wall_seconds"],
                                   completed_utc=stored["completed_utc"], cache_recovered=reused)
                    if stop.is_set():
                        return
                    write(path, receipt, immutable=True)
                attempts.append(dict(receipt, receipt=identity(path)))
                if number == 0 and not receipt["retained"]["case_status_counts"].get("TIMEOUT_CPU"):
                    break
            if stop.is_set():
                return
            result = {k: job[k] for k in ("job_id", "case_id", "collection", "code_sha256", "references")}
            result.update(verdict(attempts), attempts=attempts, binding_sha256=binding, completed_utc=now())
            write(run / "raw/jobs" / f"{job_id}.json", result, immutable=True)
            events.put(dict(job_id=job_id, status="COMPLETE", outcome=result["outcome"]))
        except Exception:
            error = dict(job_id=job_id, session=session, created_utc=now(), status="ENGINEERING_ERROR",
                         traceback=traceback.format_exc(), binding_sha256=binding)
            write(run / "raw/errors" / f"{job_id}.{session}.json", error, immutable=True)
            events.put(error)


def progress_page(run, status):
    path = run / "ONE_PAGE.md"
    old = path.read_text() if path.exists() else "# PRIV-IND 补全\n"
    marker = "<!-- private_completion_progress -->"
    if marker in old:
        old = old.split(marker, 2)[-1].lstrip("\n")
    eta = "UNKNOWN" if status["eta_seconds"] is None else f"约 {status['eta_seconds']/3600:.2f} 小时（按已完成速度）"
    local_time = datetime.fromisoformat(status["updated_utc"]).astimezone(timezone(timedelta(hours=8))).isoformat()
    text = (f"{marker}\n**{local_time}（北京时间）：{status['status']}，已完成 {status['completed_jobs']}/{status['total_jobs']}。** "
            f"累计保守 CPU 计费 {status['budget_cpu_seconds']/3600:.3f}/70 小时；预计剩余 {eta}；"
            f"可用空间 {status['free_gib']:.2f} GiB。\n{marker}\n\n")
    path.write_text(text + old)


def run(args):
    global _tasks
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, interrupted)
        return execute(args, run)


def interrupted(*_):
    raise KeyboardInterrupt("SIGTERM")


def private_definition(path):
    text = Path(path).read_text()
    return text.split("## 3. 事先确定的判据\n", 1)[1].split("\n## 4.", 1)[0]


def execute(args, run):
    global _tasks
    baseline = cgroup_cpu()
    tree_baseline = process_cpu()
    usage_baseline = sum(getattr(resource.getrusage(who), name)
                         for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN) for name in ("ru_utime", "ru_stime"))
    config = read(run / "config.yaml")
    if config["cpu_limit_seconds"] != 70 * 3600 or config["min_free_gib"] != 20 or not 1 <= args.workers <= 16:
        raise ValueError("Unchanged 70 CPU-hour / 20 GiB boundary and at most 16 workers required")
    if current_machine_identity() != config["machine"]:
        raise ValueError("Execution machine differs from configured identity")
    expected_limits = dict(wall_seconds=72, cpu_seconds=60, memory_bytes=1073741824,
                           output_bytes=8388608, code_bytes=131072, input_bytes=67108864)
    if config["limits"] != expected_limits:
        raise ValueError("Frozen production execution limits changed")
    definition = run / "definition_before_execution.md"
    for ref in config["source_identities"]:
        if Path(ref["path"]).name == "FREEZE_ADDENDUM_LLAMA.md":
            if private_definition(definition) != private_definition(ref["path"]):
                raise ValueError("PRIV-IND definition differs from the pre-execution snapshot")
        else:
            checked(ref)
    jobs = list(map(json.loads, JOBS.open()))
    if len(jobs) != 2867 or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Unique job denominator differs")
    for job in jobs:
        if hashlib.sha256(job["code_text"].encode()).hexdigest() != job["code_sha256"]:
            raise ValueError("Saved code bytes changed")
    sources = [identity(path) for path in [__file__, private_sequence.__file__, lc.__file__, stdio.__file__,
               ROOT / "exec_exit/execute/evaluate_dev.py", ROOT / "exec_exit/execute/run_tests.py",
               ROOT / "exec_exit/pipeline/postfreeze_evaluate.py", ROOT / "exec_exit/analysis/private_overlap.py",
               stdio.LEGACY_EXECUTOR, lc.official_evaluator_path()]]
    binding_data = dict(config=identity(run / "config.yaml"), sources=sources, jobs=identity(JOBS),
                        definition=identity(definition), private_definition_sha256=hashlib.sha256(private_definition(definition).encode()).hexdigest(),
                        editable_document_rule="Historical full-document identity remains in config; current section 3 must equal the pre-execution definition snapshot, whose actual SHA is bound here. Other document sections may change.",
                        overlap=identity(OVERLAP), machine=config["machine"], limits=config["limits"])
    write(run / "execution_config.json", binding_data, immutable=True)
    binding = identity(run / "execution_config.json")["sha256"]
    for path in (run / "raw/workers").glob("*.json"):
        previous = read(path)
        if previous["binding_sha256"] != binding:
            raise ValueError("Previous worker identity differs")
        if process_start(previous["pid"]) == previous["start_ticks"]:
            raise RuntimeError("Previous owned worker is still alive; recover that process group before resuming")
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    session_dir = run / "raw/sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    prior_cpu = 0.0
    for path in sorted(session_dir.glob("*.json")):
        old = read(path)
        charged = old.get("cgroup_cpu_seconds", 0)
        if not old.get("closed"):
            if old["machine_fingerprint"] != config["machine"]["fingerprint"]:
                raise ValueError("Cannot reconcile interrupted CPU accounting on another machine")
            if baseline < old["cgroup_start_seconds"]:
                raise ValueError("cgroup counter reset; interrupted CPU cost cannot be reconciled")
            charged = max(charged, baseline - old["cgroup_start_seconds"])
            old.update(closed=True, recovered_utc=now(), cgroup_cpu_seconds=charged, status="UNKNOWN_INTERRUPTED")
            write(path, old)
        prior_cpu += charged
    session_path = session_dir / (session + ".json")
    current = dict(session=session, started_utc=now(), cgroup_start_seconds=baseline,
                   machine_fingerprint=config["machine"]["fingerprint"], closed=False,
                   accounting="Container cgroup CPU includes unrelated background work; conservative budget protection")
    write(session_path, current)
    started = time.monotonic()
    if prior_cpu >= config["cpu_limit_seconds"] or shutil.disk_usage(run).free < 20 * 1024**3:
        current.update(closed=True, status="STOPPED_RESOURCE_GUARD", cgroup_cpu_seconds=cgroup_cpu()-baseline)
        write(session_path, current)
        return current
    _tasks, _ = load_inputs(jobs)
    (run / "runtime").mkdir(parents=True, exist_ok=True)
    stdio.make_sandbox(run / "runtime/template_stdio")
    lc.make_sandbox(run / "runtime/template_lc")
    completed = set()
    for path in (run / "raw/jobs").glob("*.json"):
        result = read(path)
        if result["binding_sha256"] != binding:
            raise ValueError("Existing job belongs to another binding")
        for attempt in result["attempts"]:
            checked(attempt["receipt"])
            checked(attempt["cache"])
        completed.add(result["job_id"])
    selected = [j for j in jobs if not args.job_id or j["job_id"] == args.job_id]
    if args.job_id and len(selected) != 1:
        raise ValueError("Smoke job must be an existing job")
    waiting = [j for j in selected if j["job_id"] not in completed]
    initial_completed = len(completed)
    context = mp.get_context("fork")
    pending, events, stop = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=worker, args=(i, session, run, pending, events, stop, binding, config["limits"]))
               for i in range(min(args.workers, len(waiting)))]
    for process in workers:
        process.start()
    for job in waiting:
        pending.put(job)
    for _ in workers:
        pending.put(None)
    errors, reason, max_tree_cpu = [], None, 0.0
    quarter = initial_completed * 4 // len(jobs)
    last_write = 0
    try:
        while any(p.is_alive() for p in workers):
            try:
                event = events.get(timeout=.5)
                if event["status"] == "COMPLETE":
                    completed.add(event["job_id"])
                else:
                    errors.append(event)
            except queue.Empty:
                pass
            spent = max(0, cgroup_cpu() - baseline)
            free = shutil.disk_usage(run).free / 1024**3
            max_tree_cpu = max(max_tree_cpu, process_cpu()-tree_baseline)
            reason = guard_reason(prior_cpu + spent, free)
            if reason:
                stop.set()
                break
            elapsed = time.monotonic() - started
            finished = len(completed) - initial_completed
            status = dict(status="RUNNING", updated_utc=now(), session=session, completed_jobs=len(completed),
                          total_jobs=len(jobs), session_wall_seconds=elapsed, budget_cpu_seconds=prior_cpu+spent,
                          process_tree_cpu_seconds=max_tree_cpu, free_gib=free,
                          eta_seconds=(len(jobs)-len(completed))*elapsed/finished if finished else None,
                          engineering_errors=len(errors))
            if elapsed - last_write >= 2:
                write(run / "progress.json", status)
                current.update(cgroup_cpu_seconds=spent, process_tree_cpu_seconds=max_tree_cpu)
                write(session_path, current)
                last_write = elapsed
            if len(completed)*4//len(jobs) > quarter:
                quarter = len(completed)*4//len(jobs)
                progress_page(run, status)
        for process in workers:
            if process.exitcode not in (None, 0):
                errors.append(dict(pid=process.pid, exitcode=process.exitcode, status="WORKER_EXIT"))
    except BaseException:
        reason = "UNKNOWN_INTERRUPTED"
        write(run / "raw/errors" / f"controller.{session}.json", dict(created_utc=now(), traceback=traceback.format_exc()), immutable=True)
        raise
    finally:
        if reason:
            stop.set()
            for process in workers:
                if process.is_alive():
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in workers:
                process.join(timeout=3)
                if process.is_alive():
                    os.killpg(process.pid, signal.SIGKILL)
        for process in workers:
            process.join()
        current.update(closed=True, finished_utc=now(), status=reason or "FINISHED",
                       cgroup_cpu_seconds=max(0, cgroup_cpu()-baseline), process_tree_cpu_seconds=max_tree_cpu,
                       controller_and_reaped_children_cpu_seconds=sum(getattr(resource.getrusage(who), name)
                           for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN) for name in ("ru_utime", "ru_stime"))-usage_baseline)
        write(session_path, current)
        pending.cancel_join_thread()
    completed = {p.stem for p in (run / "raw/jobs").glob("*.json")}
    spent = prior_cpu + current["cgroup_cpu_seconds"]
    result = dict(status=reason or ("COMPLETE" if len(completed)==len(jobs) else "SMOKE_COMPLETE" if args.job_id and args.job_id in completed else "INCOMPLETE_ENGINEERING_ERRORS"),
                  updated_utc=now(), session=session, completed_jobs=len(completed), total_jobs=len(jobs),
                  budget_cpu_seconds=spent, process_tree_cpu_seconds=max_tree_cpu,
                  free_gib=shutil.disk_usage(run).free/1024**3, eta_seconds=None,
                  engineering_errors=errors, binding_sha256=binding, gpu_used=False)
    for path in (run / "raw/attempts").glob(f"*.{session}.started.json"):
        attempt = read(path)
        if not (run / "raw/attempts" / f"{attempt['job_id']}.{attempt['attempt']}.json").exists():
            write(path.with_name(path.name.replace(".started.json", ".interrupted.json")),
                  dict(attempt, status="UNKNOWN_INTERRUPTED", reason=reason or "ENGINEERING_ERROR", ended_utc=now()), immutable=True)
    write(run / "progress.json", result)
    write(run / "summary.json", result)
    progress_page(run, result)
    for source in sources:
        checked(source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--job-id", help="One existing job for the required smoke; its receipt is reused by the full run")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
