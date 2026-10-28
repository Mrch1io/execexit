"""Blind heldout segmented trajectories with native metrics; never execute programs.

The explicit blind public-only projection and source hashes are fixed before execute.
This module neither selects a dataset nor reads private tests. Existing Dev36
generation modules remain unchanged. Each active slot owns an entire two-stage
question; stage 1 has priority only because its stage 0 has already finished.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

from exec_exit.gen.engine_preflight import MODEL, ROOT, VLLM, SNAPSHOT_RPC_METHOD, gpu_inventory, save, sha
from exec_exit.gen.probes import bounded_queue
from exec_exit.gen.host_environment import collect_host_environment
from exec_exit.gen.traces import (ENVIRONMENT, LEGACY, Protocol, StreamWriter, atomic_jsonl,
    generate_case, render_prompt, supervise_process, verify_preflight)


def safe_id(case_id):
    if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+", case_id):
        raise ValueError("Unsafe task identity")
    return case_id.replace(":", "_")


def load_public_cases(path):
    from exec_exit.data.blind_hv6 import load_blind_cases
    rows=load_blind_cases(path)
    for row in rows: safe_id(row['case_id'])
    return [{key: row[key] for key in ('case_id','task_id','question_content')} for row in rows]


class NativeCounters:
    """Client-side sums over vLLM's public custom-stat-logger interface.

    num_preempted_reqs increments for each delivered PREEMPTED EngineCoreEvent.
    It counts repeated preemptions, not unique requests. No logging I/O in record.
    Stats callbacks are native observations; output token count remains separate.
    """
    instances = []

    def __init__(self, vllm_config=None, engine_index=0):
        self.engine_index = engine_index
        self.callbacks = self.iterations = self.preemptions = 0
        self.prompt_tokens = self.generation_tokens = 0
        self.metric_schema_valid = True
        self.instances.append(self)

    def log_engine_initialized(self):
        pass

    def log(self):
        pass

    def record_sleep_state(self, is_awake, level):
        pass

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        self.callbacks += 1
        if iteration_stats is None:
            return
        self.iterations += 1
        fields = ("num_preempted_reqs", "num_prompt_tokens", "num_generation_tokens")
        values = [getattr(iteration_stats, name, None) for name in fields]
        if any(type(value) is not int or value < 0 for value in values):
            self.metric_schema_valid = False
            return
        self.preemptions += values[0]
        self.prompt_tokens += values[1]
        self.generation_tokens += values[2]

    @classmethod
    def snapshot(cls):
        good = bool(cls.instances) and all(x.metric_schema_valid and x.iterations for x in cls.instances)
        return dict(status="OBSERVED" if good else "UNKNOWN", engines=len(cls.instances),
            preemption_events=sum(x.preemptions for x in cls.instances) if good else "UNKNOWN",
            preemption_definition="Each delivered PREEMPTED EngineCoreEvent; repeated preemption counts repeatedly",
            iteration_callbacks=sum(x.iterations for x in cls.instances),
            native_prompt_tokens=sum(x.prompt_tokens for x in cls.instances) if good else "UNKNOWN",
            native_output_tokens=sum(x.generation_tokens for x in cls.instances) if good else "UNKNOWN",
            physical_recomputed_tokens="UNKNOWN", source="vllm.v1.metrics.stats.IterationStats")


RUNNER_FIELDS = ("runner_class", "runner_module", "runner_source_sha256", "use_v2_model_runner",
    "max_model_len", "async_scheduling", "compile_mode", "cudagraph_mode", "enable_prefix_caching",
    "dtype", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes", "block_size")


def validate_identity(config):
    for path, expected in config["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Frozen source/input changed: " + path)


def progress(run_dir, manifest, status, *, active=None, deadline=None):
    done = [x for x in manifest["tasks"] if x["status"] == "COMPLETE"]
    elapsed = time.time() - manifest.get("generation_started_epoch", time.time())
    session_done = len(done) - manifest.get("completed_before_session", 0)
    row = dict(status=status, completed=len(done), planned=len(manifest["tasks"]),
        completed_tokens=sum(x.get("generated_tokens", 0) for x in done),
        active=active or {}, updated_epoch=time.time(),
        eta_remaining_s=elapsed / session_done * (len(manifest["tasks"])-len(done)) if session_done > 0 else None,
        eta_basis="Completed questions per session wall time; heterogeneous lengths make early estimates unstable")
    if deadline is not None:
        row["phase_deadline_epoch"] = deadline
    save(run_dir / "manifest.json", manifest)
    save(run_dir / "progress.json", row)


def submission(run_dir, case_id, config_identity, session_index):
    with (run_dir / "raw/submissions.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(case_id=case_id, config_identity=config_identity,
            session_index=session_index, epoch=time.time())) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(args):
    from transformers import AutoTokenizer
    run_dir = args.run_dir.resolve()
    preflight = verify_preflight(args.preflight_dir)
    cases = load_public_cases(args.public_cases)
    if type(args.sample_index) is not int or args.sample_index < 0:
        raise ValueError("Nonnegative integer sample_index required")
    if not 1 <= args.concurrency <= preflight["engine"]["max_num_seqs"]:
        raise ValueError("Concurrency exceeds frozen engine capacity")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("New runs cannot overwrite or silently retry an existing run; use recover_pending")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    sizes = [len(render_prompt(tokenizer, case)[1]) for case in cases]
    if any(size + 28000 + 5 + 2048 > 40960 for size in sizes):
        raise ValueError("Full forced-answer protocol exceeds capacity for a selected public question")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "checkpoints", "source"):
        (run_dir / name).mkdir()
    paths = [Path(__file__), ROOT / "exec_exit/data/blind_hv6.py", ROOT / "exec_exit/data/heldout_inventory.py", ROOT / "exec_exit/gen/traces.py", ROOT / "exec_exit/gen/probes.py",
        ROOT / "exec_exit/gen/engine_preflight.py", LEGACY, args.public_cases.resolve(),
        args.preflight_dir.resolve() / "config.yaml", args.preflight_dir.resolve() / "summary.json",
        MODEL / "config.json", MODEL / "generation_config.json", MODEL / "tokenizer.json"]
    paths += [VLLM / name for name in ("v1/metrics/stats.py", "v1/metrics/loggers.py", "v1/engine/async_llm.py")]
    config = dict(schema="exec_exit.blind_trajectories.v1", model=str(MODEL),
        public_cases=str(args.public_cases.resolve()), preflight_dir=str(args.preflight_dir.resolve()),
        expected_gpu_uuid=preflight["expected_gpu_uuid"], engine=preflight["engine"],
        preflight_runner_snapshot=preflight["actual_runner_snapshot"],
        concurrency=args.concurrency, serial_questions=False, repeats_per_question=1,
        submission_order="Selected manifest order; bounded complete-question queue; each question runs stages 0 then 1",
        protocol=Protocol().__dict__, sample_index=args.sample_index,
        seed_rule=f"legacy seed_for(case_id,{args.sample_index},stage)",
        question_count=len(cases), prompt_tokens_min=min(sizes), prompt_tokens_max=max(sizes),
        question_timeout_s=args.question_timeout_s, no_total_run_quota=True,
        timeout_basis="Per-question failure guard only, including contention; no total run budget",
        environment=ENVIRONMENT, host_environment=collect_host_environment(),
        source_sha256={str(path.resolve()): sha(path) for path in paths},
        metrics_capture="Custom native StatLogger; enables native metrics despite disable_log_stats=True; native Prometheus also initialized",
        metrics_difference_from_dev36="Dev36 had logging disabled; counters observation added to record requested preemptions",
        private_tests_read=False, public_test_payloads_read=False, heldout_question_text_read=True,
        question_programs_executed=0, blind_generation_only=True, contribution_or_online_equivalence_claim=False)
    config["config_identity"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(run_dir / "config.yaml", config)
    (run_dir / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()+"\n")
    for path in paths:
        if path.suffix == ".py":
            name = str(path.resolve()).replace("/", "_").lstrip("_")
            (run_dir / "source" / name).write_bytes(path.read_bytes())
    manifest = dict(schema="exec_exit.blind_manifest.v1", session_index=0,
        tasks=[dict(case_id=case["case_id"], candidate_index=i, status="PENDING") for i, case in enumerate(cases)])
    progress(run_dir, manifest, "PREPARED")
    save(run_dir / "summary.json", dict(status="PREPARED", gpu_used=False, completed=0, planned=len(cases)))


def recover_pending(run_dir):
    """Recover completed checkpoints; never re-submit an attempted incomplete case."""
    config = json.loads((run_dir / "config.yaml").read_text())
    validate_identity(config)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    submitted = set()
    journal = run_dir / "raw/submissions.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row["config_identity"] != config["config_identity"] or row["case_id"] in submitted:
                raise ValueError("Submission journal inconsistent")
            submitted.add(row["case_id"])
    for entry in manifest["tasks"]:
        path = run_dir / "checkpoints" / (safe_id(entry["case_id"])+".json")
        if path.exists():
            row = json.loads(path.read_text())
            if (row.get("status") != "COMPLETE" or row.get("config_identity") != config["config_identity"]
                    or row.get("case_id") != entry["case_id"]
                    or entry.get("checkpoint_sha256", sha(path)) != sha(path)):
                raise ValueError("Checkpoint identity mismatch")
            entry.update(status="COMPLETE", checkpoint=str(path.relative_to(run_dir)), checkpoint_sha256=sha(path),
                generated_tokens=row["generated_tokens"], branch=row["branch"])
        elif entry["status"] != "PENDING" or entry["case_id"] in submitted:
            raise ValueError("Attempted incomplete question requires an explicit retry decision")
    manifest["session_index"] += 1
    progress(run_dir, manifest, "PREPARED")


async def run_gpu(run_dir):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.usage.usage_lib import UsageContext
    config = json.loads((run_dir / "config.yaml").read_text())
    validate_identity(config)
    verify_preflight(config["preflight_dir"])
    cases = load_public_cases(config["public_cases"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if [x["case_id"] for x in cases] != [x["case_id"] for x in manifest["tasks"]]:
        raise ValueError("Manifest order or denominator changed")
    entries = {x["case_id"]: x for x in manifest["tasks"]}
    writer = StreamWriter(run_dir / "raw" / f"stream_{manifest['session_index']}.jsonl")
    active, observed_counts = {}, {}
    engine = reporter = None
    stop = asyncio.Event()
    start, usage0, child0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    engine_start = generation_start = generation_end = None
    result_status, failure, drained = "FAILED", None, False
    summary = dict(status="RUNNING", gpu_used=False, process_id=os.getpid(), started_epoch=time.time(),
        completed=0, planned=len(cases), cuda_busy_seconds="UNKNOWN")
    save(run_dir / "summary.json", summary)
    try:
        config["observed_gpu"] = gpu_inventory(config["expected_gpu_uuid"])
        save(run_dir / "config.yaml", config)
        progress(run_dir, manifest, "ENGINE_STARTING", deadline=time.time()+1800)
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
        eos = json.loads((Path(config["model"]) / "generation_config.json").read_text())["eos_token_id"]
        eos = set(eos if isinstance(eos, list) else [eos])
        NativeCounters.instances = []
        engine_start = time.perf_counter()
        summary["gpu_used"] = "UNKNOWN_ENGINE_STARTING"
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=config["model"], **config["engine"]),
            usage_context=UsageContext.OPENAI_API_SERVER, stat_loggers=[NativeCounters])
        summary["gpu_used"] = True
        snapshots = await engine.collective_rpc(SNAPSHOT_RPC_METHOD)
        if len(snapshots) != 1:
            raise RuntimeError("Exactly one native worker required")
        snapshot = snapshots[0]
        for field in RUNNER_FIELDS:
            if snapshot.get(field) != config["preflight_runner_snapshot"].get(field):
                raise RuntimeError("Passed preflight worker differs: " + field)
        config.update(actual_model_runner=snapshot["runner_class"], actual_runner_snapshot=snapshot)
        save(run_dir / "config.yaml", config)
        writer.add("worker_snapshot", snapshot=snapshot, gpu=config["observed_gpu"])
        manifest.update(generation_started_epoch=time.time(), completed_before_session=sum(e["status"] == "COMPLETE" for e in entries.values()))
        generation_start = time.perf_counter()
        factory = lambda values: SamplingParams(**values, output_kind=RequestOutputKind.DELTA)

        async def launched(spec):
            case_id = spec["case_id"]
            await asyncio.to_thread(submission, run_dir, case_id, config["config_identity"], manifest["session_index"])
            entries[case_id].update(status="RUNNING", started_epoch=time.time())
            active[case_id] = dict(stage=None, observed_stage_tokens=0, deadline_epoch=time.time()+config["question_timeout_s"])

        async def operation(spec):
            case_id = spec["case_id"]
            def observe(stage, kind, tokens, cap):
                active[case_id].update(stage=stage, kind=kind, observed_stage_tokens=tokens, stage_cap=cap)
                observed_counts[(case_id, stage)] = tokens
                if reporter is not None and reporter.done() and reporter.exception():
                    raise RuntimeError("Progress writer failed") from reporter.exception()
            case = {key: spec[key] for key in ("case_id", "task_id", "question_content")}
            try:
                return await asyncio.wait_for(generate_case(engine, tokenizer, writer, case, eos, factory, observe,
                    sample_index=config.get("sample_index", 0)),
                    timeout=config["question_timeout_s"])
            except BaseException as exc:
                entries[case_id].update(status="FAILED", error=repr(exc))
                raise

        async def completed(spec, row):
            row["config_identity"] = config["config_identity"]
            path = run_dir / "checkpoints" / (safe_id(row["case_id"])+".json")
            await asyncio.to_thread(save, path, row)
            entries[row["case_id"]].update(status="COMPLETE", generated_tokens=row["generated_tokens"], branch=row["branch"],
                checkpoint=str(path.relative_to(run_dir)), checkpoint_sha256=sha(path))
            active.pop(row["case_id"], None)

        async def report():
            while not stop.is_set():
                deadline = min((x["deadline_epoch"] for x in active.values()), default=time.time()+config["question_timeout_s"])
                await asyncio.to_thread(progress, run_dir, json.loads(json.dumps(manifest)), "GENERATING",
                    active={key: dict(value) for key, value in active.items()}, deadline=deadline)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass
        pending = [dict(case, candidate_index=i) for i, case in enumerate(cases) if entries[case["case_id"]]["status"] != "COMPLETE"]
        reporter = asyncio.create_task(report())
        await bounded_queue(pending, config["concurrency"], operation, completed, launched)
        generation_end = time.perf_counter()
        if engine.output_processor.has_unfinished_requests():
            raise RuntimeError("Native requests remain after all questions completed")
        result_status = "COMPLETE"
    except BaseException as exc:
        failure = repr(exc)
        writer.add("failure", error=failure)
        raise
    finally:
        primary = sys.exc_info()[1]
        cleanup = None
        if generation_start is not None and generation_end is None:
            generation_end = time.perf_counter()
        if reporter is not None:
            stop.set()
            try:
                await reporter
            except BaseException as exc:
                cleanup = exc
        try:
            if engine is not None:
                drained = not engine.output_processor.has_unfinished_requests()
                summary["native_metrics"] = NativeCounters.snapshot()
                engine.shutdown(timeout=15)
        except BaseException as exc:
            cleanup = cleanup or exc
        try:
            writer.close()
        except BaseException as exc:
            cleanup = cleanup or exc
        if cleanup is not None:
            result_status, failure = "FAILED_CLEANUP", failure or repr(cleanup)
        done = [x for x in manifest["tasks"] if x["status"] == "COMPLETE"]
        rows = [json.loads((run_dir / x["checkpoint"]).read_text()) for x in done]
        await asyncio.to_thread(atomic_jsonl, run_dir / "raw/trajectories.jsonl", rows)
        calls = [call for row in rows for call in row["calls"]]
        wall = generation_end-generation_start if generation_start is not None else None
        output_tokens = sum(observed_counts.values())
        cached = [x["num_cached_tokens"] for x in calls if x["num_cached_tokens"] is not None]
        usage1, child1 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        summary.update(status=result_status, error=failure, completed=len(rows), planned=len(cases),
            generated_tokens_completed=sum(x["generated_tokens"] for x in rows),
            session_observed_output_tokens=output_tokens, generation_stage_wall_seconds=wall,
            session_output_tokens_per_generation_second=output_tokens/wall if wall else None,
            total_prompt_tokens_completed=sum(x["prompt_tokens"] for x in calls),
            native_cached_tokens_completed=sum(cached) if len(cached)==len(calls) else "UNKNOWN",
            native_cached_tokens_missing_calls=len(calls)-len(cached), native_frontend_drained=drained,
            preemption_count=summary.get("native_metrics", {}).get("preemption_events", "UNKNOWN"),
            cancelled_inflight_unobserved_tokens="UNKNOWN_IF_FAILED", cuda_busy_seconds="UNKNOWN",
            ended_epoch=time.time(), process_wall_seconds=time.perf_counter()-start,
            engine_ownership_wall_seconds=time.perf_counter()-engine_start if engine_start is not None else None,
            cpu_user_seconds=usage1.ru_utime-usage0.ru_utime, cpu_system_seconds=usage1.ru_stime-usage0.ru_stime,
            waited_child_cpu_user_seconds=child1.ru_utime-child0.ru_utime,
            waited_child_cpu_system_seconds=child1.ru_stime-child0.ru_stime,
            recording=writer.statistics(), probes_generated=0)
        save(run_dir / "summary.json", summary)
        progress(run_dir, manifest, result_status)
        if cleanup is not None and primary is None:
            raise cleanup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-cases", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--question-timeout-s", type=float, default=7200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.environ.update(ENVIRONMENT)
    run_dir = args.run_dir.resolve()
    if args.worker:
        asyncio.run(run_gpu(run_dir))
        return 0
    if args.resume:
        recover_pending(run_dir)
    else:
        if args.public_cases is None or args.preflight_dir is None:
            parser.error("Explicit blind public projection and passed preflight required")
        prepare(args)
    if not args.execute:
        return 0
    command = [sys.executable, "-B", "-m", "exec_exit.gen.blind_native", "--run-dir", str(run_dir), "--worker"]
    with (run_dir / "worker.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=os.environ.copy())
        save(run_dir / "supervisor.json", dict(status="RUNNING", supervisor_pid=os.getpid(),
            worker_pid=process.pid, started_epoch=time.time(), command=command))
        supervision = supervise_process(process, lambda: json.loads((run_dir / "progress.json").read_text()))
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.update(worker_returncode=process.returncode, supervisor_timeout=supervision["timed_out"],
        supervisor_deadline_policy=supervision)
    if process.returncode != 0 or supervision["timed_out"] or summary["status"] != "COMPLETE":
        summary["status"] = "FAILED_SUPERVISOR_TIMEOUT" if supervision["timed_out"] else "FAILED"
    save(run_dir / "summary.json", summary)
    sup = json.loads((run_dir / "supervisor.json").read_text())
    save(run_dir / "supervisor.json", sup | dict(status=summary["status"], ended_epoch=time.time(),
        worker_returncode=process.returncode, deadline_policy=supervision))
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
