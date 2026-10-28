"""Full v0 greedy probes on the H-v6 AtCoder blind cohort.

No online reflection-comment abort. Exact original candidates/prefixes are
selected by the explicit public-only blind projection. Original generation
modules are reused unchanged; source/index provenance is retained per probe.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from collections import Counter

from exec_exit.gen.engine_preflight import FORCED_ANSWER_CONTROL, MODEL, ROOT, VLLM, SNAPSHOT_RPC_METHOD, gpu_inventory, save, sha
from exec_exit.gen.traces import ENVIRONMENT, StreamWriter, atomic_jsonl, supervise_process, verify_preflight
from exec_exit.gen.probes import (generate_probe, sampling_for, bounded_queue, validate_identity,
    checkpoint_path, record_submission, submitted_ids, write_progress)
from exec_exit.gen.probes_v1 import load_candidates as load_all_candidates
from exec_exit.gen.dev_pipeline import NativeCounters
from exec_exit.gen.host_environment import collect_host_environment



def selected_cases(dataset_dir):
    from exec_exit.data.blind_hv6 import load_blind_cases
    dataset_dir=Path(dataset_dir).resolve()
    config=json.loads((dataset_dir/'config.yaml').read_text())
    path=dataset_dir/'data/runtime_tasks.json'
    if sha(path)!=config['runtime_tasks_sha256']:
        raise ValueError('Frozen blind runtime projection changed')
    cases=load_blind_cases(path)
    return cases,cases


def load_candidates(trajectory_dir, annotations_dir, dataset_dir, tokenizer):
    cases, selected = selected_cases(dataset_dir)
    trajectories = [json.loads(line) for line in (Path(trajectory_dir)/"raw/trajectories.jsonl").read_text().splitlines()]
    if [row["case_id"] for row in trajectories] != [row["case_id"] for row in cases]:
        raise ValueError("Native trajectories do not preserve the full selected blind manifest order")
    all_specs = load_all_candidates(trajectory_dir, annotations_dir, tokenizer, expected_cases=63)
    selected_ids = {row["case_id"] for row in selected}
    specs = []
    for source in all_specs:
        if source["case_id"] in selected_ids:
            specs.append(dict(source, source_candidate_index=source["candidate_index"],
                candidate_index=len(specs), source_cohort="hv6_atcoder"))
    seen_order = list(dict.fromkeys(spec["case_id"] for spec in specs))
    if seen_order != [row["case_id"] for row in selected if row["case_id"] in set(seen_order)]:
        raise ValueError("Selected candidate order differs from selected task order")
    return specs


def environment_audit(reference_config):
    """Read-only current machine/model comparison against the recent same-card run."""
    reference_config = Path(reference_config).resolve()
    reference = json.loads(reference_config.read_text())
    host, expected_host = collect_host_environment(), reference["host_environment"]
    fields = ("cpu_model", "affinity_logical_cpus", "affinity_cpu_ids", "cpu_quota", "hostname", "boot_id",
        "gpu_inventory", "software_versions", "python_executable", "python_version")
    checks = {key: dict(expected=expected_host.get(key, "UNKNOWN"), observed=host.get(key, "UNKNOWN"),
        matches=key in expected_host and host.get(key) == expected_host[key]) for key in fields}
    if host["collection_errors"] or not all(item["matches"] for item in checks.values()):
        raise RuntimeError("Machine/software identity differs: " + json.dumps(checks, ensure_ascii=False))
    if str(MODEL) != reference["model"]:
        raise RuntimeError("Stable model path differs from reference")
    artifacts = {}
    for path in [MODEL/"config.json", MODEL/"generation_config.json", MODEL/"tokenizer.json"]:
        expected, actual = reference["source_sha256"].get(str(path.resolve())), sha(path)
        if expected is None or actual != expected:
            raise RuntimeError("Model/config/tokenizer identity differs: " + str(path))
        artifacts[str(path.resolve())] = actual
    runner = reference["actual_runner_snapshot"]
    if sha(runner["runner_source"]) != runner["runner_source_sha256"]:
        raise RuntimeError("Installed model-runner source differs from reference")
    weights = [dict(name=path.name, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        for path in sorted(MODEL.glob("*.safetensors"))]
    if not weights:
        raise RuntimeError("No weights at stable model path")
    return dict(status="MATCH", reference_config=str(reference_config), reference_config_sha256=sha(reference_config),
        host_environment=host, host_checks=checks, model_realpath=str(MODEL.resolve()),
        model_artifact_sha256=artifacts, weight_file_metadata=weights,
        weight_identity_scope="Same stable resolved model directory and config/tokenizer; weight files stat-recorded, not rehashed",
        actual_model_runner=reference["actual_model_runner"], runner_source_sha256=runner["runner_source_sha256"])


def source_identity(args):
    paths = [Path(args.trajectory_dir)/name for name in ("config.yaml", "summary.json", "raw/trajectories.jsonl")]
    paths += [Path(args.annotations_dir)/name for name in ("config.yaml", "summary.json", "raw/annotations.jsonl")]
    paths += [Path(args.dataset_dir)/name for name in ("config.yaml", "summary.json", "data/runtime_tasks.json")]
    paths += [Path(args.preflight_dir)/name for name in ("config.yaml", "summary.json")]
    paths += [Path(args.reference_config), MODEL/"config.json", MODEL/"generation_config.json", MODEL/"tokenizer.json"]
    paths += [ROOT/"exec_exit/gen"/name for name in ("blind_probes_v0.py", "probes.py", "probes_v1.py",
        "traces.py", "engine_preflight.py", "dev_pipeline.py", "host_environment.py")]
    paths += [ROOT/"exec_exit/data/blind_hv6.py", ROOT/"exec_exit/data/heldout_inventory.py", ROOT/"exec_exit/replay/early_abort.py", ROOT/"exec_exit/analysis/trajectory_annotations.py"]
    paths += [VLLM/name for name in ("v1/metrics/stats.py", "v1/metrics/loggers.py", "v1/engine/async_llm.py")]
    return {str(path.resolve()): sha(path) for path in paths}


def prepare(args):
    from transformers import AutoTokenizer
    run_dir = args.run_dir.resolve()
    preflight = verify_preflight(args.preflight_dir)
    environment = environment_audit(args.reference_config)
    reference = json.loads(Path(args.reference_config).read_text())
    sample_index = json.loads((args.trajectory_dir / "config.yaml").read_text())["sample_index"]
    if type(sample_index) is not int or sample_index not in range(4):
        raise ValueError("Sample indices are 0 through 3")
    if preflight["engine"] != reference["engine"] or args.concurrency != 8:
        raise RuntimeError("Exact same engine and concurrency 8 required")
    if preflight["engine"].get("logprobs_mode") != "raw_logprobs":
        raise RuntimeError("Original raw_logprobs capture is required")
    observed = gpu_inventory(preflight["expected_gpu_uuid"])
    cases, selected = selected_cases(args.dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    specs = load_candidates(args.trajectory_dir, args.annotations_dir, args.dataset_dir, tokenizer)
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.resume:
            raise ValueError("Existing run; no overwrite or silent retry")
        config = json.loads((run_dir/"config.yaml").read_text())
        validate_identity(config)
        for key in ("trajectory_dir", "annotations_dir", "dataset_dir", "preflight_dir", "reference_config"):
            if str(Path(getattr(args, key)).resolve()) != config[key]:
                raise ValueError("Resume input path differs: " + key)
        if config["concurrency"] != args.concurrency:
            raise ValueError("Resume concurrency changed")
        manifest = json.loads((run_dir/"manifest.json").read_text())
        submitted = submitted_ids(run_dir, config["config_identity"])
        if [entry["probe_id"] for entry in manifest["probes"]] != [spec["probe_id"] for spec in specs]:
            raise ValueError("Resume denominator/order changed")
        for spec, entry in zip(specs, manifest["probes"]):
            path = checkpoint_path(run_dir, spec)
            if path.exists():
                row = json.loads(path.read_text())
                if (row["status"] != "COMPLETE" or row["config_identity"] != config["config_identity"]
                        or row["probe_id"] != spec["probe_id"]
                        or (entry.get("checkpoint_sha256") and sha(path) != entry["checkpoint_sha256"])):
                    raise ValueError("Resume checkpoint identity mismatch")
                entry.update(status="COMPLETE", checkpoint_sha256=sha(path), checkpoint=str(path.relative_to(run_dir)),
                    generated_tokens=row["call"]["token_count"], finish_reason=row["call"]["finish_reason"], stop_reason=row["call"]["stop_reason"])
            elif entry["status"] != "PENDING" or spec["probe_id"] in submitted:
                raise ValueError("Attempted incomplete probe cannot be silently retried")
        manifest["session_index"] += 1
        write_progress(run_dir, manifest, "PREPARED")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "checkpoints", "source"):
        (run_dir/name).mkdir()
    config = dict(schema="exec_exit.blind_full_probes.v0", model=str(MODEL),
        trajectory_dir=str(args.trajectory_dir.resolve()), annotations_dir=str(args.annotations_dir.resolve()),
        dataset_dir=str(args.dataset_dir.resolve()), preflight_dir=str(args.preflight_dir.resolve()),
        reference_config=str(args.reference_config.resolve()), concurrency=args.concurrency,
        global_submission_order="Canonical H-v6 AtCoder63 order; positions increasing; bounded refill queue",
        selected_cohort="hv6_atcoder", selected_task_ids=[row["case_id"] for row in selected],
        question_count=len(selected), source_question_count=len(cases), probe_count=len(specs),
        questions_with_candidates=len({spec["case_id"] for spec in specs}),
        selected_difficulty_counts=dict(Counter(row["difficulty"] for row in selected)),
        expected_gpu_uuid=preflight["expected_gpu_uuid"], observed_gpu_at_prepare=observed,
        engine=preflight["engine"], preflight_runner_snapshot=preflight["actual_runner_snapshot"],
        sample_index=sample_index,
        sampling_protocol=sampling_for(selected[0]["case_id"]) | {"seed": f"legacy seed_for(case_id,{sample_index},1)"},
        control_text=FORCED_ANSWER_CONTROL, control_token_ids=tokenizer.encode(FORCED_ANSWER_CONTROL, add_special_tokens=False),
        seed_rule="SHA256('reasoning_capacity_v1:{case_id}:"+str(sample_index)+":1') first 8 hex digits as integer",
        early_abort_protocol="DISABLED_V0_FULL_DRAFT", source_sha256=source_identity(args),
        host_environment=environment["host_environment"], environment_comparison=environment,
        probe_timeout_s=args.probe_timeout_s, startup_timeout_s=1800, terminal_exit_grace_s=30,
        environment=ENVIRONMENT, clock="host_perf_counter_ns; multi-token message shares actual receive time",
        raw_logprobs="Original v0 generate_probe, logprobs=1, actual rank-one values with native IDs",
        metrics_capture="Same native StatLogger as recent dev121; PREEMPTED event counts, not unique victims",
        submission_durability="Fsynced per-probe intent; incomplete attempted probes never silently retried",
        no_v1_suffix_reconstruction=True, no_claim_same_greedy_prefix_across_batching=True,
        public_test_payloads_read=False, private_test_payloads_read=False, heldout_question_text_read=True,
        question_programs_executed=0, blind_generation_only=True, no_total_run_quota=True)
    config["config_identity"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(run_dir/"config.yaml", config)
    (run_dir/"commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()+"\n")
    for path in map(Path, config["source_sha256"]):
        if path.suffix == ".py":
            (run_dir/"source"/str(path).replace("/", "_").lstrip("_")).write_bytes(path.read_bytes())
    manifest = dict(schema="exec_exit.blind_probe_manifest.v0", session_index=0,
        tasks=[dict(case_id=row["case_id"], difficulty=row["difficulty"], source_cohort=row["source_cohort"],
            planned_probes=sum(spec["case_id"] == row["case_id"] for spec in specs)) for row in selected],
        probes=[{key: spec[key] for key in ("probe_id", "case_id", "candidate_index", "source_candidate_index", "position_generated_tokens")}
            | dict(status="PENDING", seed=spec["sampling_parameters"]["seed"]) for spec in specs])
    write_progress(run_dir, manifest, "PREPARED")
    save(run_dir/"summary.json", dict(status="PREPARED", gpu_used=False, completed=0, planned=len(specs), selected_question_count=len(selected)))


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
    environment_audit(config["reference_config"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    specs = load_candidates(config["trajectory_dir"], config["annotations_dir"], config["dataset_dir"], tokenizer)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if [entry["probe_id"] for entry in manifest["probes"]] != [spec["probe_id"] for spec in specs]:
        raise RuntimeError("Frozen selected-cohort candidate population/order changed")
    if len(specs) != config["probe_count"]:
        raise RuntimeError("Frozen selected-cohort candidate count changed")
    prior_submitted = submitted_ids(run_dir, config["config_identity"])
    if any(entry["status"] != "COMPLETE" and entry["probe_id"] in prior_submitted for entry in manifest["probes"]):
        raise RuntimeError("Previously submitted incomplete probe cannot be reattempted")
    entries = {entry["probe_id"]: entry for entry in manifest["probes"]}
    writer = StreamWriter(run_dir / "raw" / f"stream_{manifest['session_index']}.jsonl")
    active, engine, reporter = {}, None, None
    observed_tokens = {entry["probe_id"]: entry.get("generated_tokens", 0)
                       for entry in manifest["probes"] if entry["status"] == "COMPLETE"}
    stop_reporter = asyncio.Event()
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    children0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    engine_start, generation_start, generation_end = None, None, None
    summary = dict(status="RUNNING", gpu_used=False, process_id=os.getpid(), started_epoch=time.time(),
                   completed=0, planned=len(specs), cuda_busy_seconds="UNKNOWN")
    save(run_dir / "summary.json", summary)
    result_status, failure = "FAILED", None
    try:
        observed = gpu_inventory(config["expected_gpu_uuid"])
        config["observed_gpu"] = observed
        save(run_dir / "config.yaml", config)
        write_progress(run_dir, manifest, "ENGINE_STARTING", deadline=time.time() + 1800)
        engine_start = time.perf_counter()
        summary["gpu_used"] = "UNKNOWN_ENGINE_STARTING"
        save(run_dir / "summary.json", summary)
        NativeCounters.instances = []
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=config["model"], **config["engine"]),
                                         usage_context=UsageContext.OPENAI_API_SERVER, stat_loggers=[NativeCounters])
        summary["gpu_used"] = True
        snapshots = await engine.collective_rpc(SNAPSHOT_RPC_METHOD)
        if len(snapshots) != 1:
            raise RuntimeError("Expected exactly one native worker")
        snapshot, expected = snapshots[0], config["preflight_runner_snapshot"]
        for field in ("runner_class", "runner_module", "runner_source_sha256", "use_v2_model_runner", "max_model_len",
                      "async_scheduling", "compile_mode", "cudagraph_mode", "enable_prefix_caching", "dtype",
                      "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes", "block_size"):
            if snapshot.get(field) != expected.get(field):
                raise RuntimeError("Native worker differs from passed preflight: " + field)
        config.update(actual_model_runner=snapshot["runner_class"], actual_runner_snapshot=snapshot)
        save(run_dir / "config.yaml", config)
        writer.add("worker_snapshot", snapshot=snapshot, gpu=observed)
        manifest["generation_started_epoch"] = time.time()
        generation_start = time.perf_counter()
        manifest["completed_before_session"] = sum(entry["status"] == "COMPLETE" for entry in manifest["probes"])
        manifest["tokens_before_session"] = sum(observed_tokens.values())

        def observe(request_id, tokens):
            active[request_id]["observed_tokens"] = tokens
            observed_tokens[request_id] = tokens
            if reporter is not None and reporter.done():
                error = reporter.exception()
                if error is not None:
                    raise RuntimeError("Progress reporter failed") from error

        async def launched(spec):
            entry = entries[spec["probe_id"]]
            if entry["status"] != "PENDING":
                raise RuntimeError("Only never-attempted probes may be submitted")
            await asyncio.to_thread(record_submission, run_dir, spec, config["config_identity"], manifest["session_index"])
            entry.update(status="RUNNING", started_epoch=time.time())
            active[spec["probe_id"]] = dict(candidate_index=spec["candidate_index"],
                started_epoch=entry["started_epoch"], deadline_epoch=time.time() + config["probe_timeout_s"], observed_tokens=0)

        async def operation(spec):
            try:
                factory = lambda params: SamplingParams(**params, output_kind=RequestOutputKind.DELTA)
                return await asyncio.wait_for(generate_probe(engine, tokenizer, writer, spec, factory, observe),
                                              timeout=config["probe_timeout_s"])
            except BaseException as error:
                entries[spec["probe_id"]].update(status="FAILED", error=repr(error))
                raise

        async def completed(spec, row):
            row["config_identity"] = config["config_identity"]
            path = checkpoint_path(run_dir, spec)
            await asyncio.to_thread(save, path, row)
            entries[spec["probe_id"]].update(status="COMPLETE", generated_tokens=row["call"]["token_count"],
                checkpoint=str(path.relative_to(run_dir)), checkpoint_sha256=sha(path),
                finish_reason=row["call"]["finish_reason"], stop_reason=row["call"]["stop_reason"])
            active.pop(spec["probe_id"], None)

        async def report():
            while not stop_reporter.is_set():
                deadline = min((item["deadline_epoch"] for item in active.values()), default=time.time() + config["probe_timeout_s"])
                # Build a compact state snapshot on the event loop; the file write
                # runs off-thread and never serializes full token/logprob arrays.
                frozen_manifest = json.loads(json.dumps(manifest))
                frozen_active = {key: dict(value) for key, value in active.items()}
                await asyncio.to_thread(write_progress, run_dir, frozen_manifest, "GENERATING",
                                        active=frozen_active, deadline=deadline)
                try:
                    await asyncio.wait_for(stop_reporter.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass

        pending = [spec for spec in specs if entries[spec["probe_id"]]["status"] != "COMPLETE"]
        reporter = asyncio.create_task(report())
        await bounded_queue(pending, config["concurrency"], operation, completed, launched)
        generation_end = time.perf_counter()
        if engine.output_processor.has_unfinished_requests():
            raise RuntimeError("Native requests remain after all probe calls ended")
        result_status = "COMPLETE"
    except BaseException as error:
        failure = repr(error)
        try:
            writer.add("failure", error=failure)
        except BaseException:
            pass
        raise
    finally:
        primary_error = sys.exc_info()[1]
        if generation_start is not None and generation_end is None:
            generation_end = time.perf_counter()
        cleanup_error = None
        if reporter is not None:
            # Await any outstanding file-write thread before the terminal write;
            # cancelling to_thread would otherwise allow stale progress to win.
            stop_reporter.set()
            try:
                await reporter
            except BaseException as error:
                cleanup_error = error
                summary["progress_recording_error"] = repr(error)
        drained = False
        try:
            if engine is not None:
                drained = not engine.output_processor.has_unfinished_requests()
                summary["native_metrics"] = NativeCounters.snapshot()
                engine.shutdown(timeout=15)
        except BaseException as error:
            cleanup_error = error
            summary["shutdown_error"] = repr(error)
        try:
            writer.close()
        except BaseException as error:
            cleanup_error = cleanup_error or error
            summary["recording_error"] = repr(error)
        if cleanup_error is not None:
            result_status = "FAILED_CLEANUP"
            failure = failure or repr(cleanup_error)
        done = [entry for entry in manifest["probes"] if entry["status"] == "COMPLETE"]
        rows = [json.loads((run_dir / entry["checkpoint"]).read_text()) for entry in done]
        await asyncio.to_thread(atomic_jsonl, run_dir / "raw/probes.jsonl", rows)
        generation_wall = generation_end - generation_start if generation_start is not None else None
        completed_tokens = sum(entry["generated_tokens"] for entry in done)
        session_tokens = completed_tokens - manifest.get("tokens_before_session", 0)
        cached_known = [row["call"]["num_cached_tokens"] for row in rows if row["call"]["num_cached_tokens"] is not None]
        usage1, children1 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        summary.update(status=result_status, error=failure, completed=len(done), planned=len(specs),
            generated_tokens_completed=completed_tokens,
            generated_tokens_observed_all_calls=sum(observed_tokens.values()),
            cancelled_inflight_unobserved_tokens="UNKNOWN_IF_ANY_CALL_FAILED_OR_WAS_CANCELLED",
            generation_stage_wall_seconds=generation_wall,
            session_completed_output_tokens=session_tokens,
            completed_output_tokens_per_generation_second=session_tokens / generation_wall if generation_wall else None,
            total_prompt_tokens_completed=sum(row["call"]["prompt_tokens"] for row in rows),
            native_cached_tokens_completed=sum(cached_known) if len(cached_known) == len(rows) else "UNKNOWN_INCOMPLETE_METADATA",
            native_cached_tokens_known_subtotal=sum(cached_known), native_cached_tokens_missing_calls=len(rows)-len(cached_known),
            native_frontend_drained=drained,
            ended_epoch=time.time(), process_wall_seconds=time.perf_counter() - started,
            engine_ownership_wall_seconds=time.perf_counter() - engine_start if engine_start is not None else None,
            cpu_user_seconds=usage1.ru_utime - usage0.ru_utime, cpu_system_seconds=usage1.ru_stime - usage0.ru_stime,
            waited_child_cpu_user_seconds=children1.ru_utime - children0.ru_utime,
            waited_child_cpu_system_seconds=children1.ru_stime - children0.ru_stime,
            resource_scope="Ownership wall time includes startup/CPU/wait; CUDA busy UNKNOWN; child CPU only waited children",
            recording=writer.statistics(), probes_truncated=sum(entry.get("finish_reason") == "length" for entry in done),
            logprob_alignment="PASS_FOR_COMPLETED_CHECKPOINTS", preemption_count=summary.get("native_metrics", {}).get("preemption_events", "UNKNOWN"),
            early_abort_protocol="DISABLED_V0_FULL_DRAFT",
            total_cost_status="OBSERVED_RECEIVED_OUTPUT_TOKENS; physical recomputation remains UNKNOWN",
            selected_question_count=config["question_count"],
            questions_with_candidates=config["questions_with_candidates"])
        save(run_dir / "summary.json", summary)
        write_progress(run_dir, manifest, result_status)
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--annotations-dir", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--reference-config", type=Path, default=ROOT/"runs/dev121_sample2_probes/config.yaml")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--probe-timeout-s", type=float, default=1800)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.environ.update(ENVIRONMENT)
    run_dir = args.run_dir.resolve()
    if args.worker:
        asyncio.run(run_gpu(run_dir))
        return 0
    if any(value is None for value in (args.trajectory_dir, args.annotations_dir, args.preflight_dir, args.dataset_dir, args.concurrency)):
        parser.error("Trajectory, annotations, passed preflight and explicit concurrency are required")
    prepare(args)
    if not args.execute:
        return 0
    command = [sys.executable, "-B", "-m", "exec_exit.gen.blind_probes_v0", "--run-dir", str(run_dir), "--worker"]
    with (run_dir / "worker.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ.copy())
        save(run_dir / "supervisor.json", dict(supervisor_pid=os.getpid(), worker_pid=process.pid,
             started_epoch=time.time(), command=command, status="RUNNING"))
        supervision = supervise_process(process, lambda: json.loads((run_dir / "progress.json").read_text()))
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.update(worker_returncode=process.returncode, supervisor_timeout=supervision["timed_out"],
                   supervisor_deadline_policy=supervision)
    if process.returncode != 0 or supervision["timed_out"] or summary["status"] != "COMPLETE":
        summary["status"] = "FAILED_SUPERVISOR_TIMEOUT" if supervision["timed_out"] else "FAILED"
    save(run_dir / "summary.json", summary)
    supervisor = json.loads((run_dir / "supervisor.json").read_text())
    supervisor.update(ended_epoch=time.time(), worker_returncode=process.returncode,
                      status=summary["status"], deadline_policy=supervision)
    save(run_dir / "supervisor.json", supervisor)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
