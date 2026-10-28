"""Greedy drafts at frozen native-trajectory candidates, without scoring.

Submission order is fixed. A global bounded in-flight queue is refilled after
completion, without a question barrier or GPU/source modifications. Raw native
token/logprob records and completed checkpoints are required, never inferred.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from exec_exit.gen.engine_preflight import (
    FORCED_ANSWER_CONTROL, MODEL, ROOT, SNAPSHOT_RPC_METHOD, gpu_inventory,
    save, seed_for, serialize_logprobs, sha, validate_prompt_token_ids,
)
from exec_exit.gen.traces import ENVIRONMENT, StreamWriter, atomic_jsonl, supervise_process, verify_preflight


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def probe_id(case_id, position):
    if not case_id.startswith("atcoder:") or not case_id.removeprefix("atcoder:").replace("_", "").isalnum():
        raise ValueError("Unexpected public task identity")
    return case_id.replace(":", "_") + f".p{position:05d}"


def sampling_for(case_id):
    return dict(temperature=0.0, top_p=0.95, top_k=20, min_p=0.0,
        min_tokens=0, ignore_eos=False, max_tokens=2048,
        seed=seed_for(case_id, 0, 1), logprobs=1, stop=["```"],
        include_stop_str_in_output=True, skip_special_tokens=True)


def load_candidates(trajectory_dir, annotations_dir, tokenizer, *, expected_probes=386, expected_cases=36):
    trajectory_dir, annotations_dir = Path(trajectory_dir).resolve(), Path(annotations_dir).resolve()
    trajectory_path = trajectory_dir / "raw/trajectories.jsonl"
    annotations_path = annotations_dir / "raw/annotations.jsonl"
    annotation_config = json.loads((annotations_dir / "config.yaml").read_text())
    snapshot = annotation_config["trajectory_snapshot"]
    if Path(snapshot["path"]).resolve() != trajectory_path or sha(trajectory_path) != snapshot["snapshot_sha256"]:
        raise ValueError("Annotations do not identify this exact trajectory snapshot")
    trajectories = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    annotations = [json.loads(line) for line in annotations_path.read_text().splitlines()]
    if len(trajectories) != expected_cases or len(annotations) != expected_cases:
        raise ValueError("The complete frozen question population is required")
    if len({row["case_id"] for row in trajectories}) != expected_cases:
        raise ValueError("Duplicate trajectory identity")
    control_ids = tokenizer.encode(FORCED_ANSWER_CONTROL, add_special_tokens=False)
    validate_prompt_token_ids(control_ids)
    if control_ids[0] != annotation_config["thinking_close_token_id"]:
        raise ValueError("Forced control tokenizer identity differs from annotations")
    specs = []
    for source, annotation in zip(trajectories, annotations):
        case_id = source["case_id"]
        if (source.get("status") != "COMPLETE" or annotation.get("status") != "COMPLETE"
                or annotation["case_id"] != case_id or annotation["task_id"] != source["task_id"]):
            raise ValueError("Question ordering or completed source identity mismatch")
        calls = [call for call in source["calls"] if call["stage"] == 0]
        if len(calls) != 1 or not calls[0]["native_finished"] or calls[0].get("error"):
            raise ValueError("Missing or incomplete native thinking call")
        thinking = calls[0]["token_ids"]
        if token_digest(thinking) != annotation["stage0_token_ids_sha256"]:
            raise ValueError("Thinking token IDs differ from annotated IDs")
        original_prompt = source["prompt_token_ids"]
        validate_prompt_token_ids(original_prompt)
        previous = 0
        for candidate in annotation["candidates"]:
            position = candidate["position_generated_tokens"]
            if type(position) is not int or not previous < position <= annotation["eligible_thinking_generated_tokens"]:
                raise ValueError("Candidates must be unique, increasing, and still within thinking")
            previous = position
            ids = original_prompt + thinking[:position] + control_ids
            validate_prompt_token_ids(ids)
            if len(ids) + 2048 > 40960:
                raise ValueError("Probe exceeds frozen capacity; no truncation or candidate deletion")
            specs.append(dict(probe_id=probe_id(case_id, position), case_id=case_id,
                task_id=source["task_id"], position_generated_tokens=position,
                candidate_index=len(specs), sample_index=0, candidate_sources=candidate["sources"],
                control_token_ids=control_ids, control_text=FORCED_ANSWER_CONTROL,
                prompt_token_ids=ids, prompt_token_ids_sha256=token_digest(ids),
                source_stage0_sha256=annotation["stage0_token_ids_sha256"],
                sampling_parameters=sampling_for(case_id)))
    if len(specs) != expected_probes:
        raise ValueError(f"Expected all {expected_probes} candidates, found {len(specs)}")
    return specs


def rank_one_values(ids, probabilities):
    if not ids:
        if probabilities not in (None, []):
            raise ValueError("Logprobs without generated token IDs")
        return []
    if probabilities is None or len(probabilities) != len(ids):
        raise ValueError("Missing or misaligned native per-token logprobs")
    values = []
    for token_id, row in zip(ids, probabilities):
        if str(token_id) not in row:
            raise ValueError("Actual generated token is absent from native logprobs")
        best = [item["logprob"] for item in row.values() if item["rank"] == 1]
        if not best or any(value != best[0] for value in best[1:]):
            raise ValueError("Native rank-one logprob is missing or inconsistent")
        if any(not math.isfinite(item["logprob"]) or item["logprob"] > 0 for item in row.values()):
            raise ValueError("Native logprob is non-finite or positive")
        values.append(best[0])
    return values


def native_metrics(output):
    metrics = getattr(output, "metrics", None)
    names = ("arrival_time", "queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts", "first_token_latency")
    return {name: getattr(metrics, name, "UNKNOWN") if metrics is not None else "UNKNOWN" for name in names}


async def generate_probe(engine, tokenizer, writer, spec, parameter_factory, observe=lambda *args: None):
    prompt = spec["prompt_token_ids"]
    validate_prompt_token_ids(prompt)
    request_id, started = spec["probe_id"], time.perf_counter_ns()
    call = dict(request_id=request_id, prompt_token_ids=prompt, prompt_tokens=len(prompt),
        sampling_parameters=spec["sampling_parameters"], seed=spec["sampling_parameters"]["seed"],
        started_ns=started, token_ids=[], raw_text="", logprobs=[], rank_one_logprobs=[],
        returned_rank_one_entry_counts=[],
        token_observation_ns=[], native_finished=False, finish_reason=None, stop_reason=None,
        num_cached_tokens=None, first_output_ns=None, last_output_ns=None, error=None,
        preemptions="UNKNOWN_NOT_EXPOSED_BY_UNCHANGED_NATIVE_OUTPUT")
    writer.add("probe_start", probe_id=request_id, case_id=spec["case_id"],
        candidate_index=spec["candidate_index"], position_generated_tokens=spec["position_generated_tokens"],
        prompt_token_ids=prompt, control_token_ids=spec["control_token_ids"],
        sampling_parameters=spec["sampling_parameters"])
    texts = []
    try:
        parameters = parameter_factory(spec["sampling_parameters"])
        call["effective_sampling_parameters"] = {
            name: parameters.get(name) if isinstance(parameters, dict) else getattr(parameters, name)
            for name in spec["sampling_parameters"]}
        async for output in engine.generate({"prompt_token_ids": prompt},
                                             parameters, request_id):
            received = time.perf_counter_ns()
            if len(output.outputs) != 1:
                raise RuntimeError("Expected exactly one completion per probe")
            completion = output.outputs[0]
            ids, offset = list(completion.token_ids), len(call["token_ids"])
            call["token_ids"].extend(ids)
            call["token_observation_ns"].extend([received] * len(ids))
            texts.append(completion.text)
            call.update(native_finished=bool(output.finished), finish_reason=completion.finish_reason,
                        stop_reason=completion.stop_reason, last_output_ns=received)
            cached = getattr(output, "num_cached_tokens", None)
            if cached is not None:
                call["num_cached_tokens"] = int(cached)
            if ids and call["first_output_ns"] is None:
                call["first_output_ns"] = received
            observe(request_id, len(call["token_ids"]))
            try:
                probabilities = serialize_logprobs(completion.logprobs)
                ranks = rank_one_values(ids, probabilities)
                rank_counts = [sum(item["rank"] == 1 for item in row.values()) for row in (probabilities or [])]
            except Exception as error:
                writer.add("probe_output_invalid_logprobs", probe_id=request_id,
                    token_offset=offset, token_ids=ids, text=completion.text,
                    native_frontend_receive_ns=received, error=repr(error),
                    raw_native_logprobs_repr=repr(completion.logprobs))
                raise
            call["logprobs"].extend(probabilities or [])
            call["rank_one_logprobs"].extend(ranks)
            call["returned_rank_one_entry_counts"].extend(rank_counts)
            writer.add("probe_output", probe_id=request_id, token_offset=offset,
                token_ids=ids, text=completion.text, logprobs=probabilities,
                rank_one_logprobs=ranks, returned_rank_one_entry_counts=rank_counts,
                native_frontend_receive_ns=received,
                num_cached_tokens=cached, finished=output.finished,
                finish_reason=completion.finish_reason, stop_reason=completion.stop_reason)
            if output.finished:
                call["native_metrics"] = native_metrics(output)
        if not call["native_finished"]:
            raise RuntimeError("Probe stream ended without a native finished event")
        count = len(call["token_ids"])
        if count == 0 or count > 2048 or len(call["rank_one_logprobs"]) != count:
            raise RuntimeError("Probe token count or native logprob alignment failed")
    except BaseException as error:
        call["error"] = repr(error)
        try:
            await asyncio.wait_for(engine.abort(request_id), timeout=15)
        except BaseException as abort_error:
            call["abort_error"] = repr(abort_error)
        raise
    finally:
        call.update(ended_ns=time.perf_counter_ns(), raw_text="".join(texts),
            token_count=len(call["token_ids"]),
            canonical_text=tokenizer.decode(call["token_ids"], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False),
            canonical_text_with_special_tokens=tokenizer.decode(call["token_ids"], skip_special_tokens=False,
                                                               clean_up_tokenization_spaces=False))
        call["first_50_visible_ns"] = call["token_observation_ns"][49] if len(call["token_ids"]) >= 50 else "NOT_REACHED"
        call["returned_rank_one_tie_positions_zero_based"] = [
            index for index, count in enumerate(call["returned_rank_one_entry_counts"]) if count > 1]
        writer.add("probe_end", probe_id=request_id, case_id=spec["case_id"], call=call)
    return {key: value for key, value in spec.items() if key not in ("prompt_token_ids", "sampling_parameters")} | dict(
        schema="exec_exit.probe.v1", status="COMPLETE", call=call, correctness="NOT_EVALUATED")


async def bounded_queue(specs, concurrency, operation, completed, launched=lambda spec: None):
    """Submit the fixed sequence with at most concurrency actual live tasks."""
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    pending, active = iter(specs), {}

    async def fill():
        while len(active) < concurrency:
            spec = next(pending, None)
            if spec is None:
                break
            launching = launched(spec)
            if inspect.isawaitable(launching):
                await launching
            active[asyncio.create_task(operation(spec))] = spec

    try:
        await fill()
        while active:
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            ordered = sorted(done, key=lambda task: active[task]["candidate_index"])
            # Resolve all ready tasks before refilling; any failure stops new submission.
            results, failure = [], None
            for task in ordered:
                spec = active.pop(task)
                try:
                    results.append((spec, task.result()))
                except BaseException as error:
                    failure = failure or error
            for spec, result in results:
                await completed(spec, result)
            if failure is not None:
                raise failure
            await fill()
    finally:
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)


def source_identity(trajectory_dir, annotations_dir, preflight_dir):
    paths = [Path(trajectory_dir) / name for name in ("config.yaml", "raw/trajectories.jsonl")]
    paths += [Path(annotations_dir) / name for name in ("config.yaml", "raw/annotations.jsonl")]
    paths += [Path(preflight_dir) / name for name in ("config.yaml", "summary.json")]
    paths += [Path(__file__), ROOT / "exec_exit/gen/traces.py", ROOT / "exec_exit/gen/engine_preflight.py",
              MODEL / "config.json", MODEL / "generation_config.json", MODEL / "tokenizer.json"]
    return {str(path.resolve()): sha(path) for path in paths}


def validate_identity(config):
    for path, expected in config["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Frozen probe source/input changed: " + path)


def checkpoint_path(run_dir, spec):
    return run_dir / "checkpoints" / (spec["probe_id"] + ".json")


def record_submission(run_dir, spec, config_identity, session_index):
    """Persist a tiny intent before submit, so an interrupted call cannot retry."""
    row = dict(probe_id=spec["probe_id"], candidate_index=spec["candidate_index"],
               config_identity=config_identity, session_index=session_index, epoch=time.time())
    with (run_dir / "raw/submissions.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def submitted_ids(run_dir, config_identity):
    path = run_dir / "raw/submissions.jsonl"
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)  # Torn records stop recovery; never guess safe retry.
        if row["config_identity"] != config_identity or row["probe_id"] in result:
            raise ValueError("Submission journal identity mismatch or duplicate attempt")
        result.add(row["probe_id"])
    return result


def write_progress(run_dir, manifest, status, *, active=None, deadline=None):
    done = [entry for entry in manifest["probes"] if entry["status"] == "COMPLETE"]
    elapsed = time.time() - manifest.get("generation_started_epoch", time.time())
    completed_tokens = sum(entry.get("generated_tokens", 0) for entry in done)
    session_completed = len(done) - manifest.get("completed_before_session", 0)
    session_tokens = completed_tokens - manifest.get("tokens_before_session", 0)
    remaining = len(manifest["probes"]) - len(done)
    progress = dict(status=status, completed=len(done), planned=len(manifest["probes"]),
        completed_tokens=completed_tokens, active=active or {}, updated_epoch=time.time(),
        completed_tokens_per_generation_wall_s=session_tokens / elapsed if elapsed > 0 else None,
        eta_remaining_s=elapsed / session_completed * remaining if session_completed > 0 else None,
        eta_basis="Completed probes per wall time including queue/capture; heterogeneous lengths make early estimates uncertain")
    if deadline is not None:
        progress["phase_deadline_epoch"] = deadline
    save(run_dir / "manifest.json", manifest)
    save(run_dir / "progress.json", progress)


def prepare(args):
    from transformers import AutoTokenizer
    run_dir = args.run_dir.resolve()
    preflight = verify_preflight(args.preflight_dir)
    if preflight["engine"].get("logprobs_mode") != "raw_logprobs":
        raise RuntimeError("Passed preflight must use original raw_logprobs mode")
    if not 1 <= args.concurrency <= preflight["engine"]["max_num_seqs"]:
        raise ValueError("Explicit concurrency exceeds the unchanged engine capacity")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    specs = load_candidates(args.trajectory_dir, args.annotations_dir, tokenizer)
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.resume:
            raise ValueError("Existing run; no overwrite or automatic retry")
        config = json.loads((run_dir / "config.yaml").read_text())
        validate_identity(config)
        if config["concurrency"] != args.concurrency:
            raise ValueError("Resume concurrency differs from frozen configuration")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        submitted = submitted_ids(run_dir, config["config_identity"])
        if [entry["probe_id"] for entry in manifest["probes"]] != [spec["probe_id"] for spec in specs]:
            raise ValueError("Resume candidate denominator/order changed")
        for spec, entry in zip(specs, manifest["probes"]):
            path = checkpoint_path(run_dir, spec)
            if path.exists():
                row = json.loads(path.read_text())
                if (row["status"] != "COMPLETE" or row["config_identity"] != config["config_identity"]
                        or row["probe_id"] != spec["probe_id"]
                        or (entry.get("checkpoint_sha256") and sha(path) != entry["checkpoint_sha256"])):
                    raise ValueError("Resume checkpoint identity mismatch")
                entry.update(status="COMPLETE", checkpoint_sha256=sha(path),
                             checkpoint=str(path.relative_to(run_dir)),
                             generated_tokens=row["call"]["token_count"],
                             finish_reason=row["call"]["finish_reason"], stop_reason=row["call"]["stop_reason"])
            elif entry["status"] != "PENDING" or spec["probe_id"] in submitted:
                raise ValueError("Failed/interrupted probe cannot be silently retried")
        manifest["session_index"] += 1
        write_progress(run_dir, manifest, "PREPARED")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "checkpoints", "source"):
        (run_dir / name).mkdir()
    config = dict(schema="exec_exit.probes_config.v1", model=str(MODEL),
        trajectory_dir=str(args.trajectory_dir.resolve()), annotations_dir=str(args.annotations_dir.resolve()),
        preflight_dir=str(args.preflight_dir.resolve()), concurrency=args.concurrency,
        global_submission_order="Original 36 question order, each question's positions increasing; refill bounded queue on completion, no question barrier",
        probe_count=len(specs), question_count=36, expected_gpu_uuid=preflight["expected_gpu_uuid"],
        engine=preflight["engine"], preflight_runner_snapshot=preflight["actual_runner_snapshot"],
        sampling_protocol=sampling_for("atcoder:abc390_a") | {"seed": "legacy seed_for(case_id,0,1); same seed for all positions of one question"},
        control_text=FORCED_ANSWER_CONTROL, control_token_ids=specs[0]["control_token_ids"],
        seed_rule="SHA256('reasoning_capacity_v1:{case_id}:0:1') first 8 hex digits as integer",
        source_sha256=source_identity(args.trajectory_dir, args.annotations_dir, args.preflight_dir),
        probe_timeout_s=args.probe_timeout_s, startup_timeout_s=1800, terminal_exit_grace_s=30,
        environment=ENVIRONMENT, clock="host_perf_counter_ns; multi-token output shares one observed receive time",
        raw_logprobs="Unmodified native logprobs=1; capture rank-one before policy filtering, no text reconstruction",
        rank_one_ties="Equal returned rank-one values accepted; returned entry count is not a full-vocabulary tie count",
        submission_durability="One off-thread fsynced compact intent before each submission; no retry without a complete checkpoint",
        source_run_process_status=json.loads((args.trajectory_dir / "summary.json").read_text())["status"],
        source_process_status_not_reclassified=True, no_gpu_state_or_online_equivalence_claim=True)
    config["config_identity"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(run_dir / "config.yaml", config)
    (run_dir / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + "\n")
    for name in ("probes.py", "traces.py", "engine_preflight.py"):
        (run_dir / "source" / name).write_bytes((ROOT / "exec_exit/gen" / name).read_bytes())
    manifest = dict(schema="exec_exit.probes_manifest.v1", session_index=0,
        probes=[{key: spec[key] for key in ("probe_id", "case_id", "candidate_index", "position_generated_tokens")}
                | dict(status="PENDING", seed=spec["sampling_parameters"]["seed"]) for spec in specs])
    write_progress(run_dir, manifest, "PREPARED")
    save(run_dir / "summary.json", dict(status="PREPARED", gpu_used=False, completed=0, planned=len(specs)))


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
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    specs = load_candidates(config["trajectory_dir"], config["annotations_dir"], tokenizer)
    manifest = json.loads((run_dir / "manifest.json").read_text())
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
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=config["model"], **config["engine"]),
                                         usage_context=UsageContext.OPENAI_API_SERVER)
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
            logprob_alignment="PASS_FOR_COMPLETED_CHECKPOINTS", preemption_count="UNKNOWN_UNCHANGED_NATIVE_OUTPUT")
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
    parser.add_argument("--concurrency", type=int)
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
    if any(value is None for value in (args.trajectory_dir, args.annotations_dir, args.preflight_dir, args.concurrency)):
        parser.error("Trajectory, annotations, passed preflight and explicit concurrency are required")
    prepare(args)
    if not args.execute:
        return 0
    command = [sys.executable, "-B", "-m", "exec_exit.gen.probes", "--run-dir", str(run_dir), "--worker"]
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
