"""Serial two-segment Qwen3 trajectories; no probes, annotations, or scoring.

The five-part real GPU preflight must pass before execution. CPU helpers and
fake-engine tests import this module without importing CUDA.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import resource
import signal
import subprocess
import sys
import threading
import time

from exec_exit.data.load_lcb import load_runtime_tasks, legacy_source_identity
from exec_exit.gen.engine_preflight import (
    CHECKS, ROOT, MODEL, MODELS, VLLM, SNAPSHOT_RPC_METHOD, gpu_inventory, save, sha,
)

LEGACY = ROOT / "research/early_exit_component/run/scripts/legacy_workflow.py"
ENVIRONMENT = {
    "EXECEXIT_MODELS": str(MODELS), "EXECEXIT_VLLM_SOURCE": str(VLLM),
    "PATH": str(Path(sys.executable).parent) + os.pathsep + os.pathsep.join(
        part for part in os.environ.get("PATH", "").split(os.pathsep)
        if part != str(Path(sys.executable).parent)),
    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_CACHE_ROOT": str(ROOT / "research/mixed_batch_upgrade_5090/engine_cache"),
    "TORCHINDUCTOR_CACHE_DIR": str(ROOT / "research/mixed_batch_upgrade_5090/engine_cache/torchinductor"),
}


@dataclass(frozen=True)
class Protocol:
    thinking_cap: int = 28000
    model_capacity: int = 40960
    forced_answer_cap: int = 2048
    forced_control: str = "</think>\n\n```python\n"


def legacy_helpers():
    spec = importlib.util.spec_from_file_location("exec_exit_legacy_messages", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.messages, module.seed_for


def params_for(seed, cap, *, think_id=None, forced=False):
    result = dict(temperature=.6, top_p=.95, top_k=20, min_p=0, min_tokens=0,
                  ignore_eos=False, seed=seed, max_tokens=cap, logprobs=None,
                  skip_special_tokens=True, include_stop_str_in_output=True)
    if think_id is not None:
        result["stop_token_ids"] = [think_id]
    if forced:
        result["stop"] = ["```"]
    return result


def render_prompt(tokenizer, case):
    """Exactly the old N render-then-encode path, including current tokenizer API."""
    messages, _ = legacy_helpers()
    rendered = tokenizer.apply_chat_template(messages(case), tokenize=False,
                    add_generation_prompt=True, enable_thinking=True)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not isinstance(rendered, str) or not isinstance(ids, list) or not all(type(x) is int for x in ids):
        raise RuntimeError("Expected explicit native rendered prompt and integer token IDs")
    return rendered, ids


def continuation(first, *, prompt_ids, close_id, eos_ids, tokenizer, protocol=Protocol()):
    """Choose only the specified branch; actual stop tokens outrank cap flags."""
    ids = first["token_ids"]
    if not first["native_finished"] or first.get("error"):
        raise RuntimeError("Incomplete first native segment; no continuation is inferred")
    if len(ids) > protocol.thinking_cap:
        raise RuntimeError("Native first segment exceeded the thinking cap")
    if ids and ids[-1] == close_id:
        if close_id in ids[:-1]:
            raise RuntimeError("Repeated thinking-close token in the stopped segment")
        prefix = prompt_ids + ids
        remaining = protocol.model_capacity - len(prefix)
        if remaining <= 0:
            raise RuntimeError("Natural-answer continuation has no context capacity")
        return dict(branch="natural_thinking_close", prefix=prefix, control=[],
                    max_tokens=remaining, forced=False, thinking_end_token_index=len(ids)-1)
    if ids and ids[-1] in eos_ids:
        return dict(branch="unexpected_eos_in_thinking", prefix=None, control=[],
                    max_tokens=0, forced=False, thinking_end_token_index=None)
    if close_id in ids:
        raise RuntimeError("Thinking stop token was not the last observed token")
    if len(ids) == protocol.thinking_cap and first["finish_reason"] == "length":
        control = tokenizer.encode(protocol.forced_control, add_special_tokens=False)
        if not control or control[0] != close_id:
            raise RuntimeError("Forced-answer control has the wrong tokenizer boundary")
        prefix = prompt_ids + ids + control
        if len(prefix) + protocol.forced_answer_cap > protocol.model_capacity:
            raise RuntimeError("Forced-answer request exceeds capacity; no silent truncation")
        return dict(branch="forced_at_cap", prefix=prefix, control=control,
                    max_tokens=protocol.forced_answer_cap, forced=True, thinking_end_token_index=None)
    raise RuntimeError("Unspecified first-segment termination; preserved without inventing a branch")


class StreamWriter:
    """Bounded background JSONL writer; flush <= 1 s, never silently drop rows."""
    def __init__(self, path):
        self.path, self.q = Path(path), queue.Queue(maxsize=4096)
        self.error, self.accepted, self.written, self.rejected = None, 0, 0, 0
        self.cpu_s = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, event, **fields):
        if self.error:
            raise RuntimeError(self.error)
        row = dict(event=event, at_ns=time.perf_counter_ns(), epoch_ns=time.time_ns(), **fields)
        try:
            self.q.put_nowait(row)
            self.accepted += 1
        except queue.Full:
            self.rejected += 1
            raise RuntimeError("Raw recording queue full; stopping instead of losing observations")

    def _run(self):
        cpu = time.thread_time()
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                flushed = time.monotonic()
                while True:
                    try:
                        row = self.q.get(timeout=1.0)
                    except queue.Empty:
                        stream.flush()
                        flushed = time.monotonic()
                        continue
                    if row is None:
                        stream.flush()
                        os.fsync(stream.fileno())
                        break
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    self.written += 1
                    if time.monotonic() - flushed >= 1.0:
                        stream.flush()
                        flushed = time.monotonic()
        except BaseException as error:
            self.error = repr(error)
        finally:
            self.cpu_s = time.thread_time() - cpu

    def close(self):
        if self.thread.is_alive():
            self.q.put(None, timeout=15)
            self.thread.join(timeout=15)
        if self.thread.is_alive() or self.error or self.accepted != self.written:
            raise RuntimeError("Raw writer did not drain: " + repr(self.statistics()))

    def statistics(self):
        return dict(accepted=self.accepted, written=self.written, rejected=self.rejected,
                    error=self.error, cpu_seconds=self.cpu_s, flush_interval_seconds=1,
                    abrupt_loss="Active queue or unflushed OS buffer may be missing after a hard kill")


async def generate_call(engine, tokenizer, writer, *, case_id, stage, kind, prompt_ids,
                        sampling, parameter_factory, progress):
    request_id = case_id.replace(":", "_") + f".stage{stage}"
    started = time.perf_counter_ns()
    row = dict(stage=stage, kind=kind, request_id=request_id, seed=sampling["seed"],
               sampling_parameters=sampling, prompt_tokens=len(prompt_ids),
               started_ns=started, token_ids=[], raw_text="", native_finished=False,
               finish_reason=None, stop_reason=None, num_cached_tokens=None,
               first_output_ns=None, last_output_ns=None, error=None)
    writer.add("call_start", case_id=case_id, request_id=request_id, stage=stage,
               kind=kind, prompt_token_ids=prompt_ids, sampling_parameters=sampling)
    texts = []
    try:
        async for output in engine.generate({"prompt_token_ids": prompt_ids},
                                             parameter_factory(sampling), request_id):
            received = time.perf_counter_ns()
            if len(output.outputs) != 1:
                raise RuntimeError("Expected one native completion per serial request")
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            offset = len(row["token_ids"])
            row["token_ids"].extend(ids)
            texts.append(completion.text)
            row["native_finished"] = bool(output.finished)
            row["finish_reason"] = completion.finish_reason
            row["stop_reason"] = completion.stop_reason
            cached = getattr(output, "num_cached_tokens", None)
            if cached is not None:
                row["num_cached_tokens"] = int(cached)
            if ids and row["first_output_ns"] is None:
                row["first_output_ns"] = received
            row["last_output_ns"] = received
            writer.add("call_output", case_id=case_id, request_id=request_id, stage=stage,
                       token_offset=offset, token_ids=ids, text=completion.text,
                       native_frontend_receive_ns=received, num_cached_tokens=cached,
                       finished=output.finished, finish_reason=completion.finish_reason,
                       stop_reason=completion.stop_reason)
            progress(stage, kind, len(row["token_ids"]), sampling["max_tokens"])
        if not row["native_finished"]:
            raise RuntimeError("Native output stream ended without a finished event")
        if len(row["token_ids"]) > sampling["max_tokens"]:
            raise RuntimeError("Native request exceeded its configured output cap")
    except BaseException as error:
        row["error"] = repr(error)
        try:
            await asyncio.wait_for(engine.abort(request_id), timeout=15)
        except BaseException as abort_error:
            row["abort_error"] = repr(abort_error)
        raise
    finally:
        row.update(ended_ns=time.perf_counter_ns(), raw_text="".join(texts),
                   token_count=len(row["token_ids"]),
                   canonical_text=tokenizer.decode(row["token_ids"], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False),
                   canonical_text_with_special_tokens=tokenizer.decode(
                       row["token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        writer.add("call_end", case_id=case_id, **row)
    return row


async def generate_case(engine, tokenizer, writer, case, eos_ids, parameter_factory,
                        progress=lambda *args: None, protocol=Protocol(), *, sample_index=0):
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("Nonnegative integer sample_index required")
    _, seed_for = legacy_helpers()
    rendered, prompt = render_prompt(tokenizer, case)
    if len(prompt) + protocol.thinking_cap > protocol.model_capacity:
        raise RuntimeError("Question prompt plus thinking cap exceeds capacity")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(close_id, int) or close_id in eos_ids:
        raise RuntimeError("Invalid thinking-close token identity")
    first = await generate_call(engine, tokenizer, writer, case_id=case["case_id"], stage=0,
        kind="thinking", prompt_ids=prompt, parameter_factory=parameter_factory, progress=progress,
        sampling=params_for(seed_for(case["case_id"], sample_index, 0), protocol.thinking_cap, think_id=close_id))
    decision = continuation(first, prompt_ids=prompt, close_id=close_id, eos_ids=eos_ids,
                            tokenizer=tokenizer, protocol=protocol)
    calls = [first]
    if decision["prefix"] is not None:
        calls.append(await generate_call(engine, tokenizer, writer, case_id=case["case_id"], stage=1,
            kind="forced_answer" if decision["forced"] else "natural_answer",
            prompt_ids=decision["prefix"], parameter_factory=parameter_factory, progress=progress,
            sampling=params_for(seed_for(case["case_id"], sample_index, 1), decision["max_tokens"], forced=decision["forced"])))
    last = calls[-1]
    return dict(event="trajectory_complete", schema="exec_exit.trajectory.v1", status="COMPLETE",
                case_id=case["case_id"], task_id=case["task_id"], prompt_token_ids=prompt,
                prompt_text=rendered,
                prompt_tokens=len(prompt), calls=calls, branch=decision["branch"],
                injected_control_token_ids=decision["control"],
                injected_control_text=protocol.forced_control if decision["forced"] else "",
                thinking_end_token_index=decision["thinking_end_token_index"],
                thinking_budget_hit=len(first["token_ids"]) == protocol.thinking_cap,
                answer_truncated=len(calls) == 2 and last["finish_reason"] == "length",
                answer_available=len(calls) == 2,
                terminal_eos_observed=bool(last["token_ids"] and last["token_ids"][-1] in eos_ids),
                generated_tokens=sum(call["token_count"] for call in calls),
                injected_control_tokens=len(decision["control"]),
                started_ns=first["started_ns"], ended_ns=last["ended_ns"],
                correctness="NOT_EVALUATED")


def verify_preflight(directory):
    directory = Path(directory).resolve()
    summary = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "config.yaml").read_text())
    if (summary.get("status") != "PASS" or not summary.get("gpu_used")
            or any(summary.get("checks", {}).get(name, {}).get("status") != "PASS" for name in CHECKS)):
        raise RuntimeError("All five real GPU preflight checks must PASS before trajectories")
    if config.get("budget_protocol") != "externally_segmented":
        raise RuntimeError("Preflight did not validate the selected two-segment protocol")
    snapshot = config.get("actual_runner_snapshot", {})
    if (snapshot.get("max_model_len") != 40960 or not snapshot.get("use_v2_model_runner")
            or "." not in config.get("actual_model_runner", "")):
        raise RuntimeError("Preflight is missing actual runner identity or capacity")
    return config


def atomic_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint_path(run_dir, case_id):
    if not case_id.startswith("atcoder:") or not case_id.removeprefix("atcoder:").replace("_", "").isalnum():
        raise ValueError("Unexpected original Dev36 case identity")
    return run_dir / "checkpoints" / (case_id.replace(":", "_") + ".json")


def update_progress(run_dir, manifest, status, **extra):
    done = [entry for entry in manifest["tasks"] if entry["status"] == "COMPLETE"]
    seconds = [entry["elapsed_s"] for entry in done]
    remaining = len(manifest["tasks"]) - len(done)
    estimate = sum(seconds) / len(seconds) * remaining if seconds else None
    progress = dict(status=status, completed=len(done), planned=len(manifest["tasks"]),
                    eta_remaining_seconds=estimate,
                    eta_basis="Mean completed-question wall time; early estimate may be unstable",
                    updated_epoch=time.time(), **extra)
    manifest["progress"] = progress
    save(run_dir / "manifest.json", manifest)
    save(run_dir / "progress.json", progress)


async def run_gpu(run_dir):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.usage.usage_lib import UsageContext
    config = json.loads((run_dir / "config.yaml").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cases = load_runtime_tasks()
    verify_preflight(config["preflight_dir"])
    if sha(__file__) != config["trajectory_source_sha256"] or sha(LEGACY) != config["legacy_helpers_sha256"]:
        raise RuntimeError("Trajectory implementation changed after run preparation")
    if tuple(t["case_id"] for t in manifest["tasks"]) != tuple(t["case_id"] for t in cases):
        raise RuntimeError("Manifest must preserve every original Dev36 task exactly once in order")
    for name, expected in ((Path(config["model"])/"config.json", config["model_config_sha256"]),
                           (Path(config["preflight_dir"])/"config.yaml", config["preflight_config_sha256"]),
                           (Path(config["preflight_dir"])/"summary.json", config["preflight_summary_sha256"])):
        if sha(name) != expected:
            raise RuntimeError("Prepared model/preflight identity changed: " + str(name))
    writer = StreamWriter(run_dir / "raw" / f"stream_{manifest['session_index']}.jsonl")
    engine, result_status, failure = None, "FAILED", None
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    ownership_start = None
    summary = dict(status="RUNNING", gpu_used=False, process_id=os.getpid(),
                   completed=sum(t["status"] == "COMPLETE" for t in manifest["tasks"]), planned=len(cases),
                   started_epoch=time.time(), cuda_busy_seconds="UNKNOWN")
    save(run_dir / "summary.json", summary)
    try:
        observed = gpu_inventory(config["expected_gpu_uuid"])
        config["observed_gpu"] = observed
        save(run_dir / "config.yaml", config)
        update_progress(run_dir, manifest, "ENGINE_STARTING", phase_deadline_epoch=time.time()+1800)
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
        generation_config = json.loads((Path(config["model"]) / "generation_config.json").read_text())
        eos_ids = generation_config["eos_token_id"]
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        ownership_start = time.perf_counter()
        summary["gpu_used"] = True
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=config["model"], **config["engine"]),
                                         usage_context=UsageContext.OPENAI_API_SERVER)
        snapshots = await engine.collective_rpc(SNAPSHOT_RPC_METHOD)
        if len(snapshots) != 1:
            raise RuntimeError("Expected exactly one native worker")
        snapshot = snapshots[0]
        expected = config["preflight_runner_snapshot"]
        for field in ("runner_class", "runner_module", "runner_source_sha256", "use_v2_model_runner", "max_model_len",
                      "async_scheduling", "compile_mode", "cudagraph_mode", "enable_prefix_caching",
                      "dtype", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes", "block_size"):
            if snapshot.get(field) != expected.get(field):
                raise RuntimeError("Actual worker differs from passed preflight: " + field)
        config["actual_model_runner"] = snapshot["runner_class"]
        config["actual_runner_snapshot"] = snapshot
        save(run_dir / "config.yaml", config)
        writer.add("worker_snapshot", snapshot=snapshot, gpu=observed)
        factory = lambda values: SamplingParams(**values, output_kind=RequestOutputKind.DELTA)
        for case, entry in zip(cases, manifest["tasks"]):
            if entry["case_id"] != case["case_id"]:
                raise RuntimeError("Manifest question identity or order changed")
            if entry["status"] == "COMPLETE":
                continue
            if entry["status"] != "PENDING":
                raise RuntimeError("Failed/interrupted question requires an explicit retry decision")
            case_started, last_progress = time.perf_counter(), 0.0
            deadline = time.time() + config["question_timeout_s"]
            entry.update(status="RUNNING", started_epoch=time.time())

            def progress(stage, kind, tokens, cap):
                nonlocal last_progress
                if time.monotonic()-last_progress >= 2:
                    update_progress(run_dir, manifest, "GENERATING", active_case=case["case_id"],
                                    stage=stage, kind=kind, observed_stage_tokens=tokens, stage_cap=cap,
                                    phase_deadline_epoch=deadline)
                    last_progress = time.monotonic()

            update_progress(run_dir, manifest, "GENERATING", active_case=case["case_id"],
                            phase_deadline_epoch=deadline)
            try:
                row = await asyncio.wait_for(generate_case(engine, tokenizer, writer, case, eos_ids,
                                             factory, progress), timeout=config["question_timeout_s"])
                if engine.output_processor.has_unfinished_requests():
                    raise RuntimeError("Native requests remain after a serial question completed")
                row["config_identity"] = config["config_identity"]
                path = checkpoint_path(run_dir, case["case_id"])
                await asyncio.to_thread(save, path, row)
                entry.update(status="COMPLETE", elapsed_s=time.perf_counter()-case_started,
                             generated_tokens=row["generated_tokens"], branch=row["branch"],
                             checkpoint=str(path.relative_to(run_dir)), checkpoint_sha256=sha(path))
                complete_rows = [json.loads(checkpoint_path(run_dir, item["case_id"]).read_text())
                                 for item in manifest["tasks"] if item["status"] == "COMPLETE"]
                await asyncio.to_thread(atomic_jsonl, run_dir / "raw/trajectories.jsonl", complete_rows)
                update_progress(run_dir, manifest, "QUESTION_COMPLETE", active_case=None,
                                phase_deadline_epoch=time.time()+config["question_timeout_s"])
            except BaseException as error:
                entry.update(status="FAILED", error=repr(error), elapsed_s=time.perf_counter()-case_started)
                update_progress(run_dir, manifest, "FAILED", active_case=case["case_id"], error=repr(error))
                raise
        result_status = "COMPLETE"
    except BaseException as error:
        failure = repr(error)
        writer.add("failure", error=failure)
        raise
    finally:
        drained = False
        try:
            if engine is not None:
                drained = not engine.output_processor.has_unfinished_requests()
                engine.shutdown(timeout=15)
        finally:
            try:
                writer.close()
            except BaseException as error:
                result_status, failure = "FAILED_RECORDING", repr(error)
            usage1 = resource.getrusage(resource.RUSAGE_SELF)
            done = [t for t in manifest["tasks"] if t["status"] == "COMPLETE"]
            summary.update(status=result_status, error=failure, completed=len(done), planned=len(cases),
                           completed_generated_tokens=sum(t["generated_tokens"] for t in done),
                           ended_epoch=time.time(), process_wall_seconds=time.perf_counter()-started,
                           engine_ownership_wall_seconds=(time.perf_counter()-ownership_start
                                                         if ownership_start is not None else None),
                           cpu_user_seconds=usage1.ru_utime-usage0.ru_utime,
                           cpu_system_seconds=usage1.ru_stime-usage0.ru_stime,
                           child_process_usage="Not included in Python worker CPU delta",
                           native_frontend_drained=drained, recording=writer.statistics(),
                           probes_generated=0)
            save(run_dir / "summary.json", summary)
            update_progress(run_dir, manifest, result_status, error=failure)


def prepare(run_dir, preflight_dir, resume=False):
    preflight = verify_preflight(preflight_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        if not resume:
            raise ValueError("Existing run; --resume only continues untouched pending questions")
        config = json.loads((run_dir / "config.yaml").read_text())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if config["trajectory_source_sha256"] != sha(__file__):
            raise RuntimeError("Resume implementation differs from the frozen run")
        if tuple(t["case_id"] for t in manifest["tasks"]) != tuple(t["case_id"] for t in load_runtime_tasks()):
            raise RuntimeError("Resume manifest does not contain the exact Dev36 population")
        complete_rows = []
        for entry in manifest["tasks"]:
            path = checkpoint_path(run_dir, entry["case_id"])
            if path.exists():
                if entry.get("checkpoint_sha256") and sha(path) != entry["checkpoint_sha256"]:
                    raise RuntimeError("Recorded checkpoint SHA mismatch")
                row = json.loads(path.read_text())
                if (row.get("config_identity") != config["config_identity"] or row.get("status") != "COMPLETE"
                        or row.get("case_id") != entry["case_id"] or row.get("task_id") != entry["case_id"]):
                    raise RuntimeError("Checkpoint identity mismatch")
                entry.update(status="COMPLETE", checkpoint=str(path.relative_to(run_dir)),
                             checkpoint_sha256=sha(path), generated_tokens=row["generated_tokens"],
                             branch=row["branch"], elapsed_s=(row["ended_ns"]-row["started_ns"])/1e9)
                complete_rows.append(row)
            elif entry["status"] != "PENDING":
                raise RuntimeError("Failed/interrupted question is not silently retried")
        atomic_jsonl(run_dir / "raw/trajectories.jsonl", complete_rows)
        manifest["session_index"] += 1
        save(run_dir / "manifest.json", manifest)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir()
    (run_dir / "checkpoints").mkdir()
    source = legacy_source_identity()
    timeout = math.ceil(40960/25 + 300)
    config = dict(schema="exec_exit.native_trajectories.v1", model=str(MODEL),
                  preflight_dir=str(Path(preflight_dir).resolve()),
                  preflight_config_sha256=sha(Path(preflight_dir)/"config.yaml"),
                  preflight_summary_sha256=sha(Path(preflight_dir)/"summary.json"),
                  expected_gpu_uuid=preflight["expected_gpu_uuid"],
                  actual_model_runner=preflight["actual_model_runner"],
                  actual_runner_identity_scope="Passed preflight; rechecked in trajectory worker",
                  preflight_runner_snapshot=preflight["actual_runner_snapshot"],
                  engine=preflight["engine"], model_realpath=str(MODEL.resolve()),
                  model_config_sha256=sha(MODEL/"config.json"), protocol=Protocol().__dict__,
                  trajectory_source_sha256=sha(__file__), legacy_helpers_sha256=sha(LEGACY),
                  sampling_base=params_for(0, 28000), source_identity=source,
                  environment=ENVIRONMENT, serial_questions=True, repeats_per_question=1,
                  question_timeout_s=timeout,
                  timeout_basis="At most 40960 generated tokens / 25 token/s + 300s preparation margin",
                  no_total_run_quota=True, seed_rule="legacy seed_for(case_id,0,stage); stage 0 then 1",
                  differences_from_old_N=["28,000-token first-segment cap instead of old 16,384 total cap",
                    "Stop first segment at actual thinking-close ID; natural answer uses remaining capacity",
                    "Only still-open cap uses explicit close/fence input plus <=2,048 generated answer tokens",
                    "Continuation seed stage=1, whereas old native driver reused stage=0",
                    "Configured capacity 40,960; trajectory recording without application delivery/scoring"],
                  version_not_committed=True)
    config["config_identity"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(run_dir / "config.yaml", config)
    (run_dir / "commit.txt").write_text(subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()+"\n")
    manifest = dict(schema="exec_exit.trajectory_manifest.v1", session_index=0,
                    tasks=[dict(case_id=case["case_id"], status="PENDING") for case in load_runtime_tasks()])
    update_progress(run_dir, manifest, "PREPARED")
    save(run_dir/"summary.json", dict(status="PREPARED", gpu_used=False, completed=0, planned=36))


def supervise_process(process, read_progress, *, wall_clock=time.time,
                      monotonic_clock=time.monotonic, sleep=time.sleep,
                      kill_process_group=os.killpg, startup_timeout_s=1800,
                      phase_grace_s=30, terminal_exit_grace_s=30):
    """Bound each active phase and give a terminal worker one finite exit grace.

    Terminal progress intentionally has no phase deadline.  Its first observed
    monotonic timestamp is latched, so it neither reuses the expired startup
    deadline nor extends the exit grace every time progress is polled.
    """
    startup_deadline = wall_clock() + startup_timeout_s
    terminal_seen_monotonic, terminal_seen_epoch, terminal_status = None, None, None
    timed_out, timeout_basis = False, None
    while process.poll() is None:
        progress = read_progress()
        now_epoch, now_monotonic = wall_clock(), monotonic_clock()
        status = progress.get("status", "")
        terminal = status == "COMPLETE" or status.startswith("FAILED")
        if terminal and terminal_seen_monotonic is None:
            terminal_seen_monotonic, terminal_seen_epoch = now_monotonic, now_epoch
            terminal_status = status
        if terminal_seen_monotonic is not None:
            expired = now_monotonic > terminal_seen_monotonic + terminal_exit_grace_s
            basis = "terminal_exit_grace"
        else:
            deadline = progress.get("phase_deadline_epoch", startup_deadline)
            expired = now_epoch > deadline + phase_grace_s
            basis = "active_phase_deadline" if "phase_deadline_epoch" in progress else "startup_deadline"
        if expired:
            # A worker may exit between the first poll and the deadline check.
            if process.poll() is not None:
                break
            timed_out, timeout_basis = True, basis
            kill_process_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                kill_process_group(process.pid, signal.SIGKILL)
                process.wait()
            break
        sleep(1)
    return dict(timed_out=timed_out, timeout_basis=timeout_basis,
                startup_deadline_epoch=startup_deadline, phase_grace_s=phase_grace_s,
                terminal_exit_grace_s=terminal_exit_grace_s, terminal_status=terminal_status,
                terminal_first_observed_epoch=terminal_seen_epoch,
                terminal_first_observed_monotonic=terminal_seen_monotonic)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    os.environ.update(ENVIRONMENT)
    if args.worker:
        asyncio.run(run_gpu(run_dir))
        return 0
    prepare(run_dir, args.preflight_dir, args.resume)
    if not args.execute:
        return 0
    command = [sys.executable, "-B", "-m", "exec_exit.gen.traces", "--run-dir", str(run_dir),
               "--preflight-dir", str(args.preflight_dir.resolve()), "--worker"]
    with (run_dir/"worker.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ.copy())
        supervision = supervise_process(process,
            lambda: json.loads((run_dir/"progress.json").read_text()))
        deadline_failure = supervision["timed_out"]
    summary = json.loads((run_dir/"summary.json").read_text())
    summary["worker_returncode"] = process.returncode
    summary["supervisor_timeout"] = deadline_failure
    summary["supervisor_deadline_policy"] = supervision
    if process.returncode != 0 or deadline_failure or summary["status"] != "COMPLETE":
        summary["status"] = "FAILED_SUPERVISOR_TIMEOUT" if deadline_failure else "FAILED"
    save(run_dir/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
