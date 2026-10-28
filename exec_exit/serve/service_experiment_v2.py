"""冻结后的C4 v2：六组轮换方法、混合标准输入比较器和实测负载。

此入口不选择题目、负载或CONF参数；打开公开数据前先核验冻结文档和方法身份。
函数调用型任务及未登记模型会显式报错，不读取隐藏测试，也不评定完整正确性。
"""

from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from exec_exit.gen.engine_preflight import ROOT, MODEL, SNAPSHOT_RPC_METHOD, save, sha, gpu_inventory
from exec_exit.gen.host_environment import collect_host_environment
from exec_exit.gen.dev_pipeline import NativeCounters, RUNNER_FIELDS, progress, validate_identity
from exec_exit.gen.traces import ENVIRONMENT, StreamWriter, atomic_jsonl, supervise_process, verify_preflight
from exec_exit.serve.c4_plan_v2 import (
    validate_spec,
    load_tasks_after_freeze_impl,
    make_schedule,
    materialize_poisson,
    run_ready_service,
    require_ds_blind_complete,
)
from exec_exit.serve.public_samples_lcbpro_v1 import (
    init_public_worker,
    warm_public_worker,
    install as install_public_worker,
)
from exec_exit.serve.service_load import run_poisson


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def method_sha(spec):
    body = {k: v for k, v in spec.items() if k not in {"freeze_record", "freeze_record_sha256"}}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def checked(path, expected):
    path = Path(path).resolve()
    if not isinstance(expected, str) or sha(path) != expected:
        raise ValueError("Exact artifact SHA mismatch: " + str(path))
    return path


def specification_template():
    #沿用只导出元数据的既定计划，不能退回不完整的v1格式或重新选题
    return read(ROOT / "runs/evening_freeze_review/c4/service_spec.json")


def freeze_gate(spec):
    """读取数据前核验实际冻结记录，仅有调用方的FROZEN标记不够。"""
    validate_spec(spec)
    frozen_path = checked(spec["freeze_record"], spec["freeze_record_sha256"])
    frozen = read(frozen_path)
    if frozen.get("state") != "FROZEN":
        raise ValueError("Actual freeze record is not FROZEN")
    document = frozen.get("freeze_document", {})
    path = checked(document["path"], document["sha256"])
    if path != (ROOT / "FREEZE.md").resolve():
        raise ValueError("Repository FREEZE.md identity required")
    included = frozen.get("included_c4")
    if not isinstance(included, dict) or included.get("spec_method_sha256") != method_sha(spec):
        raise ValueError("Exact C4 specification not in frozen inclusion record")
    for name, digest in spec["implementation_sources"].items():
        checked(name, digest)
    return {
        "freeze_record": str(frozen_path),
        "freeze_record_sha256": sha(frozen_path),
        "freeze_document": document,
        "spec_method_sha256": method_sha(spec),
    }


def load_tasks_after_freeze(spec):
    identity = freeze_gate(spec)
    return load_tasks_after_freeze_impl(spec), identity


def schedule(spec):
    return make_schedule(spec)


def planned_requests(entries):
    return sum(1 if e["kind"] == "serial" else e["scheduled_requests"] for e in entries)


async def dispatch(
    entry, spec, tasks, engine, writer, backend_args, previous, *, backend=run_ready_service, poisson=run_poisson
):
    """调度器不读取源文件、不选题，也不接触未来标签。"""
    if entry["kind"] == "serial":
        case_id = entry["case_id"]
        task = tasks[case_id]
        mode = entry["mode"]
        extra = {}
        if mode == "H" and spec["serial"]["H"]["schedule"] == "MATCHED_SYNC":
            sync = previous.get((case_id, "HYB-B-SYNC"))
            if not sync or sync.get("status") != "COMPLETE":
                return {
                    "status": "NOT_RUN_DEPENDENCY_FAILED",
                    "case_id": case_id,
                    "mode": mode,
                    "reason": "Corresponding sync did not complete; no invented matching schedule",
                }
            extra["matched_positions"] = [p for pause in sync["pauses"] for p in pause["executed_nominal_positions"]]
        if mode == "H":
            extra["h_schedule"] = spec["serial"]["H"]["schedule"]
        if mode == "CONF-50":
            extra["confidence_settings"] = spec["serial"]["CONF-50"]
        case = {k: task[k] for k in ("case_id", "task_id", "question_content")}
        record = await backend(
            mode=mode,
            engine=engine,
            writer=writer,
            case=case,
            task=task,
            sample_index=spec["sample_index"],
            **extra,
            **backend_args,
        )
        previous[case_id, mode] = record
        return record

    async def adapted(engine, **kwargs):
        return await backend(engine=engine, **kwargs)

    return await poisson(
        engine,
        mode=entry["mode"],
        arrivals=entry["arrivals"],
        tasks=tasks,
        writer=writer,
        run_identity="c4-poisson-" + str(entry["experiment_index"]),
        sample_index=spec["sample_index"],
        backend=adapted,
        **backend_args,
    )


def prepare(args):
    gpu_lock = Path(os.environ["EXECEXIT_GPU_LOCK"]).expanduser().resolve()
    spec = read(args.spec)
    tasks, freeze = load_tasks_after_freeze(spec)
    preflight = verify_preflight(args.preflight_dir)
    if spec["expected_gpu_uuid"] != preflight["expected_gpu_uuid"]:
        raise ValueError("Frozen GPU differs from preflight")
    if preflight["engine"]["max_model_len"] != 40960:
        raise ValueError("Registered capacity differs from preflight")
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "source", "checkpoints", "runtime"):
        (out / name).mkdir()
    paths = [
        Path(__file__),
        args.spec.resolve(),
        args.preflight_dir.resolve() / "config.yaml",
        args.preflight_dir.resolve() / "summary.json",
        Path(spec["freeze_record"]),
        ROOT / "FREEZE.md",
    ]
    paths += [Path(ref["path"]) for ref in spec["public_sources"].values()]
    paths += [Path(spec[key]["path"]) for key in ("roster", "schedule", "unit_poisson_arrivals")]
    paths += [
        ROOT / f"exec_exit/{x}.py"
        for x in (
            "serve/heldout_service_template",
            "serve/c4_plan_v2",
            "serve/public_samples_lcbpro_v1",
            "execute/lcbpro_comparator",
            "serve/service_load",
            "serve/resume_backend",
            "serve/conf50_probe",
            "serve/confidence_probe_positions",
            "serve/async_backend",
            "serve/async_probe",
            "serve/hybrid_probe_positions",
            "gen/traces",
            "gen/probes",
            "gen/probes_v1",
            "gen/dev_pipeline",
            "gen/engine_preflight",
            "gen/host_environment",
            "data/load_lcb",
            "execute/extract",
            "execute/run_tests",
            "execute/sandbox",
            "execute/machine_identity",
            "replay/replay",
            "replay/policies",
            "replay/early_abort",
            "analysis/trajectory_annotations",
        )
    ]
    from exec_exit.execute.sandbox import LEGACY_EXECUTOR, LIMITS_PATH
    from exec_exit.data.load_lcb import official_evaluator_path
    from exec_exit.gen.traces import LEGACY

    paths += [LEGACY_EXECUTOR, LIMITS_PATH, official_evaluator_path(), LEGACY]
    paths += [MODEL / name for name in ("config.json", "generation_config.json", "tokenizer.json")]
    config = {
        "schema": "exec_exit.c4.postfreeze_experiment.v2",
        "created_utc": stamp(),
        "spec": str(args.spec.resolve()),
        "spec_sha256": sha(args.spec),
        "freeze_identity": freeze,
        "model": str(MODEL),
        "engine": preflight["engine"],
        "expected_gpu_uuid": preflight["expected_gpu_uuid"],
        "preflight_runner_snapshot": preflight["actual_runner_snapshot"],
        "host_environment": collect_host_environment(),
        "source_sha256": {str(p.resolve()): sha(p) for p in paths},
        "environment": ENVIRONMENT,
        "case_timeout_seconds": 5400,
        "startup_timeout_seconds": 1800,
        "GPU_lock": str(gpu_lock),
        "disk_min_free_gib": 20,
        "arrival_seed_rule": "Frozen unit exponential vector, scale by measured S/c; paired policy vectors exact",
        "planned_question_count": len(tasks),
        "full_correctness": "Not evaluated here; separately attach frozen offline complete-test results",
        "private_test_content_read": False,
        "heldout_public_samples_only": True,
        "no_automatic_retry": True,
    }
    save(out / "config.yaml", config)
    (out / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
    for p in paths:
        if p.suffix == ".py":
            shutil.copyfile(p, out / "source" / p.name)
    manifest = {"tasks": schedule(spec)}
    progress(out, manifest, "PREPARED")
    save(
        out / "summary.json",
        {"status": "PREPARED", "gpu_used": False, "planned_experiments": len(manifest["tasks"]), "completed": 0},
    )


async def run_gpu(out):
    cfg = read(out / "config.yaml")
    spec = read(cfg["spec"])
    freeze_gate(spec)
    require_ds_blind_complete(spec)
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.usage.usage_lib import UsageContext
    from exec_exit.analysis.trajectory_annotations import ByteLevelTokenCodec

    install_public_worker()
    cfg = read(out / "config.yaml")
    validate_identity(cfg)
    spec = read(cfg["spec"])
    tasks, _ = load_tasks_after_freeze(spec)
    model = Path(cfg["model"])
    manifest = read(out / "manifest.json")
    comparator_specs = {cid: t["comparator_spec"] for cid, t in tasks.items() if t.get("comparator_spec") is not None}
    if any(r["status"] != "PENDING" for r in manifest["tasks"]):
        raise ValueError("Never silently rerun an attempted service experiment")
    writer = StreamWriter(out / "raw/stream_0.jsonl")
    engine = pool = None
    records = []
    previous = {}
    engine_start = None
    started = time.perf_counter()
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    child0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    status = "FAILED"
    error = None
    summary = {"status": "RUNNING", "gpu_used": False, "completed": 0, "planned_experiments": len(manifest["tasks"])}
    save(out / "summary.json", summary)
    try:
        cfg["observed_gpu"] = gpu_inventory(cfg["expected_gpu_uuid"])
        save(out / "config.yaml", cfg)
        progress(out, manifest, "ENGINE_STARTING", deadline=time.time() + 1800)
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
        extraction = Tokenizer.from_file(str(model / "tokenizer.json"))
        codec = ByteLevelTokenCodec(model / "tokenizer.json")
        eos = read(model / "generation_config.json")["eos_token_id"]
        eos = set(eos if isinstance(eos, list) else [eos])
        NativeCounters.instances = []
        engine_start = time.perf_counter()
        summary["gpu_used"] = "UNKNOWN_ENGINE_STARTING"
        save(out / "summary.json", summary)
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(model=cfg["model"], **cfg["engine"]),
            usage_context=UsageContext.OPENAI_API_SERVER,
            stat_loggers=[NativeCounters],
        )
        summary["gpu_used"] = True
        save(out / "summary.json", summary)
        snapshots = await engine.collective_rpc(SNAPSHOT_RPC_METHOD)
        if len(snapshots) != 1 or any(
            snapshots[0].get(k) != cfg["preflight_runner_snapshot"].get(k) for k in RUNNER_FIELDS
        ):
            raise RuntimeError("Actual runner differs from preflight")
        cfg.update(actual_runner_snapshot=snapshots[0], actual_model_runner=snapshots[0]["runner_class"])
        save(out / "config.yaml", cfg)
        writer.add("worker_snapshot", snapshot=snapshots[0], gpu=cfg["observed_gpu"])
        factory = lambda values: SamplingParams(**values, output_kind=RequestOutputKind.DELTA)
        manifest["generation_started_epoch"] = time.time()
        for entry in manifest["tasks"]:
            if entry["kind"] == "poisson":
                materialize_poisson(entry, spec, previous)
                save(
                    out / "runtime" / ("load_" + str(entry["concurrency_target"]) + "_" + entry["mode"] + ".json"),
                    entry,
                )
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError("PAUSED_DISK_BELOW_20_GIB")
            if engine.output_processor.has_unfinished_requests():
                raise RuntimeError("Prior arm not drained; no overlap between measurements")
            reset_start = time.perf_counter_ns()
            if await engine.reset_prefix_cache() is not True:
                raise RuntimeError("Native prefix reset not acknowledged")
            writer.add(
                "prefix_cache_reset",
                experiment_index=entry["experiment_index"],
                started_ns=reset_start,
                finished_ns=time.perf_counter_ns(),
            )
            scope = spec["runtime"]["public_execution_cache_scope"]
            if pool is None:
                directory = (
                    out
                    / "runtime"
                    / ("shared" if scope == "shared_experiment" else f"arm_{entry['experiment_index']:04d}")
                )
                pool = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=init_public_worker,
                    initargs=(str(directory), comparator_specs),
                )
                identity = await asyncio.get_running_loop().run_in_executor(pool, warm_public_worker)
                writer.add("public_runtime_identity", experiment_index=entry["experiment_index"], identity=identity)
            duration = 5400 + (entry["arrivals"][-1]["intended_arrival_seconds"] if entry["kind"] == "poisson" else 0)
            entry.update(status="RUNNING", started_epoch=time.time())
            progress(
                out,
                manifest,
                "GENERATING",
                active={k: v for k, v in entry.items() if k != "arrivals"},
                deadline=time.time() + duration,
            )
            arguments = {
                "tokenizer": tokenizer,
                "extraction_tokenizer": extraction,
                "codec": codec,
                "pool": pool,
                "eos_ids": eos,
                "parameter_factory": factory,
            }
            try:
                record = await asyncio.wait_for(
                    dispatch(entry, spec, tasks, engine, writer, arguments, previous), duration
                )
                entry["status"] = "COMPLETE" if record["status"] == "COMPLETE" else "FAILED_SERVICE"
            except Exception as exc:
                record = {
                    "status": "FAILED_SERVICE",
                    "error": repr(exc),
                    "experiment_index": entry["experiment_index"],
                    "mode": entry["mode"],
                    "case_id": entry.get("case_id"),
                    "planned_request_count": 1 if entry["kind"] == "serial" else len(entry["arrivals"]),
                    "latency": "UNKNOWN_NO_COMPLETE_RECORD",
                    "partial_stream_retained": True,
                }
                entry["status"] = "FAILED_SERVICE"
                writer.add("service_failure", **record)
            if entry["kind"] == "poisson" and "records" not in record:
                record["records"] = [
                    {
                        "status": "INCOMPLETE_SERVICE",
                        "request_index": a["request_index"],
                        "case_id": a["case_id"],
                        "intended_arrival_seconds": a["intended_arrival_seconds"],
                        "cost": "PARTIAL_STREAM_ONLY",
                    }
                    for a in entry["arrivals"]
                ]
            path = out / "checkpoints" / f"experiment_{entry['experiment_index']:04d}.json"
            save(path, record)
            records.append({"experiment": dict(entry), "record": record})
            entry.update(checkpoint=str(path.relative_to(out)), checkpoint_sha256=sha(path), ended_epoch=time.time())
            progress(out, manifest, "GENERATING", deadline=time.time() + 5400)
            if scope == "per_arm":
                await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=False)
                pool = None
        if engine.output_processor.has_unfinished_requests():
            raise RuntimeError("Service front end not drained")
        status = "COMPLETE" if all(e["status"] == "COMPLETE" for e in manifest["tasks"]) else "FAILED_SERVICE_PRESENT"
    except BaseException as exc:
        error = repr(exc)
        writer.add("failure", error=error)
        raise
    finally:
        primary = sys.exc_info()[1]
        cleanup = None
        try:
            if engine is not None:
                summary["native_frontend_drained"] = not engine.output_processor.has_unfinished_requests()
                summary["native_metrics"] = NativeCounters.snapshot()
                engine.shutdown(timeout=15)
        except BaseException as exc:
            cleanup = exc
        try:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=False)
        except BaseException as exc:
            cleanup = cleanup or exc
        try:
            writer.close()
        except BaseException as exc:
            cleanup = cleanup or exc
        if cleanup is not None:
            status = "FAILED_CLEANUP"
            error = error or repr(cleanup)
        atomic_jsonl(out / "raw/experiments.jsonl", records)
        #N/S或引擎启动失败时没有到达时间，但各计划负载的冻结分母仍为40
        planned = planned_requests(manifest["tasks"])
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary.update(
            status=status,
            error=error,
            completed=len(records),
            planned_requests=planned,
            experiment_statuses=dict(Counter(e["status"] for e in manifest["tasks"])),
            all_planned_requests_retained_in_manifest=True,
            recording=writer.statistics(),
            wall_seconds=time.perf_counter() - started,
            engine_ownership_wall_seconds=time.perf_counter() - engine_start if engine_start else None,
            host_cpu_seconds=cpu.ru_utime + cpu.ru_stime - cpu0.ru_utime - cpu0.ru_stime,
            waited_child_cpu_seconds=child.ru_utime + child.ru_stime - child0.ru_utime - child0.ru_stime,
            gpu_kernel_seconds="UNKNOWN",
            gpu_per_request_seconds="UNKNOWN",
            public_sample_only=True,
            private_test_content_read=False,
        )
        save(out / "summary.json", summary)
        save(out / "resource_ledger.json", summary)
        progress(out, manifest, status)
        if cleanup is not None and primary is None:
            raise cleanup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--preflight-dir", type=Path, default=ROOT / "runs/engine_preflight_segmented_tokenids")
    p.add_argument("--template", type=Path)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--prepared", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    os.environ.update(ENVIRONMENT)
    if a.template:
        with a.template.open("x") as f:
            json.dump(specification_template(), f, ensure_ascii=False, indent=2)
            f.write("\n")
        return 0
    if a.validate:
        if not a.spec:
            p.error("--spec required")
        print(json.dumps(freeze_gate(read(a.spec))))
        return 0
    if not a.run_dir:
        p.error("--run-dir required")
    out = a.run_dir.resolve()
    if a.worker:
        asyncio.run(run_gpu(out))
        return 0
    if not a.prepared:
        if not a.spec:
            p.error("--spec required for preparation")
        prepare(a)
    if not a.execute:
        return 0
    cfg = read(out / "config.yaml")
    freeze_gate(read(cfg["spec"]))
    require_ds_blind_complete(read(cfg["spec"]))
    save(out / "queue.json", {"status": "WAITING_GPU_LOCK", "pid": os.getpid(), "queued_utc": stamp()})
    with Path(cfg["GPU_lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_identity(cfg)
        freeze_gate(read(cfg["spec"]))
        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
            save(out / "queue.json", {"status": "PAUSED_DISK_BELOW_20_GIB"})
            return 2
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            raise RuntimeError("Unexpected GPU process despite exclusive lock")
        save(out / "queue.json", {"status": "LOCK_ACQUIRED", "pid": os.getpid(), "acquired_utc": stamp()})
        command = [
            sys.executable,
            "-B",
            "-m",
            "exec_exit.serve.service_experiment_v2",
            "--run-dir",
            str(out),
            "--worker",
        ]
        with (out / "worker.log").open("a") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy()
            )
            save(
                out / "supervisor.json",
                {
                    "status": "RUNNING",
                    "worker_pid": process.pid,
                    "supervisor_pid": os.getpid(),
                    "command": command,
                    "started_epoch": time.time(),
                },
            )
            supervision = supervise_process(process, lambda: read(out / "progress.json"))
        result = read(out / "summary.json")
        result.update(
            worker_returncode=process.returncode,
            supervisor_deadline_policy=supervision,
            supervisor_timeout=supervision["timed_out"],
        )
        if process.returncode != 0 or supervision["timed_out"]:
            result["status"] = "FAILED"
        save(out / "summary.json", result)
        save(out / "resource_ledger.json", result)
        save(
            out / "supervisor.json",
            read(out / "supervisor.json")
            | {"status": result["status"], "worker_returncode": process.returncode, "ended_epoch": time.time()},
        )
        print(json.dumps(result))
        return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
