"""C4 冻结后的服务实验配置与后端入口。

模板不选择题目或负载，也不读取测试内容。执行前必须补全方法配置。
重启沿用相同前缀和种子，通过 H 对照验证，并实测 token 是否一致。
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random

POLICIES=("N","BF@4096","CONF-50","HYB-B-SYNC","H","HYB-B-ASYNC")
BACKENDS={name:"IMPLEMENTED_PENDING_REAL_GPU_CHECK" for name in POLICIES}
RESTART="Exact original prompt IDs + all observed generated IDs; same original segment seed; remaining segment budget; no retokenization"


def template():
    return {"schema":"exec_exit.c4.postfreeze_service.v1","state":"DRAFT",
        "freeze_record":None,"freeze_record_sha256":None,
        "model":"qwen3_8b","public_runtime_projection":None,"public_runtime_sha256":None,
        "postfreeze_public_tests_projection":None,"postfreeze_public_tests_sha256":None,
        "selected_case_ids":None,"target_question_count":"approximately 40; exact count requires selected manifest",
        "sample_index":None,"task_protocol":None,"expected_gpu_uuid":None,
        "serial":{"policies":list(POLICIES),"policy_order":None,"prefix_cache_reset_policy":None,
            "CONF-50":{"k":None,"minimum_thinking_tokens":None,"threshold_strictly_greater_than":0.95},
            "HYB-B":{"k":4,"minimum_thinking_tokens":0,"endpoint":4096,"probe_protocol":"v1"},
            "synchronous_pause_resume_implementation":RESTART,
            "H":{"schedule":"Corresponding sync actually executed nominal positions only", "artificial_wait":False,
                "unreachable":"Record unmatched; no replacement or retry to force equality", "scope":"Matched experimental control, not online candidate"}},
        "poisson":{"policies":["N","HYB-B-ASYNC"],"rates_requests_per_second":[None,None],
            "arrival_seed":None,"request_case_order":None,"repetitions":None,
            "clock_origin":"actual client intended arrival; all scheduled requests retained",
            "trace_pairing":"same generated arrival timestamps and same public case IDs for both policies"},
        "recording":{"client_transport":"same-host independent IPC; not HTTP/network",
            "payload":"same actual program-text schema for every policy, server logs excluded",
            "gpu_time_per_request":"UNKNOWN unless native attribution becomes available; do not substitute request latency",
            "cancelled_tokens":"record all observed IDs including post-abort tail; unexposed tail UNKNOWN",
            "completion":"single delivery plus full model/CPU drain; no dropped scheduled requests"},
        "backend_readiness":dict(BACKENDS),"execution_status":"NOT_EXECUTED"}


def issues(spec,*,require_frozen=True):
    result=[]
    if spec.get("schema")!="exec_exit.c4.postfreeze_service.v1":
        result.append("Unsupported specification schema")
    if require_frozen and spec.get("state")!="FROZEN":
        result.append("FREEZE_REQUIRED")
    for key in ("freeze_record","freeze_record_sha256","public_runtime_projection","public_runtime_sha256","postfreeze_public_tests_projection","postfreeze_public_tests_sha256","selected_case_ids","sample_index","task_protocol","expected_gpu_uuid"):
        if spec.get(key) is None:
            result.append("REQUIRED:"+key)
    ids=spec.get("selected_case_ids")
    if ids is not None and (not ids or len(ids)!=len(set(ids))):
        result.append("Explicit nonempty unique case IDs required")
    serial=spec.get("serial",{})
    if serial.get("policies")!=list(POLICIES):
        result.append("All six serial controls must remain present")
    order=serial.get("policy_order")
    if order is None or len(order)!=len(POLICIES) or set(order)!=set(POLICIES):
        result.append("REQUIRED:serial.policy_order")
    for key in ("prefix_cache_reset_policy","synchronous_pause_resume_implementation"):
        if serial.get(key) is None:
            result.append("REQUIRED:serial."+key)
    if serial.get("synchronous_pause_resume_implementation")!=RESTART:
        result.append("Unregistered pause/resume semantics")
    if serial.get("H")!=template()["serial"]["H"]:
        result.append("H matched-control semantics differ")
    conf=serial.get("CONF-50",{})
    if conf.get("k") not in (1,2,4):
        result.append("REQUIRED:CONF-50 frozen k")
    if conf.get("minimum_thinking_tokens") not in (0,2048):
        result.append("REQUIRED:CONF-50 frozen minimum")
    if conf.get("threshold_strictly_greater_than")!=.95:
        result.append("CONF-50 threshold differs from locked rule")
    if serial.get("HYB-B")!={"k":4,"minimum_thinking_tokens":0,"endpoint":4096,"probe_protocol":"v1"}:
        result.append("HYB-B differs from locked rule")
    load=spec.get("poisson",{})
    rates=load.get("rates_requests_per_second")
    if not isinstance(rates,list) or len(rates)!=2 or any(type(r) not in (int,float) or r<=0 or not math.isfinite(r) for r in rates) or len(set(rates))!=2:
        result.append("REQUIRED:two distinct positive Poisson loads")
    if load.get("policies")!=["N","HYB-B-ASYNC"]:
        result.append("Both Poisson controls required")
    if type(load.get("arrival_seed")) is not int:
        result.append("REQUIRED:arrival seed")
    if type(load.get("repetitions")) is not int or load["repetitions"]<1:
        result.append("REQUIRED:Poisson repetitions")
    request_ids=load.get("request_case_order")
    if not request_ids or ids is None or not set(request_ids)<=set(ids):
        result.append("REQUIRED:explicit scheduled request case order")
    for name,value in BACKENDS.items():
        if value not in {"IMPLEMENTED_PENDING_REAL_GPU_CHECK","GPU_VALIDATED"}:
            result.append(name+":"+value)
    return result


def paired_poisson_arrivals(case_ids,rate,seed):
    """各方法共用同一到达日程，不根据运行结果调整到达时间。"""
    if type(rate) not in (int,float) or rate<=0 or not math.isfinite(rate) or type(seed) is not int:
        raise ValueError("Explicit positive rate and integer seed required")
    rng=random.Random(seed)
    elapsed=0.
    rows=[]
    for index,case_id in enumerate(case_ids):
        elapsed+=rng.expovariate(rate)
        rows.append({"request_index":index,"case_id":case_id,"intended_arrival_seconds":elapsed})
    return rows


async def run_ready_service(mode,**backend_arguments):
    """为六个实验组调用后端，调用者须提供已冻结的公开任务。

    编排入口必须先完成全部就绪检查；这里不读取题目或测试，也不筛选题目。
    """
    if mode not in POLICIES:
        raise NotImplementedError("Unregistered policy")
    if mode=="H" and not isinstance(backend_arguments.get("matched_positions"),list):
        raise ValueError("H requires corresponding sync actual executed nominal schedule")
    from exec_exit.serve.resume_backend import run_case
    return await run_case(mode=mode,**backend_arguments)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    t=sub.add_parser("template")
    t.add_argument("--output",required=True,type=Path)
    v=sub.add_parser("validate")
    v.add_argument("--config",required=True,type=Path)
    args=p.parse_args()
    if args.action=="template":
        with args.output.open("x") as f:
            json.dump(template(),f,ensure_ascii=False,indent=2)
            f.write("\n")
        return 0
    spec=json.loads(args.config.read_text())
    missing=issues(spec)
    print(json.dumps({"status":"NOT_RUNNABLE" if missing else "SPECIFIED","issues":missing,"data_or_tests_opened":False,"gpu_used":False},ensure_ascii=False,indent=2))
    return 2 if missing else 0


if __name__=="__main__":
    raise SystemExit(main())
