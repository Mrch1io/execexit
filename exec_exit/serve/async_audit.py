"""Independent stream, public visibility and client receipt audit for C4 Dev.

No model, submitted code or private test is executed. This checks observable
records, not unexposed GPU work or causal/performance superiority.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from exec_exit.gen.audit_probes import digest,read,rows,write
from exec_exit.gen.engine_preflight import ROOT


def coverage(records):
    """Executed paths only; a failed PUB check still covers the checker path."""
    visits=[v for r in records for v in r["visits"]]
    checks=[v["public"] for v in visits if "public" in v]
    recovered=[c for r in records for c in r["probe_calls"].values()]
    accepted=[r for r in records if r["selected_kind"]=="PROBE"]
    cancelled=[r for r in accepted if any(r["token_account"].get(rid,{}).get("abort_ns") is not None for rid in r["native_calls"])]
    return {"services":len(records),"probes_recovered":len(recovered),"probes_with_actual_tokens":sum(c["token_count"]>0 for c in recovered),
        "public_checks_completed":len(checks),"public_suites_actually_executed":sum(c.get("cache_reused") is False for c in checks),
        "public_pass_checks":sum(c["status"]=="PASS" for c in checks),"probe_deliveries":len(accepted),
        "accepted_main_abort_requests":len(cancelled),"gpu_accept_then_abort_path":"EXERCISED" if cancelled else "NOT_EXERCISED",
        "unexposed_gpu_tail":"UNKNOWN_WHEN_ABORTED","success_or_win_required_for_smoke":False}


def smoke_coverage_complete(observed):
    return (observed["probes_with_actual_tokens"]>=1 and observed["probes_recovered"]>=1
        and observed["public_checks_completed"]>=1 and observed["public_suites_actually_executed"]>=1)


def check_service(record,events,require):
    identity=record["service_identity"]
    require(record["status"]=="COMPLETE" and not record["coordinator_errors"],identity+": complete without masked coordinator error")
    require(record["delivery_count"]==record["client"]["delivery_count"]==1,identity+": one delivery")
    bytype=defaultdict(list)
    for event in events:bytype[event["event"]].append(event)
    for key in ("server_request_received","server_delivery_queued","server_send_returned","delivery","client_receipt"):
        require(len(bytype[key])==1,identity+": exactly one "+key)
    receipt={k:v for k,v in bytype["client_receipt"][0].items() if k not in ("at_ns","epoch_ns","service_identity")}
    require(receipt==record["client"],identity+": actual client acknowledgement")
    require(record["client"]["client_submit_ns"]<=record["server_receive_ns"]<=record["server_accept_ns"]<=record["client"]["client_receive_complete_ns"],identity+": monotonic visibility ordering")
    require(record["server_accept_ns"]<=record["drained_ns"],identity+": complete drain after acceptance")
    extraction=record["selected"]["extraction"]
    packet={"kind":record["selected_kind"],"case_id":record["case_id"],"deliverable":bool(extraction["deliverable"]),
        "code_text":extraction.get("code_text"),"extraction_status":extraction["status"],"server_accept_ns":record["server_accept_ns"]}
    payload=json.dumps(packet,ensure_ascii=False).encode()
    require(hashlib.sha256(payload).hexdigest()==record["client"]["response_sha256"] and len(payload)==record["client"]["response_bytes"],identity+": identical actual answer transport schema/payload")
    require(extraction["online_visible_at"]<=record["server_accept_ns"],identity+": extraction actually visible before acceptance")
    if record["selected_kind"]=="PROBE":
        accepted=[v for v in record["visits"] if v["status"]=="ACCEPTED"]
        require(len(accepted)==1 and accepted[0]["public"]["status"]=="PASS",identity+": one qualifying public result")
        public=accepted[0]["public"]
        require(public["server_check_queued_ns"]<=public["worker_visible_start_ns"]<=public["worker_complete_ns"]<=public["server_check_visible_ns"]<=record["server_accept_ns"],identity+": real public check completed before acceptance")
    starts={};ends={};offsets=Counter();texts=defaultdict(list);visible_native=[];prompt=None;probe_count=0
    expected_calls=record["native_calls"]|record["probe_calls"]
    for event in events:
        kind=event["event"]
        if kind in ("call_start","probe_start"):
            rid=event.get("request_id",event.get("probe_id"));require(rid not in starts,identity+": unique native request start")
            starts[rid]=event
            if kind=="call_start" and event["stage"]==0:prompt=event["prompt_token_ids"]
            if kind=="probe_start":
                p=event["position_generated_tokens"];probe_count+=1
                require(prompt is not None and 0<p<=min(len(visible_native),4096),identity+": no future/post4096 probe")
                require(event["prompt_token_ids"]==prompt+visible_native[:p]+event["control_token_ids"],identity+": exact already-observed prefix plus control")
                require(p in record["candidate_observation"]["selected_hyb_b_positions"],identity+": frozen online opportunity")
        elif kind in ("call_output","probe_output"):
            rid=event.get("request_id",event.get("probe_id"));call=expected_calls[rid];ids=event["token_ids"];offset=offsets[rid]
            require(rid in starts and rid not in ends,identity+": output inside request lifecycle")
            require(event["token_offset"]==offset and ids==call["token_ids"][offset:offset+len(ids)],identity+": raw delta alignment")
            require(call["started_ns"]<=event["native_frontend_receive_ns"]<=call["ended_ns"],identity+": native output timestamp bounds")
            offsets[rid]+=len(ids);texts[rid].append(event["text"])
            if kind=="call_output" and event["stage"]==0:visible_native.extend(ids)
        elif kind in ("call_end","probe_end"):
            rid=event.get("request_id",event.get("probe_id"));require(rid in starts and rid not in ends,identity+": exactly one call end")
            ends[rid]=event
            actual=event["call"] if kind=="probe_end" else {k:v for k,v in event.items() if k not in ("event","at_ns","epoch_ns","service_identity")}
            require(actual==expected_calls[rid],identity+": final call equals independent stream")
    require(set(starts)==set(ends)==set(expected_calls),identity+": all call lifecycles accounted")
    require(probe_count<=3 and (record["mode"]!="N" or probe_count==0),identity+": finite probes and unchanged N")
    for rid,call in expected_calls.items():
        require(offsets[rid]==call["token_count"]==len(call["token_ids"]) and "".join(texts[rid])==call["raw_text"],identity+": full token/text concatenation")
        account=record["token_account"].get(rid)
        if account is None:
            require(call["token_count"]==0 and bool(call["error"]),identity+": no unobserved successful request")
        else:
            require(account["observed_tokens"]==call["token_count"],identity+": abort tail included in token cost")
            if account["abort_ns"] is not None:require(account["unobserved_cancelled_tokens"]=="UNKNOWN",identity+": unexposed cancellation remains unknown")
    require(sum(offsets.values())==record["observed_generated_tokens"],identity+": total real token accounting")
    require(sum(a["observed_after_abort"] for a in record["token_account"].values())==record["observed_post_abort_tokens"],identity+": observed abort tail denominator")
    require(record["private_tests_read"] is False and record["future_labels_or_trajectories_read"] is False,identity+": public-only scope declared")
    return {"service_identity":identity,"status":"PASS","calls":len(starts),"probes":probe_count,"observed_tokens":sum(offsets.values()),
        "client_e2e_seconds":record["client"]["client_e2e_seconds"],"selected_kind":record["selected_kind"],
        "client_received_before_final_drain":record["client"]["client_receive_complete_ns"]<record["drained_ns"],
        "post_abort_observed_tokens":record["observed_post_abort_tokens"],"unobserved_cancelled_tokens":record["unobserved_cancelled_tokens"]}


def audit(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    summary=read(source/"summary.json");config=read(source/"config.yaml");manifest=read(source/"manifest.json")
    if summary["status"]!="COMPLETE" or summary["worker_returncode"]!=0 or summary["supervisor_timeout"] or not summary["native_frontend_drained"]:
        raise ValueError("Actual run must complete and drain before independent audit")
    output.mkdir(parents=True,exist_ok=False)
    for part in ("raw","source"):(output/part).mkdir()
    streams=sorted((source/"raw").glob("stream_*.jsonl"));start=time.perf_counter();cpu=time.process_time()
    write(output/"config.yaml",{"source_run":str(source),"source_config_sha256":digest(source/"config.yaml"),"source_summary_sha256":digest(source/"summary.json"),
        "source_files_sha256":{str(p.relative_to(source)):digest(p) for p in streams},"created_utc":datetime.now(timezone.utc).isoformat(),"auditor_sha256":digest(__file__),"gpu_used":False})
    (output/"commit.txt").write_text(subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True));shutil.copyfile(__file__,output/"source"/Path(__file__).name)
    checks=0;details=[];result={"status":"FAILED","records_status":"FAIL","gpu_used":False}
    def require(value,message):
        nonlocal checks
        checks+=1
        if not value:raise AssertionError(message)
    try:
        for path,expected in config["source_sha256"].items():require(digest(path)==expected,"Prepared source/config changed: "+path)
        records=list(rows(source/"raw/services.jsonl"));require(len(records)==len(manifest["tasks"])==summary["planned"]==summary["completed"],"All services retained")
        require(len({r["service_identity"] for r in records})==len(records),"Unique service identities")
        groups=defaultdict(list);n=0;resets=[]
        for path in streams:
            for event in rows(path):
                n+=1
                if event["event"]=="worker_snapshot":continue
                if event["event"]=="prefix_cache_reset":resets.append((event["case_id"],event["mode"]));continue
                require(event.get("service_identity") is not None,"Tagged service event required")
                groups[event["service_identity"]].append(event)
        require(resets==[(r["case_id"],r["mode"]) for r in records],"Every service received same cold prefix reset")
        require(set(groups)=={r["service_identity"] for r in records},"No orphan event groups")
        for record,entry in zip(records,manifest["tasks"]):
            require((record["case_id"],record["mode"])==(entry["case_id"],entry["mode"]),"Frozen service order")
            require(entry["status"]=="COMPLETE" and read(source/entry["checkpoint"])==record and digest(source/entry["checkpoint"])==entry["checkpoint_sha256"],"Checkpoint manifest integrity")
            details.append(check_service(record,groups[record["service_identity"]],require))
        writer=summary["recording"];require(writer["accepted"]==writer["written"]==n and writer["rejected"]==0 and writer["error"] is None,"Writer no loss")
        require(sum(r["observed_generated_tokens"] for r in records)==summary["observed_generated_tokens"],"Run token ledger")
        result.update(status="PASS",records_status="PASS",services=len(records),questions=len({r["case_id"] for r in records}),stream_rows=n,
            observed_tokens=summary["observed_generated_tokens"],observed_post_abort_tokens=summary["observed_after_abort_tokens"],
            source_process_status=summary["status"],correctness="PUB acceptance only; complete Dev scoring is separate",performance="Engineering only; no causal benefit claim",
            actual_gpu_busy_seconds="UNKNOWN",services_detail=details,actual_path_coverage=coverage(records))
    except Exception as exc:result["error"]=repr(exc)
    result.update(checks=checks,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
    (output/"raw/checks.jsonl").write_text("".join(json.dumps(r)+"\n" for r in details));write(output/"summary.json",result);write(output/"resource_ledger.json",result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=audit(a.source,a.output);print(json.dumps(r));return 0 if r["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
