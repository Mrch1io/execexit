"""为N和在线HYB-B提供原生AsyncLLM与隔离的公开测试后端。

不载入历史轨迹、候选标签、隐藏测试或未来到达信息。
IPC客户端提交与接收使用同一主机单调时钟，不代表HTTP或网络时延。
所有已观测输出保留在原始日志中。
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import struct
import time
from exec_exit.execute.extract import extract_record
from exec_exit.execute.run_tests import ExecutionCache,run_tests
from exec_exit.execute.sandbox import frozen_limits,make_sandbox
from exec_exit.gen.engine_preflight import FORCED_ANSWER_CONTROL,seed_for
from exec_exit.gen.probes import sampling_for
from exec_exit.gen.probes_v1 import generate_probe
from exec_exit.gen.traces import generate_case,render_prompt
from exec_exit.serve.async_probe import TokenAccount,serve
from exec_exit.serve.hybrid_probe_positions import HybridPositions

_jail=_cache=None


def init_public_worker(directory):
    global _jail,_cache
    root=Path(directory);_jail=make_sandbox(root/str(os.getpid()));_cache=ExecutionCache(root/"execution_cache")


def warm_public_worker():return {"pid":os.getpid(),"runtime":_jail.runtime_manifest}


def public_job(code,task_id,tests):
    started=time.perf_counter_ns()
    result=run_tests(code.encode("utf8"),task_id=task_id,tests=tests,visibility="public",
        limits=frozen_limits()|{"input_bytes":64*1024**2},sandbox=_jail,cache=_cache)
    return result|{"worker_visible_start_ns":started,"worker_complete_ns":time.perf_counter_ns(),
        "worker_pid":os.getpid(),"private_tests_read":False}


def frame(value):
    payload=json.dumps(value,ensure_ascii=False).encode("utf8")
    if len(payload)>64*1024**2:raise ValueError("IPC payload exceeds explicit 64MiB engineering bound")
    return struct.pack("!Q",len(payload))+payload


def recv_exact(sock,n):
    chunks=[]
    while n:
        part=sock.recv(n)
        if not part:raise EOFError("Client transport closed early")
        chunks.append(part);n-=len(part)
    return b"".join(chunks)


def client_once(sock,request):
    """由独立本地客户端记录实际提交与完整接收时间。"""
    try:
        started=time.perf_counter_ns();sock.sendall(frame(request|{"client_submit_ns":started}))
        n=struct.unpack("!Q",recv_exact(sock,8))[0]
        if n>64*1024**2:raise ValueError("IPC response too large")
        payload=recv_exact(sock,n);received=time.perf_counter_ns()
        answer=json.loads(payload)
        sock.sendall(frame({"event":"client_receipt","client_submit_ns":started,"client_receive_complete_ns":received,
            "client_e2e_seconds":(received-started)/1e9,"client_pid":os.getpid(),"answer_kind":answer["kind"],
            "response_sha256":hashlib.sha256(payload).hexdigest(),"response_bytes":n,"delivery_count":1}))
    finally:sock.close()


async def recv_frame(sock):
    loop=asyncio.get_running_loop()
    async def exact(n):
        chunks=[]
        while n:
            part=await loop.sock_recv(sock,n)
            if not part:raise EOFError("Server transport closed early")
            chunks.append(part);n-=len(part)
        return b"".join(chunks)
    n=struct.unpack("!Q",await exact(8))[0]
    if n>64*1024**2:raise ValueError("IPC frame too large")
    return json.loads(await exact(n))


class ObservedEngine:
    """记录每块原生返回数据；中止返回不代表GPU已完成。"""
    def __init__(self,engine,account,journal):
        self.engine=engine;self.account=account;self.journal=journal;self.active=set();self.forbid_new=False
    async def generate(self,prompt,params,request_id):
        if self.forbid_new:raise RuntimeError("Accepted answer prohibits new continuation requests")
        self.active.add(request_id);native_finished=False
        try:
            async for output in self.engine.generate(prompt,params,request_id):
                if len(output.outputs)!=1:raise ValueError("One output per native request required")
                self.account.observe(request_id,list(output.outputs[0].token_ids));native_finished=bool(output.finished)
                yield output
        finally:
            self.active.discard(request_id);self.account.finished(request_id,native_finished)
    async def abort(self,request_id):
        self.account.abort_requested(request_id)
        self.journal("native_abort_requested",request_id=request_id,visible_ns=time.perf_counter_ns())
        await asyncio.wait_for(self.engine.abort(request_id),15)
        self.journal("native_abort_returned",request_id=request_id,visible_ns=time.perf_counter_ns(),semantics="Frontend abort submitted; unexposed device work UNKNOWN")
    async def abort_all(self):
        self.forbid_new=True
        for rid in list(self.active):await self.abort(rid)


class CaseWriter:
    def __init__(self,writer,main_queue,identity):
        self.writer=writer;self.main_queue=main_queue;self.identity=identity;self.calls={};self.probe_calls={}
    def add(self,event,**fields):
        self.writer.add(event,service_identity=self.identity,**fields)
        if event=="call_output":
            self.main_queue.put_nowait({"token_ids":fields["token_ids"],"finished":False})
        elif event=="call_end":self.calls[fields["request_id"]]=dict(fields)
        elif event=="probe_end":self.probe_calls[fields["probe_id"]]=fields["call"]


async def run_case(engine,tokenizer,extraction_tokenizer,codec,writer,pool,case,task,mode,
                   eos_ids,parameter_factory,*,sample_index=0):
    if mode not in {"N","HYB-B-ASYNC"}:raise ValueError("Only explicit N/async engineering modes")
    identity=f'{case["case_id"]}.{mode}.s{sample_index}'
    account=TokenAccount();main_queue=asyncio.Queue(maxsize=4096);wrapped=CaseWriter(writer,main_queue,identity)
    journal=lambda event,**fields:writer.add(event,service_identity=identity,**fields)
    main_engine=ObservedEngine(engine,account,journal);probe_engine=ObservedEngine(engine,account,journal)
    positions=HybridPositions(codec);specials=tokenizer.all_special_ids
    main_cancelled=False;probe_cancelled=False;native_record=None;client=None;transport_task=None
    parent_sock,child_sock=socket.socketpair();parent_sock.setblocking(False)
    request={"case_id":case["case_id"],"mode":mode,"question_content":case["question_content"]}
    client=multiprocessing.get_context("spawn").Process(target=client_once,args=(child_sock,request));client.start();child_sock.close()
    delivered_packet=None
    async def send_answer(packet):
        await asyncio.get_running_loop().sock_sendall(parent_sock,frame(packet))
        journal("server_send_returned",visible_ns=time.perf_counter_ns())
    def deliver(kind,value,now_ns):
        nonlocal transport_task,delivered_packet
        if transport_task is not None:raise AssertionError("More than one client delivery")
        extraction=value["extraction"]
        delivered_packet={"kind":kind,"case_id":case["case_id"],"deliverable":bool(extraction["deliverable"]),
            "code_text":extraction.get("code_text"),"extraction_status":extraction["status"],"server_accept_ns":now_ns}
        transport_task=asyncio.create_task(send_answer(delivered_packet))
        journal("server_delivery_queued",kind=kind,visible_ns=now_ns)
    async def main_job():
        nonlocal native_record
        try:
            native_record=await generate_case(main_engine,tokenizer,wrapped,case,eos_ids,parameter_factory,
                lambda *args:None,sample_index=sample_index)
            if not main_cancelled:
                extraction=await asyncio.to_thread(extract_record,native_record,"N",extraction_tokenizer)
                extraction.update(online_visible_at=time.perf_counter_ns(),observation_scope="ACTUAL_ONLINE_EXTRACTION",clock="host_perf_counter_ns")
                await main_queue.put({"token_ids":[],"finished":True,"result":{"record":native_record,"extraction":extraction}})
        except BaseException as exc:
            if not main_cancelled:await main_queue.put({"error":exc})
            else:journal("main_cancelled_drain",error=str(exc),visible_ns=time.perf_counter_ns())
        finally:await main_queue.put(None)
    async def main_stream():
        job=asyncio.create_task(main_job())
        try:
            while True:
                event=await main_queue.get()
                if event is None:return
                if "error" in event:raise event["error"]
                yield event
        finally:await job
    async def abort_main():
        nonlocal main_cancelled
        main_cancelled=True;await main_engine.abort_all()
    async def abort_draft():
        nonlocal probe_cancelled
        probe_cancelled=True;await probe_engine.abort_all()
    async def draft(position,prefix):
        nonlocal probe_cancelled
        probe_cancelled=False
        rid=case["case_id"].replace(":","_")+f".c4.p{position:05d}"
        values=sampling_for(case["case_id"]);values["seed"]=seed_for(case["case_id"],sample_index,1)
        spec={"case_id":case["case_id"],"task_id":case["task_id"],"probe_id":rid,"candidate_index":position,
            "position_generated_tokens":position,"prompt_token_ids":prompt_ids+list(prefix)+control_ids,
            "control_token_ids":control_ids,"sampling_parameters":values}
        try:
            result=await generate_probe(probe_engine,tokenizer,wrapped,spec,parameter_factory,codec=codec,special_ids=specials)
            extraction=await asyncio.to_thread(extract_record,result,"probe",extraction_tokenizer)
            extraction.update(online_visible_at=time.perf_counter_ns(),observation_scope="ACTUAL_ONLINE_EXTRACTION",clock="host_perf_counter_ns")
            return {"record":result,"extraction":extraction,"deliverable":extraction["deliverable"],"comment_aborted":result["protocol_rejected"]}
        except Exception:
            if not probe_cancelled:raise
            return {"record":{"call":wrapped.probe_calls.get(rid,{})},"deliverable":False,"cancelled_after_native_delivery":True}
    async def check_public(output):
        loop=asyncio.get_running_loop();queued=time.perf_counter_ns()
        result=await loop.run_in_executor(pool,public_job,output["extraction"]["code_text"],case["task_id"],task["public_test_cases"])
        return result|{"server_check_queued_ns":queued,"server_check_visible_ns":time.perf_counter_ns()}
    try:
        submitted=await asyncio.wait_for(recv_frame(parent_sock),60);server_receive=time.perf_counter_ns()
        if any(submitted[k]!=v for k,v in request.items()):raise ValueError("IPC submitted request identity mismatch")
        journal("server_request_received",client_submit_ns=submitted["client_submit_ns"],server_receive_ns=server_receive)
        prompt_text,prompt_ids=render_prompt(tokenizer,case)
        control_ids=tokenizer.encode(FORCED_ANSWER_CONTROL,add_special_tokens=False)
        result=await serve(main_stream(),positions=positions if mode=="HYB-B-ASYNC" else lambda *a:[],draft=draft,
            check_public=check_public,abort_main=abort_main,abort_draft=abort_draft,journal=journal,on_delivery=deliver)
        if transport_task is None:raise RuntimeError("No real transport handoff")
        await transport_task;receipt=await asyncio.wait_for(recv_frame(parent_sock),60)
        if receipt["delivery_count"]!=1 or receipt["client_submit_ns"]!=submitted["client_submit_ns"]:raise ValueError("Client accounting mismatch")
        if receipt["client_receive_complete_ns"]<result["delivered_ns"]:raise ValueError("Client received before acceptance")
        payload=frame(delivered_packet)[8:]
        if receipt["response_sha256"]!=hashlib.sha256(payload).hexdigest() or receipt["response_bytes"]!=len(payload):raise ValueError("Actual client answer payload mismatch")
        journal("client_receipt",**{k:v for k,v in receipt.items() if k!="event"})
        await asyncio.to_thread(client.join,10)
        if client.is_alive() or client.exitcode!=0:raise RuntimeError("Client did not exit cleanly")
        if result["errors"]:
            journal("delivered_but_coordinator_failed",errors=result["errors"],client_received=True)
            raise RuntimeError("Client delivered, but unexpected coordinator errors: "+repr(result["errors"]))
        return {"schema":"exec_exit.c4.async_dev.v1","status":"COMPLETE","service_identity":identity,"case_id":case["case_id"],"mode":mode,
            "client":receipt,"server_receive_ns":server_receive,"server_accept_ns":result["delivered_ns"],"drained_ns":result["drained_ns"],
            "selected_kind":result["kind"],"selected":result["value"],"native_record":native_record,
            "native_calls":wrapped.calls,"probe_calls":wrapped.probe_calls,
            "visits":result["visits"],"coordinator_errors":result["errors"],"token_account":account.requests,
            "observed_generated_tokens":sum(r["observed_tokens"] for r in account.requests.values()),
            "observed_post_abort_tokens":sum(r["observed_after_abort"] for r in account.requests.values()),
            "unobserved_cancelled_tokens":"UNKNOWN" if any(r["abort_ns"] is not None for r in account.requests.values()) else 0,
            "candidate_observation":positions.metadata(),"delivery_count":1,"client_transport":"Independent same-host IPC process; not HTTP/network",
            "gpu_time_per_request":"UNKNOWN_NATIVE_INTERFACE_NOT_ATTRIBUTED","private_tests_read":False,"future_labels_or_trajectories_read":False}
    finally:
        if transport_task is not None and not transport_task.done():
            transport_task.cancel()
            await asyncio.gather(transport_task,return_exceptions=True)
        if client and client.is_alive():
            parent_sock.close();await asyncio.to_thread(client.join,2)
            if client.is_alive():client.terminate();await asyncio.to_thread(client.join,2)
        else:parent_sock.close()
