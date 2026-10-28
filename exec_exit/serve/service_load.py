"""Poisson N/async execution primitive with actual intended-arrival accounting.

The caller supplies frozen public tasks, an initialized engine/pool and the
paired arrival schedule. No dataset is read, selected or scored here. Repeated
case IDs get unique native request IDs without changing case-based seeds.
"""
from __future__ import annotations
import asyncio
import time
from exec_exit.serve.async_backend import run_case


class RequestNamespace:
    def __init__(self,engine,namespace):
        self.engine=engine;self.namespace=namespace;self.request_map={}
    def actual(self,request_id):
        actual=self.namespace+":"+request_id
        self.request_map[request_id]=actual
        return actual
    async def generate(self,prompt,params,request_id):
        async for output in self.engine.generate(prompt,params,self.actual(request_id)):yield output
    async def abort(self,request_id):return await self.engine.abort(self.actual(request_id))


class RequestWriter:
    def __init__(self,writer,request_key,intended_arrival_ns):
        self.writer=writer;self.request_key=request_key;self.intended=intended_arrival_ns
    def add(self,event,**fields):
        self.writer.add(event,workload_request_key=self.request_key,intended_arrival_ns=self.intended,**fields)


async def run_poisson(engine,*,mode,arrivals,tasks,writer,run_identity,sample_index=0,backend=run_case,**backend_arguments):
    if mode not in ("N","HYB-B-ASYNC"):raise ValueError("Poisson experiment only has frozen N/async controls")
    if not arrivals or [a["request_index"] for a in arrivals]!=list(range(len(arrivals))):raise ValueError("All explicitly scheduled requests required")
    if any(a["case_id"] not in tasks for a in arrivals):raise ValueError("No implicit task loading")
    if any(a["intended_arrival_seconds"]<0 for a in arrivals) or any(a["intended_arrival_seconds"]>b["intended_arrival_seconds"] for a,b in zip(arrivals,arrivals[1:])):raise ValueError("Monotonic nonnegative intended arrival schedule required")
    origin=time.perf_counter_ns();results=[None]*len(arrivals)
    async def one(item):
        index=item["request_index"];intended=origin+round(item["intended_arrival_seconds"]*1e9)
        await asyncio.sleep(max(0,(intended-time.perf_counter_ns())/1e9))
        actual_start=time.perf_counter_ns();key=run_identity+":"+str(index)
        scoped=RequestNamespace(engine,key);wrapped=RequestWriter(writer,key,intended);task=tasks[item["case_id"]]
        wrapped.add("scheduled_request_released",request_index=index,actual_release_ns=actual_start)
        case={"case_id":task["task_id"],"task_id":task["task_id"],"question_content":task["question_content"]}
        try:
            record=await backend(scoped,writer=wrapped,case=case,task=task,mode=mode,sample_index=sample_index,**backend_arguments)
            received=record["client"]["client_receive_complete_ns"]
            if received<intended:raise ValueError("Client receipt cannot precede intended arrival")
            result={"status":"COMPLETE","request_index":index,"case_id":case["case_id"],"workload_request_key":key,
                "intended_arrival_ns":intended,"actual_release_ns":actual_start,"client_receive_complete_ns":received,
                "intended_arrival_to_client_complete_seconds":(received-intended)/1e9,"native_request_id_map":scoped.request_map,"service_record":record}
        except Exception as exc:
            result={"status":"FAILED_SERVICE","request_index":index,"case_id":case["case_id"],"workload_request_key":key,
                "intended_arrival_ns":intended,"actual_release_ns":actual_start,"error":repr(exc),"native_request_id_map":scoped.request_map,
                "latency":"UNKNOWN_NO_COMPLETE_CLIENT_RECEIPT","cost":"PARTIAL_OBSERVATIONS_RETAINED_IN_RAW_STREAM"}
            wrapped.add("scheduled_request_failed",error=repr(exc))
        results[index]=result
    await asyncio.gather(*(one(a) for a in arrivals))
    if any(r is None for r in results):raise AssertionError("Scheduled denominator lost")
    return {"status":"COMPLETE" if all(r["status"]=="COMPLETE" for r in results) else "FAILED_SERVICE_PRESENT",
        "planned":len(arrivals),"complete":sum(r["status"]=="COMPLETE" for r in results),"origin_ns":origin,"drained_ns":time.perf_counter_ns(),
        "all_scheduled_requests_retained":True,"records":results,"gpu_time_per_request":"UNKNOWN_NATIVE_INTERFACE_NOT_ATTRIBUTED"}
