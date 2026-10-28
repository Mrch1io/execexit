"""Same-token-prefix/same-segment-seed restart and synchronous public probes.

Physical restart calls are logged separately from the logical two-segment N.
No cached future trajectory or private correctness label enters this backend.
"""
from __future__ import annotations
import asyncio,copy,hashlib,json,multiprocessing,socket,sys,time
from exec_exit.gen.engine_preflight import FORCED_ANSWER_CONTROL,seed_for
from exec_exit.gen.probes import sampling_for
from exec_exit.gen.probes_v1 import generate_probe
from exec_exit.gen.traces import generate_case,render_prompt
from exec_exit.execute.extract import extract_record
from exec_exit.serve.async_backend import (ObservedEngine,client_once,recv_frame,frame,public_job,run_case as async_case)
from exec_exit.serve.async_probe import TokenAccount
from exec_exit.serve.hybrid_probe_positions import HybridPositions

class AcceptedProbe(Exception):
    def __init__(self,value):self.value=value

class MatchingPositions:
    """H control receives only sync's already-executed nominal position list."""
    def __init__(self,positions,close_id=151668,eos_ids=(151643,151645)):
        if positions!=sorted(set(positions)) or any(type(p) is not int or not 0<p<=4096 for p in positions):raise ValueError('Explicit ordered finite HYB schedule required')
        self.targets=positions;self.emitted=[];self.stops={close_id,*eos_ids};self.max_thinking=0;self.closed=False
    def __call__(self,ids):
        boundary=next((i for i,t in enumerate(ids) if t in self.stops),len(ids))
        self.max_thinking=max(self.max_thinking,boundary);self.closed=boundary<len(ids)
        out=[p for p in self.targets if p<=boundary and p not in self.emitted];self.emitted+=out;return out
    def metadata(self):return dict(selected_hyb_b_positions=self.emitted,matched_target_positions=self.targets,unreachable_positions=[p for p in self.targets if p not in self.emitted],max_inspected_thinking_tokens=self.max_thinking,schedule_source='Corresponding completed sync service actual nominal pause positions only; no labels or token text')

class RecordWriter:
    def __init__(self,writer,identity):self.writer=writer;self.identity=identity;self.calls={};self.probe_calls={}
    def add(self,event,**fields):
        self.writer.add(event,service_identity=self.identity,**fields)
        if event=='call_end':self.calls[fields['request_id']]=dict(fields)
        elif event=='probe_end':self.probe_calls[fields['probe_id']]=fields['call']

class ResumeEngine:
    """Pause an actual native request, drain, then re-submit its exact IDs.

The wrapper preserves the logical stage for generate_case/extraction. Physical
request boundaries, returned abort tails and still-unknown device work remain
separately observable; same random-number index is not token equality proof.
"""
    def __init__(self,engine,account,journal,positions,on_pause):
        self.engine=engine;self.account=account;self.journal=journal;self.positions=positions;self.on_pause=on_pause
        self.active={};self.physical={};self.pauses=[];self.stage_ids={};self.accepted=False
    async def abort(self,logical_id):
        actual=self.active.get(logical_id)
        if actual is not None:
            self.account.abort_requested(actual);await asyncio.wait_for(self.engine.abort(actual),15)
    async def generate(self,prompt,params,request_id):
        original=list(prompt['prompt_token_ids']);ids=[];self.stage_ids[request_id]=ids
        stage=0 if request_id.endswith('.stage0') else 1;attempt=0;max_tokens=params.max_tokens
        if params.min_tokens!=0 or params.presence_penalty!=0 or params.frequency_penalty!=0 or params.repetition_penalty!=1:raise ValueError('Output-history-dependent sampling cannot be silently restarted')
        while True:
            remaining=max_tokens-len(ids)
            if remaining<=0:raise RuntimeError('No observed natural terminal event at exhausted segment budget')
            actual=request_id+f'.resume{attempt}';physical_params=copy.deepcopy(params);physical_params.max_tokens=remaining
            submitted=time.perf_counter_ns();prefix=original+ids
            row=dict(request_id=actual,logical_request_id=request_id,stage=stage,restart_index=attempt,prompt_token_ids=list(prefix),seed=physical_params.seed,max_tokens=remaining,original_segment_max_tokens=max_tokens,kept_generated_prefix_tokens=len(ids),token_ids=[],started_ns=submitted,first_output_ns=None,finished=False,aborted=False,finish_reason=None)
            self.physical[actual]=row;self.active[request_id]=actual
            self.journal('physical_start',**copy.deepcopy(row))
            if attempt and self.pauses[-1].get('resume_submitted_ns') is None:
                pause=self.pauses[-1];pause['resume_submitted_ns']=submitted;pause['abort_to_resume_submit_seconds']=(submitted-pause['abort_requested_ns'])/1e9 if pause['abort_requested_ns'] is not None else None
                pause['resume_prompt_token_count']=len(prefix);pause['resume_prompt_sha256']=hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
                self.journal('resume_submitted',pause_index=len(self.pauses)-1,**pause)
            iterator=self.engine.generate({'prompt_token_ids':prefix},physical_params,actual).__aiter__();paused=False;terminal=False;drained=False
            def observe(out,after_abort=False):
                if len(out.outputs)!=1:raise ValueError('One completion required')
                completion=out.outputs[0];delta=list(completion.token_ids);offset=len(row['token_ids']);row['token_ids']+=delta;ids.extend(delta)
                now=time.perf_counter_ns()
                if delta and row['first_output_ns'] is None:
                    row['first_output_ns']=now
                    if attempt:
                        pause=self.pauses[-1];pause['resume_first_output_ns']=now;pause['resume_submit_to_first_output_seconds']=(now-submitted)/1e9
                        pause['abort_to_first_resumed_output_seconds']=(now-pause['abort_requested_ns'])/1e9 if pause['abort_requested_ns'] is not None else None
                self.account.observe(actual,delta);row.update(finished=bool(out.finished),finish_reason=completion.finish_reason)
                cached=getattr(out,'num_cached_tokens',None)
                if cached is not None:row['num_cached_tokens']=cached
                self.journal('physical_output',request_id=actual,logical_request_id=request_id,stage=stage,token_offset=offset,token_ids=delta,text=completion.text,native_frontend_receive_ns=time.perf_counter_ns(),finished=bool(out.finished),finish_reason=completion.finish_reason,num_cached_tokens=cached,after_abort=after_abort)
                return out
            try:
                async for output in iterator:
                    observe(output);yield output
                    terminal=bool(output.finished)
                    opportunities=self.positions(ids) if stage==0 else []
                    if not opportunities:
                        if terminal:break
                        continue
                    paused=True;requested=time.perf_counter_ns();before=len(ids)
                    pause=dict(nominal_positions=list(opportunities),pause_started_ns=requested,abort_requested_ns=requested if not terminal else None,request_already_finished=terminal,observed_prefix_before_abort_tokens=before,resume_submitted_ns=None,probe_wall_seconds=0.0,public_wall_seconds=0.0)
                    self.pauses.append(pause)
                    self.journal('pause_requested',pause_index=len(self.pauses)-1,**pause)
                    if not terminal:
                        row['aborted']=True;self.account.abort_requested(actual)
                        await asyncio.wait_for(self.engine.abort(actual),15);pause['abort_returned_ns']=time.perf_counter_ns()
                        async for tail in iterator:
                            observe(tail,True)
                            # A natural terminal event racing with abort is real.
                            if tail.finished and tail.outputs[0].finish_reason in ('stop','length'):terminal=True;yield tail
                            else:
                                visible=copy.copy(tail);visible.finished=False;visible.outputs=[copy.copy(tail.outputs[0])];visible.outputs[0].finish_reason=None;visible.outputs[0].stop_reason=None;yield visible
                        drained=True
                    extra=self.positions(ids) if stage==0 else []
                    opportunities=sorted(set(opportunities+extra));pause['nominal_positions']=list(opportunities);pause['additional_positions_observed_during_drain']=extra
                    pause.update(drained_ns=time.perf_counter_ns(),kept_prefix_tokens_after_drain=len(ids),observed_abort_tail_tokens=len(ids)-before,unobserved_abort_tail_tokens='UNKNOWN' if row['aborted'] else 0,natural_terminal_raced_abort=terminal and row['aborted'])
                    self.journal('pause_drained',pause_index=len(self.pauses)-1,**pause)
                    await self.on_pause(opportunities,list(ids),pause)
                    break
            finally:
                primary=sys.exc_info()[1];cleanup_error=None
                try:
                    if not terminal and not drained:
                        row['aborted']=True;self.account.abort_requested(actual)
                        await asyncio.wait_for(self.engine.abort(actual),15)
                        async def cleanup_drain():
                            async for tail in iterator:observe(tail,True)
                        await asyncio.wait_for(cleanup_drain(),15);drained=True
                    await asyncio.wait_for(iterator.aclose(),15)
                except BaseException as exc:cleanup_error=exc;row['cleanup_error']=repr(exc)
                if cleanup_error is None:self.active.pop(request_id,None)
                self.account.finished(actual,terminal and not row['aborted']);row['ended_ns']=time.perf_counter_ns();row['frontend_drained']=terminal or drained;self.journal('physical_end',**copy.deepcopy(row))
                if cleanup_error is not None and primary is None:raise cleanup_error
            if terminal:return
            if not paused:raise RuntimeError('Physical request ended without terminal event or intentional pause')
            attempt+=1

async def run_case(engine,tokenizer,extraction_tokenizer,codec,writer,pool,case,task,mode,eos_ids,parameter_factory,*,sample_index=0,matched_positions=None,confidence_settings=None):
    if mode in ('N','HYB-B-ASYNC'):return await async_case(engine,tokenizer,extraction_tokenizer,codec,writer,pool,case,task,mode,eos_ids,parameter_factory,sample_index=sample_index)
    if mode not in ('H','HYB-B-SYNC','BF@4096','CONF-50'):raise ValueError('Unregistered serial service mode')
    if mode!='H' and matched_positions is not None:raise ValueError('Only H can use another arm matching positions')
    identity=f'{case["case_id"]}.{mode}.s{sample_index}';account=TokenAccount();wrapped=RecordWriter(writer,identity)
    journal=lambda event,**fields:writer.add(event,service_identity=identity,**fields)
    if mode=='CONF-50':
        from exec_exit.serve.confidence_probe_positions import CandidatePositions
        if not isinstance(confidence_settings,dict) or set(confidence_settings)!={'k','minimum_thinking_tokens','threshold_strictly_greater_than'} or confidence_settings['threshold_strictly_greater_than']!=.95:raise ValueError('Explicit frozen CONF k/m and strict .95 threshold required')
        positions=CandidatePositions(codec,k=confidence_settings['k'],minimum=confidence_settings['minimum_thinking_tokens'])
    else:positions=MatchingPositions([4096] if mode=='BF@4096' else matched_positions) if mode=='BF@4096' or matched_positions is not None else HybridPositions(codec)
    native_record=None;visits=[];selected=None;transport=None;client=None;delivered_ns=None
    prompt_text,prompt_ids=render_prompt(tokenizer,case);control=tokenizer.encode(FORCED_ANSWER_CONTROL,add_special_tokens=False)
    probe_engine=ObservedEngine(engine,account,journal)
    parent,child=socket.socketpair();parent.setblocking(False)
    request=dict(case_id=case['case_id'],mode=mode,question_content=case['question_content'])
    async def on_pause(opportunities,prefix,pause):
        nonlocal selected,delivered_ns,transport
        pause['executed_nominal_positions']=[]
        if mode=='H':pause['executed_nominal_positions']=list(opportunities);return
        for p in opportunities:
            pause['executed_nominal_positions'].append(p)
            visit=dict(position=p,started_ns=time.perf_counter_ns(),pause_index=len(main_engine.pauses)-1);visits.append(visit)
            values=sampling_for(case['case_id']);values['seed']=seed_for(case['case_id'],sample_index,1)
            rid=case['case_id'].replace(':','_')+f'.sync.p{p:05d}'
            spec=dict(case_id=case['case_id'],task_id=case['task_id'],probe_id=rid,candidate_index=p,position_generated_tokens=p,prompt_token_ids=prompt_ids+prefix[:p]+control,control_token_ids=control,sampling_parameters=values)
            generator=generate_probe
            if mode=='CONF-50':
                from exec_exit.serve.conf50_probe import generate_conf50_probe
                generator=generate_conf50_probe
            started=time.perf_counter_ns();record=await generator(probe_engine,tokenizer,wrapped,spec,parameter_factory,codec=codec,special_ids=tokenizer.all_special_ids);pause['probe_wall_seconds']+=(time.perf_counter_ns()-started)/1e9
            extraction=await asyncio.to_thread(extract_record,record,'probe',extraction_tokenizer);extraction.update(online_visible_at=time.perf_counter_ns(),observation_scope='ACTUAL_ONLINE_EXTRACTION',clock='host_perf_counter_ns')
            value=dict(record=record,extraction=extraction,deliverable=extraction['deliverable'],comment_aborted=record['protocol_rejected']);visit.update(draft=value,draft_complete_ns=time.perf_counter_ns())
            if not value['deliverable'] or value['comment_aborted']:visit['status']='DRAFT_REJECTED';continue
            if mode=='CONF-50' and record['confidence_rejected']:visit['status']='CONFIDENCE_REJECTED';continue
            if mode in ('BF@4096','CONF-50'):visit['status']='ACCEPTED_WITHOUT_PUBLIC_CHECK';raise AcceptedProbe(value)
            started=time.perf_counter_ns();visit['public_started_ns']=started
            public=await asyncio.get_running_loop().run_in_executor(pool,public_job,extraction['code_text'],case['task_id'],task['public_test_cases']);visible=time.perf_counter_ns();public.update(server_check_queued_ns=started,server_check_visible_ns=visible)
            pause['public_wall_seconds']+=(visible-started)/1e9;visit.update(public=public,public_complete_ns=visible)
            if public['status']!='PASS':visit['status']='PUBLIC_REJECTED_OR_UNKNOWN';continue
            visit['status']='ACCEPTED';raise AcceptedProbe(value)
    main_engine=ResumeEngine(engine,account,journal,positions,on_pause)
    try:
        client=multiprocessing.get_context('spawn').Process(target=client_once,args=(child,request));client.start();child.close()
        submitted=await asyncio.wait_for(recv_frame(parent),60);server_receive=time.perf_counter_ns()
        if any(submitted[k]!=v for k,v in request.items()):raise ValueError('IPC request identity mismatch')
        journal('server_request_received',client_submit_ns=submitted['client_submit_ns'],server_receive_ns=server_receive)
        try:
            native_record=await generate_case(main_engine,tokenizer,wrapped,case,eos_ids,parameter_factory,lambda *a:None,sample_index=sample_index)
            extraction=await asyncio.to_thread(extract_record,native_record,'N',extraction_tokenizer);extraction.update(online_visible_at=time.perf_counter_ns(),observation_scope='ACTUAL_ONLINE_EXTRACTION',clock='host_perf_counter_ns')
            selected=dict(record=native_record,extraction=extraction);kind='N'
        except AcceptedProbe as accepted:selected=accepted.value;kind='PROBE'
        delivered_ns=time.perf_counter_ns();extraction=selected['extraction']
        packet=dict(kind=kind,case_id=case['case_id'],deliverable=bool(extraction['deliverable']),code_text=extraction.get('code_text'),extraction_status=extraction['status'],server_accept_ns=delivered_ns)
        journal('server_delivery_queued',kind=kind,visible_ns=delivered_ns);journal('delivery',kind=kind,visible_ns=delivered_ns)
        await asyncio.get_running_loop().sock_sendall(parent,frame(packet));journal('server_send_returned',visible_ns=time.perf_counter_ns())
        receipt=await asyncio.wait_for(recv_frame(parent),60);payload=frame(packet)[8:]
        if receipt['delivery_count']!=1 or receipt['client_submit_ns']!=submitted['client_submit_ns'] or receipt['response_sha256']!=hashlib.sha256(payload).hexdigest() or receipt['response_bytes']!=len(payload):raise ValueError('Actual IPC delivery mismatch')
        journal('client_receipt',**{k:v for k,v in receipt.items() if k!='event'});await asyncio.to_thread(client.join,10)
        if client.is_alive() or client.exitcode!=0:raise RuntimeError('Client failed to exit')
        for pause in main_engine.pauses:
            end=pause.get('resume_submitted_ns') or delivered_ns;pause['pause_until_resume_or_delivery_seconds']=(end-pause['pause_started_ns'])/1e9
            pause['host_control_seconds_excluding_probe_and_public'] = pause['pause_until_resume_or_delivery_seconds']-pause['probe_wall_seconds']-pause['public_wall_seconds']
            pause['timing_scope']='Abort-to-resume is inclusive; probe/public are subintervals, not additive to that total. First resumed output also includes resumed prefill/decode.'
        return dict(schema='exec_exit.c4.resume_dev.v1',status='COMPLETE',service_identity=identity,case_id=case['case_id'],mode=mode,client=receipt,server_receive_ns=server_receive,server_accept_ns=delivered_ns,drained_ns=time.perf_counter_ns(),selected_kind=kind,selected=selected,native_record=native_record,native_calls=wrapped.calls,probe_calls=wrapped.probe_calls,physical_calls=main_engine.physical,pauses=main_engine.pauses,visits=visits,coordinator_errors=[],token_account=account.requests,observed_generated_tokens=sum(r['observed_tokens'] for r in account.requests.values()),observed_post_abort_tokens=sum(r['observed_after_abort'] for r in account.requests.values()),unobserved_cancelled_tokens='UNKNOWN' if any(r['abort_ns'] is not None for r in account.requests.values()) else 0,candidate_observation=positions.metadata(),delivery_count=1,client_transport='Independent same-host IPC process; not HTTP/network',gpu_time_per_request='UNKNOWN_NATIVE_INTERFACE_NOT_ATTRIBUTED',private_tests_read=False,future_labels_or_trajectories_read=False,resume_protocol='Exact prompt+all returned token IDs; same original segment seed and remaining budget',matching_control=matched_positions is not None)
    finally:
        for logical in list(main_engine.active):await main_engine.abort(logical)
        parent.close();child.close()
        if client and client.is_alive():await asyncio.to_thread(client.join,2)
        if client and client.is_alive():client.terminate();await asyncio.to_thread(client.join,2)
