"""Frozen metadata-only C4 plan adapter. Data loading occurs only after freeze_gate.

No missing H decision, missing serial N latency, or generic LCB comparison is
silently filled. Same paired Poisson vector is mechanically scaled after N.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

BASE=['N','BF@4096','CONF-50','HYB-B-SYNC','HYB-B-ASYNC','H']

def checked(ref):
    path=Path(ref['path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
        raise ValueError('Frozen artifact differs: '+str(path))
    return path

def read_ref(ref):return json.loads(checked(ref).read_text())

def validate_spec(spec):
    if spec.get('schema')!='exec_exit.c4.postfreeze_service.v2':raise ValueError('Explicit v2 C4 specification required')
    if spec.get('model')!='qwen3_8b' or spec.get('task_protocol')!='stdin' or spec.get('sample_index')!=0:
        raise ValueError('Only Qwen stdin sample0 registered')
    for key in ('freeze_record','freeze_record_sha256','expected_gpu_uuid'):
        if not spec.get(key):raise ValueError('REQUIRED:'+key)
    roster=read_ref(spec['roster']);schedule=read_ref(spec['schedule']);unit=read_ref(spec['unit_poisson_arrivals'])
    ids=[r['case_id'] for r in roster]
    if len(ids)!=40 or len(set(ids))!=40 or ids!=spec['selected_case_ids']:raise ValueError('Exactly frozen 40 unique questions required')
    serial=spec['serial']; h=serial['H']['schedule']
    if h not in ('OWN_ONLINE','MATCHED_SYNC'):raise ValueError('H_SCHEDULE_REQUIRED')
    if serial['CONF-50']!={'k':1,'minimum_thinking_tokens':0,'threshold_strictly_greater_than':.95}:raise ValueError('Frozen CONF settings differ')
    if serial['HYB-B']!={'k':4,'minimum_thinking_tokens':0,'endpoint':4096,'probe_protocol':'v1'}:raise ValueError('Frozen HYB settings differ')
    if serial['prefix_cache_reset_policy']!='before_each_serial_arm_and_poisson_run':raise ValueError('Uniform prefix reset required')
    if spec['runtime']!={'public_execution_cache_scope':'per_arm','public_process_pool_workers':1,'case_timeout_seconds':5400,'max_model_len':40960}:
        raise ValueError('Registered execution controls differ')
    serial_rows=[r for r in schedule if r['kind']=='serial'];loads=[r for r in schedule if r['kind']=='poisson']
    if len(serial_rows)!=240 or len(loads)!=4:raise ValueError('All 240 serial arms and 4 loads required')
    for i,cid in enumerate(ids):
        arms=[r['mode'] for r in serial_rows if r['case_id']==cid]
        shift=i%6;expected=BASE[-shift:]+BASE[:-shift] if shift else list(BASE)
        if arms!=expected:raise ValueError('Exact cyclic order differs')
        if h=='MATCHED_SYNC' and arms.index('H')<arms.index('HYB-B-SYNC'):
            raise ValueError('Matching H requires corresponding SYNC first; cyclic schedule unresolved')
    if [(r['concurrency_target'],r['mode']) for r in loads]!=[(4,'N'),(4,'HYB-B-ASYNC'),(8,'HYB-B-ASYNC'),(8,'N')]:
        raise ValueError('Paired load order differs')
    if len(unit)!=40 or {r['case_id'] for r in unit}!=set(ids) or [r['request_index'] for r in unit]!=list(range(40)):
        raise ValueError('All 40 scheduled requests required')
    elapsed=0.
    for row in unit:
        gap=row['unit_exponential_gap']
        if not math.isfinite(gap) or gap<=0:raise ValueError('Positive finite exponential gap required')
        elapsed+=gap
        if elapsed!=row['cumulative_unit_exponential']:raise ValueError('Arrival sum differs')

def make_schedule(spec):
    result=copy.deepcopy(read_ref(spec['schedule']))
    for row in result:row['status']='PENDING'
    return result

def require_ds_blind_complete(spec):
    """Queue metadata only; pending DS work cannot be displaced by formal C4."""
    dependency=spec['execution_dependencies']
    rows=json.loads(Path(dependency['registry']).read_text())
    found={row['ticket_id']:row for row in rows}
    evidence=[]
    for name in dependency['tickets']:
        row=found.get(name)
        if not row or row.get('status')!='COMPLETE':raise ValueError('C4_WAIT_ALL_DS_BLIND:'+name)
        summary=Path(row['result_summary'])
        if json.loads(summary.read_text()).get('status')!='COMPLETE':raise ValueError('C4_WAIT_DS_SUMMARY:'+name)
        evidence.append(dict(ticket_id=name,summary=str(summary),summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest()))
    return evidence

def materialize_poisson(entry,spec,previous):
    values=[]
    for cid in spec['selected_case_ids']:
        record=previous.get((cid,'N'))
        if not record or record.get('status')!='COMPLETE':raise ValueError('All 40 serial N services must complete before deriving S')
        value=record['client']['client_e2e_seconds']
        if not math.isfinite(value) or value<=0:raise ValueError('Incomplete/invalid serial N client latency')
        values.append(value)
    S=sum(values)/len(values);c=entry['concurrency_target']
    arrivals=[dict(request_index=r['request_index'],case_id=r['case_id'],
        intended_arrival_seconds=r['cumulative_unit_exponential']*S/c) for r in read_ref(spec['unit_poisson_arrivals'])]
    entry.update(serial_N_mean_e2e_seconds=S,rate=c/S,arrivals=arrivals,
        arrival_realization_sha256=hashlib.sha256(json.dumps(arrivals,sort_keys=True,separators=(',',':')).encode()).hexdigest())

def load_tasks_after_freeze_impl(spec):
    """Caller MUST first validate the actual frozen record; only PUB is decoded."""
    from exec_exit.data.blind_hv6 import spans,decode_allowed
    from exec_exit.pipeline.postfreeze_evaluate import decoded_tests
    from exec_exit.execute.lcbpro_comparator import validate_spec as validate_comparator
    from collections import Counter
    sources=spec['public_sources'];roster=read_ref(spec['roster']);ids={r['case_id'] for r in roster}
    ac=read_ref(sources['hv6_runtime']);lc=read_ref(sources['lcbpro_runtime'])
    if ac['split']!='heldout_blind_generation' or lc['split']!='heldout_blind_generation':raise ValueError('Real heldout projection required')
    runtime={r['case_id']:r for r in ac['cases']+lc['cases'] if r['case_id'] in ids}
    if set(runtime)!=ids:raise ValueError('Selected runtime task missing')
    public={};audit=Counter()
    for raw in checked(sources['hv6_source']).open('rb'):
        positions=spans(raw)
        platform=decode_allowed(raw,positions,'platform',{'platform'},audit)
        qid=decode_allowed(raw,positions,'question_id',{'question_id'},audit);cid=platform+':'+qid
        if cid not in ids:continue
        value=decode_allowed(raw,positions,'public_test_cases',{'public_test_cases'},audit)
        public[cid]=decoded_tests(value,'stdin')
    lc_public=read_ref(sources['lcbpro_public'])
    if lc_public.get('visibility')!='public_examples_from_question_statement':raise ValueError('Explicit statement public samples required')
    for r in lc_public['tasks']:
        cid=r['task_id']
        if cid in ids:
            public[cid]=[{'input':x['input'],'output':x['output']} for x in r['examples']]
    specs=read_ref(sources['lcbpro_comparators'])['specs']
    comps={r['task_id']:validate_comparator(r) for r in specs if r['task_id'] in ids}
    tasks={}
    for cid in spec['selected_case_ids']:
        row=runtime[cid]
        if row['task_id']!=cid or row['task_type']!='stdin':raise ValueError('Exact stdin identity required')
        if cid.startswith('codeforces:') and cid not in comps:raise ValueError('Missing frozen LCB comparator')
        tests=public[cid]
        if not tests or any(set(t)!={'input','output'} or any(not isinstance(v,str) for v in t.values()) or len(t['input'].encode())>64*1024**2 for t in tests):
            raise ValueError('Exact nonempty public stdin samples within64MiB required')
        tasks[cid]=dict(case_id=cid,task_id=cid,question_content=row['question_content'],task_type='stdin',
            public_test_cases=tests,comparator_spec=comps.get(cid))
    return tasks

async def run_ready_service(mode,**kwargs):
    from exec_exit.serve.resume_backend import run_case
    if mode not in BASE:raise ValueError('Unregistered arm')
    h=kwargs.pop('h_schedule',None)
    if mode=='H':
        if h=='OWN_ONLINE':
            if kwargs.get('matched_positions') is not None:raise ValueError('Independent H cannot use other arm positions')
        elif h=='MATCHED_SYNC':
            if not isinstance(kwargs.get('matched_positions'),list):raise ValueError('Matched H requires actual sync positions')
        else:raise ValueError('H method not defined')
    return await run_case(mode=mode,**kwargs)
