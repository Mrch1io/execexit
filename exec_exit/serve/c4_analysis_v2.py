"""Descriptive C4 latency/token/prefix analysis; no program/test execution.

Uses actual online records only. No blind-generation labels are joined. Missing
services retain the fixed denominator and suppress a complete primary estimate.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from exec_exit.serve.resume_audit import main_ids, compare_tokens

def quantile(values,q):
    values=sorted(values)
    if not values:return None
    x=(len(values)-1)*q;i=int(x);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(x-i)

def paired_ratio(numerator,denominator,*,repetitions=10000,seed=20261016):
    if len(numerator)!=len(denominator) or not numerator or any(not math.isfinite(v) or v<=0 for v in numerator+denominator):
        raise ValueError('Complete finite positive paired observations required')
    n=len(numerator);rng=random.Random(seed);values=[]
    for _ in range(repetitions):
        idx=[rng.randrange(n) for _ in range(n)]
        values.append(sum(numerator[i] for i in idx)/sum(denominator[i] for i in idx))
    return dict(ratio_of_means=sum(numerator)/sum(denominator),ci95_percentile=[quantile(values,.025),quantile(values,.975)],
        pairs=n,bootstrap_repetitions=repetitions,bootstrap_seed=seed)

def nominal(record):
    return [p for pause in record.get('pauses',[]) for p in pause['executed_nominal_positions']]

def main_before(record,cutoff,stage=None):
    out=[]
    for call in sorted(record['native_calls'].values(),key=lambda c:c['stage']):
        if stage is not None and call['stage']!=stage:continue
        ids=call['token_ids']
        if not ids:continue
        first,last=call.get('first_output_ns'),call.get('last_output_ns')
        if not isinstance(first,int) or not isinstance(last,int):return None
        if last<=cutoff:out.extend(ids)
        elif first>cutoff:continue
        else:return None  # A straddling multi-message call needs actual stream audit; no interpolation.
    return out

def prefix_check(sync,h,n,asynchronous=None):
    # Different arms run at different wall times. Compare the H portion whose
    # own visible generated positions precede SYNC's last visible main prefix,
    # not unrelated timestamps across runs.
    cutoff=sync.get('server_accept_ns')
    sync_stage0=main_before(sync,cutoff,0) if isinstance(cutoff,int) else None
    sync_all=main_before(sync,cutoff) if isinstance(cutoff,int) else None
    h_stage0=main_ids(h,0)
    common=dict(h_vs_n=compare_tokens(main_ids(h),main_ids(n)),
        async_vs_n=compare_tokens(main_ids(asynchronous),main_ids(n)) if asynchronous else 'UNKNOWN_MISSING_ASYNC',
        async_scope='All actually returned main tokens including observed cancellation tail; actual overlap only')
    if sync_stage0 is None or sync_all is None:
        return common|dict(sync_vs_h='UNKNOWN_PREACCEPT_TOKEN_BOUNDARY',nominal_positions_equal='UNKNOWN_PREACCEPT_TOKEN_BOUNDARY',
            reason='Missing timestamp or native call straddles acceptance; use actual stream audit, no timestamp interpolation')
    boundary=len(sync_stage0)
    a=nominal(sync);b=[p for p in nominal(h) if p<=boundary]
    return dict(sync_selected_kind=sync['selected_kind'],sync_main_stage0_visible_tokens=boundary,
        schedule_comparison_scope='Nominal positions through SYNC accepted/end stage0 observed prefix; H positions beyond this boundary listed separately',
        sync_positions=a,h_positions_in_sync_prefix=b,h_later_positions=[p for p in nominal(h) if p>boundary],
        nominal_positions_equal=a==b,
        sync_vs_h=compare_tokens(sync_all,main_ids(h)[:len(sync_all)]),
        sync_vs_h_thinking=compare_tokens(sync_stage0,h_stage0[:boundary]),
        boundary_evidence='Whole observed calls included only when last_output_ns <= server_accept_ns; no straddling interpolation',
        caveat='Identical nominal lists do not establish identical prefix/state; H later prefix excluded from SYNC/H agreement',**common)

def reject_group(record):
    visits=record.get('visits',[])
    generated=[v for v in visits if v.get('status')!='NOT_STARTED_AFTER_DELIVERY' and ('draft' in v or 'started_ns' in v)]
    if not generated:return 'NO_STARTED_PROBE'
    if any(v.get('status') in ('ACCEPTED','ACCEPTED_WITHOUT_PUBLIC_CHECK') for v in generated):return 'ACCEPTED'
    allowed={'DRAFT_REJECTED','PUBLIC_REJECTED_OR_UNKNOWN','CONFIDENCE_REJECTED'}
    if all(v.get('status') in allowed and (v.get('status')!='PUBLIC_REJECTED_OR_UNKNOWN' or v.get('public',{}).get('status')=='FAIL') for v in generated):return 'ALL_COMPLETED_PROBES_REJECTED'
    return 'NO_ACCEPTANCE_BUT_NOT_ALL_DECISIONS_KNOWN_REJECTED'

def p5_description(ids,serial,mode):
    n=sum(serial[cid,'N']['observed_generated_tokens'] for cid in ids)
    a=sum(serial[cid,mode]['observed_generated_tokens'] for cid in ids)
    latency_ratio=sum(serial[cid,mode]['client']['client_e2e_seconds'] for cid in ids)/sum(serial[cid,'N']['client']['client_e2e_seconds'] for cid in ids)
    latency_saving=1-latency_ratio;token_saving=1-a/n if n else None
    groups=Counter();sub=[];unknown=[];known=[]
    for cid in ids:
        group=reject_group(serial[cid,mode]);groups[group]+=1
        if group in ('ALL_COMPLETED_PROBES_REJECTED','NO_ACCEPTANCE_BUT_NOT_ALL_DECISIONS_KNOWN_REJECTED'):
            sub.append(cid)
            (known if group=='ALL_COMPLETED_PROBES_REJECTED' else unknown).append(cid)
    ratio=sum(serial[cid,mode]['client']['client_e2e_seconds'] for cid in sub)/sum(serial[cid,'N']['client']['client_e2e_seconds'] for cid in sub) if sub else None
    known_ratio=sum(serial[cid,mode]['client']['client_e2e_seconds'] for cid in known)/sum(serial[cid,'N']['client']['client_e2e_seconds'] for cid in known) if known else None
    return dict(descriptive_only=True,latency_saving=latency_saving,observed_token_saving=token_saving,
        latency_saving_at_least_80percent_observed_token_saving=latency_saving>=.8*token_saving if token_saving is not None else None,
        probe_decision_groups=dict(groups),nonaccepting_with_actual_probe_case_ids=sub,
        known_all_rejected_case_ids=known,unknown_or_raced_decision_case_ids=unknown,
        subgroup_definition='P5: >=1 actual probe and all clearly rejected; no pending, cancelled, unknown or acceptance. Nonaccepting is supplementary only',
        all_rejected_latency_ratio=known_ratio,all_rejected_overhead_at_most_5percent=known_ratio<=1.05 if known_ratio is not None else 'UNKNOWN_NO_IDENTIFIABLE_ALL_REJECTED_REQUEST',
        subgroup_identification='PARTIAL_UNKNOWN_REQUESTS_SEPARATELY_LISTED' if unknown else 'COMPLETE',
        nonaccepting_latency_ratio_supplementary=ratio,
        unknown_inflight_work='Unexposed cancellation remains UNKNOWN; no physical token guarantee')

def analyze(spec,records):
    ids=spec['selected_case_ids'];serial={};concurrent={}
    for row in records:
        e,r=row['experiment'],row['record']
        key=(e['case_id'],e['mode']) if e['kind']=='serial' else (e['concurrency_target'],e['mode'])
        target=serial if e['kind']=='serial' else concurrent
        if key in target:raise ValueError('Duplicate service experiment')
        target[key]=r
    result=dict(status='COMPLETE',planned_questions=len(ids),primary=None,serial={},poisson={},prefix_checks=[],
        correctness='NOT_SCORED_HERE; actual delivered programs require separate frozen full-test scoring',
        tokens='observed_generated_tokens already includes observed cancelled/post-abort output; never add tail twice',
        no_drop_missing=True,missing=[],P5={'descriptive_only':True})
    by_mode={}
    for mode in spec['serial']['base_order']:
        values=[];tokens=[];unknown_cancel=0
        for cid in ids:
            r=serial.get((cid,mode))
            if not r or r.get('status')!='COMPLETE' or not isinstance(r.get('client',{}).get('client_e2e_seconds'),(int,float)) or not math.isfinite(r['client']['client_e2e_seconds']) or r['client']['client_e2e_seconds']<=0:
                result['missing'].append(dict(case_id=cid,mode=mode));continue
            values.append(r['client']['client_e2e_seconds']);tokens.append(r['observed_generated_tokens'])
            unknown_cancel+=r.get('unobserved_cancelled_tokens')=='UNKNOWN'
        result['serial'][mode]=dict(planned=len(ids),complete=len(values),mean_e2e_seconds=sum(values)/len(values) if len(values)==len(ids) else 'UNKNOWN_INCOMPLETE',
            observed_token_total=sum(tokens),unknown_cancelled_work_requests=unknown_cancel)
        if len(values)==len(ids):by_mode[mode]=values
    if 'N' in by_mode and 'HYB-B-ASYNC' in by_mode:
        result['primary']=paired_ratio(by_mode['HYB-B-ASYNC'],by_mode['N'])
        result['P5']['ASYNC']=p5_description(ids,serial,'HYB-B-ASYNC')
    else:result['status']='INCOMPLETE_PRIMARY_NO_COMPLETE_CASE_DELETION'
    if 'N' in by_mode and 'HYB-B-SYNC' in by_mode:
        result['secondary_sync_ratio']=paired_ratio(by_mode['HYB-B-SYNC'],by_mode['N'])
        result['P5']['SYNC']=p5_description(ids,serial,'HYB-B-SYNC')
    for cid in ids:
        if all(serial.get((cid,m),{}).get('status')=='COMPLETE' for m in ('N','H','HYB-B-SYNC')):
            result['prefix_checks'].append(dict(case_id=cid,**prefix_check(serial[cid,'HYB-B-SYNC'],serial[cid,'H'],serial[cid,'N'],serial.get((cid,'HYB-B-ASYNC')))))
    for c in (4,8):
        modes={};counts={}
        for mode in ('N','HYB-B-ASYNC'):
            r=concurrent.get((c,mode),{});rs=r.get('records',[])
            ok=[x for x in rs if x.get('status')=='COMPLETE' and isinstance(x.get('intended_arrival_to_client_complete_seconds'),(int,float)) and math.isfinite(x['intended_arrival_to_client_complete_seconds']) and x['intended_arrival_to_client_complete_seconds']>0]
            counts[mode]=dict(planned=len(ids),complete=len(ok))
            if len(rs)==len(ids) and len(ok)==len(ids) and {x['case_id'] for x in rs}==set(ids) and [x['request_index'] for x in rs]==list(range(len(ids))):
                modes[mode]=[x['intended_arrival_to_client_complete_seconds'] for x in rs]
        values=dict(status='INCOMPLETE' if len(modes)!=2 else 'COMPLETE',counts=counts,ratio={})
        if len(modes)==2:
            nr=concurrent[c,'N']['records'];ar=concurrent[c,'HYB-B-ASYNC']['records']
            if [(x['request_index'],x['case_id']) for x in nr]!=[(x['request_index'],x['case_id']) for x in ar]:raise ValueError('Paired concurrent request order differs')
            a,n=modes['HYB-B-ASYNC'],modes['N']
            values['ratio']={'mean':sum(a)/sum(n),'p50':quantile(a,.5)/quantile(n,.5),'p90':quantile(a,.9)/quantile(n,.9)}
        result['poisson'][str(c)]=values
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.run_dir/'config.yaml').read_text());spec=json.loads(Path(cfg['spec']).read_text())
    from exec_exit.serve.service_experiment_v2 import freeze_gate
    freeze_gate(spec)
    rows=[json.loads(line) for line in (a.run_dir/'raw/experiments.jsonl').read_text().splitlines() if line]
    summary=analyze(spec,rows);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (a.output/'config.yaml').write_text(json.dumps(dict(source=str(a.run_dir),spec=cfg['spec'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpu_used=False,test_execution=False),indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
