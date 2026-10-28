"""Heldout artifact adapter around frozen C1/C2 functions; no scoring or tuning.

Imports and preparation do not inspect outcomes. Execution is queued after the
model's complete scoring/replay, under the GPU resource lock to protect C4.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lines(path,values):
    with Path(path).open('w') as stream:
        for value in values:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def checked(ref):
    p=Path(ref['path'])
    if sha(p)!=ref['sha256']:raise ValueError('Changed input/source identity: '+str(p))
    return p
def unique(values,key):
    result={r[key]:r for r in values}
    if len(result)!=len(values):raise ValueError('Duplicate '+key)
    return result
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def lock(fd):
    if type(fd)is not int or fd<0:raise ValueError('Inherited resource lock required')
    if Path(os.readlink(f'/proc/self/fd/{fd}')).resolve()!=Path('/root/autodl-tmp/gpu.lock'):raise ValueError('Wrong resource lock')
    if 'FLOCK' not in Path(f'/proc/self/fdinfo/{fd}').read_text():raise ValueError('Descriptor does not own flock')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


def signal_rows(probe,evaluation,account,*,sample,difficulty,ordinal):
    """Project one complete draft and its saved virtual-v1 prefix separately."""
    from exec_exit.analysis.c1c2_development import confidence_values,failure_type
    if probe['probe_id']!=account['subject_id'] or probe['probe_id']!=evaluation['subject_id']:
        raise ValueError('Probe/evaluation/account identity mismatch')
    if account['received_generated_tokens']!=len(probe['call']['token_ids']):raise ValueError('Received-token count mismatch')
    rejected=account['protocol_rejected']
    if type(rejected)is not bool:raise ValueError('Unknown virtual-abort decision')
    full_signals,reasons=confidence_values(probe,evaluation)
    public=evaluation.get('public_result',{}).get('status')
    deliverable=bool(evaluation['extraction'].get('deliverable'))
    category_view=copy.deepcopy(evaluation)
    failure_sources={}
    for key in ('public_result','private_result'):
        suite=category_view.get(key,{})
        if suite.get('case_status_sequence'):
            failure_sources[key]='RECORDED_SEQUENCE'
        else:
            kinds=[k for k,v in suite.get('case_status_counts',{}).items()
                   if v>0 and k not in ('PASS','NOT_EVALUATED_AFTER_FAILURE')]
            if len(kinds)==1:
                suite['case_status_sequence']=kinds
                failure_sources[key]='UNIQUE_FAILURE_KIND_IN_SAVED_COUNTS'
            else:
                failure_sources[key]='NO_FAILURE_OR_ORDER_UNKNOWN'
    category,statuses=failure_type(category_view)
    full_signals.update(PUB=1 if public=='PASS' else 0 if public=='FAIL' else None,
        PUB_GATE=int(deliverable and public=='PASS'),NOT_ABORTED=int(not rejected))
    common=dict(sample=sample,case_id=probe['case_id'],subject_id=probe['probe_id'],difficulty=difficulty,
        position=probe['position_generated_tokens'],probe_ordinal=ordinal,
        generated_tokens=len(probe['call']['token_ids']),code_sha256=evaluation.get('code_sha256'))
    complete=dict(common,signals=full_signals,signal_availability=reasons,deliverable=deliverable,
        outcome=evaluation['final_outcome'],public_status=public or 'UNKNOWN',protocol_rejected=False,
        virtual_rule_rejected=rejected,counterfactual_complete_correctness=evaluation['final_outcome'],
        failure_category=category,failure_statuses=statuses,failure_category_source=failure_sources,
        corpus='complete_v0_auxiliary')
    virtual=copy.deepcopy(complete)
    virtual.update(corpus='virtual_v1_pipeline',protocol_rejected=rejected,
                   generated_tokens=account['effective_v1_tokens'])
    if rejected:
        cutoff=account['effective_v1_tokens']
        if not 0<cutoff<=len(probe['call']['token_ids']):raise ValueError('Invalid saved virtual cutoff')
        prefix=copy.deepcopy(probe)
        for key in ('token_ids','rank_one_logprobs'):
            prefix['call'][key]=prefix['call'].get(key,[])[:cutoff]
        prefix['call']['native_finished']=False
        rejected_evaluation=copy.deepcopy(evaluation)
        rejected_evaluation['extraction']['deliverable']=False
        conf,why=confidence_values(prefix,rejected_evaluation)
        conf.update(PUB=None,PUB_GATE=0,NOT_ABORTED=0)
        virtual.update(signals=conf,signal_availability=why,deliverable=False,outcome='FAIL',
            public_status='NOT_EVALUATED_VIRTUAL_ABORT',failure_category='PROTOCOL_REJECTION',
            failure_statuses=[],counterfactual_complete_correctness=evaluation['final_outcome'])
    if virtual['outcome']=='PASS' and (rejected or not virtual['deliverable']):raise ValueError('PASS without valid v1 delivery')
    return virtual,complete


def opportunity(native,annotation,native_evaluation,probes,*,sample,difficulty,token_metadata):
    first=next(c for c in native['calls'] if c['stage']==0)
    terminal={token_metadata['thinking_close_token_id'],*token_metadata['eos_token_ids']}
    end=next((i for i,t in enumerate(first['token_ids']) if t in terminal),len(first['token_ids']))
    if end!=annotation['eligible_thinking_generated_tokens']:raise ValueError('Model-specific thinking boundary differs from audited annotation')
    if sum(len(c['token_ids']) for c in native['calls'])!=native['generated_tokens']:raise ValueError('Native token ledger mismatch')
    passing=[r for r in probes if r['outcome']=='PASS']
    earliest=min(passing,key=lambda r:r['position']) if passing else None
    native_outcome=native_evaluation['final_outcome']
    bounds={}
    for include in (True,False):
        options=[(native['generated_tokens'],'N:'+native['case_id'])] if native_outcome=='PASS' else []
        options += [(r['position']+(r['generated_tokens'] if include else 0),r['subject_id']) for r in passing]
        cost,selected=min(options) if options else (native['generated_tokens'],'N:'+native['case_id'])
        bounds['delivery_charged' if include else 'delivery_free']=dict(tokens=cost,selected=selected,outcome='PASS' if options else native_outcome)
    unknown=native_outcome=='UNKNOWN' or any(r['outcome']=='UNKNOWN' for r in probes)
    return dict(sample=sample,case_id=native['case_id'],difficulty=difficulty,native_generated_tokens=native['generated_tokens'],
        native_thinking_tokens=end,native_outcome=native_outcome,thinking_cap_hit=native.get('thinking_budget_hit',False),
        candidate_count=len(probes),correct_probe_count=len(passing),first_full_pass_probe_id=earliest['subject_id'] if earliest else None,
        first_full_pass_position=earliest['position'] if earliest else None,
        observed_position_margin=end-earliest['position'] if earliest else None,
        margin_interpretation='Cap-relative margin; natural end censored' if native.get('thinking_budget_hit') else 'Observed native-end-relative position margin',
        absence=('UNKNOWN_INCOMPLETE_LABELS' if unknown else 'NO_PASS_OBSERVED_AT_TESTED_CANDIDATES') if not passing else None,
        later_probe_failures_after_first_pass=sum(r['position']>earliest['position'] and r['outcome']=='FAIL' for r in probes) if earliest else None,
        unknown_outcomes=unknown,bounds=bounds)


def load_job(job,config,task_map):
    base=Path(config['scoring_run'])/'jobs'/job['job_id']
    evaluation_dirs={kind:base/(job['job_id']+'_'+kind+'_evaluation') for kind in ('N','probe')}
    replay=base/(job['job_id']+'_replay')
    for path in [base/'summary.json',replay/'summary.json',*[d/'summary.json' for d in evaluation_dirs.values()]]:
        if read(path).get('status')!='COMPLETE':raise ValueError('Incomplete scoring/replay dependency '+str(path))
    refs={key:job['artifacts'][key] for key in ('native','probes','annotations')}
    data_paths={key:checked(ref) for key,ref in refs.items()}
    for key,directory in evaluation_dirs.items():data_paths[key+'_evaluation']=directory/'raw/evaluations.jsonl'
    data_paths['probe_accounts']=replay/'raw/probe_accounts.jsonl'
    identities={key:dict(path=str(path),sha256=sha(path)) for key,path in data_paths.items()}
    natives=unique(rows(data_paths['native']),'case_id');probes=unique(rows(data_paths['probes']),'probe_id')
    annotations=unique(rows(data_paths['annotations']),'case_id');ne=unique(rows(data_paths['N_evaluation']),'case_id')
    pe=unique(rows(data_paths['probe_evaluation']),'subject_id');accounts=unique(rows(data_paths['probe_accounts']),'subject_id')
    if not (set(natives)==set(annotations)==set(ne)==set(job['case_ids'])):raise ValueError('Question denominator mismatch')
    if not (set(probes)==set(pe)==set(accounts)):raise ValueError('Full candidate scoring/account denominator mismatch')
    expected={(cid,c['position_generated_tokens']) for cid,a in annotations.items() for c in a['candidates']}
    if {(p['case_id'],p['position_generated_tokens']) for p in probes.values()}!=expected or len(probes)!=len(expected):raise ValueError('All frozen candidates required')
    grouped=defaultdict(list)
    for p in probes.values():grouped[p['case_id']].append(p)
    virtual=[];complete=[];opportunities=[]
    sample=str(job['sample_index'])
    for case,native in natives.items():
        difficulty=task_map[case]['difficulty'];local=[]
        for ordinal,p in enumerate(sorted(grouped[case],key=lambda r:r['position_generated_tokens']),1):
            v,c=signal_rows(p,pe[p['probe_id']],accounts[p['probe_id']],sample=sample,difficulty=difficulty,ordinal=ordinal)
            v['collection']=c['collection']=job['dataset'];virtual.append(v);complete.append(c);local.append(v)
        o=opportunity(native,annotations[case],ne[case],local,sample=sample,difficulty=difficulty,token_metadata=config['token_metadata'])
        o['collection']=job['dataset'];opportunities.append(o)
    return virtual,complete,opportunities,identities


def c1_summaries(opportunities):
    from exec_exit.analysis.c1c2_development import bounds_summary
    # The frozen function treats non-PASS as False. Never call it on UNKNOWN
    # labels: keep the full denominator and explicit UNKNOWN estimates instead.
    result={}
    for difficulty in ('all','easy','medium','hard'):
        selected=[r for r in opportunities if difficulty=='all' or r['difficulty']==difficulty]
        if not selected or any(r['unknown_outcomes'] for r in selected):
            result[difficulty]=dict(status='UNKNOWN_INCOMPLETE_LABELS' if selected else 'UNKNOWN_EMPTY_STRATUM',
                questions=len({r['case_id'] for r in selected}),question_samples=len(selected),
                unknown_question_samples=sum(r['unknown_outcomes'] for r in selected),
                fixed_denominator_retained=True,delivery_charged=None,delivery_free=None)
        else:
            # frozen bounds_summary always traverses all difficulties. Calling
            # it on the whole population preserves original defaults/estimates.
            result[difficulty]=None
    complete_keys=[d for d,v in result.items() if v is None]
    if complete_keys:
        # Each planned collection contains all three difficulties. If any has
        # UNKNOWN, retain UNKNOWN rather than quietly using complete cases in
        # a helper which assumes complete labels. Known raw values remain saved.
        if len(complete_keys)==4:
            return {k:dict(status='COMPLETE',**v) for k,v in bounds_summary(opportunities).items()}
        for d in complete_keys:
            result[d]=dict(status='UNKNOWN_OTHER_STRATUM_INCOMPLETE',questions=len({r['case_id'] for r in opportunities if d=='all' or r['difficulty']==d}),
                question_samples=sum(d=='all' or r['difficulty']==d for r in opportunities),fixed_denominator_retained=True,
                delivery_charged=None,delivery_free=None,reason='Do not call frozen complete-case helper with any UNKNOWN; raw known observations remain available')
    return result


def verify_prepare(config):
    frozen=read(checked(config['freeze']))
    if frozen['state']!='FROZEN' or frozen['git_commit']!=config['frozen_commit']:raise ValueError('Frozen identity mismatch')
    inventory={x['path']:x['sha256'] for x in frozen['source_inventory']['files']}
    for key,ref in config['frozen_functions'].items():
        checked(ref)
        if inventory.get(ref['path'])!=ref['sha256']:raise ValueError('Unfrozen function '+key)
    for ref in config['model_metadata'].values():checked(ref)
    tok=read(config['model_metadata']['tokenizer']['path']);gen=read(config['model_metadata']['generation_config']['path'])
    close=next(t['id'] for t in tok['added_tokens'] if t['content']=='</think>')
    eos=gen['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
    if dict(thinking_close_token_id=close,eos_token_ids=eos)!=config['token_metadata']:raise ValueError('Locked model tokens changed')
    contract=read(checked(config['statistics_contract']))
    reference=read(checked(config['reference_question_contract']))
    primary=read(checked(frozen['primary_spec']))
    if primary['statistics_contract']!=config['reference_question_contract']:
        raise ValueError('Question reference not bound to original frozen primary spec')
    if contract['questions']!=reference['questions'] or contract['samples_by_collection']!=reference['samples_by_collection']:
        raise ValueError('Model diagnostic question/sample population differs from frozen population')
    if len(contract['questions'])!=199 or contract['samples_by_collection']!={'hv6_atcoder':4,'hv6_leetcode':4,'lcbpro_2025':2}:raise ValueError('Frozen denominator changed')
    if config['model']=='dsr1_distill_qwen7b' and frozen['included_secondary_models'].get(config['model'])!=contract['samples_by_collection']:
        raise ValueError('DS secondary scope not included in frozen receipt')
    if len({q['case_id'] for q in contract['questions']})!=199:raise ValueError('Duplicate question across collections')
    jobs=[]
    for ref in config['job_manifests']:jobs+=read(checked(ref))['jobs']
    if len(jobs)!=10 or sum(len(j['case_ids']) for j in jobs)!=650:raise ValueError('Expected frozen 10 jobs /650 question samples')
    actual=Counter((j['dataset'],c,j['sample_index']) for j in jobs for c in j['case_ids'])
    expected=Counter((q['collection'],q['case_id'],s) for q in contract['questions'] for s in range(contract['samples_by_collection'][q['collection']]))
    if actual!=expected or any(j['model']!=config['model'] or j['probe_mode']!='virtual_v1' for j in jobs):raise ValueError('Job sample/model/protocol identity mismatch')
    return jobs,contract


def execute(manifest,output,fd):
    lock(fd);config=read(manifest);jobs,contract=verify_prepare(config)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU active at CPU analysis boundary')
    if read(Path(config['scoring_run'])/'summary.json').get('status')!='COMPLETE':raise RuntimeError('Wait for complete model scoring and replay')
    for ref in config['source_sha256'].items():checked(dict(path=ref[0],sha256=ref[1]))
    output.mkdir(parents=True,exist_ok=False);(output/'raw').mkdir();(output/'source').mkdir()
    save(output/'config.yaml',config);(output/'commit.txt').write_text(config['frozen_commit']+'\n');shutil.copy2(__file__,output/'source'/Path(__file__).name)
    os.nice(19);start=time.perf_counter();cpu=time.process_time();stage='load_inputs';completed={}
    try:
        from exec_exit.analysis.grouped_signals import folds_for_questions,compare_signal
        from exec_exit.analysis.c1c2_development import plot_c1
        from exec_exit.gen.host_environment import collect_host_environment
        save(output/'machine_identity.json',collect_host_environment())
        tasks={q['case_id']:q for q in contract['questions']};folds=folds_for_questions(tasks)
        save(output/'raw/fold_assignment.json',folds)
        pipeline=[];full=[];opportunities=[];identities={}
        for job in jobs:
            v,c,o,ids=load_job(job,config,tasks);pipeline+=v;full+=c;opportunities+=o;identities[job['job_id']]=ids
        save(output/'raw/input_identities.json',identities)
        if len(opportunities)!=650:raise ValueError('Expected all650 planned native records')
        lines(output/'raw/c1_opportunities.jsonl',opportunities);lines(output/'raw/probe_signals.jsonl',pipeline+full)
        populations={'pooled':list(tasks)}|{'collection:'+c:[q['case_id'] for q in contract['questions'] if q['collection']==c] for c in contract['samples_by_collection']}
        stage='c1_aggregation'
        c1={name:c1_summaries([r for r in opportunities if r['case_id'] in ids]) for name,ids in populations.items()}
        save(output/'raw/c1_summary.json',dict(status='COMPLETE',model=config['model'],c1=c1,
            question_samples=len(opportunities),opportunities_sha256=sha(output/'raw/c1_opportunities.jsonl')))
        completed['raw/c1_summary.json']=dict(path=str(output/'raw/c1_summary.json'),sha256=sha(output/'raw/c1_summary.json'))
        stage='c2_analysis'
        metrics=[];predictions=[];models=[];gaps=[]
        for population,ids in populations.items():
            for corpus,data in [('virtual_v1_pipeline',pipeline),('complete_v0_auxiliary',full)]:
                data=[r for r in data if r['case_id'] in ids]
                for scope,subset in [('pipeline',data),('complete_deliverable',[r for r in data if r['deliverable']])]:
                    for signal in ('CONF-50','CONF-full','PUB','PUB_GATE','NOT_ABORTED'):
                        try:result,pred,model=compare_signal(subset,signal,folds)
                        except (ValueError,RuntimeError) as exc:
                            # No fitting retry, changed folds, imputation, extra
                            # sample or hyperparameter tuning on heldout labels.
                            result=dict(status='UNKNOWN_UNCOMPUTABLE_WITH_FROZEN_FIT',signal_name=signal,error=repr(exc),source_rows=len(subset));pred=[];model=[]
                            gaps.append(dict(population=population,corpus=corpus,scope=scope,**result))
                        tag=dict(population=population,corpus=corpus,scope=scope)
                        metrics.append(dict(result,**tag));predictions.extend(dict(r,**tag) for r in pred);models.extend(dict(r,**tag) for r in model)
                        print(json.dumps(dict(event='SIGNAL_COMPLETE',**tag,signal=signal,status=result['status'])),flush=True)
        false_accept=[r for r in full if r['public_status']=='PASS' and r['outcome']=='FAIL']
        false_reject=[r for r in full if r['virtual_rule_rejected'] and r['outcome']=='PASS']
        lines(output/'raw/signal_metrics.jsonl',metrics);lines(output/'raw/oof_predictions.jsonl',predictions);lines(output/'raw/fold_models.jsonl',models)
        lines(output/'raw/false_acceptances.jsonl',false_accept);lines(output/'raw/false_rejections.jsonl',false_reject)
        summary=dict(status='COMPLETE',model=config['model'],scope='heldout_diagnostic_separate_from_primary_policy_inference',
            questions=199,question_samples=650,probes=len(pipeline),samples_by_collection=contract['samples_by_collection'],
            c1=c1,c2=metrics,analysis_gaps=gaps,false_acceptance_categories=dict(Counter(r['failure_category'] for r in false_accept)),
            false_rejected_complete_correct_drafts=len(false_reject),
            no_model_or_candidate_execution=True,test_payload_read=False,
            uncertainty='Frozen C1/C2 question-cluster bootstrap seed20261014/10000; C2 fixed5fold OOF, no model-refit bootstrap; not primary stratified seed20261018',
            signal_scope='Virtual-v1 pipeline and complete-v0 auxiliary reported separately; not an online-cancellation equivalence claim',
            missing_data='C2 matched known signal/label coverage explicit; no UNKNOWN->FAIL; C1 unresolved population estimates UNKNOWN with fixed denominator')
        # Completed analysis is immutable across plotting/reporting failures.
        # summary.json is the whole pipeline status, never its only copy.
        save(output/'analysis_summary.json',summary)
        completed['analysis_summary.json']=dict(path=str(output/'analysis_summary.json'),sha256=sha(output/'analysis_summary.json'))
        stage='plot'
        save(output/'summary.json',dict(status='RUNNING',phase=stage,analysis_summary='analysis_summary.json'))
        plot_c1(opportunities,output)
        stage='report'
        text=['# C1/C2 留出诊断', '',config['model']+'；199题、650个预定题目样本。与另一模型分开。', '',
              '| 范围 | 难度 | 状态 | 题数 | 原生平均token | 免费探测、交付收费平均token | 上限平均通过题数 |', '|---|---|---|---:|---:|---:|---:|']
        for name,group in c1.items():
            for diff,r in group.items():
                b=r.get('delivery_charged') or {};text.append(f"| {name} | {diff} | {r['status']} | {r['questions']} | {b.get('native_mean_generated_tokens','UNKNOWN')} | {b.get('mean_generated_tokens','UNKNOWN')} | {b.get('mean_pass_count','UNKNOWN')} |")
        text+=['', '完整C2分数、匹配覆盖、逐题OOF和置信区间见 raw/signal_metrics.jsonl。置信度是诊断分数，未用于新策略或调参。', '',
               '免费探测及完美停止是不可部署上限；余量未扣失败探测或检查成本。未观察到合格候选不赋零；28k截尾单列。', '',
               '![余量](c1_opportunity_margin.png)']
        (output/'ONE_PAGE.md').write_text('\n'.join(text)+'\n')
        save(output/'summary.json',summary|dict(pipeline_complete=True,analysis_summary='analysis_summary.json'))
        return 0
    except BaseException as exc:
        failure=dict(status='FAILED_ENGINEERING',failed_stage=stage,error=repr(exc),
            pipeline_complete=False,completed_stage_outputs=completed,
            analysis_complete='analysis_summary.json' in completed)
        save(output/'failure.json',failure)
        save(output/'summary.json',failure)
        (output/'raw/error.txt').write_text(traceback.format_exc());return 1
    finally:
        save(output/'resource_ledger.json',dict(wall_seconds=time.perf_counter()-start,self_cpu_seconds=time.process_time()-cpu,
             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_used=False,new_candidate_executions=0,
             new_generated_tokens=0,heldout_test_payload_read=False,derived_labels_read=True,shared_resource_lock=True,
             heavy_cpu_during_c4=False,blas_threads={k:os.getenv(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parent-lock-fd',type=int,required=True)
    a=parser.parse_args()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    return execute(a.manifest.resolve(),a.output.resolve(),a.parent_lock_fd)


if __name__=='__main__':raise SystemExit(main())
