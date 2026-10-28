"""Explicit artifact-manifest scoring/replay; heldout decoding is deferred past freeze.

Current development rehearsals may reuse complete execution-cache observations.
H-v6 AtCoder/LeetCode support both registered models; no second set is inferred.
This module never generates model output and never makes inclusion decisions.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os,pickletools,shutil,subprocess,sys,time,zlib
from pathlib import Path
from collections import Counter
from exec_exit.gen.host_environment import collect_host_environment
from exec_exit.gen.engine_preflight import MODELS
from exec_exit.execute.run_tests import ExecutionCache
from exec_exit.execute.machine_identity import current_machine_identity
from exec_exit.execute.evaluate_dev import limits_for_run,adjudicate
from exec_exit.replay.replay import read_rows,unique_index
from exec_exit.replay.replay_development import (validate_contract,adapt_probe,adapt_complete_v0_probe,final_outcomes,reject_private_signal_fields,received_message_tail,population_summaries,write_rows)
from exec_exit.replay.policies import NativeCost
from exec_exit.replay.policies_v1 import choose_v1,choose_oracle_v1,select_candidates
from exec_exit.replay.hybrid_v1 import choose_hybrid
from exec_exit.analysis.trajectory_annotations import ByteLevelTokenCodec
ROOT=Path(__file__).resolve().parents[2]
MODEL_PATHS={'qwen3_8b':MODELS/'Qwen3-8B','dsr1_distill_qwen7b':MODELS/'DeepSeek-R1-Distill-Qwen-7B'}
DATA=ROOT/'research/code_feedback_admission/run/sources/dataset/test6.jsonl'
DATA_SHA='bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5'
CONTRACT=ROOT/'runs/dev121_replay_v1_contract/config.yaml'
CONTRACT_SHA='42dc98ecd1ab03d6f9705e5b7bafc74110434f97cc7b3ae8c1c7176cab67115e'
LIMITS=limits_for_run(64)
REGISTRY={'hv6_atcoder':'stdin','hv6_leetcode':'functional'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ident(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def checked(item):
 p=Path(item['path']).resolve()
 if sha(p)!=item['sha256']:raise ValueError('Artifact SHA mismatch: '+str(p))
 return p
def validate_job_sources(job):
 artifacts={k:checked(v) for k,v in job['artifacts'].items()}
 runtime=read(artifacts['public_runtime'])
 if [r['case_id'] for r in runtime['cases']]!=job['case_ids']:raise ValueError('Public task order differs from explicit job')
 expected_split='development' if job['split']=='development' else 'heldout_blind_generation'
 if runtime['split']!=expected_split or any(r.get('split',runtime['split'])!=expected_split or r['task_type']!=job['task_type'] for r in runtime['cases']):raise ValueError('Actual task split/type differs; no relabeling')
 if any(r.get('task_id')!=r['case_id'] for r in runtime['cases']):raise ValueError('Executor task_id must identify the same selected case')
 runs={artifacts[k].parent.parent for k in ('native','probes')}
 for directory in runs:
  cfg=read(directory/'config.yaml');summary=read(directory/'summary.json')
  if Path(cfg['model']).resolve()!=MODEL_PATHS[job['model']].resolve():raise ValueError('Wrong model corpus')
  if type(cfg.get('sample_index')) is not int or cfg['sample_index']!=job['sample_index']:raise ValueError('Generation sample_index differs from explicit job')
  source_hashes={Path(k).resolve():v for k,v in cfg['source_sha256'].items()}
  for name in ('config.json','generation_config.json','tokenizer.json'):
   model_file=(MODEL_PATHS[job['model']]/name).resolve()
   if source_hashes.get(model_file)!=sha(model_file):raise ValueError('Generation model artifact identity differs: '+name)
  if summary['status']!='COMPLETE' or summary.get('worker_returncode')!=0 or summary.get('supervisor_timeout'):raise ValueError('Incomplete or failed generation corpus')
 audits=set()
 for item in job['integrity_audits']:
  path=checked(item);summary=read(path);cfg=read(path.parent/'config.yaml')
  if summary.get('records_status',summary.get('status'))!='PASS':raise ValueError('Generation audit not PASS')
  audits.add(Path(cfg['source_run']).resolve())
 if audits!=runs:raise ValueError('Independent audits must identify exactly both source runs')
 return artifacts
def specs():
 if sha(CONTRACT)!=CONTRACT_SHA:raise ValueError('Frozen contract source changed')
 c=read(CONTRACT);validate_contract(c)
 return c,c['specifications']+[dict(name=p,policy=p,k=4,minimum=0) for p in ('HYB-A','HYB-B')]
def gate(manifest,freeze=None):
 if manifest.get('schema')!='exec_exit.postfreeze_jobs.v1' or manifest.get('limits')!=LIMITS:raise ValueError('Explicit locked manifest/limits required')
 contract,policies=specs();names=[p['name'] for p in policies]
 if manifest.get('required_policy_names')!=names:raise ValueError('Unregistered or incomplete policy set; never silently omit an ablation')
 if not manifest.get('jobs'):raise ValueError('No explicit included job')
 for job in manifest['jobs']:
  if type(job.get('sample_index')) is not int or job['sample_index']<0:raise ValueError('Explicit nonnegative integer sample_index required')
  if job.get('model') not in MODEL_PATHS or job.get('task_type') not in ('stdin','functional'):raise ValueError('UNREGISTERED model/task adapter')
  if job.get('split')=='development':
   if job.get('dataset')!='development_explicit':raise ValueError('Development must use explicit previously seen ID whitelist')
   from exec_exit.data.load_dev121 import load_public,DATASET
   from exec_exit.data.leetcode import load_runtime
   allowed={t['task_id'] for t in load_public(DATASET)}|{t['case_id'] for t in load_runtime(ROOT/'runs/leetcode_dev62_projection/data/runtime_tasks.json')}
   if not job.get('case_ids') or not set(job['case_ids'])<=allowed:raise ValueError('Unknown or heldout task cannot be relabeled development')
  elif job.get('split')=='heldout':
   # Do this BEFORE reading a generation corpus or opening any test source.
   if freeze is None or freeze.get('state')!='FROZEN':raise ValueError('HELDOUT_REFUSED_BEFORE_FREEZE')
   freeze_doc=checked(freeze['freeze_document'])
   if freeze_doc!=(ROOT/'FREEZE.md').resolve():raise ValueError('Explicit repository FREEZE.md identity required')
   if job.get('dataset') not in REGISTRY:raise ValueError('UNREGISTERED second heldout: selection required')
   if REGISTRY[job['dataset']]!=job['task_type']:raise ValueError('Task adapter mismatch')
   key=job['model']+'::'+job['dataset'];component=freeze.get('included_components',{}).get(key)
   if not component or component.get('case_ids')!=job.get('case_ids') or job.get('sample_index') not in component.get('sample_indices',[]):raise ValueError('Job absent from exact frozen inclusion/sample manifest')
   frozen_artifacts=component.get('artifacts_by_sample',{}).get(str(job['sample_index']))
   if not frozen_artifacts or frozen_artifacts!=job.get('artifacts'):raise ValueError('Job artifacts differ from exact frozen sample identities')
   if freeze.get('limits')!=LIMITS or freeze.get('required_policy_names')!=names:raise ValueError('Freeze execution/policy mismatch')
   if job.get('dataset_source')!=dict(path=str(DATA),sha256=DATA_SHA):raise ValueError('Unregistered H-v6 source identity')
  else:raise ValueError('Explicit unchanged split required')
 return contract,policies

def decoded_tests(value,task_type):
 try:values=json.loads(value)
 except json.JSONDecodeError:
  raw=zlib.decompress(base64.b64decode(value,validate=True));strings=[];ops=[]
  for op,arg,_ in pickletools.genops(raw):
   if op.name not in {'PROTO','FRAME','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','MEMOIZE','STOP'}:raise ValueError('Executable pickle opcode forbidden')
   ops.append(op.name)
   if op.name in {'BINUNICODE','SHORT_BINUNICODE','BINUNICODE8'}:strings.append(arg)
  if len(strings)!=1 or ops[-1:]!=['STOP']:raise ValueError('Expected data-only string envelope')
  values=json.loads(strings[0])
 if not isinstance(values,list) or not values:raise ValueError('No complete nonempty test suite')
 out=[]
 for t in values:
  if set(t)!={'input','output','testtype'} or t['testtype']!=task_type or not all(isinstance(t[k],str) for k in ['input','output']):raise ValueError('Test task schema mismatch')
  if task_type=='functional':
   for line in t['input'].split('\n'):json.loads(line)
   json.loads(t['output'])
  out.append(dict(input=t['input'],output=t['output']))
 return out

def delayed_hv6_projection(manifest,job,freeze,out):
 gate(manifest,freeze)
 if job['split']!='heldout':raise ValueError('This decoder is only the postfreeze H-v6 handler')
 validate_job_sources(job)
 source=checked(job['dataset_source']);runtime=read(checked(job['artifacts']['public_runtime']))
 tasks=runtime['cases'];ids=job['case_ids']
 if runtime['split']!='heldout_blind_generation' or [t['case_id'] for t in tasks]!=ids or len(ids)!=len(set(ids)):raise ValueError('Actual heldout public projection/order mismatch')
 from exec_exit.data.blind_hv6 import spans,decode_allowed
 registry={};audit=Counter()
 for raw in source.open('rb'):
  positions=spans(raw);platform=decode_allowed(raw,positions,'platform',{'platform'},audit);qid=decode_allowed(raw,positions,'question_id',{'question_id'},audit);cid=platform+':'+qid
  if cid not in ids:continue
  public=decode_allowed(raw,positions,'public_test_cases',{'public_test_cases'},audit);private=decode_allowed(raw,positions,'private_test_cases',{'private_test_cases'},audit)
  registry[cid]=dict(public=decoded_tests(public,job['task_type']),private=decoded_tests(private,job['task_type']))
 if set(registry)!=set(ids):raise ValueError('Do not delete missing heldout tasks')
 if job['task_type']=='functional':
  for task in tasks:
   r=registry[task['case_id']]
   for suite in (r['public'],r['public']+r['private']):
    payload=dict(code='',tests=suite,fn_name=task['func_name'],limits=LIMITS)
    if len(json.dumps(payload,ensure_ascii=False).encode())+6*LIMITS['code_bytes']>LIMITS['input_bytes']:raise ValueError('Functional harness aggregate input capacity requires engineering review; per-test limits unchanged')
 result=save_projection(out,tasks,registry,'heldout',audit)
 cfg=read(result/'config.yaml');cfg.update(frozen_job_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),source_dataset=job['dataset_source'],projection_files_sha256={n:sha(result/'data'/n) for n in ('public_tasks.json','dev_private.jsonl','registry.json')});write(result/'config.yaml',cfg)
 return result

def save_projection(out,tasks,registry,split,audit=None):
 out=Path(out)
 if out.exists():raise FileExistsError('Fresh postfreeze test projection required')
 (out/'data').mkdir(parents=True);(out/'raw').mkdir()
 selected=[dict(t,public_test_cases=registry[t['case_id']]['public']) for t in tasks]
 write(out/'data/public_tasks.json',dict(split=split,tasks=selected));write_rows(out/'data/dev_private.jsonl',[dict(task_id=t['case_id'],actual_split=split,tests=registry[t['case_id']]['private']) for t in tasks]);write(out/'data/registry.json',dict(actual_split=split,tasks=registry))
 write(out/'config.yaml',dict(actual_split=split,internal_private_field_name='dev_private retained solely for existing executor schema; actual_split is authoritative',task_count=len(tasks),host_environment=collect_host_environment(),source_sha256=sha(__file__)))
 write(out/'raw/decode_counts.json',dict(audit or {}));write(out/'summary.json',dict(status='COMPLETE',actual_split=split,questions=len(tasks),public_tests=sum(len(r['public']) for r in registry.values()),private_tests=sum(len(r['private']) for r in registry.values())))
 return out

def configure_evaluator(job,tasks,registry):
 from exec_exit.execute import evaluate_dev as ev
 ev.load_public=lambda _:tasks;ev.normalized_tests=lambda values:values
 ev.load_private=lambda *_:{cid:r['private'] for cid,r in registry.items()}
 if job['model']=='dsr1_distill_qwen7b':
  from exec_exit.gen import dsr1_protocol as ds
  if job['task_type']=='functional':
   # Set only the new process's model-specific control. No source mutation.
   ds.CONTROL='\n</think>\n\n```python\nclass Solution'
  ev.extract_record=ds.extract_record;ev.CodeTokenMapper=ds.DSCodeTokenMapper
 if job['task_type']=='functional':
  from exec_exit.execute import leetcode_call as lc
  from exec_exit.execute import evaluate_leetcode as lce
  ev.make_sandbox=lambda path,template_dir=None:lc.make_sandbox(path,template_dir)
  ev.execution_identity=lc.execution_identity
  ev.load_private=lambda *_:{cid:r['public']+r['private'] for cid,r in registry.items()}
  functions={t['case_id']:t['func_name'] for t in tasks}
  original_init=ev._initialize_worker
  def initialize(*args):
   original_init(*args)
   # Imported observer has separate module globals; bind to this worker's exact
   # registry/cache/identity. The observer retains LC-specific cache descriptors.
   for key in ('_output','_cache','_limits','_sandbox','_identity','_registry','_capacity_violations'):setattr(lce,key,getattr(ev,key))
   lce._functions=functions
  ev._initialize_worker=initialize;ev.observe_suite=lce.observe_suite
 return ev

def score_one(manifest,job,freeze,projection,source_kind,out):
 gate(manifest,freeze)
 if os.environ.get('EXEC_EXIT_POSTFREEZE_FLOCK_HELD')!='1':raise RuntimeError('Explicit resource lock boundary required')
 validate_job_sources(job)
 p=Path(projection);cfg=read(p/'config.yaml')
 if cfg.get('frozen_job_sha256')!=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest() or cfg.get('source_dataset')!=job['dataset_source']:raise ValueError('Scoring projection is not bound to this frozen job/source')
 if cfg.get('projection_files_sha256')!={n:sha(p/'data'/n) for n in ('public_tasks.json','dev_private.jsonl','registry.json')}:raise ValueError('Scoring projection changed after decoding')
 tasks=read(p/'data/public_tasks.json')['tasks'];registry=read(p/'data/registry.json')['tasks']
 if [t['case_id'] for t in tasks]!=job['case_ids']:raise ValueError('Frozen selected scoring population mismatch')
 ev=configure_evaluator(job,tasks,registry);write0=ev._write
 def w(path,v):
  if Path(path).name=='config.yaml':v=dict(v,actual_split=job['split'],model_family=job['model'],task_type=job['task_type'],postfreeze_adapter_sha256=sha(__file__),frozen_job=job['job_id'],private_visibility='heldout_private' if job['split']=='heldout' else 'development_private',heldout_or_reserved_read=job['split']=='heldout',heldout_labels_never_policy_input=True)
  return write0(path,v)
 ev._write=w;kind='native' if source_kind=='N' else 'probes';source=checked(job['artifacts'][kind])
 return ev.main(['--input',str(source),'--source-kind',source_kind,'--output',str(out),'--private-projection',str(p),'--dataset',str(p),'--cache-root',job['cache_root'],'--model',str(MODEL_PATHS[job['model']]),'--workers',str(job.get('workers',8)),'--input-capacity-mib','64','--capacity-policy','execute-output-limit'])

def verify_cached_evaluation(directory,ids):
 directory=Path(directory);config=read(directory/'config.yaml');summary=read(directory/'summary.json')
 if summary['status']!='COMPLETE' or config['limits']!=LIMITS:raise ValueError('Complete identical-limit cache source required')
 identity=config['execution_identity'];machine=identity['machine']
 if machine!=current_machine_identity():raise ValueError('Cross-host execution cache reuse forbidden')
 cache=ExecutionCache(config['cache_root']);evaluations=read_rows(directory/'raw/evaluations.jsonl');receipts=[]
 if any(r['case_id'] not in ids for r in evaluations):raise ValueError('Cached scores outside selected development population')
 for r in evaluations:
  if not r['extraction']['deliverable']:
   if r['final_outcome']!='FAIL':raise ValueError('Invalid cached delivery mislabeled')
   continue
  attempts=r['timeout_robustness']['attempts'];result=adjudicate(attempts)
  if result['final_outcome']!=r['final_outcome'] or result['unstable']!=r['unstable']:raise ValueError('Cached adjudication mismatch')
  for a in attempts:
   for field in ['public_result','private_result']:
    ref=a[field];stored=cache.get(ref['cache_key'])
    if stored is None or stored['code_sha256']!=r['code_sha256'] or stored['identity']!=identity:raise ValueError('Missing or incompatible immutable suite observation')
    value=stored['result']
    if (value['status'],value['all_passed'],value['total'])!=(ref['status'],ref['all_passed'],ref['total']):raise ValueError('Cache result differs from attached evaluation')
    receipts.append(dict(subject_id=r['subject_id'],cache_key=ref['cache_key'],cache_path=str(cache._path(ref['cache_key'])),cache_sha256=sha(cache._path(ref['cache_key'])),new_execution=False))
 signals=read_rows(directory/'raw/signals_public.jsonl');labels=read_rows(directory/'raw/dev_labels.jsonl')
 by={r['subject_id']:r for r in evaluations}
 if len(by)!=len(evaluations) or {r['subject_id'] for r in signals}!={r['subject_id'] for r in labels} or {r['subject_id'] for r in signals}!=set(by):raise ValueError('Cached signal/label population mismatch')
 for row in signals+labels:
  if any(by[row['subject_id']].get(k)!=v for k,v in row.items()):raise ValueError('Cached projection differs from execution record')
 return signals,labels,receipts

def replay(job,signals_rows,labels_rows,out,contract,policies):
 if out.exists():raise FileExistsError('Fresh replay destination required')
 (out/'raw').mkdir(parents=True)
 artifacts={k:checked(v) for k,v in job['artifacts'].items()};natives=unique_index(read_rows(artifacts['native']),'case_id');probes=unique_index(read_rows(artifacts['probes']),'probe_id');annotations=read_rows(artifacts['annotations']);runtime=read(artifacts['public_runtime']);tasks=runtime['cases'];ids=job['case_ids'];taskmap={t['case_id']:t for t in tasks}
 if set(natives)!=set(ids) or set(taskmap)!=set(ids):raise ValueError('Native/public denominator mismatch')
 expected={(r['case_id'],c['position_generated_tokens']) for r in annotations for c in r['candidates']}
 if {(p['case_id'],p['position_generated_tokens']) for p in probes.values()}!=expected or len(probes)!=len(expected):raise ValueError('Complete candidate coverage required')
 signals=unique_index(signals_rows,'subject_id');subjects={'N:'+c for c in ids}|set(probes)
 if set(signals)!=subjects:raise ValueError('Signals must cover every planned subject')
 for signal in signals.values():reject_private_signal_fields(signal)
 outcomes,labels=final_outcomes(labels_rows,subjects);model=MODEL_PATHS[job['model']];codec=ByteLevelTokenCodec(model/'tokenizer.json');special={r['id'] for r in read(model/'tokenizer.json').get('added_tokens',[]) if r.get('special')};grouped={c:[] for c in ids};accounts=[]
 for r in probes.values():
  if job['probe_mode']=='actual_v1':p,acct=adapt_probe(r,signals[r['probe_id']],codec,special)
  elif job['probe_mode']=='virtual_v1':p,acct=adapt_complete_v0_probe(r,signals[r['probe_id']],codec,special,virtual_abort=True)
  else:raise ValueError('Unregistered stopping protocol')
  grouped[r['case_id']].append(p);accounts.append(acct)
 for values in grouped.values():values.sort(key=lambda p:p.public.position)
 paths=[]
 for cid in ids:
  n=natives[cid];total=sum(len(c['token_ids']) for c in n['calls'])
  if total!=n['generated_tokens']:raise ValueError('Native token cost mismatch')
  cost=NativeCost('N:'+cid,total,len(n['injected_control_token_ids']))
  for spec in policies:
   selected=grouped[cid] if spec['k'] is None else select_candidates(grouped[cid],k=spec['k'],minimum=spec['minimum'])
   if spec['policy'] in ('HYB-A','HYB-B'):choice=choose_hybrid(spec['policy'],cost,grouped[cid])
   elif spec['policy']=='ORACLE':choice=choose_oracle_v1(cost,selected,outcomes)
   else:choice=choose_v1(spec['policy'],cost,selected,threshold=contract['defaults']['confidence_threshold'])
   tails=[]
   for v in choice['visited']:
    tail=received_message_tail(probes[v['subject_id']]['call'],v['charged_probe_generated_tokens']);v.update(received_same_message_extra_tokens=tail,post_cancel_unobserved_tokens='UNKNOWN_NOT_EXECUTED');tails.append(tail)
   choice.update(spec,case_id=cid,difficulty=taskmap[cid]['difficulty'],outcome=outcomes[choice['selected_subject_id']],selected_label_unstable=labels[choice['selected_subject_id']]['unstable'],received_same_message_extra_tokens=sum(t for t in tails if t is not None),unknown_message_tail_visits=sum(t is None for t in tails));paths.append(choice)
 nl={c:outcomes['N:'+c] for c in ids};summary=dict(status='COMPLETE',questions=len(ids),probes=len(probes),configurations=len(policies),actual_split=job['split'],probe_mode=job['probe_mode'],strategies=[dict(s,populations=population_summaries([p for p in paths if p['name']==s['name']],nl,{job['dataset']:ids})) for s in policies],cost_scope=contract['primary_token_axis'],new_generation=False)
 write(out/'config.yaml',dict(job=job,contract=ident(CONTRACT),source_sha256=sha(__file__),model_tokenizer=ident(model/'tokenizer.json'),host_environment=collect_host_environment()));write_rows(out/'raw/policy_paths.jsonl',paths);write_rows(out/'raw/probe_accounts.jsonl',accounts);write(out/'summary.json',summary)
 return summary

def run(manifest_path,output,freeze_path=None,cache_only=False):
 start=time.perf_counter();cpu=time.process_time();manifest=read(manifest_path);freeze=read(freeze_path) if freeze_path else None;contract,policies=gate(manifest,freeze)
 if not cache_only and os.environ.get('EXEC_EXIT_POSTFREEZE_FLOCK_HELD')!='1':raise RuntimeError('Run complete scoring under the documented GPU-resource flock boundary')
 if output.exists():raise FileExistsError('Fresh pipeline result directory required')
 for d in ['raw','source']:(output/d).mkdir(parents=True)
 write(output/'config.yaml',dict(manifest=ident(manifest_path),freeze_manifest=ident(freeze_path) if freeze_path else None,cache_only=cache_only,host_environment=collect_host_environment(),source_sha256=sha(__file__),limits=LIMITS));shutil.copyfile(__file__,output/'source/postfreeze_evaluate.py');(output/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True));results=[]
 try:
  for job in manifest['jobs']:
   validate_job_sources(job)
   if cache_only:
    if job['split']!='development':raise ValueError('Current cache rehearsal only development')
    signals=[];labels=[];receipts=[]
    for value in job['cached_evaluations']:
     directory=checked(value).parent;s,l,r=verify_cached_evaluation(directory,job['case_ids']);signals+=s;labels+=l;receipts+=r
    write_rows(output/'raw'/f"{job['job_id']}_cache_receipts.jsonl",receipts)
   else:
    if job['split']!='heldout':raise ValueError('Current non-cache handler is explicitly postfreeze H-v6; development smoke uses verified cache')
    projection=delayed_hv6_projection(manifest,job,freeze,output/(job['job_id']+'_tests'))
    signals=[];labels=[]
    for kind in ['N','probe']:
     evaluation=output/(job['job_id']+'_'+kind+'_evaluation')
     command=[sys.executable,'-B','-m','exec_exit.pipeline.postfreeze_evaluate','score-job','--manifest',str(manifest_path),'--freeze-manifest',str(freeze_path),'--job-id',job['job_id'],'--projection',str(projection),'--source-kind',kind,'--output',str(evaluation)]
     with (output/'raw'/f"{job['job_id']}_{kind}_score.log").open('x') as f:rc=subprocess.call(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,EXECEXIT_MODELS=str(MODELS)))
     if rc or read(evaluation/'summary.json')['status']!='COMPLETE':raise RuntimeError('Scoring failed; no automatic retry')
     signals+=read_rows(evaluation/'raw/signals_public.jsonl');labels+=read_rows(evaluation/'raw/dev_labels.jsonl')
   destination=output/(job['job_id']+'_replay');summary=replay(job,signals,labels,destination,contract,policies)
   results.append(dict(job_id=job['job_id'],split=job['split'],questions=summary['questions'],configurations=summary['configurations'],replay=str(destination),score_source='VERIFIED_EXISTING_EXECUTION_CACHE' if cache_only else 'EXECUTED_OR_CACHE_BY_IMMUTABLE_IDENTITY'))
  result=dict(status='COMPLETE',jobs=results,cache_only_rehearsal=cache_only,new_question_program_executions=0 if cache_only else 'SEE_EVALUATION_SUMMARIES',new_model_generation=False,heldout_tests_read=any(j['split']=='heldout' for j in manifest['jobs']))
  write(output/'summary.json',result)
  (output/'ONE_PAGE.md').write_text('# 一键评分与回放入口演练\n\n'+('已核验既有缓存并重新运行全部43配置回放。' if cache_only else '按明确冻结manifest完成评分和全部已注册配置回放。')+'\n\n'+''.join(f"- {r['job_id']}：{r['questions']}题，{r['configurations']}配置，{r['score_source']}。\n" for r in results)+'\nH-v6两题型×两模型使用延迟加载handler；冻结状态/纳入范围/文件hash全部匹配后才允许解码测试。未知第二集合保持未注册；新增冻结消融未注册时拒绝，不自动丢弃。\n')
  return result
 except BaseException as exc:write(output/'summary.json',dict(status='FAILED',error=repr(exc),completed_jobs=results));raise
 finally:write(output/'resource_ledger.json',dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,gpu_used=False,cache_only=cache_only,new_generation=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','score-job']);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--freeze-manifest',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--cache-only',action='store_true');p.add_argument('--job-id');p.add_argument('--projection',type=Path);p.add_argument('--source-kind',choices=['N','probe']);a=p.parse_args()
 if a.action=='score-job':
  m=read(a.manifest);f=read(a.freeze_manifest) if a.freeze_manifest else None;gate(m,f);jobs=[j for j in m['jobs'] if j['job_id']==a.job_id]
  if len(jobs)!=1:raise ValueError('Unique explicit frozen job required')
  return score_one(m,jobs[0],f,a.projection,a.source_kind,a.output)
 print(json.dumps(run(a.manifest,a.output,a.freeze_manifest,a.cache_only),ensure_ascii=False))
if __name__=='__main__':raise SystemExit(main())
