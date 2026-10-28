"""Separate public projections and Dev62 call-based tests.

Heldout preparation decodes only public metadata, question, starter and func_name;
all test payloads are structurally skipped. Never instantiate LCB's eager loader.
"""
from __future__ import annotations
import argparse,ast,base64,hashlib,json,pickletools,subprocess,time,zlib
from collections import Counter
from pathlib import Path
from exec_exit.data.blind_hv6 import spans,decode_allowed,FIELDS
from exec_exit.gen.host_environment import collect_host_environment
ROOT=Path(__file__).resolve().parents[2]
SOURCES={
 'development':(ROOT/'runs/dev_dataset_v1/source/lcb/test5.jsonl','7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214',62),
 'heldout_blind_generation':(ROOT/'research/code_feedback_admission/run/sources/dataset/test6.jsonl','bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5',63)}
RUNTIME_FIELDS=FIELDS|{'case_id','task_id','question_content','starter_code','func_name','split','task_type','source_cohort'}

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def allowed_row(raw,audit,*,question=True):
 pos=spans(raw);meta={k:decode_allowed(raw,pos,k,FIELDS,audit) for k in FIELDS}
 if meta['platform']!='leetcode':return None,pos
 if not question:return meta,pos
 q=decode_allowed(raw,pos,'question_content',{'question_content'},audit)
 starter=decode_allowed(raw,pos,'starter_code',{'starter_code'},audit)
 metadata=decode_allowed(raw,pos,'metadata',{'metadata'},audit)
 if isinstance(metadata,str):metadata=json.loads(metadata)
 name=metadata.get('func_name')
 if not isinstance(name,str) or not name.isidentifier() or not isinstance(q,str) or not q.strip() or not isinstance(starter,str) or not starter.strip():raise ValueError('LC question/starter/public function metadata unavailable')
 # Official starters intentionally end at an empty method signature. Validate
 # their public structure with a synthetic body, but preserve original bytes
 # in the model prompt and projection.
 try:tree=ast.parse(starter)
 except IndentationError:tree=ast.parse(starter+'\n        pass\n')
 if not any(isinstance(n,ast.ClassDef) and n.name=='Solution' and any(isinstance(m,ast.FunctionDef) and m.name==name for m in n.body) for n in tree.body):raise ValueError('Official starter and function metadata disagree')
 cid='leetcode:'+meta['question_id'];return dict(meta,case_id=cid,task_id=cid,question_content=q,starter_code=starter,func_name=name,task_type='functional'),pos

def load_runtime(path):
 path=Path(path);data=json.loads(path.read_text());rows=data['cases'];split=data['split']
 if split not in SOURCES or not rows or len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Explicit unique LC projection required')
 cfg=json.loads((path.parent.parent/'config.yaml').read_text())
 if cfg['runtime_tasks_sha256']!=sha(path) or cfg['split']!=split or cfg['question_count']!=len(rows):raise ValueError('LC projection identity mismatch')
 if split=='heldout_blind_generation' and len(rows)!=63:raise ValueError('Whole heldout LC63 required')
 if split=='development' and len(rows) not in (1,62):raise ValueError('Only full Dev62 or predetermined one-question smoke admitted')
 for r in rows:
  if set(r)!=RUNTIME_FIELDS or r['split']!=split or r['platform']!='leetcode' or r['task_type']!='functional' or r['case_id']!=r['task_id'] or r['case_id']!='leetcode:'+r['question_id']:raise ValueError('LC whitelist/identity mismatch')
 return rows

def normalize_tests(values):
 if not isinstance(values,list) or not values:raise ValueError('Nonempty functional test suite required')
 result=[]
 for t in values:
  if set(t)!={'input','output','testtype'} or t['testtype']!='functional' or not all(isinstance(t[k],str) for k in ('input','output')):raise ValueError('Explicit functional JSON-string tests required')
  for line in t['input'].split('\n'):json.loads(line)
  json.loads(t['output']);result.append({k:t[k] for k in ('input','output')})
 return result

def decode_development_tests(value):
 try:data=json.loads(value);encoding='json'
 except json.JSONDecodeError:
  raw=zlib.decompress(base64.b64decode(value,validate=True));strings=[];ops=[]
  for op,arg,_ in pickletools.genops(raw):
   if op.name not in {'PROTO','FRAME','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','MEMOIZE','STOP'}:raise ValueError('Executable pickle opcode forbidden')
   ops.append(op.name)
   if op.name in {'BINUNICODE','SHORT_BINUNICODE','BINUNICODE8'}:strings.append(arg)
  if len(strings)!=1 or not ops or ops[-1]!='STOP':raise ValueError('Expected data-only JSON-string envelope')
  data=json.loads(strings[0]);encoding='base64_zlib_single_json_string_no_unpickle'
 return normalize_tests(data),encoding

def prepare(output,split,smoke=False):
 output=Path(output)
 if output.exists():raise FileExistsError('New projection directory required')
 if smoke and split!='development':raise ValueError('Smoke must use development only')
 source,expected,count=SOURCES[split];start=time.perf_counter();cpu=time.process_time()
 if sha(source)!=expected:raise ValueError('Pinned dataset identity changed')
 audit=Counter();cases=[];all_ids=set()
 with source.open('rb') as f:
  for raw in f:
   row,_=allowed_row(raw,audit)
   if row is not None:
    if row['case_id'] in all_ids:raise ValueError('Duplicate source LC ID')
    all_ids.add(row['case_id']);cases.append(dict(row,split=split,source_cohort='test5_leetcode' if split=='development' else 'hv6_leetcode'))
 if len(cases)!=count:raise ValueError('Whole LC population count differs')
 if split=='heldout_blind_generation':
  dev_ids=set()
  with SOURCES['development'][0].open('rb') as f:
   for raw in f:
    row,_=allowed_row(raw,Counter(),question=False)
    if row:dev_ids.add('leetcode:'+row['question_id'])
  if all_ids&dev_ids:raise ValueError('Development/heldout overlap')
 selected=cases[:1] if smoke else cases
 (output/'data').mkdir(parents=True);(output/'raw').mkdir();(output/'source').mkdir()
 path=output/'data/runtime_tasks.json';write(path,dict(split=split,cases=selected))
 cfg=dict(schema='exec_exit.leetcode_public_projection.v1',split=split,source=str(source),source_sha256=expected,
  source_question_count=count,question_count=len(selected),selection='First original source-order development LC question' if smoke else 'All original source-order LC questions; no resampling',
  runtime_tasks_sha256=sha(path),difficulty_counts=dict(Counter(r['difficulty'] for r in selected)),
  public_test_payloads_read=False,private_test_payloads_read=False,host_environment=collect_host_environment(),source_code_sha256=sha(__file__))
 write(output/'config.yaml',cfg);load_runtime(path)
 write(output/'raw/decode_counts.json',dict(audit));(output/'source/leetcode.py').write_bytes(Path(__file__).read_bytes())
 (output/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
 write(output/'resource_ledger.json',dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,gpu_used=False))
 write(output/'summary.json',dict(status='COMPLETE',question_count=len(selected),source_question_count=count,split=split,tests_decoded=0))
 return cfg

def project_development_tests(dataset,output):
 start=time.perf_counter();cpu=time.process_time()
 dataset,output=Path(dataset),Path(output);rows=load_runtime(dataset/'data/runtime_tasks.json')
 if any(r['split']!='development' for r in rows):raise ValueError('Frozen heldout has no test-loading API')
 if output.exists():raise FileExistsError('New test projection required')
 ids={r['case_id'] for r in rows};source,expected,_=SOURCES['development']
 if sha(source)!=expected:raise ValueError('Development source identity changed')
 registry={};audit=Counter();encodings=Counter()
 with source.open('rb') as f:
  for raw in f:
   meta,pos=allowed_row(raw,audit,question=False)
   if meta is None or 'leetcode:'+meta['question_id'] not in ids:continue
   cid='leetcode:'+meta['question_id']
   pub=decode_allowed(raw,pos,'public_test_cases',{'public_test_cases'},audit)
   private=decode_allowed(raw,pos,'private_test_cases',{'private_test_cases'},audit)
   pub=normalize_tests(json.loads(pub));private,encoding=decode_development_tests(private);encodings[encoding]+=1
   registry[cid]=dict(public=pub,dev_private=private)
 if set(registry)!=ids:raise ValueError('Development test population incomplete')
 (output/'raw').mkdir(parents=True);(output/'data').mkdir()
 path=output/'data/tests.json';write(path,dict(split='development',tasks=registry))
 write(output/'config.yaml',dict(schema='exec_exit.leetcode_dev_tests.v1',dataset=str(dataset.resolve()),runtime_tasks_sha256=sha(dataset/'data/runtime_tasks.json'),source_sha256=expected,tests_sha256=sha(path),split='development',host_environment=collect_host_environment(),source_code_sha256=sha(__file__)))
 write(output/'raw/decode_counts.json',dict(audit));write(output/'summary.json',dict(status='COMPLETE',questions=len(rows),public_tests=sum(len(t['public']) for t in registry.values()),private_tests=sum(len(t['dev_private']) for t in registry.values()),encodings=dict(encodings)))
 (output/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
 write(output/'resource_ledger.json',dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,gpu_used=False))
 (output/'source').mkdir();(output/'source/leetcode.py').write_bytes(Path(__file__).read_bytes())
 return registry

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('public','tests'));p.add_argument('--output',required=True,type=Path);p.add_argument('--split',choices=tuple(SOURCES),default='development');p.add_argument('--smoke',action='store_true');p.add_argument('--dataset',type=Path);a=p.parse_args()
 if a.action=='public':prepare(a.output,a.split,a.smoke)
 else:project_development_tests(a.dataset,a.output)
 print(json.dumps(json.loads((a.output/'summary.json').read_text())))
