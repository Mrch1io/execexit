"""Official LCB call-based evaluator in a separate chroot/uid/seccomp child.

The official call/comparison bodies are copied unchanged from pinned trusted AST.
One Solution instance lives for an entire suite. Method/compile hooks add per-case
CPU guards; official alarms provide the per-case wall guard. No candidate exec
occurs in the host. The separate PUB signal and PUB+private full suite have distinct
cache descriptors. This is a call-based adaptation, not stdio lifecycle equivalence.
"""
from __future__ import annotations
import ast,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from exec_exit.data.load_lcb import official_evaluator_path
from exec_exit.execute.sandbox import LEGACY_EXECUTOR,clone_template
from exec_exit.execute.machine_identity import current_machine_identity
ROOT=Path(__file__).resolve().parents[2]
_MODULE=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def legacy():
 global _MODULE
 if _MODULE is None:
  spec=importlib.util.spec_from_file_location('exec_exit_lc_isolated_executor',LEGACY_EXECUTOR);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
  needle='"getrandom","uname","sysinfo","getrusage","getrlimit",'
  if m.LAUNCHER.count(needle)!=1:raise ValueError('Pinned launcher insertion location changed')
  m.LAUNCHER=m.LAUNCHER.replace(needle,needle+'\n    "alarm","setitimer","getitimer",')
  _MODULE=m
 return _MODULE

def make_sandbox(path,template=None,*,python_executable=None):
 if python_executable is None:
  python_executable=os.environ['EXECEXIT_SANDBOX_PYTHON']
 if template is not None:clone_template(template,path)
 return legacy().PublicCodeExecutor(path,python_executable=python_executable)

def official_subset():
 path=official_evaluator_path();tree=ast.parse(path.read_text())
 names={'truncatefn','timeout_handler','get_function','compile_code','grade_call_based','run_test','reliability_guard'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.ClassDef) and n.name in {'TimeoutException','CODE_TYPE'} or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='import_string' for t in n.targets)]
 if len(nodes)!=10:raise ValueError('Pinned official call-based AST dependencies changed')
 imports='import json,sys,signal,faulthandler,platform,time\nfrom types import ModuleType\nfrom enum import Enum\nfrom datetime import datetime\n'
 return imports+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'

HARNESS=r'''
import json,sys,time,signal,hashlib
payload=json.loads(sys.stdin.read())
ns={}
exec(compile(OFFICIAL_SOURCE,'pinned_lcb_testing_util.py','exec'),ns)
original_stdout,original_stderr=sys.stdout,sys.stderr
measurements=[]
class OutputLimitError(Exception):pass
class Sink:
    encoding='utf-8'
    def __init__(self):self.bytes=0;self.buffer=self
    def write(self,text):
        n=len(text if isinstance(text,bytes) else text.encode('utf-8'));self.bytes+=n
        if self.bytes>payload['limits']['output_bytes']:raise OutputLimitError('OUTPUT_LIMIT')
        return len(text)
    def flush(self):pass
    def isatty(self):return False

def cpu_timeout(signum,frame):raise ns['TimeoutException']('CPU_LIMIT')
signal.signal(signal.SIGPROF,cpu_timeout)
original_compile=ns['compile_code'];original_get=ns['get_function']
def timed_call(fn,phase,index,args):
    sink=Sink();before_wall=time.perf_counter();before_cpu=time.process_time()
    sys.stdout=sys.stderr=sink
    signal.setitimer(signal.ITIMER_PROF,payload['limits']['cpu_seconds'])
    try:
        result=fn(*args)
        if phase=='case':
            try: size=len(json.dumps(result,ensure_ascii=False).encode())
            except (TypeError,ValueError):size=len(repr(result).encode())
            if size+sink.bytes>payload['limits']['output_bytes']:raise OutputLimitError('OUTPUT_LIMIT')
        return result
    finally:
        signal.setitimer(signal.ITIMER_PROF,0)
        sys.stdout,sys.stderr=original_stdout,original_stderr
        measurements.append(dict(phase=phase,test_index=index,wall_seconds=time.perf_counter()-before_wall,cpu_seconds=time.process_time()-before_cpu,printed_bytes=sink.bytes))

def compile_hook(code,timeout):return timed_call(original_compile,'compile',None,(code,timeout))
def get_hook(compiled,fn_name):
    fn=original_get(compiled,fn_name)
    if fn is None:return None
    def method(*args):return timed_call(fn,'case',sum(r['phase']=='case' for r in measurements),args)
    return method
ns['compile_code']=compile_hook;ns['get_function']=get_hook
try:
    result=ns['run_test']({'input_output':json.dumps({'inputs':[r['input'] for r in payload['tests']], 'outputs':[r['output'] for r in payload['tests']], 'fn_name':payload['fn_name']})},test=payload['code'],debug=False,timeout=int(payload['limits']['wall_seconds']))
    response={'official_result':result,'measurements':measurements}
except BaseException as exc:
    response={'harness_error':repr(exc),'measurements':measurements}
finally:
    signal.setitimer(signal.ITIMER_PROF,0);signal.alarm(0);sys.stdout,sys.stderr=original_stdout,original_stderr
original_stdout.write(json.dumps(response,ensure_ascii=False)+'\n');original_stdout.flush()
'''

def execution_identity(sandbox):
 m=sandbox.runtime_manifest
 return dict(machine=current_machine_identity(),interpreter={k:m[k] for k in ('python','python_version','python_sha256','libpython_sha256')},
  comparator=dict(name='official_LCB_run_test_call_based_unchanged_comparison',source_sha256=sha(official_evaluator_path()),adapter_sha256=sha(__file__)),
  executor=dict(legacy_source_sha256=sha(LEGACY_EXECUTOR),adapter_sha256=sha(__file__),runtime_fingerprint=m['fingerprint'],
   extra_allowed_syscalls=['alarm','setitimer','getitimer'],lifecycle='One Solution instance per suite; independent PUB and PUB+private full suites'))

def evaluate_suite(sandbox,code,tests,limits,visibility,fn_name):
 if not isinstance(code,bytes) or not code or len(code)>limits['code_bytes'] or not tests:raise ValueError('Bounded raw code and nonempty tests required')
 if visibility not in {'public','dev_full'}:raise ValueError('No heldout execution admitted before freeze')
 if not isinstance(fn_name,str) or not fn_name.isidentifier():raise ValueError('Explicit public function identity required')
 for t in tests:
  if set(t)!={'input','output'} or any(not isinstance(v,str) for v in t.values()) or len(t['input'].encode())>limits['input_bytes']:raise ValueError('Functional test input exceeds limits or schema')
 # Original limits apply to each method call, and independently to compilation.
 # The outer kernel/process guard is cumulative across these N+1 bounded phases.
 total=dict(limits,cpu_seconds=max(1,int((len(tests)+1)*limits['cpu_seconds']+1)),wall_seconds=(len(tests)+1)*limits['wall_seconds']+10)
 payload=dict(code=code.decode('utf-8'),tests=tests,fn_name=fn_name,limits=limits)
 source='OFFICIAL_SOURCE='+repr(official_subset())+'\n'+HARNESS
 with sandbox.lock:
  raw=sandbox._run_case(source,json.dumps(payload,ensure_ascii=False),legacy().ExecutionLimits(**total))
 cases=[];metadata={};measurements=[]
 if raw['status']=='EXECUTED':
  try:
   response=json.loads(raw['stdout']);measurements=response.get('measurements',[])
   if 'harness_error' in response:raise RuntimeError(response['harness_error'])
   values,metadata=response['official_result']
   if not isinstance(values,list) or not 1<=len(values)<=len(tests):raise ValueError('Invalid official fail-fast result cardinality')
   for index,value in enumerate(values):
    if value is True:status='PASS'
    elif value is False or value==-2:status='WRONG_ANSWER'
    elif value==-3:status='TIMEOUT_CPU' if 'CPU_LIMIT' in metadata.get('error','') else 'TIMEOUT_WALL'
    elif value==-4:
     error=metadata.get('error','')+' '+metadata.get('error_message','')
     status='TIMEOUT_CPU' if 'CPU_LIMIT' in error else 'OUTPUT_LIMIT' if 'OUTPUT_LIMIT' in error else 'RUNTIME_ERROR'
    else:raise ValueError('Unknown official result code')
    cases.append(dict(test_index=index,status=status))
   if any(row['status']!='PASS' for row in cases[:-1]):raise ValueError('Official evaluator did not fail fast')
  except Exception as exc:
   return dict(status='UNKNOWN',all_passed=None,total=len(tests),evaluated=0,not_evaluated_after_failure=0,cases=[],error=repr(exc),child=raw,visibility=visibility,limits=limits,outer_guard=total,wall_seconds=raw['wall_seconds'],cpu_seconds=raw['cpu_seconds'],child_wall_seconds=raw['wall_seconds'],child_cpu_seconds=raw['cpu_seconds'])
 else:
  if raw['status']=='SANDBOX_SETUP_ERROR':raise RuntimeError('Isolation setup failed; no unsafe fallback')
  # A process failure is not given a fabricated first-test label.
  return dict(status='UNKNOWN',all_passed=None,total=len(tests),evaluated=0,not_evaluated_after_failure=0,cases=[],error='Whole suite child failed before trusted final record',child=raw,visibility=visibility,limits=limits,outer_guard=total,wall_seconds=raw['wall_seconds'],cpu_seconds=raw['cpu_seconds'],child_wall_seconds=raw['wall_seconds'],child_cpu_seconds=raw['cpu_seconds'])
 count=len(cases);passed=count==len(tests) and all(r['status']=='PASS' for r in cases)
 return dict(status='PASS' if passed else 'FAIL',all_passed=passed,total=len(tests),evaluated=count,not_evaluated_after_failure=len(tests)-count,
  cases=cases+[dict(test_index=i,status='NOT_EVALUATED_AFTER_FAILURE') for i in range(count,len(tests))],
  wall_seconds=raw['wall_seconds'],child_wall_seconds=raw['wall_seconds'],cpu_seconds=raw['cpu_seconds'],child_cpu_seconds=raw['cpu_seconds'],host_thread_cpu_seconds=0,
  official_metadata=metadata,method_measurements=measurements,visibility=visibility,stop_on_failure=True,limits=limits,outer_guard=total,
  isolation=sandbox.runtime_manifest,code_sha256=hashlib.sha256(code).hexdigest(),lifecycle='Same Solution instance through entire ordered suite',
  official_source_sha256=sha(official_evaluator_path()),public_compatibility='Pinned official call-based mode; CPU hook and isolated process resource enforcement recorded separately')
