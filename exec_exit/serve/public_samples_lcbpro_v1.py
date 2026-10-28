"""C4 public-only worker: same per-task LCB comparator as frozen offline scoring.

No data files are loaded here. A post-freeze caller supplies
public samples and static comparator specs. Heldout test archives are unreachable.
"""
from collections import Counter
import hashlib
import os
from pathlib import Path
import time

from exec_exit.data.load_dev121 import normalized_tests
from exec_exit.execute import lcbpro_comparator as comparator
from exec_exit.execute.run_tests import ExecutionCache, execution_cache_key, run_tests
from exec_exit.execute.sandbox import frozen_limits, make_sandbox

_jail = _cache = None
_specs = {}

def init_public_worker(directory, specs):
    global _jail, _cache, _specs
    root=Path(directory)
    _specs={key:comparator.validate_spec(value) for key,value in specs.items()}
    if any(key!=value['task_id'] for key,value in _specs.items()):
        raise ValueError('Comparator task identity mismatch')
    _jail=make_sandbox(root/str(os.getpid()))
    _cache=ExecutionCache(root/'execution_cache')

def warm_public_worker():
    return dict(pid=os.getpid(),runtime=_jail.runtime_manifest,
        per_task_comparators={key:comparator.comparator_identity(value) for key,value in _specs.items()})

def lcb_public(code, task_id, tests, limits):
    spec=_specs[task_id]
    identity=comparator.execution_identity(_jail,spec)
    descriptor=dict(task_id=task_id,protocol='lcbpro_per_task_comparator_v1',visibility='public',
        stop_on_failure=True,comparator=comparator.comparator_identity(spec))
    key=execution_cache_key(code,task=descriptor,tests=tests,resource_limits=limits,**identity)
    start=time.perf_counter_ns();stored=_cache.get(key);reused=stored is not None
    if not reused:
        result=comparator.evaluate_suite(_jail,code,tests,limits,'public',spec=spec)
        stored=dict(result=result,task=descriptor,code_sha256=hashlib.sha256(code).hexdigest(),
            observed_offline_started_ns=start,observed_offline_finished_ns=time.perf_counter_ns(),
            online_visible_at='AVAILABLE_TO_WORKER_AFTER_CHECK',identity=identity)
        _cache.put(key,stored)
    observed=stored['result']
    return dict(status=observed['status'],all_passed=observed['all_passed'],total=observed['total'],
        evaluated=observed['evaluated'],not_evaluated_after_failure=observed['not_evaluated_after_failure'],
        case_status_counts=dict(Counter(r['status'] for r in observed['cases'])),
        original_execution_wall_seconds=observed['wall_seconds'],original_execution_cpu_seconds=observed['cpu_seconds'],
        child_cpu_seconds=observed['child_cpu_seconds'],child_wall_seconds=observed['child_wall_seconds'],
        cache_key=key,cache_reused=reused,lookup_and_current_execution_wall_seconds=(time.perf_counter_ns()-start)/1e9,
        observed_offline_started_ns=stored['observed_offline_started_ns'],
        observed_offline_finished_ns=stored['observed_offline_finished_ns'],
        online_visible_at='AVAILABLE_TO_WORKER_AFTER_CHECK',comparator=descriptor['comparator'])

def public_job(code, task_id, tests):
    started=time.perf_counter_ns();projected=normalized_tests(tests)
    limits=frozen_limits()|dict(input_bytes=64*1024**2)
    if task_id in _specs:
        result=lcb_public(code.encode('utf8'),task_id,projected,limits)
    elif task_id.startswith('atcoder:'):
        result=run_tests(code.encode('utf8'),task_id=task_id,tests=projected,visibility='public',
            limits=limits,sandbox=_jail,cache=_cache)
    else:
        raise ValueError('No frozen per-task comparator; refusing generic fallback: '+task_id)
    return result|dict(worker_visible_start_ns=started,worker_complete_ns=time.perf_counter_ns(),
        worker_pid=os.getpid(),private_tests_read=False)

def install():
    """Fresh worker process only: original source files and bound jobs unchanged."""
    from exec_exit.serve import async_backend, resume_backend
    async_backend.public_job=public_job
    resume_backend.public_job=public_job
