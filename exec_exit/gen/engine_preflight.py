"""Native vLLM W0 preflight, with CPU-only source audit as the default.

No vLLM source changes or model-runner substitutions are performed.  A selected
budget protocol is required before GPU execution.  The installed V2 runner's
native thinking_token_budget rejection is a real blocker, not a GPU test pass.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import sysconfig
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
#模型目录与源码审计位置可配置，默认使用本仓库和当前Python环境
ENV = Path(sys.prefix)
MODELS = Path(os.environ.get('EXECEXIT_MODELS', ROOT / 'models')).expanduser().resolve()
MODEL = MODELS / 'Qwen3-8B'
VLLM = Path(os.environ.get('EXECEXIT_VLLM_SOURCE', Path(sysconfig.get_path('purelib')) / 'vllm')).expanduser().resolve()
OLD_CONFIG = ROOT / 'research/early_exit_component/run/configs/engine.json'
CHECKS = ('logprobs', 'prefix_cache', 'thinking_budget', 'abort', 'shared_prefix_concurrency')
PROTOCOLS = ('native_thinking', 'externally_segmented', 'total_output_cap')
SNAPSHOT_EXTENSION = 'exec_exit.gen.engine_preflight.WorkerSnapshotExtension'
SNAPSHOT_RPC_METHOD = 'exec_exit_worker_snapshot'
FORCED_ANSWER_CONTROL = '</think>\n\n```python\n'
PREFLIGHT_PROMPTS = {
    'basic': 'Count slowly from 1 to 100 and explain the pattern.',
    'shared': ('Shared immutable public context. ' * 220) + '\nExplain why addition is associative.',
    'natural': 'Reply with exactly the word OK, and then stop.',
    'forced': ('Write a Python program that prints OK. Return one complete Python code block, '
               'close the block with three backticks, and stop.'),
}


def seed_for(case_id, repeat=0, stage=0):
    """Use the existing seed derivation; stage 1 is explicit for either answer branch."""
    payload = f'reasoning_capacity_v1:{case_id}:{repeat}:{stage}'.encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def validate_prompt_token_ids(prompt_ids):
    if (not isinstance(prompt_ids, list) or not prompt_ids
            or any(type(token) is not int or token < 0 for token in prompt_ids)):
        raise TypeError('prompt_token_ids must be a nonempty list of nonnegative Python ints; '
                        'apply_chat_template must use explicit return_dict=False')


def chat_prompt_ids(tokenizer, content):
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
    validate_prompt_token_ids(ids)
    return ids


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """JSON is also valid YAML 1.2; config.yaml remains machine-readable."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def source_record(path, terms):
    path = Path(path)
    lines = path.read_text().splitlines()
    return dict(path=str(path), sha256=sha(path), matches=[
        dict(line=i, text=line) for i, line in enumerate(lines, 1)
        if any(term in line for term in terms)])


def thinking_guard_ast(path=VLLM / 'v1/engine/input_processor.py'):
    """Extract only the exact installed preflight guard; never import CUDA."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, ast.If)
                  and ast.unparse(node.test) == 'params.thinking_token_budget is not None']
    if len(candidates) != 1:
        raise ValueError('Expected one native thinking-token-budget validation guard')
    node = copy.deepcopy(candidates[0])
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), node


def reproduce_budget_guard(budget=28000, use_v2=True, reasoning_enabled=True,
                           path=VLLM / 'v1/engine/input_processor.py'):
    """Execute the installed guard with inert CPU objects, not a model engine.

    This verifies the native validation branch only.  It cannot establish GPU
    runtime support, actual runner identity, throughput, or generation semantics.
    """
    module, node = thinking_guard_ast(path)
    context = dict(params=SimpleNamespace(thinking_token_budget=budget),
                   self=SimpleNamespace(use_v2_model_runner=use_v2,
                       vllm_config=SimpleNamespace(reasoning_config=
                           SimpleNamespace(enabled=reasoning_enabled))))
    result = dict(kind='CPU_EXTRACTED_INSTALLED_GUARD', gpu_used=False,
                  budget=budget, use_v2_model_runner=use_v2,
                  reasoning_enabled=reasoning_enabled,
                  source_path=str(path), source_sha256=sha(path),
                  first_line=node.lineno, last_line=node.end_lineno)
    try:
        exec(compile(module, str(path), 'exec'), context)
    except ValueError as error:
        return result | dict(status='REJECTED', exception_type='ValueError', message=str(error))
    return result | dict(status='NOT_REJECTED_BY_THIS_GUARD', message=None)


def source_audit(vllm_root=VLLM):
    vllm_root = Path(vllm_root)
    inputs = [
        ('sampling_params.py', ('thinking_token_budget', 'max_tokens', 'logprobs:')),
        ('v1/engine/input_processor.py', ('thinking_token_budget', 'use_v2_model_runner')),
        ('v1/engine/async_llm.py', ('async def abort(', 'async def reset_prefix_cache(',
                                  'async def collective_rpc(', 'async def generate(')),
        ('v1/engine/core_client.py', ('async def collective_rpc_async(',)),
        ('v1/executor/uniproc_executor.py', ('class UniProcExecutor', 'def collective_rpc(')),
        ('v1/worker/gpu/model_runner.py', ('class GPUModelRunner',)),
    ]
    guard = reproduce_budget_guard(path=vllm_root / 'v1/engine/input_processor.py')
    native_blocked = guard['status'] == 'REJECTED' and 'not yet supported by the V2' in guard['message']
    return dict(gpu_used=False, native_budget_on_v2='UNSUPPORTED' if native_blocked else 'UNKNOWN',
                guard_reproduction=guard,
                sources=[source_record(vllm_root / path, terms) for path, terms in inputs],
                boundary='Source/API evidence is not a runtime/GPU pass.')


def worker_snapshot(worker):
    """Read a native worker, called by its bound extension method at a boundary."""
    runner = worker.model_runner
    config = worker.vllm_config
    klass = type(runner)
    source = inspect.getsourcefile(klass)
    return dict(pid=os.getpid(), at_ns=time.perf_counter_ns(),
        clock='host_perf_counter_ns', runner_module=klass.__module__,
        runner_class=klass.__module__ + '.' + klass.__qualname__,
        runner_source=source, runner_source_sha256=sha(source) if source else None,
        use_v2_model_runner=bool(worker.use_v2_model_runner),
        async_scheduling=config.scheduler_config.async_scheduling,
        compile_mode=str(config.compilation_config.mode),
        cudagraph_mode=str(config.compilation_config.cudagraph_mode),
        enable_prefix_caching=config.cache_config.enable_prefix_caching,
        max_model_len=config.model_config.max_model_len,
        dtype=str(config.model_config.dtype),
        max_num_seqs=config.scheduler_config.max_num_seqs,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        block_size=config.cache_config.block_size,
        kv_cache_memory_bytes=config.cache_config.kv_cache_memory_bytes)


class WorkerSnapshotExtension:
    """Native worker extension: string RPC, no callable/pickle transport.

    vLLM adds this mixin to the existing worker class via worker_extension_cls.
    Consequently ``self`` is the native worker, not WorkerWrapperBase.  The
    wrapper resolves this named method through its normal __getattr__ path.
    """

    def exec_exit_worker_snapshot(self):
        return worker_snapshot(self)


def snapshot_engine_kwargs(engine):
    existing = engine.get('worker_extension_cls', '')
    if existing not in ('', SNAPSHOT_EXTENSION):
        raise ValueError('Refusing to replace an existing worker extension')
    return dict(engine, worker_extension_cls=SNAPSHOT_EXTENSION)


def serialize_logprobs(rows):
    if rows is None:
        return None
    return [{str(token): dict(logprob=float(value.logprob), rank=value.rank,
                              decoded_token=value.decoded_token)
             for token, value in row.items()} for row in rows]


def verify_logprobs(token_ids, rows):
    if not rows or len(rows) != len(token_ids):
        raise AssertionError('Native logprobs are absent or not aligned to true output tokens')
    for token, row in zip(token_ids, rows):
        if str(token) not in row or not any(value['rank'] == 1 for value in row.values()):
            raise AssertionError('Sampled token or top-one entry missing')
        if any(not math.isfinite(value['logprob']) or value['logprob'] > 0 for value in row.values()):
            raise AssertionError('Invalid native logprob')


def gpu_inventory(expected_uuid):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                    '--format=csv,noheader,nounits'], text=True)
    rows = [dict(zip(('uuid', 'name', 'memory_total_mib', 'driver_version'),
                     [value.strip() for value in line.split(',')]))
            for line in text.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0]['uuid'] != expected_uuid:
        raise RuntimeError('Current GPU is not the unique expected full UUID')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name',
                                        '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError('GPU already has compute processes; do not overlap preflight')
    return rows[0]


class Events:
    def __init__(self, path):
        self.path, self.pending = Path(path), []
        self.count = 0

    def add(self, event, **fields):
        row = dict(event=event, at_ns=time.perf_counter_ns(),
                   epoch_ns=time.time_ns(), clock='host_perf_counter_ns')
        row.update(fields)
        self.pending.append(row)

    def flush(self):
        written = 0
        try:
            with self.path.open('a') as stream:
                for row in self.pending:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    written += 1
        finally:
            self.count += written
            del self.pending[:written]


def flush_preserving_primary_error(events, summary):
    """Report recording failure without replacing a generation/API root cause."""
    try:
        events.flush()
    except Exception as error:
        summary.setdefault('recording_errors', []).append(dict(
            error_type=type(error).__name__, error=str(error), pending_rows=len(events.pending)))
        if 'error_type' not in summary:
            summary.update(status='FAILED_RECORDING', error_type=type(error).__name__, error=str(error))
        return error
    return None


async def generate_one(engine, tokenizer, events, request_id, prompt_ids, *,
                       max_tokens=16, logprobs=None, ignore_eos=False, abort_after=None,
                       seed=None, stop_token_ids=None, stop=None,
                       include_stop_str_in_output=False):
    validate_prompt_token_ids(prompt_ids)
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    sampling = dict(temperature=0.6, top_p=0.95, top_k=20,
        max_tokens=max_tokens, ignore_eos=ignore_eos, logprobs=logprobs,
        seed=seed, stop_token_ids=stop_token_ids, stop=stop,
        include_stop_str_in_output=include_stop_str_in_output)
    params = SamplingParams(**sampling, output_kind=RequestOutputKind.DELTA,
                            skip_special_tokens=False)
    start = time.perf_counter_ns()
    events.add('request_start', request_id=request_id, prompt_token_ids=prompt_ids,
               sampling=sampling)
    ids, texts, probabilities = [], [], []
    cached, finish, stop_reason, aborted, finished = 0, None, None, False, False
    abort_ns, last_ns = None, None
    async for output in engine.generate({'prompt_token_ids': prompt_ids}, params, request_id):
        received = time.perf_counter_ns()
        cached = max(cached, int(output.num_cached_tokens or 0))
        for completion in output.outputs:
            chunk = list(completion.token_ids)
            probs = serialize_logprobs(completion.logprobs)
            ids.extend(chunk)
            texts.append(completion.text)
            if probs:
                probabilities.extend(probs)
            finish = completion.finish_reason or finish
            if completion.stop_reason is not None:
                stop_reason = completion.stop_reason
            events.add('output', request_id=request_id, received_ns=received,
                       token_ids=chunk, text=completion.text, logprobs=probs,
                       num_cached_tokens=cached, finish_reason=completion.finish_reason,
                       stop_reason=completion.stop_reason, finished=output.finished)
        last_ns, finished = received, output.finished
        if abort_after is not None and len(ids) >= abort_after and not output.finished and not aborted:
            abort_ns = time.perf_counter_ns()
            events.add('abort_call', request_id=request_id, visible_tokens=len(ids))
            await engine.abort(request_id)
            events.add('abort_return', request_id=request_id,
                       generated_but_unobserved_inflight_tokens='UNKNOWN')
            aborted = True
    result = dict(request_id=request_id, start_ns=start, end_ns=time.perf_counter_ns(),
        last_output_ns=last_ns, prompt_tokens=len(prompt_ids), output_token_ids=ids,
        text=''.join(texts), token_count=len(ids), finish_reason=finish, stop_reason=stop_reason,
        token_ids_decoded_text=tokenizer.decode(ids, skip_special_tokens=False),
        stream_finished=finished, num_cached_tokens=cached, logprobs=probabilities,
        abort_requested=aborted, abort_call_ns=abort_ns)
    events.add('request_end', **result)
    return result


async def gpu_checks(config, run_dir):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.usage.usage_lib import UsageContext

    events = Events(run_dir / 'raw/gpu_events.jsonl')
    summary = json.loads((run_dir / 'summary.json').read_text())
    summary['source_audit_resources'] = dict(summary['resources'])
    summary['resources'] = dict(gpu_seconds='UNKNOWN_KERNEL_BUSY_TIME')
    summary.update(status='RUNNING', gpu_used=False, started_epoch=time.time(),
                   worker_pid=os.getpid())
    save(run_dir / 'summary.json', summary)
    engine = None
    started = time.perf_counter()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    engine_start, engine_ready, engine_end = None, None, None
    try:
        config['observed_gpu'] = gpu_inventory(config['expected_gpu_uuid'])
        save(run_dir / 'config.yaml', config)
        tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
        engine_start = time.perf_counter()
        summary.update(gpu_used='UNKNOWN_DURING_ENGINE_STARTUP', engine_start_epoch=time.time())
        save(run_dir / 'summary.json', summary)
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=config['model'], **config['engine']),
                                         usage_context=UsageContext.OPENAI_API_SERVER)
        engine_ready = time.perf_counter()
        summary.update(gpu_used=True, engine_ready_epoch=time.time())
        save(run_dir / 'summary.json', summary)
        snapshots = await engine.collective_rpc(SNAPSHOT_RPC_METHOD)
        if len(snapshots) != 1:
            raise RuntimeError('Expected exactly one native worker snapshot')
        snapshot = snapshots[0]
        config['actual_model_runner'] = snapshot['runner_class']
        config['actual_runner_snapshot'] = snapshot
        save(run_dir / 'config.yaml', config)
        events.add('worker_snapshot', **snapshot)
        events.flush()
        if not (snapshot['use_v2_model_runner'] and snapshot['async_scheduling']
                and snapshot['compile_mode'] == '3'
                and snapshot['cudagraph_mode'] == 'FULL_AND_PIECEWISE'):
            raise RuntimeError('Current native runner/async/compile/graph configuration differs')
        if snapshot['max_model_len'] != 40960 or not snapshot['enable_prefix_caching']:
            raise RuntimeError('Requested capacity or prefix cache unavailable')
        prompt = chat_prompt_ids(tokenizer, PREFLIGHT_PROMPTS['basic'])

        async def checked(name, operation):
            row = await asyncio.wait_for(operation(), timeout=config['check_timeout_s'])
            summary['checks'][name] = dict(status='PASS', evidence=row)
            events.flush()
            save(run_dir / 'summary.json', summary)

        async def check_logprobs():
            row = await generate_one(engine, tokenizer, events, 'preflight_logprobs', prompt,
                                     max_tokens=16, ignore_eos=True, logprobs=1)
            verify_logprobs(row['output_token_ids'], row['logprobs'])
            return dict(request_id=row['request_id'], token_positions=row['token_count'],
                        distribution=config['engine']['logprobs_mode'])

        await checked('logprobs', check_logprobs)

        shared = chat_prompt_ids(tokenizer, PREFLIGHT_PROMPTS['shared'])

        async def check_cache():
            if not await engine.reset_prefix_cache():
                raise RuntimeError('Prefix cache reset failed at drained boundary')
            cold = await generate_one(engine, tokenizer, events, 'preflight_cache_cold', shared)
            warm = await generate_one(engine, tokenizer, events, 'preflight_cache_warm', shared)
            if warm['num_cached_tokens'] <= 0:
                raise RuntimeError('No native prefix-cache hit on identical prompt')
            return dict(cold_cached_tokens=cold['num_cached_tokens'],
                        warm_cached_tokens=warm['num_cached_tokens'], prompt_tokens=len(shared))

        await checked('prefix_cache', check_cache)

        async def check_budget():
            protocol = config['budget_protocol']
            if protocol == 'total_output_cap':
                row = await generate_one(engine, tokenizer, events, 'preflight_budget_cap', prompt,
                                         max_tokens=16, ignore_eos=True)
                if row['token_count'] != 16 or row['finish_reason'] != 'length':
                    raise RuntimeError('Native max_tokens cap did not stop at its small threshold')
                return dict(protocol=protocol, configured_full_cap=28000,
                            executed_smoke_cap=16, full_28000_boundary_executed=False)
            if protocol != 'externally_segmented':
                raise RuntimeError('Native thinking budget is unsupported on the V2 runner')

            end_ids = tokenizer.encode('</think>', add_special_tokens=False)
            if len(end_ids) != 1:
                raise RuntimeError('Thinking end must have one exact token ID for stop_token_ids')
            end_id = end_ids[0]
            config['two_segment_protocol']['observed_think_end_token_ids'] = end_ids
            save(run_dir / 'config.yaml', config)
            eos_ids = json.loads((Path(config['model']) / 'generation_config.json').read_text())['eos_token_id']
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            # A full-budget SamplingParams construction is checked, while the
            # actual mechanism smoke uses short synthetic prefixes and caps.
            full = SamplingParams(max_tokens=28000, temperature=.6, top_p=.95, top_k=20,
                stop_token_ids=[end_id], include_stop_str_in_output=True,
                seed=seed_for('preflight_natural_close', 0, 0))
            assert full.max_tokens == 28000

            natural_case = 'preflight_natural_close'
            natural_prompt = chat_prompt_ids(tokenizer, PREFLIGHT_PROMPTS['natural'])
            synthetic_tail = tokenizer.encode(
                'The instruction asks for exactly one word, OK. No calculation is needed. '
                'My reasoning is complete; the final answer is OK.\n', add_special_tokens=False)
            natural_prompt += synthetic_tail
            first = await generate_one(engine, tokenizer, events, natural_case + '.thinking',
                natural_prompt, max_tokens=128, stop_token_ids=[end_id],
                include_stop_str_in_output=True, seed=seed_for(natural_case, 0, 0))
            if (not first['output_token_ids'] or first['output_token_ids'][-1] != end_id
                    or first['stop_reason'] != end_id or first['finish_reason'] != 'stop'
                    or '</think>' not in first['text']):
                raise RuntimeError('NATURAL_CLOSE_SMOKE_NOT_OBSERVED: expected retained </think> stop ID and text')
            natural_prefix = natural_prompt + first['output_token_ids']
            natural_remaining = config['engine']['max_model_len'] - len(natural_prefix)
            if natural_remaining <= 0:
                raise RuntimeError('No context room for natural answer continuation')
            answer = await generate_one(engine, tokenizer, events, natural_case + '.answer',
                natural_prefix, max_tokens=natural_remaining,
                include_stop_str_in_output=True, seed=seed_for(natural_case, 0, 1))
            if (not answer['output_token_ids'] or answer['output_token_ids'][-1] not in eos_ids
                    or answer['finish_reason'] != 'stop'):
                raise RuntimeError('NATURAL_EOS_SMOKE_NOT_OBSERVED: continuation must end at native EOS')

            forced_case = 'preflight_forced_answer'
            forced_prompt = chat_prompt_ids(tokenizer, PREFLIGHT_PROMPTS['forced'])
            forced_prompt += tokenizer.encode(
                'I need to reason through the requested program. The first step is',
                add_special_tokens=False)
            capped = await generate_one(engine, tokenizer, events, forced_case + '.thinking',
                forced_prompt, max_tokens=1, stop_token_ids=[end_id],
                include_stop_str_in_output=True, seed=seed_for(forced_case, 0, 0))
            if (capped['token_count'] != 1 or capped['finish_reason'] != 'length'
                    or end_id in capped['output_token_ids']):
                raise RuntimeError('OPEN_THINKING_CAP_SMOKE_NOT_OBSERVED: expected one-token cap while still thinking')
            control_ids = tokenizer.encode(FORCED_ANSWER_CONTROL, add_special_tokens=False)
            forced_prefix = forced_prompt + capped['output_token_ids'] + control_ids
            if len(forced_prefix) + 2048 > config['engine']['max_model_len']:
                raise RuntimeError('Forced answer smoke does not fit its exact 2048-token budget')
            forced = await generate_one(engine, tokenizer, events, forced_case + '.answer',
                forced_prefix, max_tokens=2048, stop=['```'],
                include_stop_str_in_output=True, seed=seed_for(forced_case, 0, 1))
            if forced['finish_reason'] != 'stop' or forced['stop_reason'] != '```':
                raise RuntimeError('CLOSING_FENCE_SMOKE_NOT_OBSERVED: forced answer must exercise the fence stop')
            return dict(protocol=protocol, configured_full_thinking_cap=full.max_tokens,
                full_28000_boundary_executed=False,
                natural=dict(case_id=natural_case, synthetic_reasoning_prefix_is_input=True,
                    executed_thinking_cap=128, retained_end_token_id=end_id,
                    prefix_passed_without_retokenization=True,
                    answer_max_tokens=natural_remaining, answer_native_eos=True,
                    thinking_seed=seed_for(natural_case, 0, 0), answer_seed=seed_for(natural_case, 0, 1)),
                forced=dict(case_id=forced_case, synthetic_reasoning_prefix_is_input=True,
                    executed_thinking_cap=1, injected_control_token_ids=control_ids,
                    answer_max_tokens=2048, matched_stop=forced['stop_reason'],
                    thinking_seed=seed_for(forced_case, 0, 0), answer_seed=seed_for(forced_case, 0, 1)),
                scope='Small synthetic protocol smoke, not a 28000-token run or quality evaluation')

        await checked('thinking_budget', check_budget)

        async def check_abort():
            row = await generate_one(engine, tokenizer, events, 'preflight_abort', prompt,
                                     max_tokens=1024, ignore_eos=True, abort_after=4)
            if not row['abort_requested'] or row['token_count'] >= 1024:
                raise RuntimeError('Abort was not observed before the request output cap')
            if engine.output_processor.has_unfinished_requests():
                raise RuntimeError('Frontend request ownership remains after abort')
            after = await generate_one(engine, tokenizer, events, 'preflight_after_abort', prompt)
            if not after['stream_finished']:
                raise RuntimeError('Engine did not serve a following request after abort')
            return dict(request_id=row['request_id'], visible_tokens=row['token_count'],
                        final_finish_reason=row['finish_reason'], next_request_completed=True,
                        discarded_inflight_tokens='UNKNOWN',
                        boundary='Abort return is not an exact GPU token-accounting boundary')

        await checked('abort', check_abort)

        async def check_concurrent():
            first, second = await asyncio.gather(
                generate_one(engine, tokenizer, events, 'preflight_concurrent_A', shared,
                             max_tokens=64, ignore_eos=True),
                generate_one(engine, tokenizer, events, 'preflight_concurrent_B', shared,
                             max_tokens=64, ignore_eos=True))
            overlap = min(first['end_ns'], second['end_ns']) - max(first['start_ns'], second['start_ns'])
            if overlap <= 0 or any(row['token_count'] != 64 for row in (first, second)):
                raise RuntimeError('Two shared-prefix requests did not overlap and complete')
            if any(row['num_cached_tokens'] <= 0 for row in (first, second)):
                raise RuntimeError('Concurrent shared-prefix requests did not report native cache hits')
            return dict(request_ids=[first['request_id'], second['request_id']],
                        shared_prompt_tokens=len(shared), frontend_lifetime_overlap_ns=overlap,
                        cached_tokens=[first['num_cached_tokens'], second['num_cached_tokens']],
                        scope='Concurrent native requests; not a proof of every same-step batch')

        await checked('shared_prefix_concurrency', check_concurrent)
        summary['status'] = 'PASS'
    except BaseException as error:
        summary.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        events.add('failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        primary_error = sys.exc_info()[1]
        shutdown_error = None
        try:
            if engine is not None:
                engine.shutdown(timeout=15)
        except BaseException as error:
            shutdown_error = error
            summary['shutdown_error'] = str(error)
            if primary_error is None:
                summary.update(status='FAILED_SHUTDOWN', error_type=type(error).__name__, error=str(error))
        finally:
            if engine_start is not None:
                engine_end = time.perf_counter()
        recording_error = flush_preserving_primary_error(events, summary)
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        child_cpu_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary['resources'].update(
            worker_region_wall_s=time.perf_counter() - started,
            worker_cpu_user_s=cpu_end.ru_utime - cpu_start.ru_utime,
            worker_cpu_system_s=cpu_end.ru_stime - cpu_start.ru_stime,
            waited_children_cpu_user_s=child_cpu_end.ru_utime - child_cpu_start.ru_utime,
            waited_children_cpu_system_s=child_cpu_end.ru_stime - child_cpu_start.ru_stime,
            waited_children_scope='RUSAGE_CHILDREN observed after engine shutdown; not all system GPU processes',
            engine_startup_and_ownership_wall_s=(engine_end - engine_start) if engine_start is not None else 0,
            engine_ready_to_shutdown_wall_s=(engine_end - engine_ready) if engine_ready is not None else 'UNKNOWN',
            cuda_kernel_busy_s='UNKNOWN',
            boundary='Engine ownership and process wall times include CPU, startup and waiting; never CUDA busy time')
        summary.update(ended_epoch=time.time(), process_wall_s=time.perf_counter() - started,
                       event_rows=events.count, cuda_busy_time_s='UNKNOWN',
                       recording=dict(buffered_until_check_boundary=True, dropped=0,
                                      pending_unwritten_rows=len(events.pending),
                                      failed=bool(summary.get('recording_errors')),
                                      failure_recovery='Abrupt process loss may lose the active check buffer'))
        save(run_dir / 'summary.json', summary)
        if shutdown_error is not None and primary_error is None:
            raise shutdown_error
        if recording_error is not None and primary_error is None:
            raise recording_error


def build_config(args, audit):
    engine = json.loads(Path(args.engine_config).read_text())['engine']
    engine = snapshot_engine_kwargs(dict(engine, max_model_len=40960))
    if engine.get('dtype') != 'bfloat16' or not engine.get('async_scheduling') or engine.get('enforce_eager'):
        raise ValueError('Expected existing BF16/async/compile engine configuration')
    if args.budget_protocol not in (None, *PROTOCOLS):
        raise ValueError('Unknown budget protocol')
    software = {}
    for package in ('vllm', 'torch', 'transformers'):
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = 'UNKNOWN'
    return dict(schema='exec_exit.engine_preflight.v1', model=str(MODEL),
        model_realpath=str(MODEL.resolve()), model_config_sha256=sha(MODEL / 'config.json'),
        expected_gpu_uuid=args.gpu_uuid, actual_model_runner='UNKNOWN_NOT_RUN',
        historical_runner_reference='vllm.v1.worker.gpu.model_runner.GPUModelRunner',
        historical_runner_is_not_current_execution_evidence=True,
        engine=engine, engine_basis=str(args.engine_config),
        engine_basis_sha256=sha(args.engine_config), budget_protocol=args.budget_protocol,
        native_thinking_budget_on_v2=audit['native_budget_on_v2'],
        two_segment_protocol=dict(
            protocol='application_layer_two_calls', thinking_cap=28000,
            thinking_stop='exact </think> token ID retained in IDs and text',
            natural_answer='continue exact full token prefix; max_tokens=40960-len(prefix); native EOS',
            natural_answer_has_2048_cap=False,
            forced_only_when='thinking cap reached with no </think> in the generated thinking prefix',
            forced_control=FORCED_ANSWER_CONTROL, forced_answer_cap=2048,
            forced_stop=['```'], include_stop_str_in_output=True,
            seed_derivation="int(sha256(f'reasoning_capacity_v1:{case_id}:{repeat}:{stage}').hexdigest()[:8],16)",
            repeat=0, thinking_stage=0, natural_or_forced_answer_stage=1,
            seed_basis='research/early_exit_component/run/scripts/legacy_workflow.py:33',
            seed_change_from_old_calls='Derivation retained; answer stage 1 is explicit under the new two-call protocol',
            smoke_boundaries='Synthetic caps of 128 and 1; actual 28000 boundary NOT executed'),
        check_timeout_s=args.check_timeout_s, run_timeout_s=args.run_timeout_s,
        source_only=not args.execute, python_executable=sys.executable,
        software_versions=software,
        source_sha256=sha(__file__),
        snapshot_interface=dict(worker_extension_cls=SNAPSHOT_EXTENSION,
                                rpc_method=SNAPSHOT_RPC_METHOD,
                                transport='native_string_rpc',
                                adds_insecure_serialization_override=False),
        visible_gpu_environment=os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET'),
        vllm_use_v2_environment=os.environ.get('VLLM_USE_V2_MODEL_RUNNER', 'UNSET'),
        process_environment={name: os.environ.get(name, 'UNSET') for name in (
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'VLLM_WORKER_MULTIPROC_METHOD', 'VLLM_ENABLE_V1_MULTIPROCESSING',
            'VLLM_NO_USAGE_STATS', 'VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR',
            'VLLM_ALLOW_INSECURE_SERIALIZATION')},
        validation_scope='Synthetic technical smoke, no task-quality or service-benefit evaluation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--engine-config', type=Path, default=OLD_CONFIG)
    parser.add_argument('--budget-protocol', choices=PROTOCOLS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-timeout-s', type=float, default=180)
    parser.add_argument('--run-timeout-s', type=float, default=1800)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if args.worker:
        config = json.loads((run_dir / 'config.yaml').read_text())
        asyncio.run(gpu_checks(config, run_dir))
        return 0
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Refusing to overwrite an existing preflight run; use a new run directory')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'raw').mkdir(exist_ok=True)
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    audit = source_audit()
    config = build_config(args, audit)
    save(run_dir / 'config.yaml', config)
    (run_dir / 'commit.txt').write_text(subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() + '\n')
    with (run_dir / 'raw/source_audit.jsonl').open('w') as stream:
        stream.write(json.dumps(dict(event='source_audit', epoch_ns=time.time_ns(), **audit)) + '\n')
    blocked = args.budget_protocol is None or args.budget_protocol == 'native_thinking'
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    summary = dict(status='BLOCKED_PROTOCOL_DECISION' if blocked else 'SOURCE_AUDIT_COMPLETE',
        gpu_execution='NOT_RUN', gpu_used=False, native_budget_on_v2=audit['native_budget_on_v2'],
        checks={name: dict(status='NOT_RUN_GPU') for name in CHECKS},
        native_budget_guard=audit['guard_reproduction'],
        budget_protocol=args.budget_protocol,
        unresolved='Select a budget protocol; native thinking_token_budget is rejected by V2' if blocked else None,
        resources=dict(cpu_wall_s=time.perf_counter() - started,
                       cpu_user_s=usage_after.ru_utime - usage_before.ru_utime,
                       cpu_system_s=usage_after.ru_stime - usage_before.ru_stime,
                       gpu_seconds=0),
        source_only_is_not_runtime_success=True)
    save(run_dir / 'summary.json', summary)
    if not args.execute:
        print(json.dumps(summary, ensure_ascii=False))
        return 2 if blocked else 0
    if blocked:
        print(json.dumps(summary, ensure_ascii=False))
        return 2
    # One finite fault-protected run.  A timeout terminates this owned process
    # group only; no retry, runner fallback, or CUDA-mode change is attempted.
    command = [sys.executable, '-m', 'exec_exit.gen.engine_preflight',
               '--run-dir', str(run_dir), '--gpu-uuid', args.gpu_uuid, '--worker']
    supervisor_started = time.perf_counter()
    supervisor_cpu = resource.getrusage(resource.RUSAGE_SELF)
    children_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    supervisor_events = Events(run_dir / 'raw/supervisor.jsonl')
    timed_out = False
    with (run_dir / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        manifest = dict(status='RUNNING', supervisor_pid=os.getpid(), worker_pid=process.pid,
                        owned_process_group=process.pid, command=command,
                        started_epoch=time.time(), run_timeout_s=args.run_timeout_s)
        save(run_dir / 'process_manifest.json', manifest)
        supervisor_events.add('worker_launched', **manifest)
        supervisor_events.flush()
        try:
            result = process.wait(timeout=args.run_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            supervisor_events.add('run_timeout', worker_pid=process.pid,
                                  run_timeout_s=args.run_timeout_s)
            supervisor_events.flush()
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result = process.returncode
    final = json.loads((run_dir / 'summary.json').read_text())
    if timed_out:
        final.update(status='FAILED_RUN_TIMEOUT', run_timeout_s=args.run_timeout_s)
    elif result != 0 and not final.get('status', '').startswith('FAILED'):
        final['status'] = 'FAILED_WORKER_EXIT'
    supervisor_end = resource.getrusage(resource.RUSAGE_SELF)
    children_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    supervisor_usage = dict(
        owned_worker_process_wall_s=time.perf_counter() - supervisor_started,
        supervisor_cpu_user_s=supervisor_end.ru_utime - supervisor_cpu.ru_utime,
        supervisor_cpu_system_s=supervisor_end.ru_stime - supervisor_cpu.ru_stime,
        waited_worker_tree_cpu_user_s=children_end.ru_utime - children_cpu.ru_utime,
        waited_worker_tree_cpu_system_s=children_end.ru_stime - children_cpu.ru_stime,
        cpu_scope='RUSAGE_CHILDREN after wait; overlaps worker-reported self/children and must not be summed with them',
        cuda_kernel_busy_s='UNKNOWN')
    final.setdefault('resources', {})['gpu_seconds'] = 'UNKNOWN_KERNEL_BUSY_TIME'
    final['supervisor_resources'] = supervisor_usage
    final['worker_returncode'] = result
    final['gpu_execution'] = 'COMPLETED' if final.get('status') == 'PASS' else 'FAILED'
    manifest.update(status=final['status'], ended_epoch=time.time(), worker_returncode=result,
                    timed_out=timed_out, resources=supervisor_usage)
    save(run_dir / 'process_manifest.json', manifest)
    supervisor_events.add('worker_exited', **manifest)
    supervisor_events.flush()
    save(run_dir / 'summary.json', final)
    print(json.dumps(final, ensure_ascii=False))
    return 3 if timed_out else (result or (0 if final.get('status') == 'PASS' else 1))


if __name__ == '__main__':
    raise SystemExit(main())
