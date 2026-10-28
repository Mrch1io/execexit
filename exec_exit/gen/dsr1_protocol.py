"""DS-R1 token-boundary adapter; the frozen Qwen sources remain unchanged.

Only model-native rendering/marker boundaries differ. Sampling, stage seeds,
budget, generated-token accounting and first-close program selection are kept.
The model's prompt-side <think> is an observed prompt token, never a generated
token. All full draft records support the existing virtual-v1 postprocessing.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import time

from exec_exit.gen import traces
from exec_exit.gen.engine_preflight import MODELS, validate_prompt_token_ids
from exec_exit.execute.extract import extract_code_bytes, first_close_final_ids, CodeTokenMapper

MODEL = MODELS / 'DeepSeek-R1-Distill-Qwen-7B'
REVISION = '916b56a44061fd5cd7d6a8fb632557ed4f724f60'
CONTROL = '\n</think>\n\n```python\n'
PROMPT_POLICY = 'same_legacy_system_and_user_messages_model_native_template'


@dataclass(frozen=True)
class Protocol:
    thinking_cap: int = 28000
    model_capacity: int = 40960
    forced_answer_cap: int = 2048
    forced_control: str = CONTROL


def render_prompt(tokenizer, case):
    messages, _ = traces.legacy_helpers()
    rendered = tokenizer.apply_chat_template(messages(case), tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    validate_prompt_token_ids(ids)
    start = tokenizer.convert_tokens_to_ids('<think>')
    if start not in ids or tokenizer.decode(ids[ids.index(start):], skip_special_tokens=False) != '<think>\n':
        raise ValueError('DS-R1 native prompt must end in its observed <think> newline')
    return rendered, ids


def control_ids(tokenizer, close_id):
    ids = tokenizer.encode(CONTROL, add_special_tokens=False)
    validate_prompt_token_ids(ids)
    # The newline is deliberately before the close token; validate the actual
    # vocabulary rather than applying Qwen's close-at-index-zero assumption.
    if ids.count(close_id) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != CONTROL:
        raise ValueError('DS-R1 control IDs do not exactly encode the selected control')
    if tokenizer.decode(ids[:ids.index(close_id)], skip_special_tokens=False) != '\n':
        raise ValueError('DS-R1 control must begin with the specified newline')
    return ids


def continuation(first, *, prompt_ids, close_id, eos_ids, tokenizer, protocol=Protocol()):
    ids = first['token_ids']
    if not first['native_finished'] or first.get('error') or len(ids) > protocol.thinking_cap:
        raise RuntimeError('Incomplete or over-budget first native segment')
    if ids and ids[-1] == close_id:
        if close_id in ids[:-1]: raise RuntimeError('Repeated close in stopped segment')
        prefix = prompt_ids + ids
        remaining = protocol.model_capacity - len(prefix)
        if remaining <= 0: raise RuntimeError('No natural answer capacity')
        return dict(branch='natural_thinking_close', prefix=prefix, control=[], max_tokens=remaining,
                    forced=False, thinking_end_token_index=len(ids)-1)
    if ids and ids[-1] in eos_ids:
        return dict(branch='unexpected_eos_in_thinking', prefix=None, control=[], max_tokens=0,
                    forced=False, thinking_end_token_index=None)
    if close_id in ids: raise RuntimeError('Close is not the final observed token')
    if len(ids) == protocol.thinking_cap and first['finish_reason'] == 'length':
        control = control_ids(tokenizer, close_id)
        prefix = prompt_ids + ids + control
        if len(prefix) + protocol.forced_answer_cap > protocol.model_capacity:
            raise RuntimeError('Full forced-answer request exceeds capacity')
        return dict(branch='forced_at_cap', prefix=prefix, control=control,
                    max_tokens=protocol.forced_answer_cap, forced=True, thinking_end_token_index=None)
    raise RuntimeError('Unspecified first-segment termination')


async def generate_case(engine, tokenizer, writer, case, eos_ids, parameter_factory,
                        progress=lambda *args: None, protocol=Protocol(), *, sample_index=0):
    if type(sample_index) is not int or sample_index < 0: raise ValueError('Invalid sample index')
    _, seed_for = traces.legacy_helpers()
    rendered, prompt = render_prompt(tokenizer, case)
    if len(prompt) + protocol.thinking_cap > protocol.model_capacity: raise ValueError('Prompt capacity')
    close = tokenizer.convert_tokens_to_ids('</think>')
    if not isinstance(close, int) or close in eos_ids: raise ValueError('Invalid close ID')
    first = await traces.generate_call(engine, tokenizer, writer, case_id=case['case_id'], stage=0,
        kind='thinking', prompt_ids=prompt, parameter_factory=parameter_factory, progress=progress,
        sampling=traces.params_for(seed_for(case['case_id'], sample_index, 0), protocol.thinking_cap, think_id=close))
    decision = continuation(first, prompt_ids=prompt, close_id=close, eos_ids=eos_ids,
                            tokenizer=tokenizer, protocol=protocol)
    calls = [first]
    if decision['prefix'] is not None:
        calls.append(await traces.generate_call(engine, tokenizer, writer, case_id=case['case_id'], stage=1,
            kind='forced_answer' if decision['forced'] else 'natural_answer', prompt_ids=decision['prefix'],
            parameter_factory=parameter_factory, progress=progress,
            sampling=traces.params_for(seed_for(case['case_id'], sample_index, 1), decision['max_tokens'], forced=decision['forced'])))
    last = calls[-1]
    return dict(event='trajectory_complete', schema='exec_exit.trajectory.v1', status='COMPLETE',
        case_id=case['case_id'], task_id=case['task_id'], prompt_token_ids=prompt, prompt_text=rendered,
        prompt_tokens=len(prompt), calls=calls, branch=decision['branch'],
        injected_control_token_ids=decision['control'], injected_control_text=CONTROL if decision['forced'] else '',
        thinking_end_token_index=decision['thinking_end_token_index'],
        thinking_budget_hit=len(first['token_ids']) == protocol.thinking_cap,
        answer_truncated=len(calls)==2 and last['finish_reason']=='length', answer_available=len(calls)==2,
        terminal_eos_observed=bool(last['token_ids'] and last['token_ids'][-1] in eos_ids),
        generated_tokens=sum(c['token_count'] for c in calls), injected_control_tokens=len(decision['control']),
        started_ns=first['started_ns'], ended_ns=last['ended_ns'], correctness='NOT_EVALUATED',
        model_family='dsr1_distill_qwen', prompt_policy=PROMPT_POLICY,
        thinking_start_origin='actual_prompt_ids_not_generated')


def _observed_prompt_think_tail(prompt, tokenizer):
    start, close = tokenizer.token_to_id('<think>'), tokenizer.token_to_id('</think>')
    starts = [i for i, value in enumerate(prompt) if value == start]
    if len(starts) != 1: return None
    tail = prompt[starts[0]:]
    if close in tail or tokenizer.decode(tail, skip_special_tokens=False) != '<think>\n': return None
    return tail


def extract_record(record, source_kind, tokenizer):
    """Use the real prompt marker plus actual output; never re-tokenize output."""
    begun=time.perf_counter_ns(); final_ids=None; generation_valid=False
    close=tokenizer.token_to_id('</think>')
    if source_kind == 'N':
        calls=record['calls']; prompt=record['prompt_token_ids']
        tail=_observed_prompt_think_tail(prompt, tokenizer)
        sequence=list(calls[0]['token_ids'])
        if len(calls)==2: sequence += record['injected_control_token_ids'] + calls[1]['token_ids']
        if tail is not None: final_ids=first_close_final_ids(tail+sequence,tokenizer)
        generation_valid=len(calls)==2 and all(c.get('native_finished') and not c.get('error') for c in calls) and calls[-1].get('finish_reason')=='stop'
    elif source_kind == 'probe':
        call=record['call']; calls=[call]; prefix=call.get('prompt_token_ids',[])
        position=record.get('position_generated_tokens'); control=record['control_token_ids']
        if (type(position) is int and position>0 and control.count(close)==1
                and tokenizer.decode(control,skip_special_tokens=False)==CONTROL
                and len(prefix)>position+len(control) and prefix[-len(control):]==control):
            original=prefix[:-(position+len(control))]
            tail=_observed_prompt_think_tail(original,tokenizer)
            if tail is not None:
                final_ids=first_close_final_ids(tail+prefix[-(position+len(control)):]+call['token_ids'],tokenizer)
        generation_valid=bool(call.get('native_finished') and not call.get('error') and call.get('finish_reason')=='stop')
    else: raise ValueError('Unknown source kind')
    text=tokenizer.decode(final_ids,skip_special_tokens=True) if final_ids is not None else None
    result=extract_code_bytes(text)
    result.update(final_text_sha256=hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
        generation_valid=generation_valid, generation_reason='answer_complete' if generation_valid else 'incomplete_truncated_or_missing_answer',
        termination=[dict(finish_reason=c.get('finish_reason'),stop_reason=c.get('stop_reason')) for c in calls],
        deliverable=generation_valid and result['status']=='PARSED', offline_started_ns=begun,
        offline_finished_ns=time.perf_counter_ns(),online_visible_at='UNKNOWN_NOT_EXECUTED_ONLINE',
        observation_scope='OFFLINE_CPU_ONLY',marker_adapter='dsr1_actual_prompt_think_first_close')
    return result


class DSCodeTokenMapper(CodeTokenMapper):
    def probe_code_tokens(self, record, extraction):
        # The parent maps only final-region bytes. Its control[1:] convention is
        # supplied with the actual suffix beginning at this model's close ID.
        close=self.codec.close_id; control=record['control_token_ids']
        if control.count(close)!=1: return dict(status='UNKNOWN',reason='invalid_control')
        mapped=dict(record,control_token_ids=control[control.index(close):])
        return super().probe_code_tokens(mapped,extraction)
