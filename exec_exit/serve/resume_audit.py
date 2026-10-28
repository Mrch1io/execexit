"""Independent development-service audit of exact-prefix physical restarts.

No model, submitted program or test suite is executed. Token agreement is an
observation, never a pass condition; unknown cancellation work remains unknown.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from exec_exit.gen.audit_probes import digest, read, rows, write
from exec_exit.gen.engine_preflight import ROOT, seed_for
from exec_exit.serve.async_audit import check_service, coverage

META = {'event', 'at_ns', 'epoch_ns', 'service_identity'}


def payload(event):
    return {k: v for k, v in event.items() if k not in META}


def compare_tokens(left, right):
    """Positional agreement on actual overlap; lengths/first divergence explicit."""
    overlap = min(len(left), len(right))
    first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    matches = sum(a == b for a, b in zip(left, right))
    return dict(left_tokens=len(left), right_tokens=len(right), overlap_tokens=overlap,
        matching_tokens=matches, overlap_agreement=matches / overlap if overlap else None,
        identical_prefix_tokens=overlap if first is None else first,
        first_divergence_generated_position_1based=None if first is None else first + 1,
        first_divergence_token_ids=None if first is None else [left[first], right[first]],
        overlap_equal=first is None, full_equal=left == right,
        length_difference=len(left) - len(right), empty_overlap='UNKNOWN_NO_COMMON_TOKEN' if not overlap else None)


def main_ids(record, stage=None):
    calls = sorted(record['native_calls'].values(), key=lambda c: c['stage'])
    return [token for call in calls if stage is None or call['stage'] == stage for token in call['token_ids']]


def prefix_comparisons(records):
    by_case = defaultdict(dict)
    for r in records:
        by_case[r['case_id']][r['mode']] = r
    result = []
    for case, modes in by_case.items():
        for left, right, scope in [('H', 'N', 'All actual main generated tokens; no length padding'),
                ('HYB-B-SYNC', 'H', 'Common actual main prefix before synchronous delivery/acceptance'),
                ('HYB-B-ASYNC', 'N', 'All returned async main tokens including observed cancellation tail; actual overlap only')]:
            if left not in modes or right not in modes:
                continue
            a, b = modes[left], modes[right]
            result.append(dict(case_id=case, left=left, right=right, comparison_scope=scope,
                interpretation='Different actual paths are possible; matching nominal schedule is not matched GPU state',
                all_generated=compare_tokens(main_ids(a), main_ids(b)),
                stages={str(s): compare_tokens(main_ids(a, s), main_ids(b, s)) for s in (0, 1)},
                left_selected_kind=a['selected_kind'], right_selected_kind=b['selected_kind']))
    return result


def actual_coverage(records):
    result = coverage(records)
    # Async abort is requested after acceptance; sync first aborts/drains the
    # main request, and only then generates/checks a possible accepted draft.
    # These are different paths and must not share an "accept then abort" label.
    sync = [r for r in records if r['mode'] == 'HYB-B-SYNC']
    result['sync_probe_deliveries_after_main_abort'] = sum(
        r['selected_kind'] == 'PROBE' and any(
            r['token_account'].get(rid, {}).get('abort_ns') is not None
            for rid in r.get('physical_calls', {})) for r in sync)
    result['coverage_scope'] = 'Observed paths; failed PUB counts as executed, UNKNOWN/fallback is not quality failure'
    return result


def check_delivery(record, events, require):
    ident = record['service_identity']
    require(record['status'] == 'COMPLETE' and not record['coordinator_errors'], ident + ': complete service')
    require(record['delivery_count'] == record['client']['delivery_count'] == 1, ident + ': one delivery')
    grouped = defaultdict(list)
    for e in events:
        grouped[e['event']].append(e)
    for name in ('server_request_received', 'server_delivery_queued', 'server_send_returned', 'delivery', 'client_receipt'):
        require(len(grouped[name]) == 1, ident + ': one ' + name)
    receipt = {k: v for k, v in grouped['client_receipt'][0].items() if k not in META - {'event'}}
    require(receipt == record['client'], ident + ': real client acknowledgement')
    require(record['client']['client_submit_ns'] <= record['server_receive_ns'] <= record['server_accept_ns'] <= record['client']['client_receive_complete_ns'], ident + ': acceptance visibility order')
    require(record['server_accept_ns'] <= record['drained_ns'], ident + ': service drain bound')
    extraction = record['selected']['extraction']
    packet = dict(kind=record['selected_kind'], case_id=record['case_id'], deliverable=bool(extraction['deliverable']),
        code_text=extraction.get('code_text'), extraction_status=extraction['status'], server_accept_ns=record['server_accept_ns'])
    encoded = json.dumps(packet, ensure_ascii=False).encode()
    require(len(encoded) == record['client']['response_bytes'] and hashlib.sha256(encoded).hexdigest() == record['client']['response_sha256'], ident + ': actual transport payload')
    require(extraction['online_visible_at'] <= record['server_accept_ns'], ident + ': extraction visible before delivery')
    require(grouped['delivery'][0]['visible_ns'] == grouped['server_delivery_queued'][0]['visible_ns'] == record['server_accept_ns'], ident + ': one actual acceptance timestamp')
    for v in record['visits']:
        if 'public' not in v:
            continue
        p = v['public']
        require(v['draft']['deliverable'] and not v['draft']['comment_aborted'], ident + ': public check requires a complete valid draft')
        require(v['draft']['extraction']['online_visible_at'] <= p['server_check_queued_ns'] <= p['worker_visible_start_ns'] <= p['worker_complete_ns'] <= p['server_check_visible_ns'], ident + ': actual public-only check visibility')
    accepted = [v for v in record['visits'] if v['status'] == 'ACCEPTED']
    if record['selected_kind'] == 'PROBE':
        require(len(accepted) == 1 and accepted[0]['public']['status'] == 'PASS', ident + ': one accepted public PASS')
        require(accepted[0]['public']['server_check_visible_ns'] <= record['server_accept_ns'], ident + ': public result before acceptance')
        require(accepted[0]['draft'] == record['selected'], ident + ': selected actual draft')
    else:
        require(not accepted and record['native_record'] is not None, ident + ': actual native delivery')


def check_resume(record, events, require):
    ident = record['service_identity']
    check_delivery(record, events, require)
    require(record['mode'] in {'H', 'HYB-B-SYNC'}, ident + ': explicit restart mode')
    expected = record['native_calls'] | record['probe_calls']
    physical = record['physical_calls']
    starts, ends, pstarts, pends = {}, {}, {}, {}
    offsets, poffsets = Counter(), Counter()
    texts, poutputs = defaultdict(list), defaultdict(list)
    visible_native, prompt = [], None
    pause_events = defaultdict(dict)
    for event in events:
        kind = event['event']
        if kind in ('call_start', 'probe_start'):
            rid = event.get('request_id', event.get('probe_id'))
            require(rid not in starts, ident + ': unique logical request start')
            starts[rid] = event
            if kind == 'call_start' and event['stage'] == 0:
                prompt = event['prompt_token_ids']
            if kind == 'probe_start':
                p = event['position_generated_tokens']
                require(prompt is not None and 0 < p <= min(len(visible_native), 4096), ident + ': probe at an observed prefix')
                require(event['prompt_token_ids'] == prompt + visible_native[:p] + event['control_token_ids'], ident + ': exact probe prefix')
                require(p in record['candidate_observation']['selected_hyb_b_positions'], ident + ': discovered opportunity')
        elif kind in ('call_output', 'probe_output'):
            rid = event.get('request_id', event.get('probe_id'))
            require(rid in starts and rid not in ends and rid in expected, ident + ': logical output lifecycle')
            call, offset, ids = expected[rid], offsets[rid], event['token_ids']
            require(event['token_offset'] == offset and ids == call['token_ids'][offset:offset + len(ids)], ident + ': exact logical deltas')
            require(call['started_ns'] <= event['native_frontend_receive_ns'] <= call['ended_ns'], ident + ': logical receive time')
            offsets[rid] += len(ids)
            texts[rid].append(event['text'])
            if kind == 'call_output' and event['stage'] == 0:
                visible_native.extend(ids)
        elif kind in ('call_end', 'probe_end'):
            rid = event.get('request_id', event.get('probe_id'))
            require(rid in starts and rid not in ends, ident + ': unique logical end')
            actual = event['call'] if kind == 'probe_end' else payload(event)
            require(actual == expected[rid], ident + ': logical stream/checkpoint identity')
            ends[rid] = event
        elif kind == 'physical_start':
            rid = event['request_id']
            require(rid not in pstarts and rid in physical, ident + ': unique physical start')
            require(event['token_ids'] == [], ident + ': physical start snapshot has no future token alias')
            pstarts[rid] = event
        elif kind == 'physical_output':
            rid = event['request_id']
            require(rid in pstarts and rid not in pends, ident + ': physical delta lifecycle')
            offset, ids, call = poffsets[rid], event['token_ids'], physical[rid]
            require(event['token_offset'] == offset and ids == call['token_ids'][offset:offset + len(ids)], ident + ': exact physical delta')
            require(call['started_ns'] <= event['native_frontend_receive_ns'] <= call['ended_ns'], ident + ': physical receive bounds')
            poffsets[rid] += len(ids)
            poutputs[rid].append(event)
        elif kind == 'physical_end':
            rid = event['request_id']
            require(rid in pstarts and rid not in pends and payload(event) == physical[rid], ident + ': physical final snapshot identity')
            pends[rid] = event
        elif kind in ('pause_requested', 'pause_drained', 'resume_submitted'):
            index = event['pause_index']
            require(kind not in pause_events[index], ident + ': unique pause stage')
            pause_events[index][kind] = event
    require(set(starts) == set(ends) == set(expected), ident + ': logical lifecycle denominator')
    require(set(pstarts) == set(pends) == set(physical), ident + ': physical lifecycle denominator')
    require(all(p['logical_request_id'] in record['native_calls'] for p in physical.values()), ident + ': no orphan physical segments')
    for rid, call in expected.items():
        require(offsets[rid] == call['token_count'] == len(call['token_ids']) and ''.join(texts[rid]) == call['raw_text'], ident + ': all logical token/text observations')
        if rid in record['native_calls']:
            allowed_interruption = record['selected_kind'] == 'PROBE' and call['stage'] == 0 and str(call.get('error', '')).startswith('AcceptedProbe(')
            require(call.get('error') is None or allowed_interruption, ident + ': no masked logical error')
            require(call['native_finished'] or allowed_interruption, ident + ': native complete or explicitly accepted interruption')
    total, restarts, logical_input, cached_values = 0, 0, 0, []
    for logical, call in record['native_calls'].items():
        physical_calls = sorted((p for p in physical.values() if p['logical_request_id'] == logical), key=lambda p: p['restart_index'])
        require(bool(physical_calls) and [p['restart_index'] for p in physical_calls] == list(range(len(physical_calls))), ident + ': consecutive physical segments')
        retained = []
        for p in physical_calls:
            rid = p['request_id']
            require(p['stage'] == call['stage'], ident + ': unchanged logical stage')
            require(p['seed'] == call['sampling_parameters']['seed'], ident + ': unchanged original segment seed')
            require(p['original_segment_max_tokens'] == call['sampling_parameters']['max_tokens'], ident + ': unchanged segment cap')
            require(p['prompt_token_ids'] == starts[logical]['prompt_token_ids'] + retained, ident + ': exact original prompt plus ALL observed tokens')
            require(p['kept_generated_prefix_tokens'] == len(retained) and p['max_tokens'] == p['original_segment_max_tokens'] - len(retained), ident + ': remaining budget, never reset')
            require(0 < p['max_tokens'] and len(p['token_ids']) <= p['max_tokens'], ident + ': physical cap')
            require(p['frontend_drained'] is True and not p.get('cleanup_error'), ident + ': physical stream cleanly drained')
            require(poffsets[rid] == len(p['token_ids']), ident + ': physical observed denominator')
            for key in ('logical_request_id', 'stage', 'restart_index', 'prompt_token_ids', 'seed', 'max_tokens', 'original_segment_max_tokens', 'kept_generated_prefix_tokens', 'started_ns'):
                require(pstarts[rid][key] == p[key], ident + ': immutable physical start ' + key)
            cached = [e['num_cached_tokens'] for e in poutputs[rid] if e.get('num_cached_tokens') is not None]
            require(p.get('num_cached_tokens') == (cached[-1] if cached else None), ident + ': actual cache metadata preserved')
            require(not cached or all(type(c) is int and 0 <= c <= len(p['prompt_token_ids']) for c in cached), ident + ': cache count bound')
            cached_values.append(cached[-1] if cached else None)
            logical_input += len(p['prompt_token_ids'])
            restarts += int(p['restart_index'] > 0)
            retained += p['token_ids']
        require(retained == call['token_ids'], ident + ': physical concatenation equals logical stream')
    require(set(record['token_account']) == set(physical) | set(record['probe_calls']), ident + ': physical-only account, no logical double charge')
    for rid, account in record['token_account'].items():
        ids = physical[rid]['token_ids'] if rid in physical else record['probe_calls'][rid]['token_ids']
        require(account['observed_tokens'] == len(ids), ident + ': all observed tokens charged once')
        if rid in physical:
            require(account['observed_after_abort'] == sum(len(e['token_ids']) for e in poutputs[rid] if e['after_abort']), ident + ': exact observed abort tail')
            require((account['abort_ns'] is not None) == physical[rid]['aborted'], ident + ': cancellation identity')
        if account['abort_ns'] is not None:
            require(account['unobserved_cancelled_tokens'] == 'UNKNOWN', ident + ': unexposed tail remains UNKNOWN')
        total += len(ids)
    require(total == record['observed_generated_tokens'] == sum(offsets.values()), ident + ': physical/logical token reconciliation')
    require(sum(a['observed_after_abort'] for a in record['token_account'].values()) == record['observed_post_abort_tokens'], ident + ': post-abort total')
    aborted = any(a['abort_ns'] is not None for a in record['token_account'].values())
    require(record['unobserved_cancelled_tokens'] == ('UNKNOWN' if aborted else 0), ident + ': cancellation total never imputed')
    require(set(pause_events) == set(range(len(record['pauses']))), ident + ': pause denominator')
    for i, pause in enumerate(record['pauses']):
        ev = pause_events[i]
        require('pause_requested' in ev and 'pause_drained' in ev, ident + ': requested/drained pause')
        require(pause['pause_started_ns'] <= pause['drained_ns'], ident + ': pause timing')
        require(pause['observed_abort_tail_tokens'] == pause['kept_prefix_tokens_after_drain'] - pause['observed_prefix_before_abort_tokens'] >= 0, ident + ': exact retained tail')
        require(pause['nominal_positions'] == sorted(set(pause['nominal_positions'])) and all(0 < p <= min(4096, pause['kept_prefix_tokens_after_drain']) for p in pause['nominal_positions']), ident + ': no future pause location')
        require(pause['executed_nominal_positions'] == pause['nominal_positions'][:len(pause['executed_nominal_positions'])], ident + ': executed opportunities are a finite ordered prefix')
        for key in ('nominal_positions', 'pause_started_ns', 'abort_requested_ns', 'observed_prefix_before_abort_tokens', 'drained_ns', 'kept_prefix_tokens_after_drain', 'observed_abort_tail_tokens', 'unobserved_abort_tail_tokens'):
            require(ev['pause_drained'][key] == pause[key], ident + ': independent drained pause ' + key)
        if pause['resume_submitted_ns'] is not None:
            require('resume_submitted' in ev and pause['drained_ns'] <= pause['resume_submitted_ns'], ident + ': resume after drain')
            require(ev['resume_submitted']['resume_submitted_ns'] == pause['resume_submitted_ns'], ident + ': exact resume time')
        else:
            require('resume_submitted' not in ev, ident + ': no invented resume')
    if record['mode'] == 'H':
        require(not record['visits'] and not record['probe_calls'] and record['selected_kind'] == 'N', ident + ': H never probes/early-delivers')
        require(all(p['probe_wall_seconds'] == p['public_wall_seconds'] == 0 for p in record['pauses']), ident + ': H no artificial probe wait')
    else:
        require([p for pause in record['pauses'] for p in pause['executed_nominal_positions']] == [v['position'] for v in record['visits']], ident + ': actually executed sync schedule')
    require(len(record['probe_calls']) <= 3, ident + ': bounded probe set')
    require(record['private_tests_read'] is False and record['future_labels_or_trajectories_read'] is False, ident + ': declared public-only service')
    pub = [v['public'] for v in record['visits'] if 'public' in v]
    return dict(service_identity=ident, status='PASS', mode=record['mode'], physical_calls=len(physical), physical_restarts=restarts,
        observed_tokens=total, logical_native_calls=len(record['native_calls']), probes=len(record['probe_calls']),
        physical_logical_input_tokens=logical_input, physical_cached_input_tokens=sum(cached_values) if all(c is not None for c in cached_values) else None,
        cache_unknown_calls=sum(c is None for c in cached_values),
        input_cost_scope='Logical request input and reported prefix cache hits; physical recompute/GPU token work UNKNOWN',
        public_check_demands=len(pub), public_original_execution_cpu_seconds=sum(p['original_execution_cpu_seconds'] for p in pub),
        public_actual_execution_cpu_seconds=sum(p['original_execution_cpu_seconds'] for p in pub if p['cache_reused'] is False),
        client_e2e_seconds=record['client']['client_e2e_seconds'], selected_kind=record['selected_kind'],
        unobserved_cancelled_tokens=record['unobserved_cancelled_tokens'], gpu_kernel_seconds='UNKNOWN')


def audit(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    summary, config, manifest = read(source/'summary.json'), read(source/'config.yaml'), read(source/'manifest.json')
    if summary['status'] != 'COMPLETE' or summary['worker_returncode'] != 0 or summary['supervisor_timeout'] or not summary['native_frontend_drained']:
        raise ValueError('Actual service run must complete and drain before audit')
    if config['schema'] != 'exec_exit.c4.resume_dev.config.v1' or config['heldout_read'] is not False or config['private_tests_in_online_path'] is not False:
        raise ValueError('Only explicit development C4 source')
    output.mkdir(parents=True, exist_ok=False)
    for part in ('raw', 'source'):
        (output/part).mkdir()
    start, cpu = time.perf_counter(), time.process_time()
    streams = sorted((source/'raw').glob('stream_*.jsonl'))
    write(output/'config.yaml', dict(source_run=str(source), source_config_sha256=digest(source/'config.yaml'), source_summary_sha256=digest(source/'summary.json'),
        source_files_sha256={str(p.relative_to(source)): digest(p) for p in streams + [source/'raw/services.jsonl', source/'manifest.json']},
        created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=digest(__file__), gpu_used=False,
        interpretation='Record identity/accounting audit only; token divergence is retained, not failure'))
    (output/'commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True))
    shutil.copyfile(__file__, output/'source'/Path(__file__).name)
    checks, details, comparisons = 0, [], []
    result = dict(status='FAILED', records_status='FAIL', gpu_used=False, new_program_tests=0)
    def require(value, message):
        nonlocal checks
        checks += 1
        if not value:
            raise AssertionError(message)
    try:
        for path, expected in config['source_sha256'].items():
            require(digest(path) == expected, 'Prepared source/config changed: ' + path)
        records = list(rows(source/'raw/services.jsonl'))
        require(len(records) == len(manifest['tasks']) == summary['planned'] == summary['completed'], 'All services in denominator')
        require(len({r['service_identity'] for r in records}) == len(records), 'Unique service identities')
        require([(r['case_id'], r['mode']) for r in records] == [(c, m) for c in config['task_ids'] for m in config['modes']], 'Predeclared task/mode order')
        groups, resets, n = defaultdict(list), [], 0
        for path in streams:
            for event in rows(path):
                n += 1
                if event['event'] == 'worker_snapshot':
                    continue
                if event['event'] == 'prefix_cache_reset':
                    resets.append((event['case_id'], event['mode']))
                    continue
                require(event.get('service_identity') is not None, 'Service-tagged event')
                groups[event['service_identity']].append(event)
        require(resets == [(r['case_id'], r['mode']) for r in records], 'Equal cold prefix reset per service')
        require(set(groups) == {r['service_identity'] for r in records}, 'No orphan service events')
        completed = {}
        for record, entry in zip(records, manifest['tasks']):
            require((record['case_id'], record['mode']) == (entry['case_id'], entry['mode']) and entry['status'] == 'COMPLETE', 'Complete frozen manifest entry')
            require(read(source/entry['checkpoint']) == record and digest(source/entry['checkpoint']) == entry['checkpoint_sha256'], 'Checkpoint identity')
            for call in record['native_calls'].values():
                require(call['seed'] == seed_for(record['case_id'], config['sample_index'], call['stage']), 'Exact original segment seed rule')
            if record['mode'] in {'N', 'HYB-B-ASYNC'}:
                details.append(check_service(record, groups[record['service_identity']], require))
            else:
                details.append(check_resume(record, groups[record['service_identity']], require))
            if record['mode'] == 'H' and config['phase'] == 'five':
                prior = completed[(record['case_id'], 'HYB-B-SYNC')]
                target = [p for pause in prior['pauses'] for p in pause['executed_nominal_positions']]
                metadata = record['candidate_observation']
                require(record['matching_control'] is True and metadata['matched_target_positions'] == target, 'H matches executed sync nominal schedule only')
                reached = metadata['selected_hyb_b_positions']
                require(reached == [p for p in target if p not in metadata['unreachable_positions']], 'H reached/UNREACHABLE partition')
                require([p for pause in record['pauses'] for p in pause['executed_nominal_positions']] == reached, 'H actually handled positions')
            completed[record['case_id'], record['mode']] = record
        writer = summary['recording']
        require(writer['accepted'] == writer['written'] == n and writer['rejected'] == 0 and writer['error'] is None, 'Writer loss-free reconciliation')
        require(sum(r['observed_generated_tokens'] for r in records) == summary['observed_generated_tokens'], 'Run observed token ledger')
        require(sum(r['observed_post_abort_tokens'] for r in records) == summary['observed_after_abort_tokens'], 'Run abort-tail ledger')
        comparisons = prefix_comparisons(records)
        result.update(status='PASS', records_status='PASS', services=len(records), questions=len(config['task_ids']), stream_rows=n,
            observed_tokens=summary['observed_generated_tokens'], physical_restarts=sum(d.get('physical_restarts', 0) for d in details),
            prefix_comparisons=comparisons, source_process_status=summary['status'], services_detail=details,
            actual_path_coverage=actual_coverage(records), correctness='Acceptance uses PUB only; complete Dev scoring is a separate process',
            performance='Engineering observation; neither output equality nor method benefit asserted', actual_gpu_busy_seconds='UNKNOWN')
    except Exception as exc:
        result['error'] = repr(exc)
    result.update(checks=checks, wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu)
    for name, values in [('checks', details), ('prefix_comparisons', comparisons)]:
        (output/'raw'/f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values))
    write(output/'summary.json', result)
    write(output/'resource_ledger.json', dict(wall_seconds=result['wall_seconds'], cpu_seconds=result['cpu_seconds'], gpu_used=False, new_program_tests=0))
    (output/'REPRODUCE.md').write_text('```bash\npython -B -m exec_exit.serve.resume_audit --source '+str(source)+' --output runs/NEW_resume_audit\n```\n\nRestore verified archived streams before re-running; use a new output directory. No model or program execution.\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.source, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
