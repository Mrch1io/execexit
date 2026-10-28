"""Read selected hidden states from saved token IDs; no generation or scoring."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoModel, DynamicCache

from exec_exit.replay.replay import read_rows

DEFINITION = '1-based decoder block output before final norm; last observed token'


class TimeLimitReached(TimeoutError):
    pass


def check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeLimitReached('Runtime limit reached; completed features retained')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(ref):
    path = Path(ref['path'])
    if file_sha(path) != ref['sha256']:
        raise ValueError('Changed bound input: ' + str(path))
    return path


def token_ids(values):
    if not values or any(type(x) is not int or x < 0 for x in values):
        raise ValueError('Nonempty actual integer token IDs required')
    return list(values)


def prepare_records(trajectory, records):
    if trajectory.get('status') != 'COMPLETE' or not trajectory['calls'][0].get('native_finished'):
        raise ValueError('Incomplete native trajectory')
    prompt = token_ids(trajectory['prompt_token_ids'])
    thinking = token_ids(trajectory['calls'][0]['token_ids'])
    probes = []
    for row in records:
        if row['case_id'] != trajectory['case_id']:
            raise ValueError('Probe belongs to a different trajectory')
        pos = row['position_generated_tokens']
        control = token_ids(row['control_token_ids'])
        if row['call']['prompt_token_ids'] != prompt + thinking[:pos] + control:
            raise ValueError('Stored probe prefix differs from actual trajectory IDs')
        draft = token_ids(row['call']['token_ids'])
        if row['call']['token_count'] != len(draft):
            raise ValueError('Draft count differs from stored token IDs')
        probes.append(dict(id=row['probe_id'], position=pos, control_ids=control, draft_ids=draft))
    return prompt, thinking, probes


def extract_trajectory(model, prompt_ids, thinking_ids, probes, layers, output,
                       identity, chunk_size=512, disk_floor_bytes=20 * 1024**3, deadline=None):
    prompt, thinking = token_ids(prompt_ids), token_ids(thinking_ids)
    if (not layers or len(set(layers)) != len(layers)
            or any(type(i) is not int or not 1 <= i <= len(model.layers) for i in layers)):
        raise ValueError('Select distinct 1-based decoder layers')
    if chunk_size < 1 or getattr(model.config, 'use_sliding_window', False):
        raise ValueError('Positive chunks and full-attention cache required')
    probes = sorted(probes, key=lambda p: p['position'])
    seen, positions = set(), set()
    for probe in probes:
        pos = probe['position']
        if (not isinstance(probe['id'], str) or not probe['id'] or probe['id'] in seen
                or type(pos) is not int or not 0 < pos <= len(thinking) or pos in positions):
            raise ValueError('Invalid, duplicate or out-of-range candidate')
        seen.add(probe['id']); positions.add(pos)
        token_ids(probe['control_ids']); token_ids(probe['draft_ids'])
        if len(prompt) + pos + len(probe['control_ids']) + len(probe['draft_ids']) > 40960:
            raise ValueError('Recorded sequence exceeds frozen context; never truncate')
    parameter = next(model.parameters())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    binding = dict(run=identity, layers=layers, chunk_size=chunk_size,
                   model_config=model.config.to_dict(), dtype=str(parameter.dtype),
                   feature_definition=DEFINITION,
                   source_sha256=digest([prompt, thinking, probes]))
    paths = {p['id']: output / (hashlib.sha256(p['id'].encode()).hexdigest() + '.pt') for p in probes}
    completed = set()
    with (output / 'features.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for probe in probes:
            path = paths[probe['id']]
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved['identity'] != binding:
                    raise ValueError('Resume input/model/feature identity changed')
                parent = len(prompt) + probe['position']
                expected = dict(candidate=parent - 1,
                                draft_end=parent + len(probe['control_ids']) + len(probe['draft_ids']) - 1)
                if saved['positions'] != expected:
                    raise ValueError('Saved feature position changed')
                for key in ('candidate', 'draft_end'):
                    value = saved[key]
                    if (value.shape != (len(layers), model.config.hidden_size)
                            or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                        raise ValueError('Incomplete saved feature')
                completed.add(probe['id'])
        reused, forwarded = len(completed), 0
        if reused == len(probes):
            return dict(status='COMPLETE', completed=reused, reused=reused, forwarded_tokens=0)
        cache, captured = DynamicCache(config=model.config), {}
        hooks, capture_end = [], False
        for layer in layers:
            def capture(_module, _args, value, layer=layer):
                if not capture_end:
                    return
                hidden = value[0] if isinstance(value, tuple) else value
                captured[layer] = hidden[0, -1].detach().to(device='cpu', dtype=torch.float32).clone()
            hooks.append(model.layers[layer - 1].register_forward_hook(capture))

        def forward(ids):
            nonlocal forwarded, capture_end
            for start in range(0, len(ids), chunk_size):
                check_deadline(deadline)
                piece = ids[start:start + chunk_size]
                capture_end = start + len(piece) == len(ids)
                captured.clear()
                batch = torch.tensor([piece], device=parameter.device, dtype=torch.long)
                model(input_ids=batch, past_key_values=cache, use_cache=True,
                      output_hidden_states=False, return_dict=True)
                forwarded += len(piece)
            if set(captured) != set(layers):
                raise RuntimeError('Selected decoder hooks did not run')
            return torch.stack([captured[layer] for layer in layers])

        status = 'COMPLETE'
        model.eval()
        try:
            with torch.inference_mode():
                forward(prompt)
                previous = 0
                for probe in probes:
                    if probe['id'] in completed:
                        continue
                    check_deadline(deadline)
                    if shutil.disk_usage(output).free < disk_floor_bytes:
                        raise OSError('Less than reserved free disk; completed shards retained')
                    pos = probe['position']
                    candidate = forward(thinking[previous:pos])
                    previous, parent = pos, len(prompt) + pos
                    if cache.get_seq_length() != parent:
                        raise RuntimeError('Native prefix cache length mismatch')
                    try:
                        draft = forward(probe['control_ids'] + probe['draft_ids'])
                    finally:
                        # In 5.17 a negative value removes tokens; a positive value is a legacy target length.
                        cache.crop(-(cache.get_seq_length() - parent))
                    if cache.get_seq_length() != parent:
                        raise RuntimeError('Draft cache was not restored to native prefix')
                    if not torch.isfinite(candidate).all() or not torch.isfinite(draft).all():
                        raise ValueError('Non-finite hidden state; no completed shard written')
                    payload = dict(identity=binding, candidate=candidate, draft_end=draft,
                                   positions=dict(candidate=parent - 1,
                                       draft_end=parent + len(probe['control_ids']) + len(probe['draft_ids']) - 1))
                    path = paths[probe['id']]; tmp = path.with_suffix('.tmp')
                    with tmp.open('wb') as stream:
                        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
                    tmp.replace(path)
                    completed.add(probe['id'])
        except TimeLimitReached:
            status = 'TIME_LIMIT'
        finally:
            for hook in hooks:
                hook.remove()
        return dict(status=status, completed=len(completed), reused=reused, forwarded_tokens=forwarded)


def check_admission(config):
    """加载模型前核对运行收据、源码与输入文件。"""
    receipt = json.loads(Path(config['receipt']).read_text())
    if receipt.get('state') not in {'FROZEN', 'EXTRACTION_ONLY'} or not receipt.get('execution_approved', receipt.get('user_confirmation')):
        raise PermissionError('Frozen or extraction-only receipt required')
    if receipt['state'] == 'EXTRACTION_ONLY' and receipt.get('allowed_operations') != ['extract', 'export_features']:
        raise PermissionError('Extraction-only receipt is limited to extraction and feature export')
    if receipt['source_sha256'].get(str(Path(__file__).resolve())) != file_sha(__file__):
        raise PermissionError('Extractor is not bound by the frozen receipt')
    ref = next((r for r in receipt['configs'] if Path(r['path']).resolve() == Path(config['_path']).resolve()), None)
    if ref is None:
        raise PermissionError('Extraction config is not frozen')
    checked(ref)
    checked(config['model_config'])
    manifest = json.loads(checked(config['model_manifest']).read_text())
    if manifest['revision'] != config['revision']:
        raise ValueError('Model revision differs from recorded weights')
    for item in manifest['files']:
        name = item.get('file', item.get('path'))
        checked(dict(path=str(Path(config['model_path']) / name), sha256=item['sha256']))
    if not config.get('layers') or not config.get('jobs'):
        raise PermissionError('Layer selection or source jobs are not fixed')
    for job in config['jobs']:
        checked(job['native'])
        checked(job['probes'])
    return receipt


def export_features(config_path, output):
    """Move saved vectors to NumPy without opening correctness labels."""
    import numpy as np
    config = json.loads(Path(config_path).read_text())
    config['_path'] = str(Path(config_path).resolve())
    check_admission(config)
    source = Path(config['output'])
    summary = json.loads((source / 'summary.json').read_text())
    if summary.get('status') != 'COMPLETE':
        raise ValueError('Extraction must complete before export')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'feature_binding.json').exists():
        binding = json.loads((output / 'feature_binding.json').read_text())
        for ref in binding['artifacts']:
            checked(ref)
        return binding
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise OSError('Less than 20 GiB reserved disk')
    bank = {f'{kind}_L{layer}': [] for kind in ('candidate', 'draft_end') for layer in config['layers']}
    index = []
    for job in config['jobs']:
        probes = read_rows(checked(job['probes']))
        for probe in probes:
            if probe['case_id'] not in job['case_ids']:
                continue
            folder = source / job['id'] / hashlib.sha256(probe['case_id'].encode()).hexdigest()
            shard = folder / (hashlib.sha256(probe['probe_id'].encode()).hexdigest() + '.pt')
            saved = torch.load(shard, map_location='cpu', weights_only=True)
            if saved['identity']['layers'] != config['layers']:
                raise ValueError('Export layers differ from extraction')
            run = saved['identity']['run']
            if run['native'] != job['native'] or run['probes'] != job['probes']:
                raise ValueError('Export token source changed')
            index.append(dict(case_id=probe['case_id'], sample=str(job['sample']),
                              subject_id=probe['probe_id'], shard_sha256=file_sha(shard)))
            for kind in ('candidate', 'draft_end'):
                for offset, layer in enumerate(config['layers']):
                    bank[f'{kind}_L{layer}'].append(saved[kind][offset].numpy())
    keys = [(r['case_id'], r['sample'], r['subject_id']) for r in index]
    if not index or len(keys) != len(set(keys)):
        raise ValueError('Empty or duplicate feature index')
    target = output / 'features.npz'
    with target.with_suffix('.tmp').open('wb') as stream:
        np.savez(stream, **{key: np.stack(values) for key, values in bank.items()})
    target.with_suffix('.tmp').replace(target)
    index_path = output / 'feature_index.json'
    index_path.write_text(json.dumps(index) + '\n')
    artifacts = [dict(path=str(p.resolve()), sha256=file_sha(p)) for p in (target, index_path)]
    binding = dict(model_path=config['model_path'], revision=config['revision'], layers=config['layers'],
                   extraction_config=dict(path=str(Path(config_path).resolve()), sha256=file_sha(config_path)),
                   extractor_sha256=file_sha(__file__), artifacts=artifacts, labels_read=False)
    (output / 'feature_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    return binding


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--export-config', type=Path)
    parser.add_argument('--export-output', type=Path)
    parser.add_argument('--parent-lock-fd', type=int)
    parser.add_argument('--max-hours', type=float,
                        help='提取运行时长上限（小时，包含模型加载）；默认不限时')
    args = parser.parse_args()
    if args.max_hours is not None and (args.max_hours <= 0 or not math.isfinite(args.max_hours * 3600)):
        parser.error('--max-hours must be finite and positive')
    deadline = None if args.max_hours is None else started + args.max_hours * 3600
    if args.export_config:
        if args.max_hours is not None:
            parser.error('--max-hours applies only to extraction')
        if not args.export_output or args.config:
            parser.error('Export requires --export-output and no --config')
        export_features(args.export_config, args.export_output)
        return
    if not args.config:
        parser.error('--config is required for extraction')
    config = json.loads(args.config.read_text()); config['_path'] = str(args.config.resolve())
    receipt = check_admission(config)
    from exec_exit.gen.host_environment import collect_host_environment
    host = collect_host_environment()
    if 'RTX 5090' not in host['gpu_inventory'] or host['gpu_inventory'].count('GPU-') != 1:
        raise PermissionError('A uniquely identified RTX 5090 is required')
    if receipt['gpu_uuid'] not in host['gpu_inventory']:
        raise PermissionError('GPU differs from frozen extraction binding')
    if not os.environ.get('EXEC_EXIT_GPU_FLOCK_HELD'):
        raise PermissionError('Use the existing GPU queue and lock')
    fd = args.parent_lock_fd
    lock = Path(os.environ['EXECEXIT_GPU_LOCK']).expanduser()
    if fd is None or (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (lock.stat().st_dev, lock.stat().st_ino):
        raise PermissionError('Inherited GPU lock descriptor required')
    if not any('FLOCK' in line and 'WRITE' in line for line in Path(f'/proc/self/fdinfo/{fd}').read_text().splitlines()):
        raise PermissionError('Inherited descriptor does not hold the GPU lock')
    import transformers
    if transformers.__version__ != '5.17.0':
        raise RuntimeError('Cache semantics were checked for transformers 5.17.0 only')
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise OSError('Less than 20 GiB reserved disk')
    model, status = None, 'COMPLETE'
    rows, input_seconds, load_seconds = [], 0.0, 0.0
    try:
        check_deadline(deadline)
        load_started = time.perf_counter()
        model = AutoModel.from_pretrained(config['model_path'], local_files_only=True,
                                         dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        torch.cuda.reset_peak_memory_stats()
        check_deadline(deadline)
        for job in config['jobs']:
            check_deadline(deadline)
            read_started = time.perf_counter()
            native = read_rows(checked(job['native']))
            probes = read_rows(checked(job['probes']))
            input_seconds += time.perf_counter() - read_started
            if not set(job['case_ids']) <= {r['case_id'] for r in native}:
                raise ValueError('Selected pilot trajectory is missing')
            for source in native:
                if source['case_id'] not in job['case_ids']:
                    continue
                check_deadline(deadline)
                selected = [p for p in probes if p['case_id'] == source['case_id']]
                prompt, thinking, drafts = prepare_records(source, selected)
                identity = dict(model=config['model_manifest'], revision=config['revision'],
                                native=job['native'], probes=job['probes'], sample=job['sample'],
                                case_id=source['case_id'], software=host['software_versions'])
                folder = output / job['id'] / hashlib.sha256(source['case_id'].encode()).hexdigest()
                begin = time.perf_counter()
                result = extract_trajectory(model, prompt, thinking, drafts, config['layers'], folder,
                                            identity, chunk_size=config.get('chunk_size', 512), deadline=deadline)
                torch.cuda.synchronize()
                row = dict(job=job['id'], case_id=source['case_id'], **result,
                           wall_seconds=time.perf_counter() - begin)
                rows.append(row)
                with (output / 'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                if result['status'] == 'TIME_LIMIT':
                    raise TimeLimitReached('Runtime limit reached; completed features retained')
    except TimeLimitReached:
        status = 'TIME_LIMIT'
    feature_seconds = sum(r['wall_seconds'] for r in rows)
    forwarded_tokens = sum(r['forwarded_tokens'] for r in rows)
    summary = dict(status=status, max_hours=args.max_hours, trajectories=rows, host=host,
                   wall_seconds=time.perf_counter() - started,
                   model_load_wall_seconds=load_seconds, input_read_wall_seconds=input_seconds,
                   feature_wall_seconds=feature_seconds, forwarded_tokens=forwarded_tokens,
                   forward_tokens_per_second=forwarded_tokens / feature_seconds if feature_seconds else None,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if model is not None else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved() if model is not None else None,
                   model_load_included=True, lm_head_executed=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if status == 'TIME_LIMIT':
        raise SystemExit('Runtime limit reached; completed features retained for resume')


if __name__ == '__main__':
    main()
