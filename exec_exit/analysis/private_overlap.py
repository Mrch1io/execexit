"""Input-only public/private overlap audit; no models, scoring, or new labels."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

from exec_exit.pipeline.postfreeze_evaluate import decoded_tests

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / 'FROZEN.json'
ERRATUM = ROOT / 'runs/lcb_erratum_scoring_prepare_v3/binding.json'


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}


def checked(ref):
    path = Path(ref['path'])
    if identity(path)['sha256'] != ref['sha256']:
        raise ValueError('Source identity changed: ' + str(path))
    return path


def strict_bytes(raw):
    value = raw.replace(b'\r\n', b'\n')
    return value[:-1] if value.endswith(b'\n') else value


def loose_bytes(raw):
    lines = raw.replace(b'\r\n', b'\n').split(b'\n')
    lines = [line.rstrip(b' \t') for line in lines]
    while lines and not lines[-1]:
        lines.pop()
    return b'\n'.join(lines)


def stream_keys(stream):
    """Binary readline preserves lone CR and bounds memory to one input line."""
    strict, loose = hashlib.sha256(), hashlib.sha256()
    strict_size = loose_size = raw_size = empty_lines = 0
    last = b''
    loose_started = False
    for raw_line in stream:
        raw_size += len(raw_line)
        line = raw_line.replace(b'\r\n', b'\n')
        strict.update(last + line[:-1])
        strict_size += len(last) + len(line) - 1
        last = line[-1:]
        body = line[:-1] if line.endswith(b'\n') else line
        body = body.rstrip(b' \t')
        if not body:
            empty_lines += 1
        else:
            separator = b'\n' * (empty_lines + int(loose_started))
            loose.update(separator + body)
            loose_size += len(separator) + len(body)
            loose_started = True
            empty_lines = 0
    if last != b'\n':
        strict.update(last)
        strict_size += len(last)
    return {'raw_bytes': raw_size,
            'strict': {'bytes': strict_size, 'sha256': strict.hexdigest()},
            'loose': {'bytes': loose_size, 'sha256': loose.hexdigest()}}


def digest_key(value):
    return {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}


def match_inputs(public, private, *, case_id, collection, source, functional=False):
    """private yields (metadata, zero-argument binary opener), in original order."""
    pub = [p.encode('utf-8') for p in public]
    strict_public = [strict_bytes(p) for p in pub]
    loose_public = [loose_bytes(p) for p in pub]
    strict_keys = [digest_key(p) for p in strict_public]
    loose_keys = [digest_key(p) for p in loose_public]
    cases = []
    for index, (meta, opener) in enumerate(private):
        with opener() as stream:
            keys = stream_keys(stream)
        strict_hits = [i for i, key in enumerate(strict_keys) if key == keys['strict']]
        loose_hits = [i for i, key in enumerate(loose_keys) if key == keys['loose']]
        if strict_hits or loose_hits:
            # A digest is only a candidate lookup; verify matched bytes directly.
            with opener() as stream:
                raw = stream.read()
            if strict_hits and any(strict_bytes(raw) != strict_public[i] for i in strict_hits):
                raise ValueError('Strict digest/byte mismatch: ' + case_id)
            if loose_hits and any(loose_bytes(raw) != loose_public[i] for i in loose_hits):
                raise ValueError('Loose digest/byte mismatch: ' + case_id)
        if strict_hits and not loose_hits:
            raise ValueError('Loose normalization must contain strict matches')
        cases.append({'private_index': index, 'old_execution_index': index + (len(pub) if functional else 0),
                      'strict_public_indices': strict_hits, 'loose_public_indices': loose_hits,
                      'remove_strict': bool(strict_hits), 'loose_additional_only': bool(loose_hits and not strict_hits),
                      'input_identity': keys, **meta})
    removed = [r['private_index'] for r in cases if r['remove_strict']]
    retained = [r['private_index'] for r in cases if not r['remove_strict']]
    extra = [r['private_index'] for r in cases if r['loose_additional_only']]
    return {'schema': 'exec_exit.private_input_overlap.v1', 'case_id': case_id, 'collection': collection,
            'public_count': len(pub), 'private_count': len(cases), 'strict_overlap_count': len(removed),
            'loose_additional_count': len(extra), 'private_keep_indices': retained,
            'private_remove_indices': removed, 'loose_additional_indices': extra,
            'remaining_private_count': len(retained), 'empty_private_remainder': not retained,
            'private_retained_count': len(retained),
            'execution_retained_indices': [r['old_execution_index'] for r in cases if not r['remove_strict']],
            'execution_total': len(cases) + (len(pub) if functional else 0),
            'old_execution_mapping': {'public_prefix_count': len(pub) if functional else 0,
                                      'mandatory_prefix_exclusion_indices': list(range(len(pub))) if functional else [],
                                      'retained_old_execution_indices': [r['old_execution_index'] for r in cases if not r['remove_strict']],
                                      'note': 'LC old private suite prepends PUB in one Solution instance; these indices do not establish independent execution equivalence.' if functional else 'Old private suite uses its original private order.'},
            'public_input_identities': [{'strict': a, 'loose': b} for a, b in zip(strict_keys, loose_keys)],
            'source': source, 'private_cases': cases}


def hv6_rows(frozen):
    components = {name: frozen['included_components']['qwen3_8b::' + name]
                  for name in ['hv6_atcoder', 'hv6_leetcode']}
    if components['hv6_atcoder']['dataset_source'] != components['hv6_leetcode']['dataset_source']:
        raise ValueError('H-v6 source mismatch')
    ref = components['hv6_atcoder']['dataset_source']
    path = checked(ref)
    selected = {cid: name for name, component in components.items() for cid in component['case_ids']}
    found = set()
    with path.open('rb') as stream:
        for raw in stream:
            row = json.loads(raw)
            cid = row['platform'] + ':' + row['question_id']
            if cid not in selected:
                continue
            if cid in found:
                raise ValueError('Duplicate source task')
            found.add(cid)
            functional = row['platform'] == 'leetcode'
            task_type = 'functional' if functional else 'stdin'
            public = decoded_tests(row['public_test_cases'], task_type)
            private = decoded_tests(row['private_test_cases'], task_type)
            inputs = (({}, lambda value=t['input'].encode('utf-8'): io.BytesIO(value)) for t in private)
            yield match_inputs([t['input'] for t in public], inputs, case_id=cid, collection=selected[cid],
                               functional=functional,
                               source={'dataset': ref, 'decoder': identity(ROOT/'exec_exit/pipeline/postfreeze_evaluate.py'),
                                       'source_row_sha256': hashlib.sha256(raw).hexdigest(),
                                       'input_bytes': 'UTF-8 encoding of unchanged decoded input strings; functional JSON is not reserialized'})
    if found != set(selected):
        raise ValueError('H-v6 task denominator incomplete')


def lcb_rows(frozen):
    component = frozen['included_components']['qwen3_8b::lcbpro_2025']
    bundle_ref = component['dataset_source']
    bundle = read(checked(bundle_ref))
    binding = read(ERRATUM)
    if checked(binding['original_freeze']) != FREEZE:
        raise ValueError('Wrong original freeze binding')
    public_ref = binding['corrected_public']
    public_data = read(checked(public_ref))
    public = {r['task_id']: r for r in public_data['tasks']}
    if bundle['test_selection'] != 'author_config_subtasks_in_declared_order':
        raise ValueError('Wrong author test-order rule')
    if [p['task_id'] for p in bundle['packages']] != component['case_ids']:
        raise ValueError('LCB frozen denominator/order mismatch')
    for package in bundle['packages']:
        cid = package['task_id']
        archive = checked(package['archive'])
        with zipfile.ZipFile(archive) as z:
            def members():
                for pair in package['pairs']:
                    member = pair['input']
                    yield {'archive_input_member': member}, lambda member=member: z.open(member, 'r')
            yield match_inputs([r['input'] for r in public[cid]['examples']], members(),
                               case_id=cid, collection='lcbpro_2025',
                               source={'scoring_bundle': bundle_ref, 'public': public_ref,
                                       'erratum_binding': identity(ERRATUM), 'archive': package['archive'],
                                       'pairs_sha256': hashlib.sha256(json.dumps(package['pairs'], sort_keys=True).encode()).hexdigest(),
                                       'test_order': bundle['test_selection'], 'only_input_members_read': True})


def run(output):
    output = Path(output)
    rows_path = output / 'overlap_tasks.jsonl'
    summary_path = output / 'overlap_summary.json'
    if rows_path.exists() or summary_path.exists():
        raise FileExistsError('Use new output paths; do not overwrite an overlap audit')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frozen = read(FREEZE)
    if frozen['state'] != 'FROZEN':
        raise ValueError('Original experiment must already be frozen')
    expected = {cid for name, value in frozen['included_components'].items() if name.startswith('qwen3_8b::') for cid in value['case_ids']}
    found, counts, empty = set(), {}, []
    with rows_path.open('x') as stream:
        for rows in [hv6_rows(frozen), lcb_rows(frozen)]:
            for row in rows:
                if row['case_id'] in found:
                    raise ValueError('Duplicate audited task')
                found.add(row['case_id'])
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                count = counts.setdefault(row['collection'], Counter())
                for key in ['public_count', 'private_count', 'strict_overlap_count', 'loose_additional_count', 'remaining_private_count']:
                    count[key] += row[key]
                count['questions'] += 1
                if row['empty_private_remainder']:
                    empty.append(row['case_id'])
                print(json.dumps({'completed': len(found), 'case_id': row['case_id']}), flush=True)
    if found != expected or len(found) != 199:
        raise ValueError('Frozen question denominator incomplete')
    result = {'status': 'COMPLETE', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'INPUT_OVERLAP_ONLY_NO_OUTCOMES', 'questions': len(found),
              'strict_rule': 'CRLF -> LF; remove exactly one final LF; retain every other byte',
              'loose_rule': 'Descriptive only: CRLF -> LF; strip trailing space/tab on each LF-delimited line; ignore all final empty lines',
              'functional_input_rule': 'Original per-argument JSON-line bytes, no JSON reserialization',
              'by_collection': counts, 'empty_private_remainder_questions': empty,
              'empty_private_remainder_count': len(empty),
              'labels_computed': False, 'scoring_or_model_execution': False, 'gpu_used': False,
              'large_decoded_files_written': False, 'wall_seconds': time.perf_counter()-started,
              'source': {'original_freeze': identity(FREEZE), 'module': identity(__file__)},
              'rows': identity(rows_path)}
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
