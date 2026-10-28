"""Explicit public-only projection for blind H-v6 generation.

Forbidden source values are structurally skipped, never JSON-decoded. No model,
program execution, extraction, or test evaluation is imported by this module.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
from exec_exit.data.heldout_inventory import FIELDS, skip_value, string_end
from exec_exit.gen.host_environment import collect_host_environment
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'research/code_feedback_admission/run/sources/dataset/test6.jsonl'
SOURCE_SHA = 'bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5'
CANDIDATES = ROOT/'runs/hv6_metadata_inventory/raw/hv6_atcoder_candidates.jsonl'
CANDIDATES_SHA = 'dfa523f6abe7a77153fe28bbddec9212c2fcfb7a25decf28a5f7f2e318bfc987'
SPLIT = 'heldout_blind_generation'
ROW_FIELDS = FIELDS | {'case_id','task_id','question_content','task_type','split','source_cohort'}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def spans(raw):
    """Decode top-level keys only; reuse the sealed structural skip functions."""
    i, seen, result = 0, set(), {}
    while i < len(raw) and raw[i] in b' \t\r\n':
        i += 1
    if i >= len(raw) or raw[i] != 123:
        raise ValueError('Expected top-level object')
    i += 1
    while True:
        while i < len(raw) and raw[i] in b' \t\r\n':
            i += 1
        if i >= len(raw):
            raise ValueError('Unclosed object')
        if raw[i] == 125:
            i += 1
            break
        end = string_end(raw, i)
        key = json.loads(raw[i:end])
        if key in seen:
            raise ValueError('Duplicate top-level key')
        seen.add(key)
        i = end
        while raw[i] in b' \t\r\n':
            i += 1
        if raw[i] != 58:
            raise ValueError('Expected colon')
        i += 1
        while raw[i] in b' \t\r\n':
            i += 1
        end = skip_value(raw, i)
        result[key] = (i, end)
        i = end
        while i < len(raw) and raw[i] in b' \t\r\n':
            i += 1
        if i >= len(raw):
            raise ValueError('Unclosed object')
        if raw[i] == 125:
            i += 1
            break
        if raw[i] != 44:
            raise ValueError('Expected comma')
        i += 1
    if raw[i:].strip():
        raise ValueError('Unexpected trailing bytes')
    return result


def decode_allowed(raw, positions, key, allowed, audit):
    if key not in allowed:
        raise ValueError('Field outside explicit whitelist')
    start, end = positions[key]
    audit[key] += 1
    return json.loads(raw[start:end])


def public_projection(raw, candidate_ids, audit):
    positions = spans(raw)
    if not FIELDS <= positions.keys():
        raise ValueError('Missing public metadata')
    metadata = {key: decode_allowed(raw, positions, key, FIELDS, audit) for key in FIELDS}
    if any(type(value) not in (str, int) for value in metadata.values()):
        raise ValueError('Metadata must be scalar')
    task_id = metadata['platform']+':'+metadata['question_id']
    question = None
    if task_id in candidate_ids:
        question = decode_allowed(raw, positions, 'question_content', {'question_content'}, audit)
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Selected public question is absent')
    return metadata, question, positions


def parser_checks():
    audit = Counter()
    metadata = dict(question_id='fixture', platform='atcoder', contest_id='fixture',
                    contest_date='2025-01-01', difficulty='easy')
    question = 'Synthetic escaped braces { } and slash \\ quote " Unicode 题面\n'
    # These invalid escaped strings would raise if any forbidden value were decoded.
    prefix = json.dumps(dict(question_content=question, **metadata), ensure_ascii=False)[:-1].encode()
    raw = prefix+b',"private_test_cases":"\\qDO_NOT_DECODE","public_test_cases":{"a":["\\q",{},[1,2]]},"starter_code":"\\q"}'
    got, text, _ = public_projection(raw, {'atcoder:fixture'}, audit)
    if got != metadata or text != question or set(audit) != FIELDS | {'question_content'}:
        raise ValueError('Selective parser synthetic projection mismatch')
    unselected = Counter()
    _, text, _ = public_projection(raw, set(), unselected)
    if text is not None or set(unselected) != FIELDS:
        raise ValueError('Unselected question text decoded')
    duplicate_rejected = False
    try:
        spans(b'{"question_id":"a","question_id":"b"}')
    except ValueError:
        duplicate_rejected = True
    if not duplicate_rejected:
        raise ValueError('Duplicate key was accepted')
    return dict(status='PASS', checks=['escaped strings and nested opaque values',
        'forbidden invalid JSON string escapes skipped without decode',
        'unselected question_content not decoded', 'duplicate top-level keys rejected'],
        selected_decode_counts=dict(audit), unselected_decode_counts=dict(unselected),
        data='SYNTHETIC_ONLY_NO_REAL_TEST_CONTENT')


def load_blind_cases(path):
    data=json.loads(Path(path).read_text())
    if set(data) != {'split','cases'} or data['split'] != SPLIT:
        raise ValueError('Explicit blind split required; development defaults forbidden')
    rows=data['cases']
    if not isinstance(rows,list) or len(rows)!=63:
        raise ValueError('Exactly 63 questions required')
    seen=set()
    for row in rows:
        if set(row)!=ROW_FIELDS or row['split']!=SPLIT or row['source_cohort']!='hv6_atcoder':
            raise ValueError('Projection contains fields outside strict generation whitelist')
        case=row['case_id']
        if case in seen or row['task_id']!=case or case!='atcoder:'+row['question_id'] or row['platform']!='atcoder':
            raise ValueError('Duplicate or inconsistent canonical identity')
        if row['task_type']!='stdin' or not isinstance(row['question_content'],str) or not row['question_content'].strip():
            raise ValueError('Missing explicit stdin public question')
        seen.add(case)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args(argv); out=a.run_dir.resolve()
    if out.exists(): raise FileExistsError('Fresh projection output required')
    start=time.perf_counter(); usage=resource.getrusage(resource.RUSAGE_SELF)
    tests=parser_checks()
    if sha(SOURCE)!=SOURCE_SHA or sha(CANDIDATES)!=CANDIDATES_SHA:
        raise ValueError('Immutable source or manifest identity changed')
    candidates=[json.loads(line) for line in CANDIDATES.read_text().splitlines()]
    ordered=[row['task_id'] for row in candidates]; ids=set(ordered)
    if len(ids)!=63 or len(ordered)!=63: raise ValueError('Exact 63 manifest required')
    dev_manifest=json.loads((ROOT/'runs/dev121_native_v1/manifest.json').read_text())
    if ids & {row['case_id'] for row in dev_manifest['tasks']}:
        raise ValueError('Heldout manifest overlaps development')
    found={}; audit=Counter(); source_rows=0
    with SOURCE.open('rb') as stream:
        for raw in stream:
            source_rows+=1
            meta,question,_=public_projection(raw,ids,audit)
            cid=meta['platform']+':'+meta['question_id']
            if question is None: continue
            if cid in found: raise ValueError('Selected source identity duplicate')
            found[cid]=dict(meta,case_id=cid,task_id=cid,question_content=question,
                task_type='stdin',split=SPLIT,source_cohort='hv6_atcoder')
    if set(found)!=ids: raise ValueError('Selected population incomplete')
    for row in candidates:
        if any(found[row['task_id']][key]!=row[key] for key in FIELDS):
            raise ValueError('Selected metadata differs from manifest')
    (out/'data').mkdir(parents=True); (out/'raw').mkdir(); (out/'source').mkdir()
    runtime=out/'data/runtime_tasks.json'
    write(runtime,dict(split=SPLIT,cases=[found[cid] for cid in ordered])); load_blind_cases(runtime)
    config=dict(schema='exec_exit.blind_projection.v1',split=SPLIT,source=str(SOURCE),source_sha256=SOURCE_SHA,
        candidates=str(CANDIDATES),candidates_sha256=CANDIDATES_SHA,runtime_tasks_sha256=sha(runtime),
        question_count=63,difficulty_counts=dict(Counter(row['difficulty'] for row in found.values())),
        decoded_source_fields=sorted(FIELDS|{'question_content'}),
        question_decode_scope='Only canonical selected IDs; top-level keys decoded to structurally skip all other values',
        source_bytes_read='Opaque source bytes read for hash and structural skip; forbidden values never decoded',
        public_test_payloads_decoded=False,private_test_payloads_decoded=False,
        question_programs_executed=0,model_used=False,host_environment=collect_host_environment())
    write(out/'config.yaml',config); write(out/'raw/parser_checks.json',tests)
    write(out/'raw/decode_counts.json',dict(source_rows=source_rows,counts=dict(audit)))
    for src in [Path(__file__),ROOT/'exec_exit/data/heldout_inventory.py']:
        (out/'source'/src.name).write_bytes(src.read_bytes())
    (out/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
    end=resource.getrusage(resource.RUSAGE_SELF)
    ledger=dict(gpu_used=False,wall_seconds=time.perf_counter()-start,cpu_user_seconds=end.ru_utime-usage.ru_utime,
        cpu_system_seconds=end.ru_stime-usage.ru_stime,source_files_sha256={str(x):sha(x) for x in [Path(__file__),ROOT/'exec_exit/data/heldout_inventory.py']})
    write(out/'resource_ledger.json',ledger)
    write(out/'summary.json',dict(status='COMPLETE',question_count=63,split=SPLIT,
        runtime_tasks_sha256=sha(runtime),synthetic_parser_status=tests['status'],
        public_test_payloads_decoded=False,private_test_payloads_decoded=False,question_programs_executed=0))
    print(json.dumps(dict(status='COMPLETE',question_count=63,runtime_tasks_sha256=sha(runtime))))
    return 0

if __name__=='__main__': raise SystemExit(main())
