"""Post-hoc ACD measurement from complete saved drafts; never execute programs."""
import argparse
import ast
from collections import Counter, defaultdict
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time
import tokenize

import numpy as np
from exec_exit.analysis.grouped_signals import percentile_ci
from exec_exit.analysis.c1c2_development import failure_type
from exec_exit.replay.replay import read_rows, write_json
from exec_exit.replay.replay_v1 import ast_signature

SEED = 20261014
REPLICATES = 10000
POLICIES = ('EG:k4:m0', 'HYB-B')
CORPUS = 'complete_v0_auxiliary'


def key(row):
    return row['collection'], row['case_id'], str(row['sample'])


def measure(native, probes):
    probes = sorted(probes, key=lambda r: r['position'])
    if len(probes) != native['candidate_count']:
        raise ValueError('Candidate denominator mismatch')
    if len({p['position'] for p in probes}) != len(probes):
        raise ValueError('Duplicate probe position')
    if [p['probe_ordinal'] for p in probes] != list(range(1, len(probes)+1)):
        raise ValueError('Probe ordering mismatch')
    labels = [p['outcome'] for p in probes]
    if not set(labels + [native['native_outcome']]) <= {'PASS', 'FAIL', 'UNKNOWN'}:
        raise ValueError('Unexpected correctness label')
    passing = [p for p in probes if p['outcome'] == 'PASS']
    adjacent = any(a == b == 'PASS' for a, b in zip(labels, labels[1:]))
    failed = native['native_outcome'] == 'FAIL'
    acd = failed and bool(passing)
    stable = failed and adjacent
    if native['native_outcome'] == 'UNKNOWN':
        acd = stable = None
    elif failed and 'UNKNOWN' in labels:
        acd = True if passing else None
        stable = True if adjacent else None
    solitary = acd is True and len(passing) == 1
    isolated = acd is True and len(passing) > 1 and stable is False
    if acd is None or (acd is True and 'UNKNOWN' in labels):
        solitary = isolated = None
    first = passing[0] if passing else None
    end = native['native_thinking_tokens']
    return dict(collection=native['collection'], case_id=native['case_id'], sample=str(native['sample']),
        difficulty=native['difficulty'], native_outcome=native['native_outcome'],
        probe_count=len(probes), correct_probe_count=len(passing), acd=acd, stable_acd=stable,
        solitary_acd=solitary, multiple_isolated_acd=isolated,
        thinking_cap_hit=native['thinking_cap_hit'], native_thinking_tokens=end,
        first_correct_probe=first['subject_id'] if first else None,
        first_correct_position=first['position'] if first else None,
        position_fraction=first['position']/end if first and end else None,
        position_margin=end-first['position'] if first else None,
        later_failures=[dict(subject_id=p['subject_id'], position=p['position'],
                            category=p['failure_category'], statuses=p['failure_statuses'])
                        for p in probes if first and p['position'] > first['position'] and p['outcome'] == 'FAIL'])


def bootstrap_counts(cases):
    # Same sorted-question multinomial draws as frozen C1/C2, retaining all samples.
    n = len(cases)
    return np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=REPLICATES)


def estimates(rows, cases=None, counts=None):
    cases = sorted({r['case_id'] for r in rows}) if cases is None else cases
    counts = bootstrap_counts(cases) if counts is None and cases else counts
    groups = defaultdict(list)
    for row in rows:
        groups[row['case_id']].append(row)
    totals = np.array([len(groups[c]) for c in cases])
    n_pass = np.array([sum(r['native_outcome'] == 'PASS' for r in groups[c]) for c in cases])
    n_fail = np.array([sum(r['native_outcome'] == 'FAIL' for r in groups[c]) for c in cases])
    result = dict(samples=len(rows), questions=len(groups), native_pass=int(n_pass.sum()),
                  native_fail=int(n_fail.sum()), metrics={})
    for field in ('acd', 'stable_acd', 'solitary_acd', 'multiple_isolated_acd'):
        values = np.array([sum(r[field] is True for r in groups[c]) for c in cases])
        unknown = sum(r[field] is None for r in rows)
        valid = bool(rows) and unknown == 0
        metric = dict(count=int(values.sum()), unknown=unknown,
            rate=float(values.sum()/totals.sum()) if valid else None,
            relative_to_native_pass=float(values.sum()/n_pass.sum()) if valid and n_pass.sum() else None)
        if valid:
            with np.errstate(divide='ignore', invalid='ignore'):
                metric['ci95'] = percentile_ci((counts @ values)/(counts @ totals))
                metric['relative_to_native_pass_ci95'] = percentile_ci((counts @ values)/(counts @ n_pass))
        if field in ('acd', 'stable_acd'):
            metric['among_native_fail'] = float(values.sum()/n_fail.sum()) if valid and n_fail.sum() else None
            if valid:
                with np.errstate(divide='ignore', invalid='ignore'):
                    metric['among_native_fail_ci95'] = percentile_ci((counts @ values)/(counts @ n_fail))
        result['metrics'][field] = metric
    return result


def paired_difference(left, right):
    a, b = {key(r): r for r in left}, {key(r): r for r in right}
    if a.keys() != b.keys():
        raise ValueError('Paired models do not share every question/sample')
    cases = sorted({r['case_id'] for r in left})
    counts = bootstrap_counts(cases)
    groups = {c: [k for k in a if k[1] == c] for c in cases}
    denominators = np.array([len(groups[c]) for c in cases])
    result = {}
    for field in ('acd', 'stable_acd'):
        if any(a[k][field] is None or b[k][field] is None for k in a):
            result[field] = dict(status='UNKNOWN_INCOMPLETE_LABELS')
            continue
        differences = np.array([sum(int(a[k][field])-int(b[k][field]) for k in groups[c]) for c in cases])
        ci = percentile_ci((counts @ differences)/(counts @ denominators))
        result[field] = dict(status='COMPLETE', difference=float(differences.sum()/len(a)), ci95=ci)
    return result


def ast_units(code):
    tree = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    return [t.string for t in tokenize.generate_tokens(io.StringIO(tree).readline)
            if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER)]


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def compare_code(first, final):
    code_a, code_b = first['extraction'].get('code_text'), final['extraction'].get('code_text')
    signatures = [ast_signature(c) for c in (code_a, code_b)]
    if None in signatures:
        return dict(status='UNKNOWN_UNPARSEABLE', category='UNPARSEABLE', ast_signatures=signatures)
    a, b = ast_units(code_a), ast_units(code_b)
    distance = edit_distance(a, b) if signatures[0] != signatures[1] else 0
    ratio = distance/max(len(a), len(b), 1)
    category = 'IDENTICAL_AST' if distance == 0 else 'SMALL_AST_EDIT' if ratio <= .1 else 'LARGE_AST_REWRITE' if ratio > .5 else 'MEDIUM_AST_EDIT'
    return dict(status='COMPLETE', category=category, normalized_distance=ratio,
                edit_distance=distance, ast_tokens=[len(a), len(b)], ast_signatures=signatures)


def probe_bin(n):
    for lo, hi, label in [(0, 0, '0'), (1, 1, '1'), (2, 4, '2-4'), (5, 8, '5-8'), (9, 16, '9-16')]:
        if lo <= n <= hi:
            return label
    return '17+'


def quantiles(values):
    values = [v for v in values if v is not None]
    return {str(q): float(np.quantile(values, q)) if values else None for q in (0, .25, .5, .75, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.config.parent
    if (out/'summary.json').exists():
        raise ValueError('Do not overwrite an existing result')
    os.nice(15)
    start, cpu = time.perf_counter(), time.process_time()
    inputs = {}

    def track(path, expected=None):
        path = Path(path)
        if str(path) not in inputs:
            with path.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            inputs[str(path)] = digest
        if expected and inputs[str(path)] != expected:
            raise ValueError('Input SHA256 mismatch: '+str(path))
        return path

    def read(path):
        return json.loads(track(path).read_text())

    def saved_rows(path):
        return read_rows(track(path))

    models = {}
    summaries = {}
    details = []
    for name, spec in config['models'].items():
        root = Path(spec['c1c2_dir'])
        old = read(root/'config.yaml')
        contract = read(track(old['statistics_contract']['path'], old['statistics_contract']['sha256']))
        natives = saved_rows(root/'raw/c1_opportunities.jsonl')
        if len(natives) != 650 or len({key(r) for r in natives}) != 650:
            raise ValueError('Expected all 650 unique native samples')
        expected = {(q['collection'], q['case_id'], str(s)) for q in contract['questions']
                    for s in range(contract['samples_by_collection'][q['collection']])}
        if {key(r) for r in natives} != expected:
            raise ValueError('Frozen question/sample inventory differs')
        by_corpus = defaultdict(lambda: defaultdict(list))
        for row in saved_rows(root/'raw/probe_signals.jsonl'):
            by_corpus[row['corpus']][key(row)].append(row)
        rows = [measure(n, by_corpus[CORPUS][key(n)]) for n in natives]
        virtual = [measure(n, by_corpus['virtual_v1_pipeline'][key(n)]) for n in natives]
        for n, v in zip(natives, virtual):
            if n['correct_probe_count'] != v['correct_probe_count']:
                raise ValueError('C1 virtual correctness cross-check failed')
        for row in rows:
            row['model'] = name
        row_map = {key(r): r for r in rows}
        policy_rows = {}
        for source in contract['sources']:
            for p in saved_rows(source['path']):
                if p['name'] not in ('N', *POLICIES):
                    continue
                k = (source['collection'], p['case_id'], str(source['sample_index']))
                identity = (k, p['name'])
                if identity in policy_rows:
                    raise ValueError('Duplicate policy path')
                policy_rows[identity] = p
        for k, row in row_map.items():
            if policy_rows[k, 'N']['outcome'] != row['native_outcome']:
                raise ValueError('Native/replay label mismatch')
            row['policies'] = {p: {f: policy_rows[k, p][f] for f in ('outcome', 'selected_subject_id', 'total_generated_tokens')} for p in POLICIES}
        identities = read(root/'raw/input_identities.json')
        jobs = []
        for ref in old['job_manifests']:
            jobs += read(track(ref['path'], ref['sha256']))['jobs']
        for job in jobs:
            selected = [r for r in rows if r['collection'] == job['dataset'] and r['sample'] == str(job['sample_index']) and r['acd'] is True]
            if not selected:
                continue
            wanted = {r['first_correct_probe'] for r in selected}
            evaluations = {}
            for kind in ('N_evaluation', 'probe_evaluation'):
                ref = identities[job['job_id']][kind]
                with track(ref['path'], ref['sha256']).open() as stream:
                    for line in stream:
                        e = json.loads(line)
                        if e['subject_id'] in wanted or (kind == 'N_evaluation' and any(r['case_id'] == e['case_id'] for r in selected)):
                            evaluations[e['subject_id']] = e
            for row in selected:
                first = evaluations[row['first_correct_probe']]
                final = evaluations['N:'+row['case_id']]
                if first['final_outcome'] != 'PASS' or final['final_outcome'] != 'FAIL':
                    raise ValueError('ACD code labels differ from signal records')
                row['code_change'] = compare_code(first, final)
                row['native_failure_category'], row['native_failure_statuses'] = failure_type(final)
                details.append(dict(model=name, collection=row['collection'], case_id=row['case_id'], sample=row['sample'],
                    first_probe=row['first_correct_probe'], comparison=row['code_change'],
                    first_code=first['extraction'].get('code_text'), final_code=final['extraction'].get('code_text'),
                    first_code_sha256=first.get('code_sha256'), final_code_sha256=final.get('code_sha256'),
                    evaluation_sources={k: identities[job['job_id']][k] for k in ('N_evaluation', 'probe_evaluation')}))
        populations = {}
        for collection in ('pooled', *contract['samples_by_collection']):
            for difficulty in ('all', 'easy', 'medium', 'hard'):
                subset = [r for r in rows if (collection == 'pooled' or r['collection'] == collection) and (difficulty == 'all' or r['difficulty'] == difficulty)]
                populations[collection+'/'+difficulty] = estimates(subset)
        rescue = {}
        for label, selected in [('ACD', [r for r in rows if r['acd'] is True]), ('non_ACD', [r for r in rows if r['acd'] is False])]:
            rescue[label] = {p: dict(samples=len(selected), rescued=sum(r['native_outcome'] == 'FAIL' and r['policies'][p]['outcome'] == 'PASS' for r in selected),
                harmed=sum(r['native_outcome'] == 'PASS' and r['policies'][p]['outcome'] == 'FAIL' for r in selected),
                unknown=sum(r['policies'][p]['outcome'] == 'UNKNOWN' for r in selected)) for p in POLICIES}
        acd = [r for r in rows if r['acd'] is True]
        mechanism = dict(samples=len(acd), thinking_cap_hit=sum(r['thinking_cap_hit'] for r in acd),
            first_position_fraction=quantiles([r['position_fraction'] for r in acd]),
            position_margin=quantiles([r['position_margin'] for r in acd]),
            uncapped_position_margin=quantiles([r['position_margin'] for r in acd if not r['thinking_cap_hit']]),
            later_failure_types=dict(Counter(f['category'] for r in acd for f in r['later_failures'])),
            samples_without_later_failed_probe=sum(not r['later_failures'] for r in acd),
            code_changes=dict(Counter(r['code_change']['category'] for r in acd)),
            normalized_ast_distance=quantiles([r['code_change'].get('normalized_distance') for r in acd]))
        summaries[name] = dict(populations=populations,
            probe_count_strata={b: estimates([r for r in rows if probe_bin(r['probe_count']) == b]) for b in ('0', '1', '2-4', '5-8', '9-16', '17+')},
            virtual_v1_description=estimates(virtual), rescue_harm=rescue, mechanism=mechanism)
        models[name] = rows
    reference = config['reference_model']
    differences = {name+' minus '+reference: paired_difference(rows, models[reference]) for name, rows in models.items() if name != reference}
    for name, rows in models.items():
        with (out/'raw'/f'{name}_samples.jsonl').open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (out/'raw/code_comparisons.jsonl').open('x') as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    write_json(out/'raw/input_identities.json', inputs)
    write_json(out/'summary.json', dict(status='COMPLETE', classification='Exploratory, post-hoc',
        corpus=CORPUS, seed=SEED, replicates=REPLICATES, models=summaries, paired_differences=differences))
    write_json(out/'resource_ledger.json', dict(wall_seconds=time.perf_counter()-start,
        cpu_seconds=time.process_time()-cpu, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_used=False, programs_executed=0, source_inputs=len(inputs), completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps({m: s['populations']['pooled/all'] for m, s in summaries.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
