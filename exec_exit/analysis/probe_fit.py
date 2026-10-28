"""Frozen, question-grouped hidden-state logistic probes; no feature extraction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from exec_exit.analysis import grouped_signals as stats, private_c2

ROOT = Path(__file__).resolve().parents[2]
LAYERS = {'qwen': [9, 18, 27, 36], 'ds': [7, 14, 21, 28], 'llama': [8, 16, 24, 32]}
CS = [0.01, 0.1, 1.0, 10.0]
POSITIONS = ['draft_end', 'candidate']
SEED, REPLICATES = 20261014, 10000


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(ref):
    path = Path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError('Changed bound file: ' + str(path))
    return path


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Refusing to replace an existing result: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def check_admission(config, stage):
    receipt = json.loads(Path(config['receipt']).read_text())
    if receipt.get('state') != 'FROZEN' or not receipt.get('execution_approved', receipt.get('user_confirmation')):
        raise PermissionError('Fitting requires a frozen probe configuration')
    #协议哈希用于匹配开发集选参记录，不要求本机保存原Markdown。
    freeze_sha256 = receipt['freeze_document']['sha256']
    sources = receipt['source_sha256']
    for source in (Path(__file__), Path(stats.__file__), Path(private_c2.__file__)):
        if sources.get(str(source.resolve())) != sha(source):
            raise PermissionError('Unbound fitting/statistical source: ' + str(source))
    if sha(stats.__file__) != private_c2.SOURCE_SHA:
        raise ValueError('Original C2 estimator has changed')
    lock = None
    if stage == 'heldout':
        if not config.get('dev_lock'):
            raise PermissionError('A Dev lock is required before reading heldout labels')
        lock = json.loads(checked(config['dev_lock']).read_text())
        if (lock.get('state') != 'DEV_LOCKED' or lock.get('model') != config['model']
                or lock.get('freeze_sha256') != freeze_sha256
                or lock.get('source_sha256') != sha(__file__)):
            raise PermissionError('Dev lock does not match this frozen method and model')
        for position in POSITIONS:
            choice = lock['choices'][position]
            if choice['layer'] not in LAYERS[config['model']] or choice['C'] not in CS:
                raise PermissionError('Dev choice is outside the frozen grid')
    elif stage != 'select':
        raise ValueError('Unknown fitting stage')
    return receipt, lock


def fit_fold(x_train, y_train, weights, x_test, C):
    scaler = StandardScaler()
    train = scaler.fit_transform(np.asarray(x_train, float), sample_weight=weights)
    estimator = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=2000,
                                   tol=1e-4, fit_intercept=True, class_weight=None)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(train, y_train, sample_weight=weights)
    failures = [str(w.message) for w in emitted if issubclass(w.category, ConvergenceWarning)]
    if failures:
        raise RuntimeError('Logistic did not converge: ' + '; '.join(failures))
    scores = estimator.predict_proba(scaler.transform(np.asarray(x_test, float)))[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError('Non-finite logistic predictions')
    return scores, dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                       coef=estimator.coef_[0].tolist(), intercept=float(estimator.intercept_[0]),
                       iterations=int(estimator.n_iter_[0]), converged=True,
                       training_weight_sum=float(np.sum(weights)))


def probe_oof(rows, x, folds, C):
    if set(folds.values()) != set(range(5)) or any(r['case_id'] not in folds for r in rows):
        raise ValueError('Original five-fold map required for all questions')
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
        raise ValueError('Finite, aligned feature matrix required')
    inventory = private_c2.fold_inventory(rows, folds)
    if any(f['status'] == 'UNKNOWN' for f in inventory):
        raise ValueError('UNKNOWN_SINGLE_CLASS_FOLD: ' + json.dumps(inventory))
    y = np.array([r['outcome'] == 'PASS' for r in rows], int)
    fold_ids = np.array([folds[r['case_id']] for r in rows])
    scores, models = np.full(len(rows), np.nan), []
    for fold in range(5):
        train, test = fold_ids != fold, fold_ids == fold
        if not test.any():
            continue
        train_rows = [r for r, use in zip(rows, train) if use]
        weights = stats.question_sample_weights(train_rows)
        prediction, model = fit_fold(x[train], y[train], weights, x[test], C)
        scores[test] = prediction
        models.append(dict(fold=fold, training_questions=sorted({r['case_id'] for r in train_rows}),
                           validation_questions=sorted({r['case_id'] for r, use in zip(rows, test) if use}),
                           **model))
    if not np.isfinite(scores).all():
        raise ValueError('OOF predictions incomplete')
    return scores, models


def auc(rows, scores):
    cases = sorted({r['case_id'] for r in rows})
    group_ids = np.array([cases.index(r['case_id']) for r in rows])
    y = np.array([r['outcome'] == 'PASS' for r in rows], int)
    q, pos, neg = stats.auc_group_matrix(y, scores, stats.question_sample_weights(rows), group_ids, len(cases))
    return float(q.sum() / (pos.sum() * neg.sum())) if pos.sum() and neg.sum() else None


def select_best(grid, total_layers):
    if not grid or any(r.get('status') != 'COMPLETE' for r in grid):
        raise RuntimeError('Dev grid incomplete; no choice or lock may be made')
    if any(r.get('auroc') is None or not np.isfinite(r['auroc']) for r in grid):
        raise ValueError('Undefined Dev AUROC')
    return min(grid, key=lambda r: (-r['auroc'], r['C'], abs(r['layer'] - total_layers / 2), r['layer']))


def lexicographic_scores(pub, scores):
    pub, scores = np.asarray(pub), np.asarray(scores, float)
    if (pub.shape != scores.shape or not np.isin(pub, [0, 1]).all()
            or not np.isfinite(scores).all()):
        raise ValueError('Known binary PUB and finite probe scores required')
    pairs = sorted(set(zip(pub.tolist(), scores.tolist())))
    rank = {pair: i for i, pair in enumerate(pairs)}
    return np.array([rank[p] for p in zip(pub.tolist(), scores.tolist())], float)


def comparison(rows, candidate, reference):
    y = np.array([r['outcome'] == 'PASS' for r in rows], int)
    left, a = stats.metrics_with_cluster_bootstrap(rows, y, candidate, seed=SEED, replicates=REPLICATES)
    right, b = stats.metrics_with_cluster_bootstrap(rows, y, reference, seed=SEED, replicates=REPLICATES)
    interval = stats.percentile_ci(a['auroc'] - b['auroc'])
    difference = left['auroc'] - right['auroc'] if left['auroc'] is not None and right['auroc'] is not None else None
    lower, upper = interval['lower'], interval['upper']
    decision = ('UNKNOWN' if lower is None or upper is None else 'SUPPORT' if lower > 0
                else 'REVERSE' if upper < 0 else 'NOT_SUPPORTED')
    return dict(status='COMPLETE', candidate_auroc=left['auroc'], reference_auroc=right['auroc'],
                difference=difference, ci=interval, decision=decision,
                drafts=len(rows), questions=len({r['case_id'] for r in rows}))


def row_key(row):
    return row['case_id'], str(row['sample']), row['subject_id']


def load_inputs(config, receipt):
    # All I/O of labels is downstream of admission (including Dev lock for heldout).
    binding = json.loads(checked(config['feature_binding']).read_text())
    extraction_ref = binding['extraction_config']
    extraction = json.loads(checked(extraction_ref).read_text())
    if not any(ref['sha256'] == extraction_ref['sha256'] and Path(ref['path']).resolve() == Path(extraction_ref['path']).resolve()
               for ref in receipt['configs']):
        raise PermissionError('Extraction config is not frozen')
    for key in ('model_path', 'revision'):
        if binding[key] != config[key] or binding[key] != extraction[key]:
            raise ValueError('Wrong feature model identity: ' + key)
    if binding['layers'] != LAYERS[config['model']] or binding['layers'] != extraction['layers']:
        raise ValueError('Feature layers differ from the frozen grid')
    extractor = str(ROOT / 'exec_exit/analysis/extract_hidden_states.py')
    if binding['extractor_sha256'] != receipt['source_sha256'].get(extractor):
        raise ValueError('Feature extractor differs from the frozen source')
    for key in ('features', 'feature_index'):
        ref = config[key]
        if not any(item['sha256'] == ref['sha256'] and Path(item['path']).resolve() == Path(ref['path']).resolve()
                   for item in binding['artifacts']):
            raise ValueError('Feature bank differs from extraction export binding: ' + key)
    index = json.loads(checked(config['feature_index']).read_text())
    keys = [row_key(r) for r in index]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate feature identities')
    mapping = {key: i for i, key in enumerate(keys)}
    bank = np.load(checked(config['features']), allow_pickle=False)
    folds = json.loads(checked(config['folds']).read_text())
    rows = read_rows(checked(config['rows']))
    rows = [r for r in rows if r.get('model', r.get('model_id', config['model'])) == config['model']
            and r['corpus'] == 'virtual_v1_pipeline' and r['deliverable']]
    if len({row_key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate label identity')
    if set(folds.values()) != set(range(5)) or any(r['case_id'] not in folds for r in rows):
        raise ValueError('Question missing from original fold assignment')
    eligible = [r for r in rows if r['outcome'] in ('PASS', 'FAIL') and row_key(r) in mapping]
    positions = np.array([mapping[row_key(r)] for r in eligible], int)
    matrices = {}
    for position in POSITIONS:
        for layer in LAYERS[config['model']]:
            values = bank[f'{position}_L{layer}']
            if values.ndim != 2 or len(values) != len(index) or not np.isfinite(values).all():
                raise ValueError('Invalid feature matrix')
            matrices[position, layer] = values[positions]
    coverage = dict(source_drafts=len(rows), matched_drafts=len(eligible),
                    unknown_labels=sum(r['outcome'] == 'UNKNOWN' for r in rows),
                    missing_features=sum(row_key(r) not in mapping for r in rows),
                    matched_questions=len({r['case_id'] for r in eligible}))
    bank.close()
    return eligible, matrices, folds, coverage


def select_dev(config, receipt, rows, matrices, folds, output):
    choices, grid = {}, []
    for position in POSITIONS:
        for layer in LAYERS[config['model']]:
            for C in CS:
                record = dict(position=position, layer=layer, C=C)
                try:
                    scores, _models = probe_oof(rows, matrices[position, layer], folds, C)
                    record.update(status='COMPLETE', auroc=auc(rows, scores))
                except (RuntimeError, ValueError) as error:
                    record.update(status='FAILED', error=repr(error))
                grid.append(record)
    save(output / 'dev_grid.json', grid)
    for position in POSITIONS:
        current = [record for record in grid if record['position'] == position]
        choices[position] = {key: value for key, value in select_best(current, LAYERS[config['model']][-1]).items()
                             if key in ('layer', 'C', 'auroc')}
    lock = dict(state='DEV_LOCKED', model=config['model'], choices=choices,
                freeze_sha256=receipt['freeze_document']['sha256'], source_sha256=sha(__file__),
                locked_at=datetime.now(timezone.utc).isoformat(),
                inputs={k: config[k] for k in ('rows', 'folds', 'features', 'feature_index', 'feature_binding')},
                fold_assignment=folds, seed=SEED)
    save(output / 'DEV_LOCK.json', lock)
    save(output / 'DEV_LOCK_RECEIPT.json', dict(state='DEV_LOCKED', model=config['model'],
         dev_lock=dict(path=str((output / 'DEV_LOCK.json').resolve()), sha256=sha(output / 'DEV_LOCK.json')),
         freeze_sha256=lock['freeze_sha256'], source_sha256=sha(__file__)))
    return dict(status='COMPLETE', choices=choices, dev_lock_sha256=sha(output / 'DEV_LOCK.json'))


def heldout(config, lock, rows, matrices, folds, output):
    results, predictions = [], []
    populations = [('pooled', list(range(len(rows))))] + [
        ('collection:' + name, [i for i, r in enumerate(rows) if r['collection'] == name])
        for name in private_c2.COLLECTIONS]
    for population, indices in populations:
        for position in POSITIONS:
            choice = lock['choices'][position]
            for pub_subset in (False, True):
                matched = [i for i in indices if rows[i]['public_status'] in ('PASS', 'FAIL')
                           and (not pub_subset or rows[i]['public_status'] == 'PASS')]
                selected = [rows[i] for i in matched]
                comparison_name = 'probe_minus_baseline_pub_pass' if pub_subset else 'probe_minus_pub'
                tag = dict(population=population, position=position, comparison=comparison_name,
                           label='PRIV-IND', corpus='virtual_v1_pipeline', scope='complete_deliverable')
                if not selected:
                    results.append(dict(**tag, status='UNKNOWN_NO_MATCHED_ROWS'))
                    continue
                x = matrices[position, choice['layer']][matched]
                inventory = private_c2.fold_inventory(selected, folds)
                try:
                    probe, models = probe_oof(selected, x, folds, choice['C'])
                    y = np.array([r['outcome'] == 'PASS' for r in selected], int)
                    pub = np.array([r['public_status'] == 'PASS' for r in selected], int)
                    if pub_subset:
                        reference, baseline_models = stats.grouped_oof(selected, y, folds)
                    else:
                        reference, baseline_models = pub, []
                    result = comparison(selected, probe, reference)
                except (RuntimeError, ValueError) as error:
                    results.append(dict(**tag, status='UNKNOWN_FIT_FAILED', error=repr(error), folds=inventory))
                    continue
                results.append(dict(**tag, **result, folds=inventory))
                if not pub_subset:
                    combo = comparison(selected, lexicographic_scores(pub, probe), pub)
                    results.append(dict(**dict(tag, comparison='probe_pub_minus_pub'), **combo))
                fit_name = population.replace(':', '_') + '_' + position + ('_pubpass' if pub_subset else '')
                save(output / 'fold_models' / (fit_name + '.json'), dict(probe=models, baseline=baseline_models))
                predictions.extend(dict(case_id=r['case_id'], sample=r['sample'], subject_id=r['subject_id'],
                                        outcome=r['outcome'], fold=folds[r['case_id']],
                                        probe=float(p), reference=float(b), **tag)
                                   for r, p, b in zip(selected, probe, reference))
    save(output / 'oof_predictions.json', predictions)
    return dict(status='COMPLETE_WITH_UNKNOWN' if any(r['status'] != 'COMPLETE' for r in results) else 'COMPLETE',
                model=config['model'], comparisons=results,
                bootstrap=dict(seed=SEED, replicates=REPLICATES, unit='question', fixed_oof=True,
                               stratified=False, model_refit_in_bootstrap=False),
                intervals='Pointwise nominal 95%; no multiplicity correction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    receipt, lock = check_admission(config, config['stage'])
    if config['model'] not in LAYERS:
        raise ValueError('Unsupported frozen model')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'config.yaml', config)
    try:
        rows, matrices, folds, coverage = load_inputs(config, receipt)
        if config['stage'] == 'select':
            summary = select_dev(config, receipt, rows, matrices, folds, output)
        else:
            summary = heldout(config, lock, rows, matrices, folds, output)
        summary['coverage'] = coverage
        save(output / 'summary.json', summary)
    except Exception as error:
        # Completed records are immutable; a recovery uses a new run directory.
        save(output / ('failure_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'),
             dict(status='FAILED', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
