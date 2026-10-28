"""PRIV-IND C2 adapter; the frozen signal estimators remain unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from exec_exit.analysis import grouped_signals


SEED = 20261014
REPLICATES = 10000
SOURCE_SHA = 'f04ddcfa9ed91bad24cff9180172a3aba3109110f9ee1de439bf17125abb4cfa'
COLLECTIONS = ('hv6_atcoder', 'hv6_leetcode', 'lcbpro_2025')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(rows, folds):
    if not folds or set(folds.values()) != set(range(5)):
        raise ValueError('The original five-fold assignment is required')
    seen = set()
    for row in rows:
        key = (row['corpus'], row['case_id'], row['sample'], row['subject_id'])
        if key in seen:
            raise ValueError(f'Duplicate signal row: {key}')
        seen.add(key)
        if row['case_id'] not in folds:
            raise ValueError(f"Question absent from original folds: {row['case_id']}")
        if row['collection'] not in COLLECTIONS or row['difficulty'] not in ('easy', 'medium', 'hard'):
            raise ValueError('Unknown collection or difficulty')
        if row['corpus'] not in ('virtual_v1_pipeline', 'complete_v0_auxiliary'):
            raise ValueError('Unknown corpus')
        if type(row['deliverable']) is not bool or row['outcome'] not in ('PASS', 'FAIL', 'UNKNOWN'):
            raise ValueError('Invalid delivery status or PRIV-IND outcome')
        if row['position'] < 0 or row['probe_ordinal'] < 1:
            raise ValueError('Invalid position or probe ordinal')
        pub, conf = row['signals']['PUB'], row['signals']['CONF-50']
        expected = {'PASS': 1, 'FAIL': 0}.get(row['public_status'])
        if pub != expected or pub not in (None, 0, 1):
            raise ValueError('PUB signal disagrees with recorded public status')
        if conf is not None and (not math.isfinite(conf) or not 0 <= conf <= 1):
            raise ValueError('Invalid CONF-50 score')


def fold_inventory(rows, folds):
    inventory = []
    for fold in sorted(set(folds.values())):
        train = [r for r in rows if folds[r['case_id']] != fold]
        test = [r for r in rows if folds[r['case_id']] == fold]
        reasons = []
        if test:
            if not train:
                reasons.append('NO_TRAINING_ROWS')
            elif len({r['outcome'] for r in train}) == 1:
                reasons.append('SINGLE_CLASS_TRAINING_FOLD')
            if len({r['outcome'] for r in test}) == 1:
                reasons.append('SINGLE_CLASS_VALIDATION_FOLD')
        inventory.append(dict(
            fold=fold, training_rows=len(train), validation_rows=len(test),
            training_questions=len({r['case_id'] for r in train}),
            validation_questions=len({r['case_id'] for r in test}),
            training_labels=dict(Counter(r['outcome'] for r in train)),
            validation_labels=dict(Counter(r['outcome'] for r in test)),
            status='UNKNOWN' if reasons else 'EVALUABLE' if test else 'NO_MATCHED_VALIDATION_ROWS',
            reasons=reasons))
    return inventory


def compare(rows, signal, folds):
    matched = [r for r in rows if r['outcome'] in ('PASS', 'FAIL') and r['signals'][signal] is not None]
    inventory = fold_inventory(matched, folds)
    coverage = dict(
        source_drafts=len(rows), source_questions=len({r['case_id'] for r in rows}),
        unknown_label_drafts=sum(r['outcome'] == 'UNKNOWN' for r in rows),
        missing_signal_drafts=sum(r['signals'][signal] is None for r in rows),
        matched_drafts=len(matched), matched_questions=len({r['case_id'] for r in matched}),
        excluded_drafts=len(rows)-len(matched),
        unknown_folds=sum(r['status'] == 'UNKNOWN' for r in inventory),
        empty_validation_folds=sum(not r['validation_rows'] for r in inventory))
    if coverage['unknown_folds']:
        # The frozen fitter permits constant training models. This adapter only
        # withholds a decision pending the explicit single-class-fold rule.
        result = dict(status='UNKNOWN_SINGLE_CLASS_FOLD', signal_name=signal,
                      reason='No alternative folds, constant-model interpretation, or refit rule selected')
        predictions, models = [], []
    else:
        try:
            result, predictions, models = grouped_signals.compare_signal(
                rows, signal, folds, seed=SEED, replicates=REPLICATES)
        except (ValueError, RuntimeError) as error:
            result = dict(status='UNKNOWN_UNCOMPUTABLE_WITH_FROZEN_FIT', signal_name=signal, error=repr(error))
            predictions, models = [], []
    return dict(result, coverage=coverage, folds=inventory), predictions, models


def analyse(rows, folds, model):
    validate(rows, folds)
    source = [r for r in rows if r['corpus'] == 'virtual_v1_pipeline' and r['deliverable']]
    results, predictions, models = [], [], []
    populations = [('pooled', source)] + [
        ('collection:'+collection, [r for r in source if r['collection'] == collection])
        for collection in COLLECTIONS]
    for population, selected in populations:
        pub_pass = [r for r in selected if r['public_status'] == 'PASS']
        for name, signal, subset in [('PUB', 'PUB', selected), ('PUB_PASS_CONF50', 'CONF-50', pub_pass)]:
            result, pred, fitted = compare(subset, signal, folds)
            tag = dict(model_id=model, population=population, comparison=name,
                       corpus='virtual_v1_pipeline', scope='complete_deliverable', label='PRIV-IND')
            result.update(tag)
            result['population_coverage'] = dict(
                eligible_drafts=len(selected), eligible_questions=len({r['case_id'] for r in selected}),
                public_pass_drafts=len(pub_pass), public_pass_questions=len({r['case_id'] for r in pub_pass}),
                public_unknown_drafts=sum(r['public_status'] not in ('PASS', 'FAIL') for r in selected))
            results.append(result)
            predictions.extend(dict(r, **tag) for r in pred)
            models.extend(dict(r, **tag) for r in fitted)
    summary = dict(status='COMPLETE_WITH_UNKNOWN' if any(r['status'] != 'COMPLETE' for r in results) else 'COMPLETE',
                   model=model, label='PRIV-IND', comparisons=results,
                   bootstrap=dict(seed=SEED, replicates=REPLICATES, unit='question', stratified=False,
                                  fixed_oof=True, model_refit_in_bootstrap=False),
                   subgroup='PUB PASS only; refit within subgroup and same known CONF-50/label coverage; original folds retained',
                   missing='UNKNOWN labels and missing signals are reported, never imputed; R1 decision remains external')
    return summary, predictions, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    source = Path(grouped_signals.__file__)
    if digest(source) != SOURCE_SHA:
        raise ValueError('Frozen grouped_signals source has changed')
    with args.rows.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    folds = json.loads(args.folds.read_text())
    validate(rows, folds)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'raw').mkdir()
    config = dict(model=args.model, rows=dict(path=str(args.rows.resolve()), sha256=digest(args.rows)),
                  folds=dict(path=str(args.folds.resolve()), sha256=digest(args.folds)),
                  frozen_function=dict(path=str(source.resolve()), sha256=SOURCE_SHA),
                  seed=SEED, replicates=REPLICATES)
    (args.out/'config.yaml').write_text(json.dumps(config, indent=2)+'\n')
    summary, predictions, models = analyse(rows, folds, args.model)
    for name, records in [('oof_predictions', predictions), ('fold_models', models)]:
        with (args.out/'raw'/f'{name}.jsonl').open('w') as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
