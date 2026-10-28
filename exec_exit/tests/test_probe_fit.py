"""Synthetic CPU checks for the registered hidden-state probe."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from exec_exit.analysis import grouped_signals as frozen
from exec_exit.analysis import probe_fit


def fixture():
    rows, folds, features = [], {}, []
    for question in range(10):
        case = f'q{question}'
        folds[case] = question % 5
        for sample in range(1 + question % 2):
            for draft in range(2 + sample):
                label = draft % 2
                rows.append(dict(
                    case_id=case, sample=str(sample),
                    subject_id=f'{case}:{sample}:{draft}',
                    outcome='PASS' if label else 'FAIL',
                    public_status='PASS' if draft else 'FAIL',
                    signals={'PUB': int(draft > 0)},
                    collection='hv6_atcoder', difficulty='medium',
                    position=1024 * (draft + 1), probe_ordinal=draft + 1,
                    corpus='virtual_v1_pipeline', deliverable=True))
                features.append([label + .03 * question, sample - .2 * draft])
    return rows, folds, np.asarray(features)


class ProbeFitTest(unittest.TestCase):
    def test_fold_uses_weighted_training_scaler_and_fixed_fit(self):
        x = np.array([[0., 10.], [1., 2.], [3., -2.], [8., 7.]])
        y = np.array([0, 1, 0, 1])
        weights = np.array([.5, .5, 1., 1.])
        test = np.array([[2., 5.], [10000., -10000.]])
        before = [a.copy() for a in (x, y, weights, test)]
        actual, record = probe_fit.fit_fold(x, y, weights, test, .1)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        expected = LogisticRegression(
            C=.1, penalty='l2', solver='lbfgs', max_iter=2000, tol=1e-4,
            fit_intercept=True, class_weight=None)
        expected.fit(scaler.transform(x), y, sample_weight=weights)
        np.testing.assert_allclose(actual, expected.predict_proba(scaler.transform(test))[:, 1])
        short, _ = probe_fit.fit_fold(x, y, weights, test[:1], .1)
        np.testing.assert_allclose(actual[:1], short)
        for value, old in zip((x, y, weights, test), before):
            np.testing.assert_array_equal(value, old)
        self.assertTrue(record)

    def test_grouped_oof_preserves_original_weights_and_question_folds(self):
        rows, folds, features = fixture()
        original = copy.deepcopy((rows, folds))
        with patch.object(probe_fit, 'fit_fold', wraps=probe_fit.fit_fold) as fitted:
            scores, models = probe_fit.probe_oof(rows, features, folds, .1)
        self.assertEqual(len(fitted.call_args_list), 5)
        for fold, call in zip(sorted(set(folds.values())), fitted.call_args_list):
            training = [r for r in rows if folds[r['case_id']] != fold]
            mask = np.array([folds[r['case_id']] != fold for r in rows])
            np.testing.assert_array_equal(call.args[0], features[mask])
            np.testing.assert_array_equal(call.args[2], frozen.question_sample_weights(training))
            np.testing.assert_array_equal(call.args[3], features[~mask])
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(models), 5)
        self.assertEqual((rows, folds), original)

    def test_nonconvergence_is_not_silently_accepted_or_retried(self):
        def incomplete_fit(*args, **kwargs):
            warnings.warn('synthetic failed convergence', ConvergenceWarning)
        x = np.array([[0.], [1.], [2.], [3.]])
        with patch.object(probe_fit.LogisticRegression, 'fit', side_effect=incomplete_fit) as fitted:
            with self.assertRaisesRegex(RuntimeError, '[Cc]onverg'):
                probe_fit.fit_fold(x, np.array([0, 1, 0, 1]), np.ones(4), x[:1], .1)
        self.assertEqual(fitted.call_count, 1)

    def test_dev_selection_uses_auc_then_regularization_then_depth(self):
        grid = [
            dict(status='COMPLETE', layer=9, C=.1, auroc=.8),
            dict(status='COMPLETE', layer=27, C=.01, auroc=.8),
            dict(status='COMPLETE', layer=9, C=.01, auroc=.8),
            dict(status='COMPLETE', layer=18, C=1., auroc=.8),
            dict(status='COMPLETE', layer=36, C=.01, auroc=.79),
        ]
        self.assertEqual(probe_fit.select_best(grid, 36)['layer'], 9)
        grid.append(dict(status='COMPLETE', layer=18, C=.01, auroc=.8))
        chosen = probe_fit.select_best(grid, 36)
        self.assertEqual((chosen['layer'], chosen['C']), (18, .01))
        grid.append(dict(status='COMPLETE', layer=36, C=10., auroc=.81))
        self.assertEqual(probe_fit.select_best(grid, 36)['layer'], 36)
        grid.append(dict(status='FAILED', layer=9, C=10.))
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            probe_fit.select_best(grid, 36)

    def test_pub_lexicographic_auc_matches_pairwise_definition(self):
        pub = np.array([0, 1, 0, 1, 1, 0])
        probe = np.array([.9, .1, .4, .1, .8, .4])
        labels = np.array([1, 0, 0, 1, 1, 0])
        scores = probe_fit.lexicographic_scores(pub, probe)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(scores[1], scores[3])
        self.assertEqual(scores[2], scores[5])
        rows = [dict(case_id=f'q{i // 2}', sample='0') for i in range(len(pub))]
        weights = frozen.question_sample_weights(rows)
        numerator = denominator = 0.
        for i in np.flatnonzero(labels):
            for j in np.flatnonzero(1 - labels):
                a, b = (pub[i], probe[i]), (pub[j], probe[j])
                factor = float(a > b) + .5 * float(a == b)
                numerator += weights[i] * weights[j] * factor
                denominator += weights[i] * weights[j]
        actual, _ = frozen.metrics_with_cluster_bootstrap(rows, labels, scores, replicates=30)
        self.assertAlmostEqual(actual['auroc'], numerator / denominator)

    def test_fitting_uses_receipt_without_protocol_file_and_checks_dev_lock_before_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(stage='heldout', model='qwen', receipt=str(root / 'receipt.json'),
                          rows={'path': str(root / 'labels_must_not_be_read')})
            path = root / 'fit.json'
            path.write_text(json.dumps(config))
            with patch.object(sys, 'argv', ['probe_fit', '--config', str(path)]), \
                    patch.object(probe_fit, 'load_inputs') as loader:
                with self.assertRaises(FileNotFoundError):
                    probe_fit.main()
                loader.assert_not_called()
                receipt_path = root / 'receipt.json'
                receipt_path.write_text(json.dumps(dict(state='EXTRACTION_ONLY', execution_approved=True)))
                with self.assertRaisesRegex(PermissionError, 'frozen probe configuration'):
                    probe_fit.main()
                loader.assert_not_called()
                receipt_path.write_text(json.dumps(dict(state='FROZEN', execution_approved=True)))
                with self.assertRaisesRegex(KeyError, 'freeze_document'):
                    probe_fit.main()
                loader.assert_not_called()
                sources = {str(Path(module.__file__).resolve()): probe_fit.sha(module.__file__)
                           for module in (probe_fit, frozen, probe_fit.private_c2)}
                receipt = dict(state='FROZEN', execution_approved=True,
                               freeze_document={'sha256': 'synthetic-freeze'}, source_sha256=sources)
                receipt_path.write_text(json.dumps(receipt))
                self.assertEqual(probe_fit.check_admission(config, 'select'), (receipt, None))
                with self.assertRaisesRegex(PermissionError, 'Dev lock'):
                    probe_fit.main()
                loader.assert_not_called()
                lock = dict(state='DEV_LOCKED', model='qwen', freeze_sha256='different-freeze',
                            source_sha256=probe_fit.sha(probe_fit.__file__),
                            choices={position: dict(layer=9, C=.1) for position in probe_fit.POSITIONS})
                lock_path = root / 'DEV_LOCK.json'
                lock_path.write_text(json.dumps(lock))
                config['dev_lock'] = dict(path=str(lock_path), sha256=probe_fit.sha(lock_path))
                with self.assertRaisesRegex(PermissionError, 'Dev lock does not match'):
                    probe_fit.check_admission(config, 'heldout')
                lock['freeze_sha256'] = receipt['freeze_document']['sha256']
                lock_path.write_text(json.dumps(lock))
                config['dev_lock']['sha256'] = probe_fit.sha(lock_path)
                self.assertEqual(probe_fit.check_admission(config, 'heldout'), (receipt, lock))
                sources[str(Path(probe_fit.__file__).resolve())] = 'changed-source'
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(PermissionError, 'Unbound fitting/statistical source'):
                    probe_fit.check_admission(config, 'select')

    def test_receipt_field_compatibility_preserves_admission_checks(self):
        config = dict(model='qwen', receipt='synthetic-receipt.json')
        sources = {str(Path(module.__file__).resolve()): probe_fit.sha(module.__file__)
                   for module in (probe_fit, frozen, probe_fit.private_c2)}
        base = dict(state='FROZEN', freeze_document={'sha256': 'synthetic-freeze'},
                    source_sha256=sources)
        cases = [
            ({'execution_approved': True}, True),
            ({'user_confirmation': True}, True),
            ({'execution_approved': True, 'user_confirmation': False}, True),
            ({'execution_approved': False, 'user_confirmation': True}, False),
            ({'execution_approved': None, 'user_confirmation': True}, False),
            ({'execution_approved': False}, False),
            ({'user_confirmation': False}, False),
            ({}, False),
        ]
        for fields, approved in cases:
            receipt = dict(base, **fields)
            with self.subTest(fields=fields), patch.object(Path, 'read_text', return_value=json.dumps(receipt)):
                if approved:
                    self.assertEqual(probe_fit.check_admission(config, 'select'), (receipt, None))
                    with self.assertRaisesRegex(PermissionError, 'Dev lock'):
                        probe_fit.check_admission(config, 'heldout')
                else:
                    with self.assertRaisesRegex(PermissionError, 'frozen probe configuration'):
                        probe_fit.check_admission(config, 'select')
        for changes, message in [
                ({'state': 'EXTRACTION_ONLY'}, 'frozen probe configuration'),
                ({'source_sha256': {}}, 'Unbound fitting/statistical source')]:
            receipt = dict(base, user_confirmation=True, **changes)
            with self.subTest(changes=changes), patch.object(Path, 'read_text', return_value=json.dumps(receipt)):
                with self.assertRaisesRegex(PermissionError, message):
                    probe_fit.check_admission(config, 'select')

    def test_export_binding_loads_current_schema_and_rejects_wrong_identity_before_labels(self):
        rows, folds, features = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def bound(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return dict(path=str(path), sha256=probe_fit.sha(path))
            model = dict(model_path='/synthetic/model', revision='synthetic-revision', layers=probe_fit.LAYERS['qwen'])
            extraction = bound('extract.json', model)
            index = bound('index.json', [{k: r[k] for k in ('case_id', 'sample', 'subject_id')} for r in rows])
            np.savez(root / 'features.npz', **{f'{position}_L{layer}': features
                     for position in probe_fit.POSITIONS for layer in probe_fit.LAYERS['qwen']})
            bank = dict(path=str(root / 'features.npz'), sha256=probe_fit.sha(root / 'features.npz'))
            extractor_sha = 'synthetic-extractor'
            binding = dict(**model, extraction_config=extraction, extractor_sha256=extractor_sha,
                           artifacts=[bank, index], labels_read=False)
            config = dict(model='qwen', model_path=model['model_path'], revision=model['revision'],
                          features=bank, feature_index=index, feature_binding=bound('binding.json', binding),
                          folds=bound('folds.json', folds))
            label_path = root / 'labels.jsonl'
            label_path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            config['rows'] = dict(path=str(label_path), sha256=probe_fit.sha(label_path))
            receipt = dict(configs=[extraction], source_sha256={
                str(probe_fit.ROOT / 'exec_exit/analysis/extract_hidden_states.py'): extractor_sha})
            actual_rows, matrices, actual_folds, coverage = probe_fit.load_inputs(config, receipt)
            self.assertEqual(actual_rows, rows)
            self.assertEqual(actual_folds, folds)
            self.assertEqual(coverage['missing_features'], 0)
            for values in matrices.values():
                np.testing.assert_array_equal(values, features)
            broken = dict(config, revision='wrong-revision')
            with patch.object(probe_fit, 'read_rows') as reader:
                with self.assertRaisesRegex(ValueError, 'model identity'):
                    probe_fit.load_inputs(broken, receipt)
                reader.assert_not_called()

    def test_failed_dev_fit_is_recorded_without_a_lock(self):
        rows, folds, features = fixture()
        matrices = {(position, layer): features for position in probe_fit.POSITIONS
                    for layer in probe_fit.LAYERS['qwen']}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(probe_fit, 'probe_oof', side_effect=RuntimeError('Logistic did not converge')):
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    probe_fit.select_dev(dict(model='qwen'), {}, rows, matrices, folds, output)
            grid = json.loads((output / 'dev_grid.json').read_text())
            self.assertEqual(len(grid), 32)
            self.assertTrue(all(r['status'] == 'FAILED' for r in grid))
            self.assertFalse((output / 'DEV_LOCK.json').exists())

    def test_single_class_fold_stays_unknown(self):
        rows, folds, features = fixture()
        for row in rows:
            if folds[row['case_id']] == 0:
                row['outcome'] = 'PASS'
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_SINGLE_CLASS_FOLD'):
            probe_fit.probe_oof(rows, features, folds, .1)

    def test_dev_lock_then_heldout_uses_frozen_bootstrap_and_subset_choice(self):
        rows, folds, features = fixture()
        matrices = {(position, layer): features for position in probe_fit.POSITIONS
                    for layer in probe_fit.LAYERS['qwen']}
        config = dict(model='qwen', **{key: {'synthetic': True} for key in (
            'rows', 'folds', 'features', 'feature_index', 'feature_binding')})
        receipt = dict(freeze_document={'sha256': 'synthetic-freeze'})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selected = probe_fit.select_dev(config, receipt, rows, matrices, folds, root / 'dev')
            self.assertEqual(selected['status'], 'COMPLETE')
            lock = json.loads((root / 'dev/DEV_LOCK.json').read_text())
            self.assertEqual(lock['fold_assignment'], folds)
            self.assertEqual(lock['source_sha256'], probe_fit.sha(probe_fit.__file__))
            with patch.object(probe_fit, 'probe_oof', wraps=probe_fit.probe_oof) as fitted:
                result = probe_fit.heldout(config, lock, rows, matrices, folds, root / 'heldout')
            self.assertEqual(result['bootstrap']['seed'], 20261014)
            self.assertEqual(result['bootstrap']['replicates'], 10000)
            for call in fitted.call_args_list:
                self.assertIn(call.args[3], [c['C'] for c in lock['choices'].values()])
                self.assertEqual(call.args[2], folds)
            main = next(r for r in result['comparisons'] if r['population'] == 'pooled'
                        and r['position'] == 'draft_end' and r['comparison'] == 'probe_minus_pub')
            self.assertEqual(main['status'], 'COMPLETE')
            prediction = json.loads((root / 'heldout/oof_predictions.json').read_text())
            selected_rows = [r for r in prediction if r['population'] == 'pooled'
                             and r['position'] == 'draft_end' and r['comparison'] == 'probe_minus_pub']
            y = np.array([r['outcome'] == 'PASS' for r in selected_rows], int)
            _, left = frozen.metrics_with_cluster_bootstrap(
                selected_rows, y, [r['probe'] for r in selected_rows], seed=20261014, replicates=10000)
            _, right = frozen.metrics_with_cluster_bootstrap(
                selected_rows, y, [r['reference'] for r in selected_rows], seed=20261014, replicates=10000)
            self.assertEqual(main['ci'], frozen.percentile_ci(left['auroc'] - right['auroc']))


if __name__ == '__main__':
    unittest.main()
