"""Fixed, question-grouped diagnostic models; never used by stop policies.

NumPy implementation avoids changing the serving environment. All preprocessing
is fit on training questions; uncertainty clusters the saved OOF predictions by
question and is conditional on these fitted folds (no bootstrap model refit).
"""
from __future__ import annotations
from collections import Counter
import numpy as np


def sigmoid(x):
    x = np.asarray(x, float)
    return np.exp(-np.logaddexp(0., -x))


def question_sample_weights(rows):
    """Each question equal total; observed samples within a question equal."""
    counts = Counter((r['case_id'], r['sample']) for r in rows)
    samples = Counter(case for case, sample in counts)
    return np.array([1. / (counts[(r['case_id'], r['sample'])] * samples[r['case_id']]) for r in rows])


def folds_for_questions(cases, seed=20261014, nfolds=5):
    ids = np.array(sorted(set(cases)), dtype=object)
    np.random.default_rng(seed).shuffle(ids)
    return {str(case): fold for fold, part in enumerate(np.array_split(ids, nfolds)) for case in part}


def fit_logistic(x, y, weights, ridge=.001, max_iter=100, tolerance=1e-9):
    """Weighted average logistic loss + L2 ridge/2, except the intercept."""
    x, y, weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    weights = weights / weights.sum()
    mean = np.sum(x * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((x-mean)**2 * weights[:, None], axis=0))
    scale[scale < 1e-12] = 1.
    z = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    coef = np.zeros(z.shape[1])
    p0 = np.clip(np.sum(weights*y), 1e-8, 1-1e-8)
    coef[0] = np.log(p0/(1-p0))
    penalty = np.ones(len(coef))*ridge
    penalty[0] = 0.
    if len(set(y)) < 2:
        return dict(coef=coef, mean=mean, scale=scale, iterations=0, converged=True,
                    single_training_class=True)
    def objective(b):
        eta = z@b
        return np.sum(weights * (np.logaddexp(0., eta)-y*eta)) + .5*np.sum(penalty*b*b)
    converged = False
    for iteration in range(1, max_iter+1):
        p = sigmoid(z@coef)
        gradient = z.T@(weights*(p-y)) + penalty*coef
        if np.max(np.abs(gradient)) < tolerance:
            converged = True
            break
        hessian = z.T@((weights*p*(1-p))[:, None]*z)+np.diag(penalty)
        direction = np.linalg.solve(hessian+np.eye(len(coef))*1e-12, gradient)
        value, step = objective(coef), 1.
        while step > 2**-30:
            candidate = coef-step*direction
            if objective(candidate) <= value-1e-4*step*np.dot(gradient, direction):
                coef = candidate
                break
            step *= .5
        else:
            raise RuntimeError('Logistic Newton line search did not converge')
    if not converged:
        converged = np.max(np.abs(z.T@(weights*(sigmoid(z@coef)-y))+penalty*coef)) < tolerance
    if not converged:
        raise RuntimeError('Logistic did not converge within fixed 100 iterations')
    return dict(coef=coef, mean=mean, scale=scale, iterations=iteration, converged=converged,
                single_training_class=False)


def predict_logistic(model, x):
    x = np.asarray(x, float)
    return sigmoid(model['coef'][0]+((x-model['mean'])/model['scale'])@model['coef'][1:])


def baseline_features(rows):
    return np.array([[r['difficulty']=='medium', r['difficulty']=='hard',
                      np.log1p(r['position']), np.log1p(r['probe_ordinal'])] for r in rows], float)


def grouped_oof(rows, y, fold_map, extra=None):
    x = baseline_features(rows)
    if extra is not None:
        x = np.column_stack((x, np.asarray(extra, float)))
    fold_ids = np.array([fold_map[r['case_id']] for r in rows])
    y = np.asarray(y, float)
    result = np.full(len(rows), np.nan)
    fitted = []
    for fold in sorted(set(fold_map.values())):
        train, test = fold_ids != fold, fold_ids == fold
        if not test.any():
            continue
        if not train.any():
            raise ValueError('No training questions for fold')
        train_rows = [r for r, keep in zip(rows, train) if keep]
        train_cases, test_cases = {r['case_id'] for r in train_rows}, {r['case_id'] for r, keep in zip(rows, test) if keep}
        assert not train_cases & test_cases
        model = fit_logistic(x[train], y[train], question_sample_weights(train_rows))
        result[test] = predict_logistic(model, x[test])
        fitted.append(dict(fold=int(fold), training_questions=sorted(train_cases), test_questions=sorted(test_cases),
                           training_rows=int(train.sum()), test_rows=int(test.sum()),
                           **{k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in model.items()}))
    if not np.isfinite(result).all():
        raise ValueError('OOF predictions incomplete')
    return result, fitted


def auc_group_matrix(y, scores, weights, group_ids, groups):
    """AUC numerator by positive-question / negative-question, ties half."""
    y, scores, weights, group_ids = map(np.asarray, (y, scores, weights, group_ids))
    order = np.argsort(scores, kind='stable')
    q = np.zeros((groups, groups))
    lower_negative = np.zeros(groups)
    start = 0
    while start < len(order):
        stop = start+1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        block = order[start:stop]
        pos = np.bincount(group_ids[block][y[block]==1], weights=weights[block][y[block]==1], minlength=groups)
        neg = np.bincount(group_ids[block][y[block]==0], weights=weights[block][y[block]==0], minlength=groups)
        q += np.outer(pos, lower_negative + .5*neg)
        lower_negative += neg
        start = stop
    pos_total = np.bincount(group_ids[y==1], weights=weights[y==1], minlength=groups)
    neg_total = np.bincount(group_ids[y==0], weights=weights[y==0], minlength=groups)
    return q, pos_total, neg_total


def percentile_ci(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return dict(lower=float(np.quantile(finite, .025)) if len(finite) else None,
                upper=float(np.quantile(finite, .975)) if len(finite) else None,
                valid_replicates=len(finite), undefined_replicates=len(values)-len(finite))


def metrics_with_cluster_bootstrap(rows, y, scores, *, threshold=.5, seed=20261014, replicates=10000):
    cases = sorted({r['case_id'] for r in rows})
    case_map = {case:i for i,case in enumerate(cases)}
    groups = np.array([case_map[r['case_id']] for r in rows])
    weights = question_sample_weights(rows)
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    q, positive, negative = auc_group_matrix(y, scores, weights, groups, len(cases))
    counts = np.random.default_rng(seed).multinomial(len(cases), np.full(len(cases), 1/len(cases)), size=replicates)
    denom = (counts@positive)*(counts@negative)
    numerator = np.sum((counts@q)*counts, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        auc_samples = numerator/denom
    total_denom = positive.sum()*negative.sum()
    auc = float(q.sum()/total_denom) if total_denom else None
    pred = scores >= threshold
    tp = np.bincount(groups, weights=weights*(pred & (y==1)), minlength=len(cases))
    fp = np.bincount(groups, weights=weights*(pred & (y==0)), minlength=len(cases))
    fn = np.bincount(groups, weights=weights*(~pred & (y==1)), minlength=len(cases))
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_samples = (counts@tp)/(counts@(tp+fp))
        recall_samples = (counts@tp)/(counts@(tp+fn))
    precision = float(tp.sum()/(tp+fp).sum()) if (tp+fp).sum() else None
    recall = float(tp.sum()/(tp+fn).sum()) if (tp+fn).sum() else None
    raw_tp, raw_fp, raw_fn = int(np.sum(pred&(y==1))), int(np.sum(pred&(y==0))), int(np.sum(~pred&(y==1)))
    result = dict(auroc=auc, auroc_ci=percentile_ci(auc_samples), precision=precision,
                  precision_ci=percentile_ci(precision_samples), recall=recall,
                  recall_ci=percentile_ci(recall_samples), threshold=threshold,
                  raw_counts=dict(tp=raw_tp, fp=raw_fp, fn=raw_fn, tn=int(np.sum(~pred&(y==0)))),
                  raw_precision=raw_tp/(raw_tp+raw_fp) if raw_tp+raw_fp else None,
                  raw_recall=raw_tp/(raw_tp+raw_fn) if raw_tp+raw_fn else None,
                  rows=len(rows), questions=len(cases), positives=int(y.sum()), negatives=int((1-y).sum()),
                  weighting='equal questions, equal covered samples within each question',
                  ci_scope='Question bootstrap of fixed OOF predictions; no model refit')
    return result, dict(auroc=auc_samples, precision=precision_samples, recall=recall_samples)


def compare_signal(rows, signal_name, fold_map, *, seed=20261014, replicates=10000):
    matched = [r for r in rows if r['signals'].get(signal_name) is not None and r['outcome'] in ('PASS','FAIL')]
    if not matched:
        return dict(signal_name=signal_name, status='UNKNOWN_NO_EVALUABLE_ROWS', source_rows=len(rows)), [], []
    y = np.array([r['outcome']=='PASS' for r in matched], int)
    signal = np.array([r['signals'][signal_name] for r in matched], float)
    baseline, baseline_models = grouped_oof(matched, y, fold_map)
    incremental, incremental_models = grouped_oof(matched, y, fold_map, extra=signal)
    output = dict(signal_name=signal_name, status='COMPLETE', source_rows=len(rows), matched_rows=len(matched),
                  excluded_missing_rows=len(rows)-len(matched), matched_questions=len({r['case_id'] for r in matched}))
    samples = {}
    for name, scores in [('signal', signal),('baseline',baseline),('baseline_plus_signal',incremental)]:
        metrics, sample = metrics_with_cluster_bootstrap(matched,y,scores,seed=seed,replicates=replicates)
        output['signal_metrics' if name=='signal' else name] = metrics
        samples[name] = sample
    output['paired_differences'] = {}
    for name in ('signal','baseline_plus_signal'):
        output['paired_differences'][name+'_minus_baseline'] = {}
        for metric in ('auroc','precision','recall'):
            a,b = output['signal_metrics' if name=='signal' else name][metric],output['baseline'][metric]
            output['paired_differences'][name+'_minus_baseline'][metric] = dict(
                difference=a-b if a is not None and b is not None else None,
                ci=percentile_ci(samples[name][metric]-samples['baseline'][metric]))
    predictions = [dict(sample=r['sample'], case_id=r['case_id'], subject_id=r['subject_id'],
                        fold=fold_map[r['case_id']], outcome=r['outcome'], signal_name=signal_name,
                        signal=float(s), baseline=float(b), baseline_plus_signal=float(i))
                   for r,s,b,i in zip(matched,signal,baseline,incremental)]
    models = [dict(model='baseline',signal=signal_name,**m) for m in baseline_models]
    models += [dict(model='baseline_plus_signal',signal=signal_name,**m) for m in incremental_models]
    return output,predictions,models
