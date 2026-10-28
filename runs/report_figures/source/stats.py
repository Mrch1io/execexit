"""对已有回放记录做按题配对统计，不运行模型或候选程序测试。

Bootstrap 判定是名义判定。CP 上界针对每题至少一次样本受损的概率；
精确覆盖要求各题受损指示独立、具有相同伯努利参数，固定异质分层不保证此条件。
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np

IMPLEMENTED = True
SCHEMA = 'exec_exit.question_statistics.v1'
ROOT = Path(__file__).resolve().parents[2]
CP_ASSUMPTIONS = ('One-sided 97.5% binomial Clopper-Pearson upper bound for the probability '
    'a question has at least one N PASS / HYB-B FAIL sample. Exact coverage requires '
    'independent Bernoulli question indicators with a common p. Fixed heterogeneous '
    'collection/difficulty quotas do not establish this assumption; no unconditional '
    'exact coverage claim for their mixture. ANY-harm is stronger than mean net '
    'pass-rate loss, depends on sample_count, and is not per-sample harm probability.')
CONDITIONAL_CP_ASSUMPTIONS = (CP_ASSUMPTIONS + ' Here the population is additionally '
    'conditioned on N passing at least one predeclared sample. A binomial-exact '
    'interpretation requires independent common-p harm indicators within this '
    'conditional population; observed eligibility does not establish those '
    'assumptions or remove sample-count/collection heterogeneity. This is a '
    'secondary diagnostic, not a new primary test or noninferiority gate.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def binomial_cdf(k,n,p):
    if k>=n or p==0: return 1.0
    if k<0 or p==1: return 0.0
    terms=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p) for i in range(k+1)]
    #先减去最大的对数概率再求和，避免小概率计算下溢
    peak=max(terms)
    return min(1.0,math.exp(peak)*math.fsum(math.exp(t-peak) for t in terms))


def clopper_pearson_upper(harmed,questions,alpha=.025):
    if not isinstance(harmed,int) or not isinstance(questions,int) or not 0<=harmed<=questions or questions<1 or not 0<alpha<1:
        raise ValueError('CP needs integer 0<=harmed<=questions, questions>0, 0<alpha<1')
    if harmed==questions: return 1.0
    if harmed==0: return -math.expm1(math.log(alpha)/questions)
    lo,hi=0.,1.
    for _ in range(100):
        mid=(lo+hi)/2
        if binomial_cdf(harmed,questions,mid)>alpha:
            lo=mid
        else: hi=mid
    return (lo+hi)/2


def minimum_zero_harm_questions(threshold=.03,alpha=.025):
    if not 0<threshold<1 or not 0<alpha<1: raise ValueError('Invalid probability')
    n=math.floor(math.log(alpha)/math.log1p(-threshold))+1
    while clopper_pearson_upper(0,n,alpha)>=threshold: n+=1
    while n>1 and clopper_pearson_upper(0,n-1,alpha)<threshold: n-=1
    return n


def stratified_indices(questions,replicates=10000,seed=20261018):
    #按题配对重采样，保留同题多次生成的相关性，不把重复采样当成新增的独立证据。
    if replicates<1 or not questions: raise ValueError('Positive draw count and population required')
    rng=np.random.Generator(np.random.PCG64(seed))
    indices=np.empty((replicates,len(questions)),dtype=np.int64)
    strata={}
    for i,q in enumerate(questions): strata.setdefault((q['collection'],q['difficulty']),[]).append(i)
    for key in sorted(strata):
        slots=np.asarray(strata[key],dtype=np.int64)
        indices[:,slots]=rng.choice(slots,size=(replicates,len(slots)),replace=True)
    return indices


def _cost(value):
    if value is None or value=='UNKNOWN': return float('nan')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Cost must be nonnegative finite or UNKNOWN')
    return float(value)


def _ci(values):
    return np.quantile(values,[.025,.975],method='linear').tolist() if np.isfinite(values).all() else None


def _float(value):
    return float(value) if np.isfinite(value) else None


def n_success_subset_harm(candidate,native,active,question_ids,*,comparison_complete,scope):
    """按计划样本计算条件受损风险，UNKNOWN 不视为失败。

    任一次 N 通过即可确定资格；排除一题则要求其全部计划 N 样本均已知失败。
    两次或四次采样计划中未安排的位置不算缺失。资格清单保留全部原题，
    不用完整案例筛选缩小总体或补出正式判定。
    """
    eligible=((native==1)&active).any(axis=1)
    all_failed=((native==0)|~active).all(axis=1)
    unknown_eligibility=~eligible&~all_failed
    known_harm=((native==1)&(candidate==0)&active).any(axis=1)
    possible_harm=(((native!=0)&(candidate!=1))&
        (~np.isfinite(native)|~np.isfinite(candidate))&active).any(axis=1)
    harm_unknown=eligible&~known_harm&possible_harm
    ids=lambda mask:[question_ids[i] for i in np.flatnonzero(mask)]
    n,h=int(eligible.sum()),int(known_harm.sum())
    if not comparison_complete:status='NO_FORMAL_DECISION_UNKNOWN'
    elif n==0:status='NOT_APPLICABLE_EMPTY_N_SUCCESS_SUBSET'
    elif scope!='heldout':status='FORMULA_ON_DEVELOPMENT_OR_SYNTHETIC_ONLY'
    else:status='COMPLETE_SECONDARY_DIAGNOSTIC'
    reportable=comparison_complete and n>0
    return dict(role='secondary',used_for_primary_decision=False,status=status,
        eligibility='At least one N PASS among all predeclared samples for the question',
        harm='At least one paired sample with N PASS and HYB-B FAIL',
        original_population_questions=len(question_ids),eligible_questions_known=n,
        eligible_questions_bounds=[n,n+int(unknown_eligibility.sum())],
        excluded_all_n_failed_questions=int(all_failed.sum()),eligibility_unknown_questions=int(unknown_eligibility.sum()),
        eligibility_known_questions=int((eligible|all_failed).sum()),
        eligibility_coverage=(int((eligible|all_failed).sum())/len(question_ids)),
        eligible_harmed_questions_known=h,eligible_harm_unknown_questions=int(harm_unknown.sum()),
        eligible_harm_known_questions=n-int(harm_unknown.sum()),
        unknown_n_sample_labels=int((~np.isfinite(native)&active).sum()),
        unknown_candidate_sample_labels=int((~np.isfinite(candidate)&active).sum()),
        harm_indicator_identified=not bool(unknown_eligibility.any() or harm_unknown.any()),
        parent_comparison_complete=bool(comparison_complete),
        harmed_proportion=h/n if reportable else None,
        upper_probability_97_5=clopper_pearson_upper(h,n) if reportable else None,
        eligible_question_ids=ids(eligible),excluded_all_n_failed_question_ids=ids(all_failed),
        eligibility_unknown_question_ids=ids(unknown_eligibility),harmed_question_ids=ids(known_harm),
        eligible_harm_unknown_question_ids=ids(harm_unknown),
        denominator_explanation='N-all-FAIL questions cannot have N PASS / HYB-B FAIL and widen the overall denominator. Removing them changes the estimand to harm among empirically N-solvable questions; this does not change the primary pooled analysis.',
        missing_rule='Any incomplete primary comparison keeps the conditional proportion and CP bound UNKNOWN; known/unknown eligibility and harm counts remain explicit. No expected missing label is treated as N-all-FAIL.',
        assumptions=CONDITIONAL_CP_ASSUMPTIONS)


def summarize(records,questions,policies,*,sample_count=4,samples_by_collection=None,replicates=10000,seed=20261018,scope='heldout'):
    """题目清单固定分母，缺失记录保留为 UNKNOWN。

    两个主比较合并全部冻结题集，每题等权，共用题集×难度分层的重采样索引。
    单独题集与难度分析为次要结果；开发集可使用实际的两次采样计划。
    """
    if not isinstance(sample_count,int) or sample_count<1 or scope not in {'heldout','development','synthetic'}:
        raise ValueError('Invalid sample_count/scope')
    if scope=='heldout' and sample_count!=4 and samples_by_collection is None:
        raise ValueError('Heldout needs four samples or an explicit frozen per-collection sample plan')
    if len(set(policies))!=len(policies) or not {'HYB-B','N','BF@4096'}<=set(policies):
        raise ValueError('Unique policies including HYB-B/N/BF@4096 required')
    if not questions or len({q['case_id'] for q in questions})!=len(questions):
        raise ValueError('Question IDs unique across collections; overlap is not independent')
    if any(q['difficulty'] not in {'easy','medium','hard'} or not q['collection'] for q in questions):
        raise ValueError('Explicit collection and recognized difficulty required')
    collections=sorted({q['collection'] for q in questions})
    plan = dict.fromkeys(collections, sample_count) if samples_by_collection is None else dict(samples_by_collection)
    if set(plan)!=set(collections) or any(type(v) is not int or v<1 for v in plan.values()):
        raise ValueError('Sample plan must explicitly cover exactly all collections with positive integer counts')
    if scope=='heldout' and any(v not in {2,4} for v in plan.values()):
        raise ValueError('Current heldout contract allows only predeclared two or four samples')
    per_question=np.array([plan[q['collection']] for q in questions],dtype=np.int64)
    maximum=int(per_question.max())
    active=np.arange(maximum)[None,:]<per_question[:,None]
    expected=int(per_question.sum())*len(policies)
    qindex={(q['collection'],q['case_id']):i for i,q in enumerate(questions)}
    pindex={p:i for i,p in enumerate(policies)}
    shape=(len(questions),maximum,len(policies))
    outcomes=np.full(shape,np.nan);tokens=np.full(shape,np.nan);seen=set()
    for r in records:
        key=(r['collection'],r['case_id'])
        if key not in qindex or r['name'] not in pindex or type(r['sample_index']) is not int or r['sample_index'] not in range(plan[r['collection']]):
            raise ValueError('Record outside fixed population/policy/sample')
        i,s,p=qindex[key],r['sample_index'],pindex[r['name']]
        if (i,s,p) in seen: raise ValueError('Duplicate question/sample/policy')
        seen.add((i,s,p))
        if r.get('difficulty',questions[i]['difficulty'])!=questions[i]['difficulty']:
            raise ValueError('Difficulty mismatch')
        if r['outcome'] not in {'PASS','FAIL','UNKNOWN'}:
            raise ValueError('UNSTABLE is a flag, not a replacement outcome')
        outcomes[i,s,p]={'PASS':1.,'FAIL':0.,'UNKNOWN':np.nan}[r['outcome']]
        tokens[i,s,p]=_cost(r.get('tokens'))
    indices=stratified_indices(questions,replicates,seed)
    counts=np.zeros((replicates,len(questions)),dtype=np.float64)
    np.add.at(counts,(np.repeat(np.arange(replicates),len(questions)),indices.ravel()),1)
    question_rows=[]
    for i,q in enumerate(questions):
        for p,name in enumerate(policies):
            ys,ts=outcomes[i,:per_question[i],p],tokens[i,:per_question[i],p]
            question_rows.append(dict(**q,name=name,sample_count=int(per_question[i]),
                outcomes=['UNKNOWN' if not np.isfinite(y) else 'PASS' if y else 'FAIL' for y in ys],
                tokens=[_float(t) for t in ts],pass_fraction=_float(ys.mean()),mean_tokens=_float(ts.mean()),
                pass_fraction_bounds=[float(np.nan_to_num(ys,nan=0).mean()),float(np.nan_to_num(ys,nan=1).mean())]))
    populations=[('pooled',list(range(len(questions))),'both_primary_comparisons')]
    for coll in collections:
        populations.append(('collection:'+coll,[i for i,q in enumerate(questions) if q['collection']==coll],'secondary'))
    for diff in ('easy','medium','hard'):
        ids=[i for i,q in enumerate(questions) if q['difficulty']==diff]
        if ids: populations.append(('difficulty:'+diff,ids,'secondary'))
    for coll in collections:
        for diff in ('easy','medium','hard'):
            ids=[i for i,q in enumerate(questions) if q['collection']==coll and q['difficulty']==diff]
            if ids: populations.append(('collection:'+coll+':difficulty:'+diff,ids,'secondary'))
    def question_mean(values,mask,denominators):
        #未安排的采样位置不计入，计划内缺失仍保留为NaN
        return np.where(mask,values,0).sum(axis=1)/denominators
    qp=question_mean(outcomes,active[:,:,None],per_question[:,None])
    qt=question_mean(tokens,active[:,:,None],per_question[:,None])
    reports=[]
    for name,selected,role in populations:
        idx=np.asarray(selected)
        ys,ts=outcomes[idx],tokens[idx]
        mask=active[idx];denominators=per_question[idx]
        def qmean(values): return question_mean(values,mask,denominators)
        def lower(values): return qmean(np.nan_to_num(values,nan=0))
        def upper(values): return qmean(np.nan_to_num(values,nan=1))
        #次要总体由完整分层组成，可共用合并总体的层内重采样计数
        bootp,boott=counts[:,idx]@qp[idx]/len(idx),counts[:,idx]@qt[idx]/len(idx)
        estimates=[]
        #使用mean而非nanmean，不能从分母中删去UNKNOWN
        for p,policy in enumerate(policies):
            estimates.append(dict(name=policy,questions=len(idx),fixed_samples=int(denominators.sum()),
                unknown_outcomes=int((np.isnan(ys[:,:,p])&mask).sum()),unknown_costs=int((np.isnan(ts[:,:,p])&mask).sum()),
                mean_passed_questions=_float(qp[idx,p].sum()),pass_rate=_float(qp[idx,p].mean()),pass_rate_ci95=_ci(bootp[:,p]),
                pass_rate_bounds=[float(lower(ys[:,:,p]).mean()),float(upper(ys[:,:,p]).mean())],
                mean_tokens=_float(qt[idx,p].mean()),mean_tokens_ci95=_ci(boott[:,p]),known_token_sum=float(np.nansum(ts[:,:,p]))))
        comparisons={}
        for ref in ('N','BF@4096'):
            a,b=pindex['HYB-B'],pindex[ref]
            ya,yb,ta,tb=ys[:,:,a],ys[:,:,b],ts[:,:,a],ts[:,:,b]
            complete=bool(all(np.isfinite(v[mask]).all() for v in (ya,yb,ta,tb)))
            denom=qmean(tb).mean()
            savings=100*(1-qmean(ta).mean()/denom) if np.isfinite(denom) and denom>0 else np.nan
            boot_savings=100*(1-boott[:,a]/boott[:,b]) if (boott[:,b]>0).all() else np.full(replicates,np.nan)
            #之前LCB Pro的样例多了空行，让一些本来正确的N程序读入时报错。
            #修正后，有3次原本看起来是HYB-B“救回”的结果，变成了双方都通过。
            #N的通过率涨得更多，HYB-B相对N的差值反而下降了。
            #这次勘误的具体过程见docs/case_studies.md。
            rescue=(ya==1)&(yb==0);harm=(ya==0)&(yb==1)
            c=dict(candidate='HYB-B',reference=ref,complete=complete,formal_eligible=complete and scope=='heldout',
                pass_difference_pp=_float(100*(qmean(ya).mean()-qmean(yb).mean())),pass_difference_pp_ci95=_ci(100*(bootp[:,a]-bootp[:,b])),
                pass_difference_pp_bounds=[100*float((lower(ya)-upper(yb)).mean()),100*float((upper(ya)-lower(yb)).mean())],
                token_saving_percent=_float(savings),token_saving_percent_ci95=_ci(boot_savings),
                mean_token_difference=_float(qmean(ta).mean()-qmean(tb).mean()),mean_token_difference_ci95=_ci(boott[:,a]-boott[:,b]),
                rescued_question_equivalents=float(qmean(rescue).sum()),harmed_question_equivalents=float(qmean(harm).sum()),
                paired_outcome_unknown=int((~(np.isfinite(ya)&np.isfinite(yb))&mask).sum()),
                rescued_pairs=[dict(case_id=questions[idx[i]]['case_id'],sample_index=int(s)) for i,s in np.argwhere(rescue)],
                harmed_pairs=[dict(case_id=questions[idx[i]]['case_id'],sample_index=int(s)) for i,s in np.argwhere(harm)])
            if ref=='N':
                harmed_q=harm.any(axis=1)
                uncertain=(~harmed_q)&(((yb!=0)&(ya!=1))&(~np.isfinite(ya)|~np.isfinite(yb))&mask).any(axis=1)
                count=int(harmed_q.sum());cp_upper=clopper_pearson_upper(count,len(idx)) if complete else None
                #不能只看平均损失有没有超过3个百分点，95%置信区间的下界也要高于-3才算通过。
                c.update(nominal_noninferiority=c['pass_difference_pp_ci95'][0]>-3 if complete else None,
                    strict_noninferiority=cp_upper<.03 if complete else None,
                    strict=dict(harmed_questions=count,questions=len(idx),possible_additional_harmed_questions=int(uncertain.sum()),
                        upper_probability_97_5=cp_upper,threshold=.03,strict_less_than=True,
                        harmed_question_ids=[questions[idx[i]]['case_id'] for i in np.flatnonzero(harmed_q)],assumptions=CP_ASSUMPTIONS),
                    decision_scope='FORMULA_ON_DEVELOPMENT_OR_SYNTHETIC_ONLY' if scope!='heldout' else 'COMPLETE' if complete else 'NO_FORMAL_DECISION_UNKNOWN')
                c['secondary_n_success_subset']=n_success_subset_harm(ya,yb,mask,
                    [questions[i]['case_id'] for i in idx],comparison_complete=complete,scope=scope)
            else: c['nominal_superiority']=c['pass_difference_pp_ci95'][0]>0 if complete else None
            comparisons[ref]=c
        reports.append(dict(population=name,role=role,questions=len(idx),question_ids=[questions[i]['case_id'] for i in idx],
            strata={f'{c}/{d}':n for (c,d),n in Counter((questions[i]['collection'],questions[i]['difficulty']) for i in idx).items()},
            estimates=estimates,comparisons=comparisons))
    ni,bf=reports[0]['comparisons']['N'],reports[0]['comparisons']['BF@4096']
    sequence=dict(primary_populations='ALL_FROZEN_COLLECTIONS_POOLED',collections=collections,
        status='NO_FORMAL_DECISION_UNKNOWN' if not ni['complete'] or not bf['complete'] else 'FORMULA_ON_DEVELOPMENT_OR_SYNTHETIC_ONLY' if scope!='heldout' else 'COMPLETE',
        nominal_ni_passed=ni['nominal_noninferiority'],strict_ni_passed=ni['strict_noninferiority'],
        nominal_bf_superiority_after_ni=bool(ni['nominal_noninferiority'] and bf['nominal_superiority']) if ni['complete'] and bf['complete'] else None,
        limitation='Nominal percentile-bootstrap levels. Fixed sequence does not repair marginal undercoverage. CP criterion is separate, not the nominal sequence gate.')
    return dict(schema=SCHEMA,scope=scope,sample_count=next(iter(set(plan.values()))) if len(set(plan.values()))==1 else None,
        samples_by_collection=plan,question_weighting='Equal question weight after own predeclared sample mean',questions=len(questions),policies=policies,
        records_expected=expected,records_observed=len(records),missing_records=expected-len(records),
        bootstrap=dict(replicates=replicates,seed=seed,generator='PCG64',strata='collection x difficulty',quantiles=[.025,.975],method='linear',
            index_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()),
        strict_precision=dict(n63_zero_harm_upper=clopper_pearson_upper(0,63),minimum_zero_harm_n=minimum_zero_harm_questions(),assumptions=CP_ASSUMPTIONS),
        sequence=sequence,populations=reports),question_rows,indices


def load_contract_records(contract,base):
    """只读取已有评分文件并核对重复记录，不读取题面或测试载荷。"""
    records={};identities=[];duplicates=0
    for source in contract['sources']:
        path=Path(source['path'])
        if not path.is_absolute(): path=base/path
        identities.append(dict(path=str(path.resolve()),sha256=sha(path),collection=source['collection'],sample_index=source['sample_index']))
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['name'] not in contract['policies']: continue
            normalized=dict(collection=source['collection'],sample_index=source['sample_index'],case_id=r['case_id'],difficulty=r['difficulty'],
                name=r['name'],outcome=r['outcome'],tokens=r.get('total_generated_tokens'),selected_label_unstable=r.get('selected_label_unstable',False),
                control_tokens=r.get('total_control_tokens','UNKNOWN'),same_message_extra_tokens=r.get('received_same_message_extra_tokens','UNKNOWN'),
                selected_subject_id=r.get('selected_subject_id'))
            key=(normalized['collection'],normalized['case_id'],normalized['sample_index'],normalized['name'])
            if key in records:
                if normalized!=records[key]: raise ValueError('Duplicate source controls disagree')
                duplicates+=1
            else: records[key]=normalized
    return list(records.values()),identities,duplicates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start,cpu=time.perf_counter(),time.process_time()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'raw').mkdir();(out/'source').mkdir()
    contract=json.loads(args.contract.read_text());write_json(out/'config.yaml',contract)
    (out/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
    for path in (Path(__file__),Path(__file__).with_name('test_stats.py')):
        if path.exists(): shutil.copy2(path,out/'source'/path.name)
    try:
        from exec_exit.gen.host_environment import collect_host_environment
        write_json(out/'machine_identity.json',collect_host_environment())
        records,sources,duplicates=load_contract_records(contract,args.contract.resolve().parent)
        write_json(out/'source_versions.json',dict(sources=sources,script_sha256=sha(__file__),numpy=np.__version__,python=sys.version,
            test_sha256=sha(Path(__file__).with_name('test_stats.py')),contract_sha256=sha(args.contract)))
        summary,question_rows,indices=summarize(records,contract['questions'],contract['policies'],sample_count=contract.get('sample_count',4),samples_by_collection=contract.get('samples_by_collection'),
            replicates=contract.get('replicates',10000),seed=contract.get('seed',20261018),scope=contract['scope'])
        summary.update(status='COMPLETE',duplicate_controls_verified=duplicates,gpu_used=False,new_model_runs=False,new_program_tests=False)
        for filename,rows in (('normalized_records.jsonl',records),('question_means.jsonl',question_rows)):
            (out/'raw'/filename).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
        (out/'raw/secondary_n_success_subset.jsonl').write_text(''.join(json.dumps(dict(population=p['population'],
            **p['comparisons']['N']['secondary_n_success_subset']),ensure_ascii=False,allow_nan=False)+'\n' for p in summary['populations']))
        np.save(out/'raw/bootstrap_question_indices.npy',indices,allow_pickle=False)
        write_json(out/'summary.json',summary)
        scope_text={'development':'开发实现核查，不是留出结论','synthetic':'合成接口测试，不是真实数据证据','heldout':'冻结留出固定轨迹评测'}[summary['scope']]
        table=['# 按题统计完整结果','',f'固定轨迹协议token；{scope_text}。名义bootstrap和假设下CP分开。集合/难度为次要分析。','']
        for pop in summary['populations']:
            table += [f"## {pop['population']}（{pop['questions']}题）",'', '| 策略 | 平均通过题数 | 平均token | 未知标签/成本 |','|---|---:|---:|---:|']
            for r in pop['estimates']: table.append(f"| {r['name']} | {r['mean_passed_questions']} | {r['mean_tokens']} | {r['unknown_outcomes']}/{r['unknown_costs']} |")
            table.append('')
        (out/'FULL_TABLES.md').write_text('\n'.join(table)+'\n')
        comparisons=summary['populations'][0]['comparisons'];strict=comparisons['N']['strict']
        conditional=comparisons['N']['secondary_n_success_subset']
        page=['# 按题统计结果','',f"COMPLETE：{summary['questions']}题，预定各集合样本数 {summary['samples_by_collection']}，{len(summary['policies'])}配置。{scope_text}。",'',
            '| 比较 | 通过率差pp [名义95%CI] | token节省% [95%CI] |','|---|---:|---:|']
        for ref,c in comparisons.items(): page.append(f"| HYB-B 对 {ref} | {c['pass_difference_pp']} {c['pass_difference_pp_ci95']} | {c['token_saving_percent']} {c['token_saving_percent_ci95']} |")
        page += ['',f"受损题 {strict['harmed_questions']}/{strict['questions']}，单侧97.5% CP上界 {strict['upper_probability_97_5']}；严格阈值 <0.03，公式判定 {comparisons['N']['strict_noninferiority']}。名义NI公式判定 {comparisons['N']['nominal_noninferiority']}。",'',
            'CP精确性依赖独立同参数Bernoulli题级指标；固定异质集合/难度配额不自动满足。受损题定义为至少一个样本N通过而HYB-B失败，比平均净差更保守。','',
            f"必报次要分析：N 至少一次通过的已知合格题 {conditional['eligible_questions_known']}，其中已知受损 {conditional['eligible_harmed_questions_known']}；条件受损比例 {conditional['harmed_proportion']}，单侧97.5% CP上界 {conditional['upper_probability_97_5']}。资格 UNKNOWN {conditional['eligibility_unknown_questions']} 题，合格题内受损状态 UNKNOWN {conditional['eligible_harm_unknown_questions']} 题。",'',
            'N 所有计划样本都失败的题不可能满足受损定义；它们扩大总体分母，因此总体受损比例不能代替本来至少答对一次题组的条件风险。此分析不改变主比较或判据；缺失/UNKNOWN不算全失败，也不通过删题补出条件区间。CP在该条件题组仍需独立、同参数假设。','',
            f"63题零损害上界 {summary['strict_precision']['n63_zero_harm_upper']:.9f}；零损害严格低于3%最少 {summary['strict_precision']['minimum_zero_harm_n']}题。该题数用于精度分析，不改变本次分析的样本范围。",'',
            '[完整配置](FULL_TABLES.md) · [全部区间/schema](summary.json) · [题级原始均值](raw/question_means.jsonl)。两个主比较都在全部冻结集合合并，按集合×难度分层且共用索引；集合单独次要。UNKNOWN不删题，正式判定留空。','',
            'token不含输入、控制、同消息尾部、未见取消在途。控制/尾部另存normalized_records。']
        (out/'ONE_PAGE.md').write_text('\n'.join(page)+'\n')
        print(json.dumps(dict(status='COMPLETE',output=str(out),questions=summary['questions'])))
    except Exception as error:
        write_json(out/'summary.json',dict(status='FAILED',error_type=type(error).__name__,error=str(error)));raise
    finally:
        write_json(out/'resource_ledger.json',dict(wall_seconds=time.perf_counter()-start,self_cpu_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_used=False,new_generated_tokens=0,new_program_tests=0,
            python=sys.executable,nice=os.getpriority(os.PRIO_PROCESS,0),thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            note='CPU existing replay analysis; nvidia-smi identity metadata only; no CUDA/model imports.'))


if __name__=='__main__': main()
