"""CPU-only C1/C2 on completed development records, without changing policies."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import tokenize
import traceback
import numpy as np

from exec_exit.analysis.grouped_signals import compare_signal, folds_for_questions, percentile_ci
from exec_exit.execute.machine_identity import current_machine_identity

ROOT = Path(__file__).resolve().parents[2]
DIFFICULTIES = ('all','easy','medium','hard')


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_rows(path, values):
    with Path(path).open('w') as stream:
        for value in values:
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def index(values,key):
    result={r[key]:r for r in values}
    if len(result)!=len(values):raise ValueError('Duplicate identity '+key)
    return result


def confidence_values(probe, evaluation):
    call=probe['call']; ids=call.get('token_ids',[]); ranks=call.get('rank_one_logprobs',[])
    def valid(values):
        return bool(values) and all(isinstance(x,(int,float)) and math.isfinite(x) and x<=0 for x in values)
    result={'CONF-50':None,'CONF-full':None}; reasons={}
    if len(ids)!=len(ranks):
        return result,{'CONF-50':'MISSING_OR_MISMATCHED_RANK_LOGPROBS','CONF-full':'MISSING_OR_MISMATCHED_RANK_LOGPROBS'}
    if len(ids)<50 and not call.get('native_finished'):
        reasons['CONF-50']='SHORT_ABORT_NO_COMPLETED_50_TOKEN_SIGNAL'
    elif valid(ranks[:50]):
        result['CONF-50']=math.exp(math.fsum(ranks[:50])/len(ranks[:50]))
        reasons['CONF-50']='KNOWN_SHORT_NATIVE_COMPLETE' if len(ids)<50 else 'KNOWN_50'
    else:reasons['CONF-50']='MISSING_OR_INVALID_LOGPROBS'
    extraction=evaluation['extraction']; mapping=extraction.get('code_token_mapping',{})
    if not extraction.get('deliverable'):
        reasons['CONF-full']='NO_COMPLETE_DELIVERABLE_CODE'
    elif mapping.get('status')!='KNOWN':reasons['CONF-full']='UNKNOWN_CODE_MAPPING'
    else:
        positions=mapping.get('generated_token_indices',[])
        if len(positions)!=len(set(positions)) or any(not 0<=i<len(ranks) for i in positions):
            raise ValueError('Invalid exact code-token mapping')
        selected=[ranks[i] for i in positions]
        if valid(selected):
            result['CONF-full']=math.exp(math.fsum(selected)/len(selected));reasons['CONF-full']='KNOWN_EXACT_CODE_TOKEN_MAPPING'
        else:reasons['CONF-full']='MISSING_OR_INVALID_CODE_LOGPROBS'
    return result,reasons


def failure_type(evaluation):
    if evaluation['final_outcome']=='PASS':return 'PASS',[]
    if not evaluation['extraction'].get('deliverable'):return 'EXTRACTION_OR_DELIVERY_FAILURE',[]
    statuses=[]
    for key in ('public_result','private_result'):
        result=evaluation.get(key,{})
        statuses.extend(x for x in result.get('case_status_sequence',[]) if x not in ('PASS','NOT_EVALUATED_AFTER_FAILURE'))
    if not statuses:return 'UNKNOWN',statuses
    mapping={'WRONG_ANSWER':'WRONG_ANSWER','CPU_TIMEOUT':'CPU_TIMEOUT','WALL_TIMEOUT':'WALL_TIMEOUT',
             'RUNTIME_ERROR':'RUNTIME_ERROR','OUTPUT_LIMIT':'OUTPUT_LIMIT','MEMORY_LIMIT':'MEMORY_LIMIT'}
    return mapping.get(statuses[0],statuses[0]),statuses


def load_sample(name,spec,task_map):
    for key in ('native_evaluation','probe_evaluation'):
        summary=json.loads((Path(spec[key])/'summary.json').read_text())
        if summary.get('status')!='COMPLETE':raise ValueError('Incomplete scoring '+spec[key])
    native=index(rows(Path(spec['native'])/'raw/trajectories.jsonl'),'case_id')
    probes=rows(Path(spec['probes'])/'raw/probes.jsonl')
    native_evals=index(rows(Path(spec['native_evaluation'])/'raw/evaluations.jsonl'),'case_id')
    evaluations=index(rows(Path(spec['probe_evaluation'])/'raw/evaluations.jsonl'),'subject_id')
    if set(native)!=set(task_map) or set(native_evals)!=set(native):raise ValueError('Task denominator mismatch')
    if set(evaluations)!={p['probe_id'] for p in probes}:raise ValueError('Probe label coverage mismatch')
    grouped=defaultdict(list)
    for p in probes:grouped[p['case_id']].append(p)
    diagnostic=[]; opportunities=[]; false_acceptances=[]
    for case,n in native.items():
        task=task_map[case]; first=next(c for c in n['calls'] if c['stage']==0)
        ids=first['token_ids']; end=next((i for i,t in enumerate(ids) if t in (151668,151643,151645)),len(ids))
        passing=[]
        for ordinal,p in enumerate(sorted(grouped[case],key=lambda p:p['position_generated_tokens']),1):
            ev=evaluations[p['probe_id']]; signals,reasons=confidence_values(p,ev)
            aborted=bool(p.get('protocol_rejected')); deliverable=bool(ev['extraction'].get('deliverable'))
            public=ev.get('public_result',{}).get('status')
            signals.update(PUB=1 if public=='PASS' else 0 if public=='FAIL' else None,
                           PUB_GATE=int(deliverable and public=='PASS'),NOT_ABORTED=int(not aborted))
            if ev['final_outcome']=='PASS' and (aborted or not deliverable):raise ValueError('PASS without complete delivery')
            category,statuses=failure_type(ev)
            item=dict(sample=name,case_id=case,subject_id=p['probe_id'],difficulty=task['difficulty'],
                      position=p['position_generated_tokens'],probe_ordinal=ordinal,generated_tokens=len(p['call']['token_ids']),
                      protocol_rejected=aborted,deliverable=deliverable,public_status=public or 'UNKNOWN',
                      outcome=ev['final_outcome'],signals=signals,signal_availability=reasons,
                      counterfactual_complete_correctness='UNKNOWN_ABORTED' if aborted else ev['final_outcome'],
                      failure_category=category,failure_statuses=statuses,code_sha256=ev.get('code_sha256'))
            diagnostic.append(item)
            if ev['final_outcome']=='PASS':passing.append(item)
            if public=='PASS' and ev['final_outcome']=='FAIL':false_acceptances.append(item)
        earliest=min(passing,key=lambda p:p['position']) if passing else None
        native_pass=native_evals[case]['final_outcome']=='PASS'
        bounds={}
        for include in (True,False):
            options=[(n['generated_tokens'],'N:'+case)] if native_pass else []
            options += [(p['position']+(p['generated_tokens'] if include else 0),p['subject_id']) for p in passing]
            cost,selected=min(options) if options else (n['generated_tokens'],'N:'+case)
            bounds['delivery_charged' if include else 'delivery_free']=dict(tokens=cost,selected=selected,
                outcome='PASS' if options else native_evals[case]['final_outcome'])
        opportunities.append(dict(sample=name,case_id=case,difficulty=task['difficulty'],native_generated_tokens=n['generated_tokens'],
            native_thinking_tokens=end,native_outcome=native_evals[case]['final_outcome'],thinking_cap_hit=n.get('thinking_budget_hit',False),
            candidate_count=len(grouped[case]),correct_probe_count=len(passing),
            first_full_pass_probe_id=earliest['subject_id'] if earliest else None,
            first_full_pass_position=earliest['position'] if earliest else None,
            observed_position_margin=end-earliest['position'] if earliest else None,
            margin_interpretation='Observed cap-relative lower bound; natural thinking end censored' if n.get('thinking_budget_hit') else 'Observed native-end-relative position margin',
            absence='NO_PASS_OBSERVED_AT_TESTED_CANDIDATES' if not passing else None,
            later_probe_failures_after_first_pass=sum(r['position']>earliest['position'] and r['outcome']=='FAIL' for r in diagnostic if r['sample']==name and r['case_id']==case) if earliest else None,
            bounds=bounds))
    return diagnostic,opportunities,false_acceptances


def bounds_summary(opportunities,seed=20261014,reps=10000):
    result={}
    for difficulty in DIFFICULTIES:
        subset=[r for r in opportunities if difficulty=='all' or r['difficulty']==difficulty]
        by_case=defaultdict(list)
        for r in subset:by_case[r['case_id']].append(r)
        cases=sorted(by_case)
        count=len(cases)
        boot=np.random.default_rng(seed).multinomial(count,np.full(count,1/count),size=reps)
        output=dict(questions=count,question_samples=len(subset),observed_first_pass=sum(r['first_full_pass_position'] is not None for r in subset),
                    no_pass_observed=sum(r['first_full_pass_position'] is None for r in subset),thinking_cap_hit=sum(r['thinking_cap_hit'] for r in subset))
        margins=[r['observed_position_margin'] for r in subset if r['observed_position_margin'] is not None]
        output['observed_margin_quantiles']={str(q):float(np.quantile(margins,q)) if margins else None for q in (0,.25,.5,.75,1)}
        output['margin_not_net_token_saving']=True
        native_cost=np.array([np.mean([r['native_generated_tokens'] for r in by_case[c]]) for c in cases])
        native_pass=np.array([np.mean([r['native_outcome']=='PASS' for r in by_case[c]]) for c in cases])
        for bound in ('delivery_charged','delivery_free'):
            costs=np.array([np.mean([r['bounds'][bound]['tokens'] for r in by_case[c]]) for c in cases])
            passes=np.array([np.mean([r['bounds'][bound]['outcome']=='PASS' for r in by_case[c]]) for c in cases])
            saved=1-costs.mean()/native_cost.mean()
            output[bound]=dict(mean_pass_count=float(passes.sum()),mean_pass_fraction=float(passes.mean()),
                mean_generated_tokens=float(costs.mean()),native_mean_generated_tokens=float(native_cost.mean()),
                native_mean_pass_count=float(native_pass.sum()),cost_saving_fraction=float(saved),
                mean_tokens_ci=percentile_ci((boot@costs)/count),
                pass_fraction_ci=percentile_ci((boot@passes)/count),
                cost_saving_fraction_ci=percentile_ci(1-(boot@costs)/(boot@native_cost)))
        result[difficulty]=output
    return result


def plot_c1(opportunities,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    colors={'easy':'#228833','medium':'#4477aa','hard':'#cc6677'}
    for diff in DIFFICULTIES[1:]:
        subset=[r for r in opportunities if r['difficulty']==diff and r['first_full_pass_position'] is not None]
        axes[0].scatter([r['first_full_pass_position'] for r in subset],[r['native_thinking_tokens'] for r in subset],s=20,alpha=.75,c=colors[diff],label=diff)
        vals=sorted(r['observed_position_margin'] for r in subset)
        if vals:axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',c=colors[diff],label=diff)
    axes[0].plot([0,28000],[0,28000],ls='--',c='gray',lw=1)
    axes[0].set(xlabel='First full-test-PASS probe position',ylabel='Observed native thinking tokens')
    axes[1].set(xlabel='Native end minus first PASS position (tokens)',ylabel='Conditional empirical CDF')
    for a in axes:a.legend();a.grid(alpha=.2)
    missing=sum(r['first_full_pass_position'] is None for r in opportunities)
    capped=sum(r['thinking_cap_hit'] for r in opportunities)
    fig.suptitle(f'Observed candidate opportunities; no PASS observed: {missing}/{len(opportunities)}; cap-hit: {capped}\nPosition margin excludes probe/check costs; not net saving')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'c1_opportunity_margin.{ext}',dpi=160)
    plt.close(fig)


def false_rejection_details():
    base=ROOT/'runs/new81_protocol_comparison_input64/raw'
    rejected=[r for r in rows(base/'misrejection.jsonl') if r['correct_complete_v0_rejected']]
    records=index(rows(ROOT/'runs/new81_probes_v0/raw/probes.jsonl'),'probe_id')
    pairs=index(rows(base/'prefix_pairs.jsonl'),'v0_subject_id')
    result=[]
    for r in rejected:
        p=records[r['subject_id']];pair=pairs[r['subject_id']]
        text=p['call']['canonical_text'];line=r['abort']['line_number'];lines=text.splitlines()
        line_text=lines[line-1] if line<=len(lines) else None
        lexical=[]
        try:
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.start[0]<=line<=tok.end[0] and tok.type in (tokenize.COMMENT,tokenize.STRING):
                    lexical.append(dict(type=tokenize.tok_name[tok.type],start=list(tok.start),end=list(tok.end),text=tok.string))
        except (tokenize.TokenError,IndentationError) as exc:
            lexical.append(dict(type='LEXER_ERROR',error=repr(exc)))
        kinds={x['type'] for x in lexical}
        context='PYTHON_COMMENT' if 'COMMENT' in kinds else 'INSIDE_STRING' if 'STRING' in kinds else 'UNKNOWN'
        result.append(dict(**r,full_trigger_line=line_text,nearby_lines=[dict(line_number=i+1,text=lines[i]) for i in range(max(0,line-3),min(len(lines),line+2))],
            lexical_context=context,lexical_tokens=lexical,
            first_divergent_generated_token_1based=pair['first_divergent_generated_token_1based'],
            actual_v1_protocol_rejected=pair['actual_v1_protocol_rejected'],
            actual_v1_decision_cutoff=pair['v1_decision_cutoff'],
            reason='Frozen lexical comment pattern matched '+repr(r['abort']['phrase'])+' after its right boundary became visible; complete v0 draft nevertheless passed tests',
            boundary='Observed full-v0 trigger evidence, not model intent; actual v1 counterfactual equivalence not inferred'))
    assert len(result)==18
    return result


def f(v,digits=3):return 'UNKNOWN' if v is None else f'{v:.{digits}f}'


def report(output,summary,metrics,rejections):
    lines=['# C1/C2：开发集固定记录分析','',
        '统计单位为题目；同题两样本一起 bootstrap。信号结果使用相同覆盖范围的按题分组 OOF logistic 对照；区间条件于已拟合的 OOF 模型，不含重拟合不确定性。','',
        '## C1：免费探测与完美停止（不可部署上限）','',
        '| 数据 | 难度 | 原生通过 | 上限通过 | 原生平均 token | 免费探测、交付收费 | 连交付也免费 | 观察到通过探测 / 题样本 | 位置余量中位数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summary['c1'].items():
        for diff,r in s.items():
            a,b=r['delivery_charged'],r['delivery_free']
            lines.append(f'| {name} | {diff} | {f(a["native_mean_pass_count"],1)} | {f(a["mean_pass_count"],1)} | {f(a["native_mean_generated_tokens"],2)} | {f(a["mean_generated_tokens"],2)} | {f(b["mean_generated_tokens"],2)} | {r["observed_first_pass"]}/{r["question_samples"]} | {f(r["observed_margin_quantiles"]["0.5"],1)} |')
    lines += ['','上限使用全部已测候选，允许完美信息在通过的 N 与探测间选低成本者；未观察到合格探测仍保留 N。第一种仅免失败探测与检查成本，第二种连交付草稿也免费。位置余量未扣探测/检查开销；28k 到顶题的自然终点截尾。','',
        '## C2：连续信号与匹配覆盖的基线','',
        '| 数据/范围 | 信号 | 覆盖行/范围行 | 题数 | 信号 AUROC [95% CI] | 基线 AUROC | 基线+信号 AUROC | 增量 AUROC [95% CI] |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        if r['status']!='COMPLETE':continue
        a,b,c=r['signal_metrics'],r['baseline'],r['baseline_plus_signal'];d=r['paired_differences']['baseline_plus_signal_minus_baseline']['auroc']
        lines.append(f'| {r["dataset"]}/{r["scope"]} | {r["signal_name"]} | {r["matched_rows"]}/{r["source_rows"]} | {r["matched_questions"]} | {f(a["auroc"])} [{f(a["auroc_ci"]["lower"])}, {f(a["auroc_ci"]["upper"])}] | {f(b["auroc"])} | {f(c["auroc"])} | {f(d["difference"])} [{f(d["ci"]["lower"])}, {f(d["ci"]["upper"])}] |')
    lines += ['','## 二值信号精确率与召回率','',
        '| 数据/范围 | 信号 | 正确接受/全部接受（原始计数） | 原始精确率 | 原始召回率 | 按题加权精确率 [95% CI] | 按题加权召回率 [95% CI] | 匹配基线 P/R（阈值0.5） |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        if r['status']!='COMPLETE' or r['signal_name'].startswith('CONF'):continue
        a,b=r['signal_metrics'],r['baseline'];c=a['raw_counts']
        lines.append(f'| {r["dataset"]}/{r["scope"]} | {r["signal_name"]} | {c["tp"]}/{c["tp"]+c["fp"]} | {f(a["raw_precision"])} | {f(a["raw_recall"])} | {f(a["precision"])} [{f(a["precision_ci"]["lower"])}, {f(a["precision_ci"]["upper"])}] | {f(a["recall"])} [{f(a["recall_ci"]["lower"])}, {f(a["recall_ci"]["upper"])}] | {f(b["precision"])}/{f(b["recall"])} |')
    lines += ['','pipeline 的标签是“实际完整交付并全测通过”；中止是交付失败，其假想完整代码正确性为 UNKNOWN。NOT_ABORTED 对该标签召回率为 1 是结构性结果，不能解释成中止没有误拒。完整 PASS 也要求 PUB 通过，因此 PUB 召回为 1 同样不能证明公共测试覆盖充分。PUB 只在实际执行样例的稿上分析；PUB_GATE 额外表示全流程接受门，未执行不冒充样例失败。complete_deliverable 限定完整可交付代码。CONF-full 不存在于中止/截断稿，不能用不同覆盖范围的分数排名判断信号强弱。辅助 v0 只含新增81题，独立列出，不混入 Dev121。','',
        'CONF-50 为前最多50个实际生成 token 的 rank-one 几何平均概率；50 token 前中止且没有完整结束的稿为 UNKNOWN。CONF-full 使用现有精确代码字节与 token 交集，排除注入、围栏及说明。','',
        '模型固定：难度 medium/hard 指示、log1p(已用思考 token)、log1p(题内探测序号)，训练折内标准化；L2=0.001，不调参。题均衡、题内样本均衡；10,000 次按题 percentile bootstrap。完整置信区间和单类/零分母次数见 summary.json。','',
        '## PUB 误接受','', '| 数据 | PUB通过但最终失败 | 分类 |','|---|---:|---|']
    for name,value in summary['false_acceptances'].items():lines.append(f'| {name} | {value["count"]} | {json.dumps(value["categories"],ensure_ascii=False)} |')
    lines += ['','仅列观察到的失败类型，资源 OUTPUT_LIMIT 与答错分开；不执行新的测试。','',
        '## 18 个完整正确 v0 稿的中止触发证据','',
        '| 草稿 | 触发token | 词 | 行号/词法上下文 | 完整触发行 | 实际v1前缀到决策一致 |','|---|---:|---|---|---|---|']
    for r in rejections:
        line=(r['full_trigger_line'] or 'UNKNOWN').replace('|','\\|').replace('`','\\`')
        lines.append(f'| {r["subject_id"]} | {r["abort"]["token_count"]} | {r["abort"]["phrase"]} | {r["abort"]["line_number"]}/{r["lexical_context"]} | `{line}` | {r["identical_through_actual_v1_decision"]} |')
    lines += ['','这些是冻结词法规则命中的可核查原因，不是对模型内在思考的解释。18 个中只有 5 个与实际 v1 已见前缀到决策一致；其余13个不能视为实际在线同前缀反事实。逐项邻近行、首处分歧与原记录路径见 raw/false_rejections18.jsonl。','',
        '![位置余量](c1_opportunity_margin.png)','',
        '资源：CPU 后处理，nice=10、BLAS 单线程；可与 GPU 任务并行，墙钟时间不用于独立吞吐比较。']
    (output/'FULL_REPORT.md').write_text('\n'.join(lines)+'\n')
    one=['# C1/C2 开发记录：一页结果','',f'分析范围：{", ".join(summary["datasets"])}。','',
         '| 数据 | N平均通过 | 免费探测上限通过 | N平均token | 免费探测、交付收费平均token | 有合格探测/题样本 | 余量中位数 |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summary['c1'].items():
        r=s['all'];a=r['delivery_charged']
        one.append(f'| {name} | {f(a["native_mean_pass_count"],1)} | {f(a["mean_pass_count"],1)} | {f(a["native_mean_generated_tokens"],2)} | {f(a["mean_generated_tokens"],2)} | {r["observed_first_pass"]}/{r["question_samples"]} | {f(r["observed_margin_quantiles"]["0.5"],1)} |')
    one += ['','**C1边界。** 上限免除全部探测查询与检查成本，不是可部署结果；另报交付也免费的更乐观版本。位置余量未扣失败草稿和测试费用；未观察到通过的题保留为缺失，28k 截尾单列。','',
            '**C2。** 5 折按题分组，10,000 次按题 bootstrap，训练折内预处理；所有信号与 logistic 基线使用相同覆盖。CONF-full 只覆盖完整可交付代码，未把缺失值当成低置信度。','',
            '| 数据/范围 | 信号 | 覆盖 | 信号AUROC | 基线AUROC | 基线+信号AUROC |','|---|---|---:|---:|---:|---:|']
    for r in metrics:
        if r['status']=='COMPLETE' and r['scope']=='complete_deliverable' and r['signal_name'] in ('CONF-50','CONF-full','PUB'):
            one.append(f'| {r["dataset"]} | {r["signal_name"]} | {r["matched_rows"]} | {f(r["signal_metrics"]["auroc"])} | {f(r["baseline"]["auroc"])} | {f(r["baseline_plus_signal"]["auroc"])} |')
    one += ['','**标签边界。** actual-v1中止稿的完整程序正确性未知；全流程标签仅评价实际交付。该标签下“未中止”召回100%由定义决定，不证明规则没有误拒；完整PASS要求PUB通过，因此PUB召回100%也不是测试覆盖充分性的证据。PUB只在测试实际执行稿上计算；全流程 PUB_GATE 单独列出。辅助 full-v0 的18个正确误拒逐一保留触发行、位置、词法上下文和前缀差异。','',
            '**误接受与误拒。** PUB误接受分答错、CPU/墙钟超时、运行错误和输出/资源上限，不合并掩盖；18个v0正确误拒只有5个实际v1前缀匹配，13个有输出分歧。','',
            '[全部难度、95%区间、二值信号P/R与18项证据](FULL_REPORT.md) · [机器与资源](resource_ledger.json) · [复跑入口](REPRODUCE.md)','',
            '![C1位置余量](c1_opportunity_margin.png)']
    (output/'ONE_PAGE.md').write_text('\n'.join(one)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contract',default='runs/c1c2_contract/config.yaml');parser.add_argument('--samples',nargs='+',default=['sample1']);parser.add_argument('--output',required=True);parser.add_argument('--include-v0-diagnostic',action='store_true')
    args=parser.parse_args();start=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_SELF)
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Require a fresh output; never overwrite completed/failed analysis')
    output.mkdir(parents=True,exist_ok=True);(output/'raw').mkdir();(output/'source').mkdir()
    contract=json.loads(Path(args.contract).read_text());task_map=index(rows('runs/dev_dataset_v1/raw/selection.jsonl'),'task_id')
    paths={}
    for name in args.samples:
        spec=contract['samples'][name]
        for key in ('native','probes','native_evaluation','probe_evaluation'):
            path=Path(spec[key])/('raw/trajectories.jsonl' if key=='native' else 'raw/probes.jsonl' if key=='probes' else 'raw/evaluations.jsonl')
            paths[name+':'+key]={'path':str(path.resolve()),'sha256':sha(path)}
    if args.include_v0_diagnostic:
        for key,path in {
            'v0_probes':'runs/new81_probes_v0/raw/probes.jsonl',
            'v0_evaluation':'runs/dev81_probes_v0_evaluation_input64_v2/raw/evaluations.jsonl',
            'v0_misrejections':'runs/new81_protocol_comparison_input64/raw/misrejection.jsonl',
            'v0_prefix_pairs':'runs/new81_protocol_comparison_input64/raw/prefix_pairs.jsonl',
            'task_manifest':'runs/dev_dataset_v1/raw/selection.jsonl',
        }.items():
            paths[key]={'path':str(Path(path).resolve()),'sha256':sha(path)}
    machine=current_machine_identity()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],capture_output=True,text=True)
    cfg=dict(schema='exec_exit.c1c2_development.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),contract=contract,contract_sha256=sha(args.contract),samples=args.samples,include_v0_diagnostic=args.include_v0_diagnostic,input_identities=paths,machine=machine,gpu_identity=gpu.stdout.strip(),python=sys.executable,invocation=sys.argv,thread_environment={k:os.getenv(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},nice=os.getpriority(os.PRIO_PROCESS,0))
    save(output/'config.yaml',cfg);(output/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    for path in (Path(__file__),Path(__file__).with_name('grouped_signals.py')):shutil.copy2(path,output/'source'/path.name)
    try:
        all_rows=[];all_opportunities=[];false_acceptances=[];datasets={};c1={}
        for name in args.samples:
            data,opportunities,errors=load_sample(name,contract['samples'][name],task_map)
            datasets[name]=data;all_rows+=data;all_opportunities+=opportunities;false_acceptances+=errors
            c1[name]=bounds_summary(opportunities)
        if len(args.samples)>1:
            datasets['pooled_question_clustered']=list(all_rows);c1['pooled_question_clustered']=bounds_summary(all_opportunities)
        if args.include_v0_diagnostic:
            spec=dict(contract['samples']['sample1']);spec['probes']='runs/new81_probes_v0';spec['probe_evaluation']='runs/dev81_probes_v0_evaluation_input64_v2'
            # Auxiliary complete-v0 corpus has a different probe denominator;
            # native inputs remain Dev121 but only selected New81 cases are kept.
            auxiliary_evals=index(rows(Path(spec['probe_evaluation'])/'raw/evaluations.jsonl'),'subject_id')
            auxiliary_probes=rows(Path(spec['probes'])/'raw/probes.jsonl');virtual=index(rows('runs/new81_protocol_comparison_input64/raw/misrejection.jsonl'),'subject_id')
            grouped=defaultdict(list)
            for p in auxiliary_probes:grouped[p['case_id']].append(p)
            aux=[]
            for case,values in grouped.items():
                for ordinal,p in enumerate(sorted(values,key=lambda x:x['position_generated_tokens']),1):
                    ev=auxiliary_evals[p['probe_id']];signals,reasons=confidence_values(p,ev);pub=ev['public_result'].get('status');deliverable=bool(ev['extraction']['deliverable'])
                    signals.update(PUB=1 if pub=='PASS' else 0 if pub=='FAIL' else None,PUB_GATE=int(deliverable and pub=='PASS'),NOT_ABORTED=int(not virtual[p['probe_id']]['virtual_v1_reject']))
                    category,statuses=failure_type(ev)
                    item=dict(sample='sample1_v0_aux',case_id=case,subject_id=p['probe_id'],difficulty=task_map[case]['difficulty'],position=p['position_generated_tokens'],probe_ordinal=ordinal,generated_tokens=len(p['call']['token_ids']),protocol_rejected=False,virtual_rule_rejected=virtual[p['probe_id']]['virtual_v1_reject'],deliverable=deliverable,public_status=pub or 'UNKNOWN',outcome=ev['final_outcome'],signals=signals,signal_availability=reasons,counterfactual_complete_correctness=ev['final_outcome'],failure_category=category,failure_statuses=statuses,code_sha256=ev.get('code_sha256'))
                    aux.append(item)
                    if pub=='PASS' and ev['final_outcome']=='FAIL':false_acceptances.append(item)
            datasets['new81_full_v0_auxiliary']=aux
        folds=folds_for_questions(task_map)
        save(output/'raw/fold_assignment.json',folds)
        save_rows(output/'raw/probe_signals.jsonl',[r for name,data in datasets.items() if name!='pooled_question_clustered' for r in data]);save_rows(output/'raw/c1_opportunities.jsonl',all_opportunities);save_rows(output/'raw/false_acceptances.jsonl',false_acceptances)
        metrics=[];predictions=[];models=[]
        for name,data in datasets.items():
            for scope,subset in [('pipeline',data),('complete_deliverable',[r for r in data if r['deliverable']])]:
                for signal in ('CONF-50','CONF-full','PUB','PUB_GATE','NOT_ABORTED'):
                    result,pred,model=compare_signal(subset,signal,folds)
                    result['dataset']=name;result['scope']=scope
                    if 'signal' in result:raise AssertionError('Unexpected signal')
                    # compare_signal uses signal for both name and metric? Keep metrics explicit.
                    metrics.append(result)
                    predictions.extend(dict(dataset=name,scope=scope,**r) for r in pred)
                    models.extend(dict(dataset=name,scope=scope,**r) for r in model)
                    print(json.dumps({'event':'SIGNAL_COMPLETE','dataset':name,'scope':scope,'signal':signal,'matched_rows':result.get('matched_rows')},ensure_ascii=False),flush=True)
        rejections=false_rejection_details();save_rows(output/'raw/false_rejections18.jsonl',rejections)
        save_rows(output/'raw/signal_metrics.jsonl',metrics);save_rows(output/'raw/oof_predictions.jsonl',predictions);save_rows(output/'raw/fold_models.jsonl',models)
        errors={name:dict(count=sum(r['sample']==name for r in false_acceptances),categories=dict(Counter(r['failure_category'] for r in false_acceptances if r['sample']==name))) for name in sorted({r['sample'] for r in false_acceptances})}
        summary=dict(status='COMPLETE',datasets=list(datasets),c1=c1,c2=metrics,false_acceptances=errors,signal_coverage={name:{key:dict(Counter(r['signal_availability'][key] for r in data)) for key in ('CONF-50','CONF-full')} for name,data in datasets.items()},false_rejections18=dict(count=len(rejections),actual_prefix_matched=sum(r['identical_through_actual_v1_decision'] for r in rejections),lexical_context=dict(Counter(r['lexical_context'] for r in rejections))),no_heldout_read=True,no_new_model_or_program_execution=True)
        plot_c1(all_opportunities,output);report(output,summary,metrics,rejections)
        save(output/'summary.json',summary)
        (output/'REPRODUCE.md').write_text('# CPU reproduction\n\nUse a new output directory.\n\n```bash\nOPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python -B -m exec_exit.analysis.c1c2_development --contract '+args.contract+' --samples '+' '.join(args.samples)+' --output runs/FRESH_C1C2'+(' --include-v0-diagnostic' if args.include_v0_diagnostic else '')+'\n```\n\nThe NumPy logistic objective, grouping, bootstrap and signal mapping are archived in source/. No policy thresholds are fitted.\n')
    except Exception as exc:
        (output/'raw/error.txt').write_text(traceback.format_exc());save(output/'summary.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        save(output/'resource_ledger.json',dict(cpu_only=True,wall_seconds=time.monotonic()-start,self_cpu_user_seconds=usage.ru_utime-cpu.ru_utime,self_cpu_system_seconds=usage.ru_stime-cpu.ru_stime,max_rss_kib=usage.ru_maxrss,machine=machine,gpu_identity_read_only=gpu.stdout.strip(),cpu_gpu_parallel_supported=True,isolated_throughput_claim=False,blas_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),new_gpu_seconds=0))


if __name__=='__main__':main()
