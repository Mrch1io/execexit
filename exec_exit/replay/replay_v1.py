"""在不可变的 Dev36 轨迹与探测记录上回放 v1 策略。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import shlex
from pathlib import Path
import statistics
import subprocess
import sys
import time

from exec_exit.analysis.trajectory_annotations import ByteLevelTokenCodec
from exec_exit.data.load_lcb import DEV36_IDS, load_legacy_dev36
from exec_exit.replay.early_abort import PHRASES, RULE, first_abort_from_token_ids
from exec_exit.replay.policies import NativeCost
from exec_exit.replay.policies_v1 import ProbeV1, choose_v1, choose_oracle_v1, free_perfect_bound, select_candidates
from exec_exit.replay.replay import read_rows, sha, write_json, unique_index, public_probe, selected_outcome

ROOT=Path(__file__).resolve().parents[2]
GRID_POLICIES=("FIRST","CONF-50","CONV-TEXT","EG","EG-2","ORACLE")


def ast_signature(code):
    if not isinstance(code,str):
        return None
    try:
        representation=ast.dump(ast.parse(code),annotate_fields=True,include_attributes=False)
    except (SyntaxError,ValueError,RecursionError):
        return None
    return hashlib.sha256(representation.encode()).hexdigest()


def groups_summary(rows, native_outcomes):
    result={}
    for difficulty in ("all","easy","medium","hard"):
        sub=[r for r in rows if difficulty=="all" or r["difficulty"]==difficulty]
        counts=Counter(r["outcome"] for r in sub)
        result[difficulty]=dict(total=len(sub),passed=counts["PASS"],failed=counts["FAIL"],unknown=counts["UNKNOWN"],
            mean_generated_tokens=statistics.mean(r["total_generated_tokens"] for r in sub),
            total_generated_tokens=sum(r["total_generated_tokens"] for r in sub),
            rescued=[r["case_id"] for r in sub if r["outcome"]=="PASS" and native_outcomes[r["case_id"]]=="FAIL"],
            harmed=[r["case_id"] for r in sub if r["outcome"]=="FAIL" and native_outcomes[r["case_id"]]=="PASS"])
    return result


def pareto_indices(rows):
    """保留实测非支配点，包括数值并列的对照。"""
    result=[]
    for i,row in enumerate(rows):
        x,y=row["mean_generated_tokens"],row["passed"]
        if not any(other["mean_generated_tokens"]<=x and other["passed"]>=y
                   and (other["mean_generated_tokens"]<x or other["passed"]>y)
                   for other in rows):
            result.append(i)
    return result


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"N":"black","BF":"#777777","FIRST":"#bd6825","CONF-50":"#8672b0",
            "CONV-TEXT":"#17a0a6","EG":"#c34343","EG-2":"#3668ad","ORACLE":"#38905c"}
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for ax,group in zip(axes.flat,("all","easy","medium","hard")):
        feasible=[]
        for entry in summary["strategies"]:
            value=entry["groups"][group]
            family="BF" if entry["policy"].startswith("BF@") else entry["policy"]
            marker="X" if family=="N" else "s" if family=="BF" else {1:"o",2:"^",4:"v"}[entry["k"]]
            ax.scatter(value["mean_generated_tokens"],value["passed"],edgecolors=colors[family],marker=marker,
                       s=70,facecolors="none" if entry.get("minimum")==2048 else colors[family],
                       alpha=.85,linewidths=1.3)
            if family!="ORACLE":
                feasible.append(value | {"name":entry["name"]})
            if family=="N":
                ax.annotate(entry["policy"],(value["mean_generated_tokens"],value["passed"]),
                            xytext=(4,4),textcoords="offset points",fontsize=7)
        frontier=sorted([feasible[i] for i in pareto_indices(feasible)],key=lambda x:x["mean_generated_tokens"])
        unique=sorted(set((x["mean_generated_tokens"],x["passed"]) for x in frontier))
        if unique:
            ax.plot([x for x,y in unique],[y for x,y in unique],color="black",lw=1,alpha=.4,linestyle="--")
        ax.set_title(f"{group.title()} ({36 if group=='all' else 12} questions)")
        ax.set_xlabel("Mean generated tokens (all failed probes and N fallback charged)")
        ax.set_ylabel("Complete test passes")
        ax.grid(alpha=.2)
        measured=[entry["groups"][group]["passed"] for entry in summary["strategies"]]
        ax.set_ylim(min(measured)-.6,max(measured)+.8)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker="o",ls="",color=color,label=family) for family,color in colors.items()]
    fig.legend(handles=handles,loc="outside lower center",ncol=8)
    fig.suptitle("Dev36 v1 fixed-record replay: measured operating points\n"
                 "Triangles: k=2/4; circles: k=1; open: m=2048. Dashed: measured deployable frontier; ORACLE excluded.",fontsize=12)
    fig.savefig(out/"pareto_v1.png",dpi=170)
    fig.savefig(out/"pareto_v1.pdf")
    plt.close(fig)
    return dict(matplotlib_version=matplotlib.__version__,panels=["all","easy","medium","hard"],
                bound_points_included=False,frontier_interpolation_claimed=False)


def write_rows(path,rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    wall,cpu=time.perf_counter(),time.process_time()
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError("Never overwrite a previous run")
    paths=dict(native=ROOT/"runs/native36/raw/trajectories.jsonl",
               probes=ROOT/"runs/probes_v0/raw/probes.jsonl",
               native_public=ROOT/"runs/native36_evaluation/raw/signals_public.jsonl",
               probe_public=ROOT/"runs/probes_v0_evaluation/raw/signals_public.jsonl",
               native_labels=ROOT/"runs/native36_evaluation/raw/dev_labels.jsonl",
               probe_labels=ROOT/"runs/probes_v0_evaluation/raw/dev_labels.jsonl",
               candidates=ROOT/"runs/native36_annotations/raw/annotations.jsonl")
    sources=[Path(__file__),Path(__file__).with_name("early_abort.py"),Path(__file__).with_name("policies_v1.py"),
             Path(__file__).with_name("policies.py"),Path(__file__).with_name("replay.py"),
             ROOT/"exec_exit/analysis/trajectory_annotations.py",ROOT/"exec_exit/data/load_lcb.py"]
    (out/"raw").mkdir(parents=True)
    (out/"source").mkdir()
    for path in sources:
        (out/"source"/path.name).write_bytes(path.read_bytes())
    config=dict(schema="exec_exit.replay_v1",started_at_utc=datetime.now(timezone.utc).isoformat(),
        input_identities={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        source_identities={str(p):sha(p) for p in sources},tokenizer=dict(path=str(args.tokenizer),sha256=sha(args.tokenizer)),
        expected_questions=36,expected_probes=386,gpu_used=False,heldout_read=False,new_tests_executed=False,
        python_executable=sys.executable,python_version=sys.version,
        protocol=dict(rule=RULE,phrases=PHRASES,line_start="Optional spaces/tabs then #; textual rule, not Python lexical classification",
            right_boundary="Actually visible non-word character or actual output end; never use a future line",
            token_count="Actual IDs, generated first token=1, include token completing visible match",
            case_insensitive=True,word_definition="Python Unicode regex \\w",
            online_equivalence="Same stored prefix only; changed batches/cancellation may change fresh online outputs",
            receive_chunk_tail="Recorded separately from ideal first visible token cutoff; cancellation tail UNKNOWN"),
        defaults=dict(k=[1,2,4],minimum=[0,2048],candidate_selection="Sort all candidates including fixed points; slice [::k]; then keep position>=minimum",
            confidence_threshold=.95,confidence_comparison="strict greater; same frozen v0 rank-one score",
            FIRST="First selected candidate only, invalid/abort => N fallback",
            BF="Exact fixed point independent of k/m; not reached => N; invalid/abort => N under common §4.3 fallback",
            CONV_TEXT="Two consecutive selected valid drafts with identical ast.dump(include_attributes=False); any invalid/abort resets",
            EG_2="Two consecutive selected valid PUB PASS drafts; any rejected probe resets",
            ORACLE="First v1-valid full-PASS draft; charge all visited probes; otherwise N",
            TEXT_and_execution_cache="AST identity only for TEXT, never changes raw-byte execution cache or labels",
            free_bounds="Perfect offline selector minimizes recorded cost among full-PASS probes and passing N; no passing option=>full N; report delivery charged and fully free delivery separately"),
        expected_EG={"k1_m0":dict(passed=20,mean_tokens_rounded=13264),"k2_m0":dict(passed=21,mean_tokens_rounded=11937)})
    write_json(out/"config.yaml",config)
    (out/"commit.txt").write_text(subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()+"\nUncommitted source snapshots and hashes are recorded.\n")
    native_records,probe_records=read_rows(paths["native"]),read_rows(paths["probes"])
    if len(native_records)!=36 or len(probe_records)!=386:
        raise ValueError("Require exact complete Dev36 / 386 corpus")
    natives=unique_index(native_records,"case_id")
    if set(natives)!=set(DEV36_IDS):
        raise ValueError("Native IDs differ from Dev36")
    annotated=read_rows(paths["candidates"])
    expected={(r["case_id"],c["position_generated_tokens"]) for r in annotated for c in r["candidates"]}
    actual={(r["case_id"],r["position_generated_tokens"]) for r in probe_records}
    if actual!=expected or len(actual)!=386:
        raise ValueError("Probe population does not exactly match all frozen candidate points")
    signals=unique_index(read_rows(paths["native_public"])+read_rows(paths["probe_public"]),"subject_id")
    labels=unique_index(read_rows(paths["native_labels"])+read_rows(paths["probe_labels"]),"subject_id")
    expected_subjects={"N:"+case for case in DEV36_IDS}|{r["probe_id"] for r in probe_records}
    if set(signals)!=expected_subjects:
        raise ValueError("Public records do not cover all and only source records")
    tasks={r["task_id"]:r for r in load_legacy_dev36()}
    codec=ByteLevelTokenCodec(args.tokenizer)
    token_data=json.loads(args.tokenizer.read_text())
    special_ids={r["id"] for r in token_data.get("added_tokens",[]) if r.get("special")}
    grouped={case:[] for case in DEV36_IDS}
    abort_rows=[]
    for record in probe_records:
        p=public_probe(record,signals[record["probe_id"]])
        match=first_abort_from_token_ids(record["call"]["token_ids"],codec,special_ids=special_ids)
        row=dict(subject_id=p.subject_id,case_id=record["case_id"],position=p.position,
                 original_generated_tokens=p.generated_tokens,abort=match.as_dict() if match else None,
                 v1_generated_tokens=p.generated_tokens if match is None else match.token_count,
                 originally_deliverable=p.deliverable)
        if match:
            times=record["call"].get("token_observation_ns",[])
            if len(times)==p.generated_tokens:
                stamp=times[match.token_count-1]
                last=match.token_count
                while last<len(times) and times[last]==stamp:
                    last+=1
                row.update(observed_message_charge_tokens=last,already_received_message_tail=last-match.token_count,
                           historical_host_receive_ns=stamp)
            else:
                row.update(observed_message_charge_tokens="UNKNOWN",already_received_message_tail="UNKNOWN")
            row["post_abort_inflight_tokens"]="UNKNOWN_NOT_EXECUTED_ONLINE"
        grouped[record["case_id"]].append(ProbeV1(p,None if match is None else match.token_count,
            ast_signature(signals[p.subject_id]["extraction"].get("code_text"))))
        abort_rows.append(row)
    for group in grouped.values():
        group.sort(key=lambda p:p.public.position)
    native_outcomes={case:selected_outcome("N:"+case,signals,labels) for case in DEV36_IDS}
    all_labels={sid:selected_outcome(sid,signals,labels) for sid in expected_subjects}
    results=[]
    bound_rows=[]
    specifications=[("N",None,None)]+[(f"BF@{b}",None,None) for b in (2048,4096,8192,16384)]
    specifications += [(policy,k,m) for policy in GRID_POLICIES for k in (1,2,4) for m in (0,2048)]
    for case in DEV36_IDS:
        native=natives[case]
        total=sum(len(call["token_ids"]) for call in native["calls"])
        if total!=native["generated_tokens"]:
            raise ValueError("Native token count mismatch")
        cost=NativeCost("N:"+case,total,len(native["injected_control_token_ids"]))
        for policy,k,m in specifications:
            probes=grouped[case] if k is None else select_candidates(grouped[case],k=k,minimum=m)
            choice=choose_oracle_v1(cost,probes,all_labels) if policy=="ORACLE" else choose_v1(policy,cost,probes)
            choice.update(case_id=case,difficulty=tasks[case]["difficulty"],k=k,minimum=m,
                          name=policy if k is None else f"{policy}:k{k}:m{m}",
                          outcome=selected_outcome(choice["selected_subject_id"],signals,labels))
            results.append(choice)
        for include in (True,False):
            bound=free_perfect_bound(cost,grouped[case],all_labels,include_delivery=include)
            bound.update(case_id=case,difficulty=tasks[case]["difficulty"],
                         name="FREE_PROBES_DELIVERY_CHARGED" if include else "FREE_ALL_DRAFTS")
            bound_rows.append(bound)
    strategies=[]
    for policy,k,m in specifications:
        name=policy if k is None else f"{policy}:k{k}:m{m}"
        rows=[r for r in results if r["name"]==name]
        strategies.append(dict(name=name,policy=policy,k=k,minimum=m,groups=groups_summary(rows,native_outcomes),
                               public_check_demands=sum(r["public_check_demands"] for r in rows),
                               offline_private_check_demands=sum(r["private_check_demands"] for r in rows)))
    bounds={name:groups_summary([r for r in bound_rows if r["name"]==name],native_outcomes)
            for name in ("FREE_PROBES_DELIVERY_CHARGED","FREE_ALL_DRAFTS")}
    checks=[]
    for k,expected_passed,expected_mean in ((1,20,13264),(2,21,11937)):
        observed=next(r for r in strategies if r["name"]==f"EG:k{k}:m0")["groups"]["all"]
        checks.append(dict(k=k,minimum=0,expected_passed=expected_passed,actual_passed=observed["passed"],
            expected_mean_rounded=expected_mean,actual_mean=observed["mean_generated_tokens"],
            mean_difference=observed["mean_generated_tokens"]-expected_mean,
            matches_rounded_expectation=observed["passed"]==expected_passed and round(observed["mean_generated_tokens"])==expected_mean))
    aborts=[r for r in abort_rows if r["abort"]]
    summary=dict(status="COMPLETE",schema="exec_exit.replay_v1",planned_questions=36,planned_probes=386,
        strategies=strategies,free_perfect_bounds=bounds,expected_result_checks=checks,
        protocol=dict(aborted_probes=len(aborts),aborted_originally_deliverable=sum(r["originally_deliverable"] is True for r in aborts),
            original_corpus_generated_tokens=sum(r["original_generated_tokens"] for r in abort_rows),
            counterfactual_v1_all_probes_generated_tokens=sum(r["v1_generated_tokens"] for r in abort_rows),
            actual_new_gpu_tokens=0,post_cancel_inflight_tokens="UNKNOWN",
            receive_message_tail_tokens=sum(r["already_received_message_tail"] for r in aborts if type(r["already_received_message_tail"]) is int)),
        interpretation="Already-seen Dev36 fixed-record exploration. No online, heldout, significance, or contribution claim.",
        costs="Generated tokens include rejected probes and N fallback, accepted delivery once. Execution/CPU/detector physical costs are not token-equivalent and not part of this Pareto axis.",
        source_test_labels="Immutable v0 evaluation; no new tests or replacement timeout labels",
        gpu_used=False,heldout_read=False)
    write_rows(out/"raw/protocol_v1_aborts.jsonl",abort_rows)
    write_rows(out/"raw/policy_paths.jsonl",results)
    write_rows(out/"raw/free_bound_paths.jsonl",bound_rows)
    if args.render:
        summary["figure"]=render(summary,out)
    summary.update(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
                   ended_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/"summary.json",summary)
    lines=["# Dev36 探测协议 v1：固定记录回放", "", "以下为已见 36 题探索结果；CPU 回放没有新增 GPU 输出、重跑测试或改写 v0 标签。", "",
           "| 策略 / k / m | 全部：通过 / 平均 token | easy | medium | hard |", "|---|---:|---:|---:|---:|"]
    for entry in strategies:
        cells=[f"{entry['groups'][g]['passed']}/{entry['groups'][g]['total']} / {entry['groups'][g]['mean_generated_tokens']:,.2f}" for g in ("all","easy","medium","hard")]
        lines.append("| "+entry["name"]+" | "+" | ".join(cells)+" |")
    lines += ["", "![Pareto](pareto_v1.png)", "", "**免费探测 + 完美停止（不可部署的上限）。**"]
    for name,value in bounds.items():
        lines.append("- "+name+"："+"；".join(f"{g} {value[g]['passed']}/{value[g]['total']}，平均 {value[g]['mean_generated_tokens']:,.2f} token" for g in ("all","easy","medium","hard")))
    lines += ["", "前一上限保留接受草稿的交付 token，后一上限连交付草稿也免费；都用离线正确性挑选最低成本的正确交付，无通过选项时保留完整 N。没有 GPU/CPU 免费或真实可实现的含义。", "",
              f"**提前中止。** {len(aborts)}/386 命中，原可交付草稿中 {summary['protocol']['aborted_originally_deliverable']} 个也被拒绝。旧完整草稿共 {summary['protocol']['original_corpus_generated_tokens']:,} token；逐 token 首次可见边界下 v1 对全部候选的回放成本 {summary['protocol']['counterfactual_v1_all_probes_generated_tokens']:,} token。实际收到同一消息的尾部另有 {summary['protocol']['receive_message_tail_tokens']} token；在线取消在途量 UNKNOWN。", "",
              "**预期核对。**"]
    for check in checks:
        lines.append(f"- EG k={check['k']}, m=0：实得 {check['actual_passed']}/36、{check['actual_mean']:,.2f} token；参考 {check['expected_passed']}/36、{check['expected_mean_rounded']:,}，均值差 {check['mean_difference']:+.2f}；四舍五入一致={check['matches_rounded_expectation']}。")
    lines += ["", "**默认口径。** 先完整候选 [::k]，再筛 m；BF 独立使用精确固定点，不足/无效回退 N；FIRST 只看首个被选择位置。CONV-TEXT 为连续有效草稿 AST 完全相同，EG-2 为连续被执行探测 PUB 通过，拒绝会重置；TEXT 不改执行缓存。CONF-50 的拒绝分支只计 50 token 或更早中止。", "",
              "**边界。** 判定等待实际可见的整词右边界，不把未来整行标签倒填到更早位置。贪心和相同原始前缀支持本记录上的截断回放，不保证新批次/在线取消下模型输出仍相同。执行 CPU、查询和控制成本不在生成 token 横轴中；完整净服务收益仍未测。", "",
              "[逐题路径](raw/policy_paths.jsonl) · [提前中止事件](raw/protocol_v1_aborts.jsonl) · [配置](config.yaml) · [完整结果](summary.json)"]
    (out/"ONE_PAGE.md").write_text("\n".join(lines)+"\n")
    command = shlex.join(["python", "-B", "-m", "exec_exit.replay.replay_v1",
                          "--output", "runs/new_v1_replay", "--tokenizer", str(args.tokenizer), "--render"])
    (out/"REPRODUCE.md").write_text("请使用新的输出目录，保留原始运行。\n\n```bash\n" + command +
        "\npython -B -m unittest exec_exit.tests.test_early_abort exec_exit.tests.test_policies_v1 -v\n```\n")
    print(json.dumps(dict(status=summary["status"],aborted=len(aborts),expected_checks=checks,wall_seconds=summary["wall_seconds"],output=str(out)),ensure_ascii=False))


if __name__=="__main__":
    main()
