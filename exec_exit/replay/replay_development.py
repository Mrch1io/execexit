"""在开发记录上回放冻结的 v1 策略，单独使用经超时复判的标签。

适配层不改变冻结策略：仅用首次运行的公开观测决策，选定后才读取整程序最终标签。
前瞻性实验契约须在读取结果前写定。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from exec_exit.analysis.trajectory_annotations import ByteLevelTokenCodec
from exec_exit.replay.early_abort import first_abort_from_token_ids
from exec_exit.replay.policies import NativeCost
from exec_exit.replay.policies_v1 import (
    ProbeV1, choose_oracle_v1, choose_v1, free_perfect_bound, select_candidates,
)
from exec_exit.replay.replay import public_probe, read_rows, sha, unique_index, write_json
from exec_exit.replay.replay_v1 import ast_signature, pareto_indices

ROOT = Path(__file__).resolve().parents[2]
DIFFICULTIES = ("all", "easy", "medium", "hard")
NON_EXECUTION = {"BF", "FIRST", "CONF-50", "CONV-TEXT"}
OUTCOMES = {"PASS", "FAIL", "UNKNOWN"}
EVALUATION_IDENTITY_FIELDS = ("limits", "execution_identity", "public_projection_sha256",
    "private_projection_sha256", "tokenizer_sha256", "extraction_rule", "public_fail_fast",
    "dev_private_fail_fast", "timeout_rule", "signal_separation")


def family(policy):
    return "BF" if policy.startswith("BF@") else policy


def policy_specifications(defaults):
    """严格使用冻结的41种配置，不根据结果追加配置。"""
    if (defaults["k"] != [1, 2, 4] or defaults["minimum"] != [0, 2048]
            or defaults["confidence_threshold"] != .95):
        raise ValueError("This adapter only supports the frozen v1 grid")
    specs = [("N", None, None)] + [(f"BF@{b}", None, None) for b in (2048, 4096, 8192, 16384)]
    specs += [(p, k, m) for p in ("FIRST", "CONF-50", "CONV-TEXT", "EG", "EG-2", "ORACLE")
              for k in defaults["k"] for m in defaults["minimum"]]
    return [dict(policy=p, k=k, minimum=m,
                 name=p if k is None else f"{p}:k{k}:m{m}") for p, k, m in specs]


def validate_contract(contract, *, root=ROOT):
    baseline_path = Path(contract["baseline_config"]["path"])
    if sha(baseline_path) != contract["baseline_config"]["sha256"]:
        raise ValueError("Sealed configuration identity changed")
    baseline = json.loads(baseline_path.read_text())
    if (contract["defaults"] != baseline["defaults"]
            or contract["protocol"] != baseline["protocol"]
            or contract["specifications"] != policy_specifications(baseline["defaults"])):
        raise ValueError("Contract no longer matches all frozen policies/defaults")
    for relative, digest in contract["fixed_policy_sources"].items():
        if sha(root / relative) != digest:
            raise ValueError("Frozen policy source changed: " + relative)
    return baseline


def validate_evaluation_metadata(config, summary, *, input_sha256, public_sha256,
                                 tokenizer_sha256, expected_subjects, reference_identity=None):
    """读取结果前拒绝未完成或不兼容的评估记录。"""
    if summary.get("status") != "COMPLETE" or summary.get("planned") != expected_subjects:
        raise ValueError("Evaluation is incomplete/failed or has the wrong denominator")
    if config.get("input_sha256") != input_sha256:
        raise ValueError("Evaluation identifies a different generation corpus")
    if (config.get("public_projection_sha256") != public_sha256
            or config.get("tokenizer_sha256") != tokenizer_sha256):
        raise ValueError("Evaluation public projection or tokenizer differs from replay")
    if any(config.get(key) is None for key in EVALUATION_IDENTITY_FIELDS):
        raise ValueError("Evaluation lacks a required execution/protocol identity")
    identity = {key: config[key] for key in EVALUATION_IDENTITY_FIELDS}
    for key in ("input_capacity_mib", "output_capacity_mib", "capacity_policy"):
        if key in config:
            identity[key] = config[key]
    if reference_identity is not None and identity != reference_identity:
        changed = [key for key in sorted(set(identity) | set(reference_identity))
                   if identity.get(key) != reference_identity.get(key) or (key in identity) != (key in reference_identity)]
        raise ValueError("Mixed evaluation limits/environment/protocol: "+", ".join(changed))
    return identity


def validate_populations(tasks, populations, *, expected_sizes=None):
    task_map = unique_index(tasks, "task_id")
    if set(populations) == {"New81"}:
        whole, parts = set(populations["New81"]), [set(populations["New81"])]
    else:
        whole = set(populations["Dev121"])
        parts = [set(populations[n]) for n in ("OldDev36", "New81", "SeenSmoke4")]
    if whole != set(task_map) or set.union(*parts) != whole:
        raise ValueError("Development populations do not cover exactly the task manifest")
    if any(a & b for i, a in enumerate(parts) for b in parts[i+1:]):
        raise ValueError("Development components overlap")
    if any(len(values) != len(set(values)) for values in populations.values()):
        raise ValueError("Duplicate question in population")
    if expected_sizes is not None and {k: len(v) for k, v in populations.items()} != expected_sizes:
        raise ValueError("Question denominators differ from prospective contract")
    if any(t["difficulty"] not in DIFFICULTIES[1:] for t in tasks):
        raise ValueError("Unrecognized difficulty; do not silently drop questions")
    return task_map


def final_outcomes(labels, expected_subjects):
    """读取超时多数裁决后的独立评分，不用于公开策略决策。"""
    index = unique_index(labels, "subject_id")
    if set(index) != set(expected_subjects):
        raise ValueError("Require one final label per planned native/probe subject")
    for row in labels:
        if row.get("final_outcome") not in OUTCOMES or type(row.get("unstable")) is not bool:
            raise ValueError("Final whole-program label/instability schema is missing")
    return {sid: row["final_outcome"] for sid, row in index.items()}, index


def reject_private_signal_fields(signal):
    #同时拒绝旧正确性字段和经超时复判的新标签，防止离线信息泄漏
    forbidden = {"dev_result", "private_result", "dev_label", "private_pass",
                 "final_outcome", "unstable", "timeout_robustness"}
    if forbidden & set(signal):
        raise ValueError("Offline correctness leaked into public policy projection")


def received_message_tail(call, cutoff):
    """计算截断token所在原生消息中，已收到的额外token数。

    仅补充记录前缀的成本核算，不假定改变在线策略后仍有相同消息批次或取消尾部。
    """
    ids, stamps = call["token_ids"], call.get("token_observation_ns", [])
    if not 0 <= cutoff <= len(ids):
        raise ValueError("Charged prefix outside received record")
    if cutoff == 0:
        return 0
    if len(stamps) != len(ids):
        return None
    last = cutoff
    while last < len(ids) and stamps[last] == stamps[cutoff-1]:
        last += 1
    return last-cutoff


def adapt_probe(record, signal, codec, special_ids):
    reject_private_signal_fields(signal)
    public = public_probe(record, signal)
    call = record["call"]
    match = first_abort_from_token_ids(call["token_ids"], codec, special_ids=special_ids)
    recorded_reject = call.get("protocol_rejected")
    if record.get("schema") != "exec_exit.probe.v2_online_abort":
        raise ValueError("Expected actual v1 online-abort probe schema")
    if recorded_reject is not bool(match) or record.get("protocol_rejected") is not recorded_reject:
        raise ValueError("Stored online reject disagrees with frozen rule")
    if match:
        if call.get("detection_token_count") != match.token_count:
            raise ValueError("Online detection cutoff differs from recorded-prefix rule")
        if signal.get("generation_valid") is not False:
            raise ValueError("Aborted probe must remain invalid regardless of parse/test result")
        tail = received_message_tail(call, match.token_count)
        if tail != call.get("extra_received_tokens_same_message"):
            raise ValueError("Online same-message tail count differs")
    else:
        tail = 0
        if call.get("native_finished") is not True:
            raise ValueError("Non-aborted probe did not finish")
    adapted = ProbeV1(public, None if match is None else match.token_count,
                      ast_signature(signal.get("extraction", {}).get("code_text")))
    account = dict(subject_id=public.subject_id, case_id=record["case_id"],
        position=public.position, received_generated_tokens=public.generated_tokens,
        effective_v1_tokens=adapted.effective_tokens, protocol_rejected=bool(match),
        abort=match.as_dict() if match else None, same_message_tail_tokens=tail,
        post_cancel_unobserved_tokens="UNKNOWN" if match else 0,
        native_finished=call.get("native_finished"))
    return adapted, account


def adapt_complete_v0_probe(record, signal, codec, special_ids, *, virtual_abort):
    """适配完整v0记录或其虚拟v1前缀，两者都不是实际在线v1运行。"""
    if record.get("schema") != "exec_exit.probe.v1":
        raise ValueError("Expected complete original-protocol v0 record")
    reject_private_signal_fields(signal)
    public = public_probe(record, signal)
    match = (first_abort_from_token_ids(record["call"]["token_ids"], codec, special_ids=special_ids)
             if virtual_abort else None)
    p = ProbeV1(public, None if match is None else match.token_count,
                ast_signature(signal.get("extraction", {}).get("code_text")))
    tail = received_message_tail(record["call"], match.token_count) if match else 0
    return p, dict(subject_id=public.subject_id, case_id=record["case_id"], position=public.position,
        received_generated_tokens=public.generated_tokens, effective_v1_tokens=p.effective_tokens,
        protocol_rejected=bool(match), abort=match.as_dict() if match else None,
        same_message_tail_tokens=tail, post_cancel_unobserved_tokens="UNKNOWN_NOT_EXECUTED" if match else 0,
        native_finished=record["call"].get("native_finished"),
        interpretation="Virtual truncation of complete v0" if virtual_abort else "Complete v0; no comment abort")


def summarize_rows(rows, native_labels):
    counts = Counter(r["outcome"] for r in rows)
    total_tokens = sum(r["total_generated_tokens"] for r in rows)
    return dict(total=len(rows), passed=counts["PASS"], failed=counts["FAIL"], unknown=counts["UNKNOWN"],
        complete_pass_count_lower=counts["PASS"], complete_pass_count_upper=counts["PASS"]+counts["UNKNOWN"],
        total_generated_tokens=total_tokens,
        mean_generated_tokens=total_tokens/len(rows) if rows else None,
        rescued=[r["case_id"] for r in rows if r["outcome"] == "PASS" and native_labels[r["case_id"]] == "FAIL"],
        harmed=[r["case_id"] for r in rows if r["outcome"] == "FAIL" and native_labels[r["case_id"]] == "PASS"],
        unknown_comparison=[r["case_id"] for r in rows if "UNKNOWN" in (r["outcome"], native_labels[r["case_id"]])],
        public_check_demands=sum(r.get("public_check_demands", 0) for r in rows),
        offline_private_check_demands=sum(r.get("private_check_demands", 0) for r in rows),
        same_message_extra_tokens_on_recorded_paths=sum(r.get("received_same_message_extra_tokens", 0) for r in rows),
        paths_with_unknown_message_tail=sum(r.get("unknown_message_tail_visits", 0) > 0 for r in rows),
        cancellation_tail="UNKNOWN_NOT_AN_ONLINE_POLICY_MEASUREMENT")


def population_summaries(rows, native_labels, populations):
    return {name: {difficulty: summarize_rows(
        [r for r in rows if r["case_id"] in cases and (difficulty == "all" or r["difficulty"] == difficulty)],
        native_labels) for difficulty in DIFFICULTIES} for name, cases in populations.items()}


def measured_envelope(strategies, population, difficulty, target_name):
    """在相同分母下比较整数总成本，不做插值。"""
    target = next(s for s in strategies if s["name"] == target_name)
    value = target["populations"][population][difficulty]
    points = []
    for strategy in strategies:
        point = strategy["populations"][population][difficulty]
        if point["total"] != value["total"]:
            raise ValueError("Cost envelope has inconsistent denominators")
        if (family(strategy["policy"]) in NON_EXECUTION
                and point["total_generated_tokens"] <= value["total_generated_tokens"]):
            points.append(dict(name=strategy["name"], **point))
    best = max((p["passed"] for p in points), default=None)
    best_upper = max((p["passed"]+p["unknown"] for p in points), default=None)
    winners = [p for p in points if p["passed"] == best]
    unknown = bool(value["unknown"] or any(p["unknown"] for p in points))
    status = ("EMPTY_POPULATION" if not value["total"] else "UNKNOWN_OUTCOMES" if unknown
              else "NO_MEASURED_ELIGIBLE_CONTROL" if best is None
              else "STRICT_PASS_ADVANTAGE" if value["passed"] > best else "NO_STRICT_PASS_ADVANTAGE")
    return dict(population=population, difficulty=difficulty, target=target_name,
        target_passed=value["passed"], target_total_generated_tokens=value["total_generated_tokens"],
        total=value["total"], eligible_controls=points, envelope_passed=best,
        target_pass_count_bounds=[value["passed"], value["passed"]+value["unknown"]],
        envelope_pass_count_bounds=None if best is None else [best, best_upper],
        target_minus_envelope_bounds=None if best is None else [value["passed"]-best_upper,
                                                               value["passed"]+value["unknown"]-best],
        tied_envelope_controls=[p["name"] for p in winners],
        target_minus_envelope_passes=None if best is None else value["passed"]-best,
        status=status, interpolated=False, N_excluded_from_primary_envelope=True)


def equal_pass_cost_comparison(strategies, population="New81", difficulty="all",
                               target_name="CONF-50:k2:m2048"):
    target = next(s for s in strategies if s["name"] == target_name)["populations"][population][difficulty]
    candidates = [dict(name=s["name"], **s["populations"][population][difficulty])
                  for s in strategies if s["policy"] != "ORACLE"]
    equal = [p for p in candidates if p["passed"] == target["passed"]]
    lowest = min(p["total_generated_tokens"] for p in equal)
    unknown = any(p["unknown"] for p in candidates)
    return dict(population=population, difficulty=difficulty, target=target_name,
        target_passed=target["passed"], target_total_generated_tokens=target["total_generated_tokens"],
        equal_pass_points=sorted(equal, key=lambda p: (p["total_generated_tokens"], p["name"])),
        least_cost_names=[p["name"] for p in equal if p["total_generated_tokens"] == lowest],
        target_is_least_cost_at_equal_pass_count=None if unknown else target["total_generated_tokens"] == lowest,
        observed_pass_count_target_is_least_cost=target["total_generated_tokens"] == lowest,
        at_least_passes_and_strictly_cheaper=[p for p in candidates if p["passed"] >= target["passed"]
                                            and p["total_generated_tokens"] < target["total_generated_tokens"]],
        interpretation="UNKNOWN_OUTCOMES" if unknown else "DESCRIPTIVE_MEASURED_DEVELOPMENT_POINTS")


def replay_development(native_records, probe_records, signals_rows, labels_rows, tasks,
                       annotations, contract, codec, special_ids, *, probe_mode="actual_v1"):
    if probe_mode not in {"actual_v1", "full_v0", "virtual_v1"}:
        raise ValueError("Unfrozen probe projection")
    populations = contract["populations"]
    task_map = validate_populations(tasks, populations)
    cases = populations[contract.get("primary_population", "Dev121")]
    natives = unique_index(native_records, "case_id")
    probes = unique_index(probe_records, "probe_id")
    if set(natives) != set(cases) or len(probes) != contract["expected_probes"]:
        raise ValueError("Missing or unexpected native/probe records")
    positions = [(r["case_id"], r["position_generated_tokens"]) for r in probe_records]
    annotated = {(r["case_id"], c["position_generated_tokens"]) for r in annotations for c in r["candidates"]}
    if set(positions) != annotated or len(set(positions)) != len(probes):
        raise ValueError("Actual probes do not cover exactly annotated positions")
    signals = unique_index(signals_rows, "subject_id")
    expected_subjects = {"N:"+case for case in cases} | set(probes)
    if set(signals) != expected_subjects:
        raise ValueError("Public observations do not cover every planned subject")
    for signal in signals_rows:
        reject_private_signal_fields(signal)
    outcomes, labels = final_outcomes(labels_rows, expected_subjects)
    native_labels = {case: outcomes["N:"+case] for case in cases}
    grouped = {case: [] for case in cases}
    probe_accounts = []
    for record in probe_records:
        if probe_mode == "actual_v1":
            adapted, account = adapt_probe(record, signals[record["probe_id"]], codec, special_ids)
        else:
            adapted, account = adapt_complete_v0_probe(record, signals[record["probe_id"]], codec,
                special_ids, virtual_abort=probe_mode == "virtual_v1")
        grouped[record["case_id"]].append(adapted)
        probe_accounts.append(account)
    for values in grouped.values():
        values.sort(key=lambda p: p.public.position)
    results, bound_rows = [], []
    for case in cases:
        native = natives[case]
        total = sum(len(call["token_ids"]) for call in native["calls"])
        if total != native["generated_tokens"]:
            raise ValueError("Native cost differs from actual saved IDs")
        cost = NativeCost("N:"+case, total, len(native["injected_control_token_ids"]))
        for spec in contract["specifications"]:
            selected = grouped[case] if spec["k"] is None else select_candidates(
                grouped[case], k=spec["k"], minimum=spec["minimum"])
            choice = (choose_oracle_v1(cost, selected, outcomes) if spec["policy"] == "ORACLE"
                      else choose_v1(spec["policy"], cost, selected,
                                     threshold=contract["defaults"]["confidence_threshold"]))
            tails = []
            for visit in choice["visited"]:
                tail = received_message_tail(probes[visit["subject_id"]]["call"],
                                             visit["charged_probe_generated_tokens"])
                visit["received_same_message_extra_tokens"] = tail
                visit["post_cancel_unobserved_tokens"] = "UNKNOWN"
                tails.append(tail)
            choice.update(spec, case_id=case, difficulty=task_map[case]["difficulty"],
                outcome=outcomes[choice["selected_subject_id"]],
                selected_label_unstable=labels[choice["selected_subject_id"]]["unstable"],
                received_same_message_extra_tokens=sum(t for t in tails if t is not None),
                unknown_message_tail_visits=sum(t is None for t in tails),
                outcome_source="independent whole-program final_outcome; public decision remains first-run")
            results.append(choice)
        for include in (True, False):
            bound = free_perfect_bound(cost, grouped[case], outcomes, include_delivery=include)
            bound.update(case_id=case, difficulty=task_map[case]["difficulty"],
                         name="FREE_PROBES_DELIVERY_CHARGED" if include else "FREE_ALL_DRAFTS")
            bound_rows.append(bound)
    strategies = [dict(spec, populations=population_summaries(
        [r for r in results if r["name"] == spec["name"]], native_labels, populations))
        for spec in contract["specifications"]]
    bounds = {name: population_summaries([r for r in bound_rows if r["name"] == name], native_labels, populations)
              for name in ("FREE_PROBES_DELIVERY_CHARGED", "FREE_ALL_DRAFTS")}
    comparisons = [measured_envelope(strategies, pop, difficulty, spec["name"])
                   for pop in populations for difficulty in DIFFICULTIES
                   for spec in contract["specifications"] if spec["policy"] in {"EG", "EG-2"}]
    label_audit = [dict(subject_id=sid, case_id=signals[sid]["case_id"],
        first_public_status=signals[sid].get("public_result", {}).get("status", "UNKNOWN"),
        final_outcome=outcomes[sid], unstable=labels[sid]["unstable"],
        timeout_robustness=labels[sid].get("timeout_robustness"),
        public_fail_but_final_pass=signals[sid].get("public_result", {}).get("status") == "FAIL" and outcomes[sid] == "PASS")
        for sid in sorted(expected_subjects) if labels[sid]["unstable"]
        or signals[sid].get("public_result", {}).get("status") == "FAIL" and outcomes[sid] == "PASS"]
    summary = dict(status="COMPLETE", schema="exec_exit.replay_development.v1", probe_mode=probe_mode, planned_questions=len(cases),
        planned_probes=len(probes), strategies=strategies, free_perfect_bounds=bounds,
        measured_nonexecution_envelopes=comparisons,
        new81_conf50_equal_pass_cost=equal_pass_cost_comparison(strategies),
        populations={name: dict(count=len(ids), case_ids=ids) for name, ids in populations.items()},
        protocol=dict(received_probe_tokens=sum(r["received_generated_tokens"] for r in probe_accounts),
            ideal_protocol_prefix_tokens=sum(r["effective_v1_tokens"] for r in probe_accounts),
            protocol_rejections=sum(r["protocol_rejected"] for r in probe_accounts),
            actual_same_message_tail_tokens=sum(r["same_message_tail_tokens"] for r in probe_accounts if r["same_message_tail_tokens"] is not None),
            unknown_same_message_tails=sum(r["same_message_tail_tokens"] is None for r in probe_accounts),
            post_cancel_unobserved_tokens="UNKNOWN"),
        final_label_counts=dict(Counter(outcomes.values())),
        unstable_subjects=sum(r["unstable"] for r in labels_rows),
        public_fail_final_pass_subjects=[r["subject_id"] for r in label_audit if r["public_fail_but_final_pass"]],
        interpretation="Fixed-record development description. Question-level units, no significance/heldout/generalization claim.",
        cost_scope=contract["primary_token_axis"])
    return summary, dict(policy_paths=results, probe_accounts=probe_accounts,
                         free_reference_paths=bound_rows, label_robustness=label_audit)


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")


def preserve_preflight_failure(args, exc, source):
    """即使运行目录尚未建好，也保留首次结构或身份校验失败。"""
    out = args.output.resolve()
    if out.exists():
        #后续分析失败由运行处理流程保存，此处不覆盖
        return
    (out/"raw").mkdir(parents=True)
    (out/"source").mkdir()
    shutil.copyfile(source, out/"source"/source.name)
    write_json(out/"config.yaml", dict(schema="exec_exit.analysis_preflight_failure.v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        source=dict(path=str(source), sha256=sha(source)), gpu_used=False, tests_executed=False))
    (out/"commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()+"\n")
    write_json(out/"summary.json", dict(status="FAILED_PREFLIGHT", error=repr(exc),
        results_produced=False,
        note="Metadata/identity failure preserved; use a new output directory after correction."))


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    colors = {"N": "black", "BF": "#777777", "FIRST": "#bd6825", "CONF-50": "#8672b0",
              "CONV-TEXT": "#17a0a6", "EG": "#c34343", "EG-2": "#3668ad", "ORACLE": "#38905c"}
    artifacts = []
    for population in ("Dev121", "OldDev36", "New81"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        for ax, difficulty in zip(axes.flat, DIFFICULTIES):
            deployable = []
            for strategy in summary["strategies"]:
                point = strategy["populations"][population][difficulty]
                fam = family(strategy["policy"])
                marker = "X" if fam == "N" else "s" if fam == "BF" else {1: "o", 2: "^", 4: "v"}[strategy["k"]]
                ax.scatter(point["mean_generated_tokens"], point["passed"], edgecolors=colors[fam],
                    facecolors="none" if strategy["minimum"] == 2048 else colors[fam],
                    marker=marker, s=70, alpha=.85)
                if fam != "ORACLE":
                    deployable.append(point)
            #用圆圈标出实测非支配点，不连线或插值
            for i in pareto_indices(deployable):
                p = deployable[i]
                ax.scatter(p["mean_generated_tokens"], p["passed"], marker="o", s=130,
                           facecolors="none", edgecolors="black", linewidths=.6)
            count = summary["strategies"][0]["populations"][population][difficulty]["total"]
            ax.set_title(f"{difficulty}: {count} questions")
            ax.set_xlabel("Mean ideal prefix tokens (trigger included); not physical GPU work")
            ax.set_ylabel("Complete passes")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.2)
        handles = [Line2D([0], [0], marker="o", ls="", color=color, label=name) for name, color in colors.items()]
        fig.legend(handles=handles, loc="outside lower center", ncol=8)
        fig.suptitle(f"{population}: frozen 41 measured v1 points\nBlack rings: deployable Pareto points; no interpolation; ORACLE is offline only.")
        for extension in ("png", "pdf"):
            filename = f"pareto_{population}.{extension}"
            fig.savefig(out/filename, dpi=170)
            artifacts.append(filename)
        plt.close(fig)
    return dict(matplotlib_version=matplotlib.__version__, files=artifacts, interpolation=False)


def reports(summary, out):
    lines = ["# Dev121：冻结 v1 完整回放", "", summary["interpretation"], "",
        "普通策略只使用首轮公共结果；下表正确性是完整程序的独立 final_outcome（含预定超时复测多数票）。主轴是 v1 首次可见触发 token 的理想前缀成本，不是物理 GPU 工作量；同消息尾部、取消未知成本和 CPU 检查另列。",
        "OldDev36 是本次 Dev121 六并发原生轨迹和实际 v1 探测中的36题子集，不是封存 native36/v0 的结果。预定 CONF-50 点来自旧封存回放，本次不调参；旧36结果变化只作描述，不归因。"]
    for population in summary["populations"]:
        lines += ["", f"## {population}", ""]
        for difficulty in DIFFICULTIES:
            lines += [f"### {difficulty}", "", "| 配置 | 通过/分母 | UNKNOWN | 平均 token | 救回/损害 |", "|---|---:|---:|---:|---:|"]
            for strategy in summary["strategies"]:
                p = strategy["populations"][population][difficulty]
                cost = "N/A" if p["mean_generated_tokens"] is None else f"{p['mean_generated_tokens']:,.2f}"
                lines.append(f"| {strategy['name']} | {p['passed']}/{p['total']} | {p['unknown']} | {cost} | {len(p['rescued'])}/{len(p['harmed'])} |")
    lines += ["", "## 实测非执行上包络", "", "只比较成本不高于对应 EG/EG-2 的实测点；所有并列控制保留，不插值。", "",
              "| 组/难度 | 执行点 | 通过数差 | 实测上包络控制 | 状态 |", "|---|---|---:|---|---|"]
    for row in summary["measured_nonexecution_envelopes"]:
        lines.append(f"| {row['population']}/{row['difficulty']} | {row['target']} | {row['target_minus_envelope_passes']} | {', '.join(row['tied_envelope_controls']) or '无合格实测点'} | {row['status']} |")
    lines += ["", "## 新增81的预定 CONF-50 点", "", "```json", json.dumps(summary["new81_conf50_equal_pass_cost"], ensure_ascii=False, indent=2), "```", "",
              "全部逐题选择、救回与损害见 summary.json 及 raw/policy_paths.jsonl。免费参考仅含 v1 有效且未中止候选加 N；不是任意方法的理论上界。",
              "公共首轮与完整程序多数票不一致见 raw/label_robustness.jsonl；它只用于离线评分，不能反向改变 EG 当时的动作。"]
    lines += ["冻结 ORACLE 也保留首轮 PUB PASS 条件；即使复测多数票最终 PASS，首轮 PUB FAIL 的草稿仍不被它接受。它不是任意停止规则的完美上界。"]
    (out/"FULL_TABLES.md").write_text("\n".join(lines)+"\n")
    short = ["# Dev121：固定 v1 回放结果", "", "全部 41 配置保留；开发集描述，不作留出确认。OldDev36 指本次六并发 Dev121 采集中的36题子集，不复用封存旧轨迹结果；变化不作归因。", "",
             "| 组 | N | CONF-50 k2/m2048 | EG k1/m0 | EG k2/m0 |", "|---|---|---|---|---|"]
    by_name = {s["name"]: s for s in summary["strategies"]}
    for population in summary["populations"]:
        cells = []
        for name in ("N", "CONF-50:k2:m2048", "EG:k1:m0", "EG:k2:m0"):
            p = by_name[name]["populations"][population]["all"]
            cells.append(f"{p['passed']}/{p['total']}; {p['mean_generated_tokens']:,.2f} token; 救/损 {len(p['rescued'])}/{len(p['harmed'])}")
        short.append(f"| {population} | "+" | ".join(cells)+" |")
    comparison = summary["new81_conf50_equal_pass_cost"]
    if comparison["interpretation"] == "UNKNOWN_OUTCOMES":
        cost_answer = ("最终通过标签仍有 UNKNOWN；仅按当前已观察通过计数，预定 CONF-50 点最低成本判定为 "
                       f"{comparison['observed_pass_count_target_is_least_cost']}，不能据此确定真实同通过数最低成本。")
    else:
        cost_answer = f"新增81预定 CONF-50 点是否为同通过数最低成本：{comparison['target_is_least_cost_at_equal_pass_count']}。"
    short += ["", cost_answer+f"当前观测计数的并列最低成本：{', '.join(comparison['least_cost_names'])}。",
              "", f"真实探测收到 {summary['protocol']['received_probe_tokens']:,} token；v1 理想前缀 {summary['protocol']['ideal_protocol_prefix_tokens']:,}；同消息额外 {summary['protocol']['actual_same_message_tail_tokens']}，取消后未暴露工作 UNKNOWN。",
              "", "主轴是含触发 token 的首次可见理想前缀成本，不是物理 GPU 成本。首轮 PUB 保持原样；最终评分使用完整程序超时多数票，差异逐条列出。这里不是在线延迟或净 GPU 节省。",
              "", "[全部控制与包络](FULL_TABLES.md) · [summary](summary.json) · [逐题选择](raw/policy_paths.jsonl) · [资源](resource_ledger.json)"]
    (out/"ONE_PAGE.md").write_text("\n".join(short)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT/"runs/dev121_replay_v1_contract/config.yaml")
    parser.add_argument("--native-dir", type=Path, default=ROOT/"runs/dev121_native_v1")
    parser.add_argument("--probes-dir", type=Path, default=ROOT/"runs/dev121_probes_v1")
    parser.add_argument("--native-eval-dir", type=Path, default=ROOT/"runs/dev121_native_evaluation_v2")
    parser.add_argument("--probe-eval-dir", type=Path, default=ROOT/"runs/dev121_probes_evaluation_v2")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT/"runs/dev121_annotations_v1")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh run directory; never overwrite prior records")
    try:
        return run(args)
    except Exception as exc:
        preserve_preflight_failure(args, exc, Path(__file__))
        raise


def run(args):
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Heavy replay/plots wait until GPU generation is idle")
    if args.output.exists():
        raise FileExistsError("Use a fresh run directory; never overwrite prior records")
    wall, cpu = time.perf_counter(), time.process_time()
    contract = json.loads(args.contract.read_text())
    validate_contract(contract)
    paths = dict(native=args.native_dir/"raw/trajectories.jsonl", probes=args.probes_dir/"raw/probes.jsonl",
        native_public=args.native_eval_dir/"raw/signals_public.jsonl", native_labels=args.native_eval_dir/"raw/dev_labels.jsonl",
        probe_public=args.probe_eval_dir/"raw/signals_public.jsonl", probe_labels=args.probe_eval_dir/"raw/dev_labels.jsonl",
        annotations=args.annotations_dir/"raw/annotations.jsonl", public_tasks=Path(contract["public_tasks"]["path"]))
    identities = {name: dict(path=str(path.resolve()), sha256=sha(path)) for name, path in paths.items()}
    if identities["public_tasks"]["sha256"] != contract["public_tasks"]["sha256"]:
        raise ValueError("Frozen task populations changed")
    eval_resources = {}
    evaluation_identity = None
    for label, directory, source in (("native", args.native_eval_dir, "native"), ("probes", args.probe_eval_dir, "probes")):
        evaluation_config = json.loads((directory/"config.yaml").read_text())
        evaluation_summary = json.loads((directory/"summary.json").read_text())
        evaluation_identity = validate_evaluation_metadata(evaluation_config, evaluation_summary,
            input_sha256=identities[source]["sha256"], public_sha256=identities["public_tasks"]["sha256"],
            tokenizer_sha256=sha(args.tokenizer), expected_subjects=121 if source == "native" else contract["expected_probes"],
            reference_identity=evaluation_identity)
        eval_resources[label] = dict(config=str(directory/"config.yaml"), config_sha256=sha(directory/"config.yaml"),
                                     summary=str(directory/"summary.json"), summary_sha256=sha(directory/"summary.json"))
    out = args.output.resolve()
    (out/"raw").mkdir(parents=True)
    (out/"source").mkdir()
    source_files = [Path(__file__), ROOT/"exec_exit/replay/replay.py", ROOT/"exec_exit/replay/replay_v1.py",
                    ROOT/"exec_exit/analysis/trajectory_annotations.py"] + [ROOT/p for p in contract["fixed_policy_sources"]]
    for path in source_files:
        shutil.copyfile(path, out/"source"/path.name)
    config = dict(schema="exec_exit.replay_development.v1", started_at_utc=datetime.now(timezone.utc).isoformat(),
        contract=dict(path=str(args.contract.resolve()), sha256=sha(args.contract)), input_identities=identities,
        source_identities={str(p): sha(p) for p in source_files}, evaluation_sources=eval_resources,
        shared_evaluation_identity=evaluation_identity,
        tokenizer=dict(path=str(args.tokenizer), sha256=sha(args.tokenizer)),
        specifications=contract["specifications"], gpu_used=False, heldout_read=False, new_tests_executed=False,
        python_executable=sys.executable, python_version=sys.version)
    write_json(out/"config.yaml", config)
    (out/"commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()+"\nUncommitted sources are archived and hashed.\n")
    try:
        tasks = json.loads(paths["public_tasks"].read_text())["tasks"]
        validate_populations(tasks, contract["populations"], expected_sizes={"Dev121": 121, "OldDev36": 36, "New81": 81, "SeenSmoke4": 4})
        codec = ByteLevelTokenCodec(args.tokenizer)
        special_ids = {r["id"] for r in json.loads(args.tokenizer.read_text()).get("added_tokens", []) if r.get("special")}
        summary, raw = replay_development(read_rows(paths["native"]), read_rows(paths["probes"]),
            read_rows(paths["native_public"])+read_rows(paths["probe_public"]),
            read_rows(paths["native_labels"])+read_rows(paths["probe_labels"]), tasks,
            read_rows(paths["annotations"]), contract, codec, special_ids)
        for name, rows in raw.items():
            write_rows(out/"raw"/(name+".jsonl"), rows)
        if args.render:
            summary["figure"] = render(summary, out)
        summary["wall_seconds"] = time.perf_counter()-wall
        summary["self_cpu_seconds"] = time.process_time()-cpu
        write_json(out/"summary.json", summary)
        write_json(out/"resource_ledger.json", dict(replay_wall_seconds=summary["wall_seconds"],
            replay_self_cpu_seconds=summary["self_cpu_seconds"], peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu_used=False, generated_tokens=0, new_tests_executed=False, evaluation_sources=eval_resources,
            note="Physical evaluation resources remain in source runs; replay token paths are counterfactual costs, not new work."))
        reports(summary, out)
        print(json.dumps(dict(status=summary["status"], output=str(out), wall_seconds=summary["wall_seconds"],
                              expected_questions=121, configurations=len(summary["strategies"]))))
    except Exception as exc:
        write_json(out/"summary.json", dict(status="FAILED_ANALYSIS", error=repr(exc),
                                           wall_seconds=time.perf_counter()-wall, self_cpu_seconds=time.process_time()-cpu))
        raise


if __name__ == "__main__":
    main()
