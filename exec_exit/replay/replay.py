"""在 Dev36 固定轨迹上回放 v0 策略，离线正确性标签单独传入。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from exec_exit.data.load_lcb import DEV36_IDS
from exec_exit.replay.policies import NativeCost, PublicProbe, choose_stop, choose_oracle

IMPLEMENTED = True
ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("N", "FIRST", "CONF-50", "EG", "ORACLE")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def unique_index(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate input identities for " + key)
    return result


def confidence_50(call):
    ids, ranks = call.get("token_ids"), call.get("rank_one_logprobs")
    if (not isinstance(ids, list) or not ids or not isinstance(ranks, list)
            or len(ranks) != len(ids)):
        return None
    prefix = ranks[:50]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value > 0 for value in prefix):
        return None
    return math.exp(math.fsum(prefix) / len(prefix))


def public_probe(record, signal):
    """仅提取公开观测，不将离线正确性标签放入策略输入。"""
    if any(key in signal for key in ("dev_result", "private_result", "dev_label", "private_pass")):
        raise ValueError("Private/development labels are forbidden in public policy signals")
    call = record["call"]
    if len(call["token_ids"]) != call["token_count"]:
        raise ValueError("Probe generated-token count differs from native IDs")
    if signal["subject_id"] != record["probe_id"] or signal["case_id"] != record["case_id"]:
        raise ValueError("Public signal identity mismatch")
    if signal["position"] != record["position_generated_tokens"]:
        raise ValueError("Public signal position differs from the generated probe")
    extraction = signal.get("extraction", {})
    return PublicProbe(record["probe_id"], record["position_generated_tokens"], len(call["token_ids"]),
        len(record["control_token_ids"]), signal.get("generation_valid"), extraction.get("parsed"),
        extraction.get("deliverable"), signal.get("public_result", {}).get("status", "UNKNOWN"),
        confidence_50(call))


def selected_outcome(subject_id, signals, labels):
    signal = signals.get(subject_id)
    if signal is None:
        return "UNKNOWN"
    extract = signal.get("extraction", {})
    #无效交付即使未运行测试也判为失败；v0协议中，长度截断的输出不能仅凭语法通过改判
    if signal.get("generation_valid") is False or extract.get("deliverable") is False:
        return "FAIL"
    if signal.get("generation_valid") is not True or extract.get("deliverable") is not True:
        return "UNKNOWN"
    if signal.get("public_result", {}).get("status") == "FAIL":
        return "FAIL"
    label = labels.get(subject_id, {}).get("dev_result", {})
    value = label.get("all_passed")
    if value is True:
        if signal.get("public_result", {}).get("status") != "PASS":
            raise ValueError("A full-pass label conflicts with the public result")
        return "PASS"
    if value is False:
        return "FAIL"
    return "UNKNOWN"


def summarize_policy(rows, native_outcomes, signals):
    counts = Counter(row["outcome"] for row in rows)
    total = len(rows)
    rescued = [r["case_id"] for r in rows if r["outcome"] == "PASS" and native_outcomes[r["case_id"]] == "FAIL"]
    harmed = [r["case_id"] for r in rows if r["outcome"] == "FAIL" and native_outcomes[r["case_id"]] == "PASS"]
    unclassified = [r["case_id"] for r in rows if "UNKNOWN" in (r["outcome"], native_outcomes[r["case_id"]])]
    selected = [signals.get(row["selected_subject_id"], {}) for row in rows]
    return dict(planned_questions=total, passed=counts["PASS"], failed=counts["FAIL"], unknown=counts["UNKNOWN"],
        full_pass_fraction=counts["PASS"] / total, known_outcome_fraction=(counts["PASS"]+counts["FAIL"]) / total,
        mean_generated_tokens=statistics.mean(row["total_generated_tokens"] for row in rows),
        total_generated_tokens=sum(row["total_generated_tokens"] for row in rows),
        mean_main_generated_tokens=statistics.mean(row["main_generated_tokens"] for row in rows),
        mean_probe_generated_tokens=statistics.mean(row["probe_generated_tokens"] for row in rows),
        total_control_tokens=sum(row["total_control_tokens"] for row in rows),
        accepted_probe_questions=sum(row["accepted_probe"] for row in rows),
        fallback_questions=sum(not row["accepted_probe"] for row in rows),
        rescued_task_ids=rescued, harmed_task_ids=harmed, unknown_comparison_task_ids=unclassified,
        selected_parseable_count=sum(x.get("extraction", {}).get("parsed") is True for x in selected),
        selected_parseable_fraction=sum(x.get("extraction", {}).get("parsed") is True for x in selected) / total,
        selected_deliverable_count=sum(x.get("extraction", {}).get("deliverable") is True for x in selected),
        selected_deliverable_fraction=sum(x.get("extraction", {}).get("deliverable") is True for x in selected) / total,
        public_check_demands=sum(row["public_check_demands"] for row in rows),
        private_check_demands=sum(row["private_check_demands"] for row in rows),
        measured_online_check_latency_seconds="UNKNOWN_NOT_MEASURED")


def replay(native_records, probe_records, public_signals, dev_labels, *, case_ids=DEV36_IDS, threshold=.95):
    natives = unique_index(native_records, "case_id")
    probes = unique_index(probe_records, "probe_id")
    signals = unique_index(public_signals, "subject_id")
    labels = unique_index(dev_labels, "subject_id")
    if set(natives) != set(case_ids):
        raise ValueError("Every planned native task must be present exactly once")
    groups = {case: [] for case in case_ids}
    for record in probes.values():
        if record["case_id"] not in groups or record["probe_id"] not in signals:
            raise ValueError("Unexpected probe task or missing public evaluation record")
        groups[record["case_id"]].append(public_probe(record, signals[record["probe_id"]]))
    outcomes, results, plot_rows = {}, [], []
    for case in case_ids:
        native = natives[case]
        native_id = "N:" + case
        if native_id not in signals:
            raise ValueError("Missing native public record; do not silently remove a planned task")
        total = sum(len(call["token_ids"]) for call in native["calls"])
        if total != native["generated_tokens"]:
            raise ValueError("Native generated-token sum is inconsistent")
        cost = NativeCost(native_id, total, len(native["injected_control_token_ids"]))
        ordered = sorted(groups[case], key=lambda item: item.position)
        #普通策略不读取正确性标签，只有oracle使用单独的离线标签映射
        choices = [choose_stop(policy, cost, ordered, threshold=threshold) for policy in POLICIES if policy != "ORACLE"]
        offline = {p.subject_id: selected_outcome(p.subject_id, signals, labels) for p in ordered}
        choices.append(choose_oracle(cost, ordered, offline))
        outcomes[case] = selected_outcome(native_id, signals, labels)
        for choice in choices:
            choice.update(case_id=case, outcome=selected_outcome(choice["selected_subject_id"], signals, labels))
            results.append(choice)
        oracle = choices[-1]
        first_call = next(call for call in native["calls"] if call["stage"] == 0)
        close = native.get("thinking_end_token_index")
        thinking_length = len(first_call["token_ids"]) if close is None else close
        position = oracle["accepted_position"]
        if position is not None and position > thinking_length:
            raise ValueError("Probe position is past the actual thinking boundary")
        status = ("OBSERVED_FULL_PASS" if position is not None else "NO_CANDIDATES" if not ordered
                  else "UNKNOWN_INCOMPLETE_TEST_COVERAGE" if "UNKNOWN" in offline.values() else "NO_OBSERVED_FULL_PASS")
        plot_rows.append(dict(case_id=case, native_thinking_generated_tokens=thinking_length,
            native_total_generated_tokens=total, first_full_pass_probe_position=position,
            opportunity_status=status, measured_candidates=len(ordered),
            unclassified_candidate_labels=sum(value == "UNKNOWN" for value in offline.values()),
            native_outcome=outcomes[case],
            raw_position_remainder=None if position is None else thinking_length-position,
            probe_generated_tokens_through_first_pass_or_all_if_absent=oracle["probe_generated_tokens"],
            probe_control_tokens=sum(row["charged_probe_control_tokens"] for row in oracle["visited"]),
            public_check_demands=oracle["public_check_demands"], private_check_demands=oracle["private_check_demands"],
            interpretation="Measured candidate/draft/suite opportunity only; position remainder is not net cost benefit"))
    stats = {policy: summarize_policy([r for r in results if r["policy"] == policy], outcomes, signals) for policy in POLICIES}
    return dict(policy_rows=results, policy_summary=stats, plot_rows=plot_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--probes-dir", type=Path, required=True)
    parser.add_argument("--native-eval-dir", type=Path, required=True)
    parser.add_argument("--probe-eval-dir", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=ROOT / "runs/pipeline_v0/config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Replay/plotting waits until GPU generation has drained")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output run, never overwrite earlier results")
    wall, cpu = time.perf_counter(), time.process_time()
    frozen = json.loads(args.pipeline_config.read_text())
    defaults = frozen["policy_defaults"]
    if (defaults["minimum_thinking_tokens"] != 0 or defaults["probe_every_n_candidates"] != 1
            or defaults["CONF_50_threshold"] != .95 or defaults["CONF_50_comparison"] != "strictly greater"):
        raise ValueError("Replay implementation is scoped to the explicit v0 defaults")
    inputs = [args.native_dir / "raw/trajectories.jsonl", args.probes_dir / "raw/probes.jsonl"]
    inputs += [directory / "raw" / name for directory in (args.native_eval_dir, args.probe_eval_dir)
               for name in ("signals_public.jsonl", "dev_labels.jsonl")]
    native, probes = read_rows(inputs[0]), read_rows(inputs[1])
    if len(native) != 36 or len(probes) != 386:
        raise ValueError("Require complete Dev36 and all 386 frozen probe records before this replay")
    native_identity = [value for key, value in frozen["sources"].items() if key.endswith("raw/trajectories.jsonl")]
    annotation_entries = [(key, value) for key, value in frozen["sources"].items() if key.endswith("raw/annotations.jsonl")]
    if len(native_identity) != 1 or sha(inputs[0]) != native_identity[0] or len(annotation_entries) != 1:
        raise ValueError("Native/candidate source identity differs from the frozen v0 configuration")
    annotation_path, expected_annotation_sha = annotation_entries[0]
    annotation_path = ROOT / annotation_path
    if sha(annotation_path) != expected_annotation_sha:
        raise ValueError("Frozen candidate annotations changed")
    annotated = read_rows(annotation_path)
    expected_points = {(row["case_id"], item["position_generated_tokens"]) for row in annotated for item in row["candidates"]}
    if {(row["case_id"], row["position_generated_tokens"]) for row in probes} != expected_points:
        raise ValueError("Probe population does not cover exactly the frozen candidate positions")
    for directory, raw_path in ((args.native_eval_dir, inputs[0]), (args.probe_eval_dir, inputs[1])):
        evaluation_config = json.loads((directory / "config.yaml").read_text())
        if evaluation_config["input_sha256"] != sha(raw_path):
            raise ValueError("Evaluation results do not identify the supplied generation corpus")
    signals = read_rows(inputs[2])+read_rows(inputs[4])
    labels = read_rows(inputs[3])+read_rows(inputs[5])
    expected = {"N:"+x["case_id"] for x in native} | {x["probe_id"] for x in probes}
    if {x["subject_id"] for x in signals} != expected:
        raise ValueError("Public evaluation population differs from frozen native/probe subjects")
    if not {x["subject_id"] for x in labels} <= expected:
        raise ValueError("Development labels include unknown subjects")
    (out / "raw").mkdir(parents=True)
    source_paths = [Path(__file__), Path(__file__).with_name("policies.py"), ROOT / "exec_exit/analysis/plot_stopping_opportunities.py"]
    config = dict(schema="exec_exit.replay_v0", pipeline_config=str(args.pipeline_config.resolve()),
        pipeline_config_sha256=sha(args.pipeline_config), defaults=defaults,
        input_identities=[dict(path=str(p.resolve()), sha256=sha(p)) for p in inputs],
        source_identities=[dict(path=str(p), sha256=sha(p)) for p in source_paths],
        python_executable=sys.executable, python_version=sys.version,
        planned_questions=36, planned_probes=386, gpu_used=False, heldout_read=False,
        interpretation="Fixed-trajectory Dev comparison; no online speed or generalization claim",
        check_cost_accounting="Demands and physical offline evaluator resources reported separately; online latency UNKNOWN",
        started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "config.yaml", config)
    (out / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()+"\nSource hashes identify the uncommitted analysis.\n")
    (out / "source").mkdir()
    for path in source_paths:
        (out / "source" / path.name).write_bytes(path.read_bytes())
    result = replay(native, probes, signals, labels, threshold=.95)
    for name, rows in (("policy_paths.jsonl", result["policy_rows"]), ("first_full_pass_opportunities.jsonl", result["plot_rows"])):
        with (out / "raw" / name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    physical = {}
    for label, directory in (("native_generation", args.native_dir), ("probe_generation", args.probes_dir),
                             ("native_evaluation", args.native_eval_dir), ("probe_evaluation", args.probe_eval_dir)):
        path = directory / "summary.json"
        physical[label] = dict(source=str(path.resolve()), sha256=sha(path), summary=json.loads(path.read_text()))
    source_coverage = {}
    for kind in ("N", "probe"):
        subset = [x for x in signals if x["source_kind"] == kind]
        source_coverage[kind] = dict(total=len(subset),
            parseable=sum(x["extraction"].get("parsed") is True for x in subset),
            parseable_fraction=sum(x["extraction"].get("parsed") is True for x in subset) / len(subset),
            deliverable=sum(x["extraction"].get("deliverable") is True for x in subset),
            deliverable_fraction=sum(x["extraction"].get("deliverable") is True for x in subset) / len(subset),
            generation_valid=sum(x.get("generation_valid") is True for x in subset),
            public_outcomes=dict(Counter(x.get("public_result", {}).get("status", "UNKNOWN") for x in subset)))
    summary = dict(status="COMPLETE_FIXED_TRAJECTORY_REPLAY", policies=result["policy_summary"],
        source_coverage=source_coverage, planned_questions=36, planned_probes=386,
        opportunity_counts=dict(Counter(x["opportunity_status"] for x in result["plot_rows"])),
        physical_corpus_resources=physical,
        old_N_sanity_reference=dict(passed=19, denominator=36,
            scope="Historical descriptive reference only: GPU/environment, budget and output protocol differ; no causal attribution to 28k"),
        resources=dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, gpu_used=False,
            scope="Replay body excludes imports, final serialization and optional plot rendering"),
        limits=["All data are already-seen Dev36", "CONF-50 costs follow the specified fixed-record prefix path; no online stop/restart equivalence claim",
                "Oracle observes offline labels but pays all preceding full draft generation", "Position remainder does not subtract draft/check costs", "Missing tests stay UNKNOWN"])
    write_json(out / "summary.json", summary)
    lines = ["# Dev36 固定轨迹回放 v0", "", "全部 36 题入分母。以下为已见开发记录回放，不是在线收益或贡献确认。", "", "| 策略 | 完整通过 / 36 | UNKNOWN | 平均生成 token | 获救 / 受损 |", "|---|---:|---:|---:|---:|"]
    for policy, stats in result["policy_summary"].items():
        lines.append(f'| {policy} | {stats["passed"]} | {stats["unknown"]} | {stats["mean_generated_tokens"]:.1f} | {len(stats["rescued_task_ids"])} / {len(stats["harmed_task_ids"])} |')
    lines += ["", "生成 token 包含被访问的失败探测，接受草稿只计一次；控制输入另列。CONF-50 低分/UNKNOWN 分支仅计最多 50 个已生成 token。FIRST 首稿无效直接回 N。长度截断即使可解析，也不是 v0 可交付程序。缺失测试为 UNKNOWN。", "", "解析成功率、可交付率、逐题获救/受损身份、全部路径与资源见 summary.json 和 raw/。检查次数是回放需求，实际离线评测 CPU/墙钟单列；不能冒充在线等待时间。", "", "旧 N 的 19/36 仅作健全性描述；不同 GPU/驱动、预算与协议，不能将差异因果归于 28k。"]
    (out / "ONE_PAGE.md").write_text("\n".join(lines)+"\n")
    if args.render:
        from exec_exit.analysis.plot_stopping_opportunities import render
        figure = render(result["plot_rows"], out, measured_candidates=386)
        summary["figure"] = figure
        write_json(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "policies": summary["policies"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
