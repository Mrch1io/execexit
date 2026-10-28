"""Draw the summary, README, and report figures from saved estimates only.

Run from any directory using this script's path. The default output directory
also refreshes runs/report_figures/{sources.json,raw/plot_data.json} and
runs/readme_figure/values.json. A custom
--output-dir writes all maintained figures there without changing repository metadata.
Historical renderers are never imported. No fitting or resampling is performed.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
C2_SOURCES = [
    ("Qwen3-8B", "post hoc", "#2474A6", "o",
     "runs/c2_label_completion/raw/qwen_completed/main/summary.json"),
    ("DS-R1-Qwen-7B", "post hoc", "#B36732", "s",
     "runs/c2_label_completion/raw/ds_completed/main/summary.json"),
    ("DS-R1-Llama-8B", "registered replication", "#31816C", "D",
     "runs/llama_execution_repair1/blind/analysis/results/private_c2.json"),
]
QWEN_SOURCE = "runs/qwen_erratum_comparison/raw/comparisons.json"
DS_SOURCE = "runs/queue_rebuild/ds_checkpoint_recovery1/result/statistics/summary.json"
C4_SOURCE = "runs/c4_recovery1/latency_analysis/summary.json"
SENSITIVITY_SOURCE = "runs/c4_recovery1/sensitivity/summary.json"
PARETO_SOURCES = [
    ("Qwen3-8B", "runs/qwen_erratum/secondary_intervals43/raw/comparisons.jsonl"),
    ("DS-R1-Qwen-7B", "runs/queue_rebuild/ds_checkpoint_recovery1/result/secondary_intervals43/raw/comparisons.jsonl"),
    ("DS-R1-Llama-8B", "runs/llama_execution_repair1/blind/analysis/results/intervals43.json"),
]
RISK_SOURCE = "runs/public_test_strength/summary.json"
LEETCODE_SOURCE = "runs/public_test_strength/posthoc_within_leetcode.json"
LLAMA_STATS_SOURCE = "runs/llama_execution_repair1/blind/analysis/results/statistics.json"
README_FIGURE_SOURCE = "runs/readme_figure/values.json"
COLLECTIONS = (("hv6_atcoder", "AtCoder"), ("hv6_leetcode", "LeetCode"), ("lcbpro_2025", "LCB Pro"))
WEIGHTING = "equal questions, equal covered samples within each question"
HIGHLIGHTS = ("N", "BF@4096", "HYB-B", "CONF-50:k1:m0", "EG:k4:m0", "ORACLE:k4:m0")
#背景、文字、蓝橙两色、网格和零线的颜色来自README的图1，其余图沿用同一套配色。
THEMES = {
    "": dict(background="#ffffff", primary="#0b0b0b", secondary="#52514e",
             blue="#2a78d6", orange="#eb6834", grid="#e9e9e7", zero="#a9a9a5",
             muted="#aaaaaa", faint="#dddddd",
             policies=("#222222", "#887b69", "#b83a36", "#8052a3", "#147e81", "#639447")),
    "_dark": dict(background="#0d1117", primary="#f0efec", secondary="#c3c2b7",
                 blue="#3987e5", orange="#d95926", grid="#222830", zero="#707983",
                 muted="#818b98", faint="#424c58",
                 policies=("#f0efec", "#b9a88d", "#ed7770", "#b58bd7", "#52b7ba", "#92bd70")),
}


def read(path):
    return json.loads((ROOT / path).read_text())


def select(rows, **fields):
    matches = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved result for {fields}; found {len(matches)}")
    return matches[0]


def interval(axis, y, value, bounds, color, marker, label, digits, signed=False):
    low, high = bounds
    if not all(math.isfinite(x) for x in (low, value, high)) or not low <= value <= high:
        raise ValueError(f"Invalid interval for {label}: {value}, {bounds}")
    axis.errorbar(value, y, xerr=[[value - low], [high - value]], fmt=marker,
                  color=color, markersize=7, capsize=4, capthick=1.5, linewidth=2,
                  markeredgecolor="white", markeredgewidth=.7, zorder=3)
    axis.text(0, y + .28, label, transform=axis.get_yaxis_transform(),
              fontsize=10.6, color="#263445", va="bottom",
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    sign = "+" if signed else ""
    number = f"{value:{sign}.{digits}f}  [{low:.{digits}f}, {high:.{digits}f}]"
    axis.text(0, y - .26, number.replace("-", "−"),
              transform=axis.get_yaxis_transform(), fontsize=10.5,
              color=color, va="top",
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})


def panel(axis, title, xlabel, limits, ticks):
    axis.set_title(title, loc="left", pad=26, fontsize=14, fontweight="bold")
    axis.set_xlabel(xlabel, labelpad=10, fontsize=11)
    axis.set_xlim(*limits)
    axis.set_xticks(ticks)
    axis.set_ylim(-.10, 3.3)
    axis.set_yticks([])
    axis.grid(axis="x", color="#E9EDF1", linewidth=.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", length=0, pad=7, labelsize=10)
    for spine in ("top", "left", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#BAC4CE")


def render_main(output, pareto):
    c2 = []
    for name, status, color, marker, path in C2_SOURCES:
        result = read(path)
        if result["status"] != "COMPLETE" or result["label"] != "PRIV-IND":
            raise ValueError(f"Expected completed PRIV-IND result: {path}")
        row = select(result["comparisons"], signal_name="PUB", population="pooled",
                     corpus="virtual_v1_pipeline", scope="complete_deliverable")
        c2.append((name, status, color, marker, row))
    qwen = select(read(QWEN_SOURCE), version="erratum", population="pooled")
    ds = select(read(DS_SOURCE)["populations"], population="pooled")
    service = read(C4_SOURCE)
    sensitivity = read(SENSITIVITY_SOURCE)
    if service["status"] != "COMPLETE" or sensitivity["status"] != "COMPLETE":
        raise ValueError("C4 requires complete saved main and sensitivity results")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none",
                         "svg.hashsalt": "execexit-main-results",
                         "savefig.facecolor": "white", "text.color": "#263445"})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.5))
    fig.subplots_adjust(left=.045, right=.98, bottom=.24, top=.78, wspace=.22)
    fig.text(.045, .935, "Signal quality, policy safety, and service latency",
             fontsize=21, fontweight="bold")
    fig.text(.045, .885, "ExecExit · saved point estimates and nominal 95% confidence intervals",
             fontsize=12, color="#596879")

    panel(axes[0], "C2  Signal discrimination", "PUB − metadata AUROC  (higher is better)",
          (-.015, .23), [0, .05, .10, .15, .20])
    axes[0].text(0, 1.025, "Qwen/DS post hoc · Llama registered replication", transform=axes[0].transAxes,
                 fontsize=9.3, color="#596879", va="bottom")
    axes[0].axvline(0, color="#84919E", linestyle=":", linewidth=1.2)
    for y, (name, status, color, marker, row) in zip((2.55, 1.45, .35), c2):
        estimate = row["paired_differences"]["signal_minus_baseline"]["auroc"]
        ci = estimate["ci"]
        interval(axes[0], y, estimate["difference"], (ci["lower"], ci["upper"]),
                 color, marker, name, 4, signed=True)

    panel(axes[1], "C3  Stopping-policy quality", "HYB-B − native pass rate  (percentage points)",
          (-4.0, 3.4), [-4, -3, -2, -1, 0, 1, 2, 3])
    axes[1].text(0, 1.025, "Qwen primary · DS secondary · Llama registered replication", transform=axes[1].transAxes,
                 fontsize=9.3, color="#596879", va="bottom")
    axes[1].axvline(0, color="#84919E", linestyle=":", linewidth=1.2)
    # The registered margin does not extend through the Llama row.
    axes[1].plot([-3, -3], [.95, 3.1], color="#AC4C4C", linestyle="--", linewidth=1.3)
    for y, estimate, name, status, color, marker in [
        (2.55, qwen["comparisons"]["N"], "Qwen3-8B", "primary", "#2474A6", "o"),
        (1.45, ds["comparisons"]["N"], "DS-R1-Qwen-7B", "registered secondary", "#B36732", "s"),
        (.35, select(pareto["DS-R1-Llama-8B"], candidate="HYB-B"),
         "DS-R1-Llama-8B", "registered replication", "#31816C", "D"),
    ]:
        if not estimate["complete"] or estimate["candidate"] != "HYB-B":
            raise ValueError(f"Incomplete or unexpected C3 policy for {name}")
        interval(axes[1], y, estimate["pass_difference_pp"],
                 estimate["pass_difference_pp_ci95"], color, marker,
                 name, 2, signed=True)

    panel(axes[2], "C4  Delivered-answer latency", "Async HYB-B / native mean latency  (lower is better)",
          (.80, 1.14), [.80, .90, 1.00, 1.10])
    axes[2].text(0, 1.025, "Registered secondary · sensitivity post-freeze", transform=axes[2].transAxes,
                 fontsize=9.3, color="#596879", va="bottom")
    axes[2].axvline(1, color="#84919E", linestyle=":", linewidth=1.2)
    for y, row, label, color, marker in [
        (2.35, service["primary"], "Main", "#2474A6", "o"),
        (.85, sensitivity["sensitivity"], "Sensitivity", "#596879", "D"),
    ]:
        interval(axes[2], y, row["ratio_of_means"], row["ci95_percentile"],
                 color, marker, f"{label} · n = {row['pairs']}", 3)

    fig.text(.045, .065,
             "Saved estimates with nominal 95% CIs; each panel has its own scale.\n"
             "The −3 pp margin applies only to Qwen primary and DS registered-secondary HYB-B vs N comparisons.",
             fontsize=10, color="#596879", linespacing=1.5)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "main_results.svg"
    fig.savefig(path, dpi=180, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(path)
    plt.close(fig)


def source_record(path, selector, fields):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                selector=selector, fields=fields)


def check_interval(value, bounds, context):
    if len(bounds) != 2 or not all(math.isfinite(v) for v in (value, *bounds)):
        raise ValueError(f"Missing/nonfinite saved interval: {context}")
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"Unordered saved interval: {context}")


def check_display(value, expected, digits, context):
    actual = f"{value:.{digits}f}"
    if actual != expected:
        raise ValueError(f"STOP: {context}: saved value rounds to {actual}, expected {expected}")


def load_pareto():
    """Select recorded policy rows; never derive a metric or an interval."""
    data, sources = {}, []
    fields = ["candidate", "reference", "pass_difference_pp", "pass_difference_pp_ci95",
              "token_saving_percent", "token_saving_percent_ci95", "complete", "role"]
    policy_names = None
    for name, path in PARETO_SOURCES:
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]
            rows = [row for row in rows if row["population"] == "pooled" and row["reference"] == "N"]
            selector = "JSONL: population=pooled; reference=N; all candidates"
        else:
            saved = read(path)
            population = select(saved["populations"], population="pooled")
            if saved["configurations"] != 43 or population["questions"] != 199:
                raise ValueError("STOP: Llama configuration/question count differs")
            rows = [dict(row, population="pooled") for row in population["comparisons"]
                    if row["reference"] == "N"]
            selector = "populations[population=pooled].comparisons[reference=N]; all candidates"
        names = {row["candidate"] for row in rows}
        if len(rows) != 43 or len(names) != 43 or not set(HIGHLIGHTS) <= names:
            raise ValueError(f"STOP: Expected all 43 unique configurations for {name}")
        if policy_names is not None and names != policy_names:
            raise ValueError("STOP: Configuration sets differ across models")
        policy_names = names
        for row in rows:
            if not row["complete"] or row["role"] != "SECONDARY_POINTWISE_NOT_SIMULTANEOUS":
                raise ValueError(f"Unexpected saved policy status: {name}, {row['candidate']}")
            for field in ("pass_difference_pp", "token_saving_percent"):
                check_interval(row[field], row[field + "_ci95"], f"{name}, {row['candidate']}, {field}")
        data[name] = rows
        sources.append(source_record(path, selector, fields))
    for candidate, expected in {
        "HYB-B": ("-0.63", "-2.39", "0.88", "10.72"),
        "BF@4096": ("-4.15", "-6.41", "-2.14", "38.18"),
    }.items():
        row = select(data["DS-R1-Llama-8B"], candidate=candidate)
        for actual, wanted in zip((row["pass_difference_pp"], *row["pass_difference_pp_ci95"],
                                   row["token_saving_percent"]), expected):
            check_display(actual, wanted, 2, f"Llama {candidate}")
    qwen = select(data["Qwen3-8B"], candidate="HYB-B")
    check_display(qwen["pass_difference_pp"], "-1.88", 2, "Qwen HYB-B point")
    check_display(qwen["pass_difference_pp_ci95"][0], "-3.39", 2, "Qwen HYB-B lower bound")
    primary = select(read(QWEN_SOURCE), version="erratum", population="pooled")
    secondary = select(read(DS_SOURCE)["populations"], population="pooled")
    if primary["questions"] != 199 or secondary["questions"] != 199:
        raise ValueError("STOP: Qwen/DS saved question count differs from 199")
    if primary["comparisons"]["N"]["nominal_noninferiority"] is not False:
        raise ValueError("STOP: Qwen saved nominal non-inferiority decision differs")
    if secondary["comparisons"]["N"]["nominal_noninferiority"] is not True:
        raise ValueError("STOP: DS saved nominal non-inferiority decision differs")
    criterion_path = "runs/report_figures/source/stats.py"
    functions = [node for node in ast.parse((ROOT / criterion_path).read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "summarize"]
    if len(functions) != 1:
        raise ValueError("STOP: Expected one statistics summarize function")
    criteria = []
    for node in ast.walk(functions[0]):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "c.update":
            criteria.extend(keyword.value for keyword in node.keywords
                            if keyword.arg == "nominal_noninferiority")
    expected = ast.parse("c['pass_difference_pp_ci95'][0] > -3 if complete else None", mode="eval").body
    if len(criteria) != 1 or ast.dump(criteria[0]) != ast.dump(expected):
        raise ValueError("STOP: Registered non-inferiority criterion source changed")
    sources.extend([
        source_record(QWEN_SOURCE, "[version=erratum, population=pooled]",
                      ["questions", "comparisons.N.nominal_noninferiority"]),
        source_record(DS_SOURCE, "populations[population=pooled]",
                      ["questions", "comparisons.N.nominal_noninferiority"]),
        source_record(criterion_path, f"summarize(): nominal_noninferiority; line {criteria[0].lineno}",
                      ["pass_difference_pp_ci95[0] > -3; strict lower-bound criterion"]),
    ])
    return data, sources


def saved_point(value, ci):
    bounds = [ci["lower"], ci["upper"]]
    check_interval(value, bounds, "public-check risk")
    return dict(value=value, ci95=bounds)


def load_public_check_risk():
    """Only complement/percentage display transforms; no new statistics."""
    overall, subgroup = read(RISK_SOURCE), read(LEETCODE_SOURCE)
    rows, sources = [], []
    selector = dict(population="pooled", signal_name="PUB", comparison="PUB", label="PRIV-IND",
                    corpus="virtual_v1_pipeline", scope="complete_deliverable")
    metric_fields = ["signal_metrics.recall", "signal_metrics.recall_ci.lower",
                     "signal_metrics.recall_ci.upper", "signal_metrics.precision",
                     "signal_metrics.precision_ci.lower", "signal_metrics.precision_ci.upper",
                     "signal_metrics.weighting", "population_coverage.public_pass_drafts",
                     "population_coverage.public_pass_questions", "excluded_missing_rows"]
    expected = [
        ("0.983", "0.315", "0.91", "0.91", "0.84", 2006, 148, 10, 91, 10, 36),
        ("0.984", "0.285", "0.76", "0.75", "0.64", 1453, 133, 9, 53, 6, 22),
        ("0.966", "0.307", "0.92", "0.89", "0.76", 1493, 133, 8, 53, 6, 18),
    ]
    for key, (name, _, _, _, path), wanted in zip(("qwen", "ds", "llama"), C2_SOURCES, expected):
        saved = read(path)
        if saved["status"] != "COMPLETE" or saved["label"] != "PRIV-IND":
            raise ValueError(f"STOP: Incomplete PRIV-IND source: {path}")
        row = select(saved["comparisons"], **selector)
        metrics, coverage = row["signal_metrics"], row["population_coverage"]
        if metrics["weighting"] != WEIGHTING:
            raise ValueError(f"STOP: Unexpected weighting for {name}")
        recall = saved_point(metrics["recall"], metrics["recall_ci"])
        # This deterministic complement, explicitly requested for the figure,
        # reverses the saved endpoints. Raw counts never enter the percentages.
        wrong = saved_point(1 - metrics["precision"],
                            dict(lower=1 - metrics["precision_ci"]["upper"],
                                 upper=1 - metrics["precision_ci"]["lower"]))
        pooled, lc = overall["M3_harm_prediction"][key], subgroup["results"][key]
        all_risk = dict(saved_point(pooled["auroc"], pooled["ci95"]),
                        harmed=pooled["harmed"], questions=pooled["problems"])
        lc_risk = dict(saved_point(lc["leetcode_auroc"], lc["leetcode_ci95"]),
                       harmed=lc["leetcode_harmed"], questions=lc["leetcode_problems"])
        for actual, target, digits in zip(
            (recall["value"], wrong["value"], all_risk["value"], lc_risk["value"], lc["collection_only_auroc"]),
            wanted[:5], (3, 3, 2, 2, 2),
        ):
            check_display(actual, target, digits, name)
        counts = (coverage["public_pass_drafts"], coverage["public_pass_questions"],
                  all_risk["harmed"], all_risk["questions"], lc_risk["harmed"], lc_risk["questions"])
        if counts != wanted[5:]:
            raise ValueError(f"STOP: Saved denominators differ for {name}: {counts}")
        for point in (recall, wrong):
            point["display_percent"] = [100 * point["value"], *[100 * v for v in point["ci95"]]]
            point["formatted"] = "{:.1f}% [{:.1f}, {:.1f}]".format(*point["display_percent"])
        for point in (all_risk, lc_risk):
            point["formatted"] = (f"{point['value']:.2f} [{point['ci95'][0]:.2f}, {point['ci95'][1]:.2f}]"
                                  f"  ·  {point['harmed']}/{point['questions']}")
        rows.append(dict(model=name, key=key, recall=recall, wrong_among_accepted=wrong,
                         public_pass_drafts=counts[0], public_pass_questions=counts[1],
                         unknown_excluded=row["excluded_missing_rows"],
                         overall=all_risk, leetcode=lc_risk,
                         collection_only=dict(value=lc["collection_only_auroc"], ci95=None,
                                              formatted=f"{lc['collection_only_auroc']:.2f}"),
                         source_path=path, selector=selector, weighting=WEIGHTING))
        sources.append(source_record(path, {"comparisons": selector}, metric_fields))
    if overall["M3_harm_prediction"]["qwen"]["harmed_by_collection"] != {"hv6_leetcode": 10}:
        raise ValueError("STOP: Qwen M3 harm is no longer confined to LeetCode")
    main = select(read(QWEN_SOURCE), version="erratum", population="pooled")
    main_counts = main["comparisons"]["N"]["strict"]
    if (main_counts["harmed_questions"], main_counts["questions"]) != (11, 199):
        raise ValueError("STOP: Qwen primary harm count differs from 11/199")
    sources.extend([
        source_record(RISK_SOURCE, "M3_harm_prediction.{qwen,ds,llama}",
                      ["auroc", "ci95.lower", "ci95.upper", "harmed", "problems", "harmed_by_collection"]),
        source_record(LEETCODE_SOURCE, "Root; results selected for qwen/ds/llama",
                      ["results.{model}.leetcode_auroc", "results.{model}.leetcode_ci95.lower",
                       "results.{model}.leetcode_ci95.upper", "results.{model}.leetcode_harmed",
                       "results.{model}.leetcode_problems", "results.{model}.collection_only_auroc", "classification"]),
        source_record(QWEN_SOURCE, "[version=erratum, population=pooled].comparisons.N.strict",
                      ["harmed_questions", "questions"]),
    ])
    return dict(models=rows, primary_harm=dict(harmed=main_counts["harmed_questions"],
                                              questions=main_counts["questions"])), sources


def load_signal_quality():
    """核对values.json与未改动的数据源一致；图只画已保存的对比值，不重新计算。"""
    points = [point for point in read(README_FIGURE_SOURCE) if point.get("figure") != "turning_points"]
    if [(p["panel"], p["signal"]) for p in points] != [("R1", "PUB")] * 3 + [("R2", "CONF-50")] * 3:
        raise ValueError("STOP: Unexpected README figure panels")
    for point, expected in zip(points, ("0.06", "0.12", "0.15", "-0.14", "-0.15", "-0.14")):
        path = point["source_path"]
        raw = (ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != point["source_sha256"]:
            raise ValueError(f"STOP: README figure source changed: {path}")
        if point["panel"] == "R1":
            rows = json.loads(raw)["comparisons"]
        else:
            rows = [json.loads(line) for line in raw.splitlines() if line]
        contrast = select(rows, **point["selector"])["paired_differences"]["signal_minus_baseline"]["auroc"]
        saved = (contrast["difference"], contrast["ci"]["lower"], contrast["ci"]["upper"])
        if saved != (point["estimate"], point["ci_low"], point["ci_high"]):
            raise ValueError(f"STOP: values.json differs from {path}")
        check_interval(point["estimate"], [point["ci_low"], point["ci_high"]], f"README {point['model']}")
        check_display(point["estimate"], expected, 2, f"README {point['model']} {point['signal']}")
    return points


def load_turning_points():
    points = []
    selector = dict(signal_name="PUB", population="pooled", corpus="virtual_v1_pipeline",
                    scope="complete_deliverable", label="PRIV-IND")
    for model, key, before, after in (
        ("Qwen3-8B", "qwen", ("0.0143", "-0.0396", "0.0700"), ("0.0649", "0.0189", "0.1153")),
        ("DeepSeek-R1-Distill-Qwen-7B", "ds", ("0.0678", "0.0112", "0.1303"), ("0.1219", "0.0780", "0.1728")),
    ):
        for phase, path, expected in (
            ("before", f"runs/c2_label_independence/raw/{key}_statistics_phase/summary.json", before),
            ("after", f"runs/c2_label_completion/raw/{key}_completed/main/summary.json", after),
        ):
            row = select(read(path)["comparisons"], **selector)
            contrast = row["paired_differences"]["signal_minus_baseline"]["auroc"]
            estimate, low, high = contrast["difference"], contrast["ci"]["lower"], contrast["ci"]["upper"]
            check_interval(estimate, [low, high], f"turning_points {model} {phase}")
            for value, wanted in zip((estimate, low, high), expected):
                check_display(value, wanted, 4, f"turning_points {model} {phase}")
            points.append(dict(panel="labels", model=model, phase=phase, estimate=estimate, ci_low=low,
                               ci_high=high, drafts=row["matched_rows"], questions=row["matched_questions"],
                               source=source_record(path, selector,
                                   ["paired_differences.signal_minus_baseline.auroc.difference",
                                    "paired_differences.signal_minus_baseline.auroc.ci.lower",
                                    "paired_differences.signal_minus_baseline.auroc.ci.upper"])))
    for phase, version, expected in (
        ("before", "original", ("-1.13", "-2.76", "0.38")),
        ("after", "erratum", ("-1.88", "-3.39", "-0.63")),
    ):
        row = select(read(QWEN_SOURCE), population="pooled", version=version)["comparisons"]["N"]
        if row["candidate"] != "HYB-B" or row["reference"] != "N" or not row["complete"]:
            raise ValueError(f"STOP: Unexpected turning_points comparison: {version}")
        estimate, (low, high) = row["pass_difference_pp"], row["pass_difference_pp_ci95"]
        check_interval(estimate, [low, high], f"turning_points Qwen HYB-B {phase}")
        for value, wanted in zip((estimate, low, high), expected):
            check_display(value, wanted, 2, f"turning_points Qwen HYB-B {phase}")
        points.append(dict(panel="erratum", model="Qwen3-8B", phase=phase, estimate=estimate,
                           ci_low=low, ci_high=high,
                           source=source_record(QWEN_SOURCE,
                               dict(population="pooled", version=version, comparison="N", candidate="HYB-B"),
                               ["comparisons.N.pass_difference_pp", "comparisons.N.pass_difference_pp_ci95"])))
    return points


def theme_axis(axis, theme):
    axis.set_facecolor(theme["background"])
    axis.tick_params(colors=theme["secondary"], labelsize=10, length=0, pad=7)
    axis.grid(color=theme["grid"], linewidth=.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_png(fig, output, name, theme):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.png"
    fig.savefig(path, dpi=200, facecolor=theme["background"],
                metadata={"Creator": "ExecExit saved-results renderer"})
    plt.close(fig)
    print(path)


def render_turning_points(points, output):
    for suffix, theme in THEMES.items():
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
            fig.patch.set_facecolor(theme["background"])
            fig.subplots_adjust(left=.19, right=.975, bottom=.22, top=.78, wspace=.55)
            for axis, title in zip(axes, ("Independent labels\nbefore and after completion", "Data correction\nbefore and after")):
                theme_axis(axis, theme)
                axis.grid(axis="y", visible=False)
                axis.axvline(0, color=theme["zero"], lw=1, zorder=2)
                axis.set_title(title, fontsize=14, color=theme["primary"], pad=18)
                axis.set_ylim(-.25, 2.35)
            axes[0].set(xlim=(-.065, .195), xticks=[-.05, 0, .05, .10, .15],
                        yticks=[1.8, .4], yticklabels=["Qwen3-8B", "DeepSeek-R1-\nDistill-Qwen-7B"])
            axes[0].set_xlabel("Public check − metadata\nAUROC difference (95% CI)", color=theme["primary"], labelpad=12)
            axes[1].set(xlim=(-4.1, 1.1), xticks=[-4, -3, -2, -1, 0, 1],
                        yticks=[1.1], yticklabels=["Qwen3-8B"])
            axes[1].set_xlabel("HYB-B − N pass-rate difference\nPercentage points (95% CI)", color=theme["primary"], labelpad=12)
            axes[1].axvline(-3, color=theme["orange"], lw=1.4, ls="--", zorder=2)
            axes[1].text(-3, -.16, "−3 pp margin", ha="center", va="bottom", fontsize=10,
                         color=theme["orange"], bbox=dict(facecolor=theme["background"], edgecolor="none", pad=2))
            for point in points:
                left = point["panel"] == "labels"
                axis, color = (axes[0], theme["blue"]) if left else (axes[1], theme["orange"])
                center = (1.8 if point["model"] == "Qwen3-8B" else .4) if left else 1.1
                before = point["phase"] == "before"
                y = center + (.25 if before else -.25)
                pale_base = theme["primary"] if suffix else theme["background"]
                faded = tuple(.48 * a + .52 * b for a, b in
                              zip(matplotlib.colors.to_rgb(color), matplotlib.colors.to_rgb(pale_base)))
                line_color = faded if before else color
                axis.plot([point["ci_low"], point["ci_high"]], [y, y], color=line_color, lw=1.4, zorder=3)
                axis.plot(point["estimate"], y, "o", markersize=6, markeredgewidth=1.5,
                          markeredgecolor=line_color, markerfacecolor=theme["background"] if before else color, zorder=4)
                digits = 4 if left else 2
                label = f"{point['estimate']:+.{digits}f} [{point['ci_low']:.{digits}f}, {point['ci_high']:.{digits}f}]"
                axis.annotate(label.replace("-", "−"), (point["estimate"], y), xytext=(0, 8 if before else -9),
                              textcoords="offset points", ha="center", va="bottom" if before else "top",
                              color=theme["secondary"], fontsize=10, zorder=5,
                              bbox=dict(facecolor=theme["background"], edgecolor="none", pad=.5))
            handles = [Line2D([], [], color=theme["muted"], marker="o", markerfacecolor=theme["background"],
                              markeredgewidth=1.5, lw=1.4, label="Before"),
                       Line2D([], [], color=theme["primary"], marker="o", lw=1.4, label="After")]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.58, .99), ncol=2,
                       frameon=False, labelcolor=theme["secondary"], fontsize=11)
            save_png(fig, output, "turning_points" + suffix, theme)


def render_signal_quality(points, output):
    bound = math.ceil(max(abs(p[k]) for p in points for k in ("ci_low", "ci_high")) / .05) * .05
    titles = ("Running the draft\non public examples", "Model's own\nconfidence")
    subtitles = ("vs. metadata baseline ·\nindependent hidden-test label",
                 "vs. metadata baseline ·\noriginal label")
    for suffix, theme in THEMES.items():
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "axes.unicode_minus": True}):
            fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), dpi=200, sharey=True)
            fig.patch.set_facecolor(theme["background"])
            fig.subplots_adjust(left=.21, right=.975, bottom=.22, top=.68, wspace=.19)
            labels = []
            for axis, color, column in zip(axes, (theme["blue"], theme["orange"]), (points[:3], points[3:])):
                axis.set_facecolor(theme["background"])
                axis.set_xlim(-bound, bound)
                axis.set_ylim(-.48, 2.75)
                axis.set_xticks([-.2, 0, .2], ["−0.20", "0", "+0.20"])
                axis.grid(axis="x", color=theme["grid"], linewidth=.55)
                axis.axvline(0, color=theme["zero"], linewidth=.7, zorder=2)
                axis.set_axisbelow(True)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.tick_params(axis="both", length=0, colors=theme["secondary"], labelsize=11, pad=6)
                axis.set_xlabel("Change in AUROC (95% CI)", color=theme["primary"], fontsize=11, labelpad=8)
                for y, point in zip((2, 1, 0), column):
                    #Matplotlib的线宽和点大小以磅为单位，这里换算成200 dpi下2像素的线和8像素的点。
                    axis.plot([point["ci_low"], point["ci_high"]], [y, y], color=color,
                              linewidth=2 * 72 / 200, solid_capstyle="round", zorder=3)
                    axis.plot(point["estimate"], y, "o", color=color, markersize=8 * 72 / 200,
                              markeredgewidth=0, zorder=4)
                    text = f"{point['estimate']:+.2f} [{point['ci_low']:.2f}, {point['ci_high']:.2f}]"
                    label = axis.annotate(text.replace("-", "−"), (point["estimate"], y), xytext=(0, 6),
                                          textcoords="offset points", ha="center", va="bottom",
                                          color=theme["secondary"], fontsize=11, zorder=5,
                                          bbox={"facecolor": theme["background"], "edgecolor": "none", "pad": .5})
                    labels.append((axis, label))
            axes[0].set_yticks([2, 1, 0], ["Qwen3-8B", "DeepSeek-R1-\nDistill-Qwen-7B",
                                          "DeepSeek-R1-\nDistill-Llama-8B"])
            axes[0].tick_params(axis="y", colors=theme["primary"], pad=12)
            for axis, title, subtitle in zip(axes, titles, subtitles):
                center = (axis.get_position().x0 + axis.get_position().x1) / 2
                fig.text(center, .97, title, ha="center", va="top", color=theme["primary"],
                         fontsize=12, fontweight="normal", linespacing=1.1)
                fig.text(center, .805, subtitle, ha="center", va="top", color=theme["secondary"],
                         fontsize=11, linespacing=1.1)
            #先绘制一次拿到文字的实际宽度，再把超出栏边的数值标签挪回栏内。
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for axis, label in labels:
                box, area = label.get_window_extent(renderer), axis.get_window_extent(renderer)
                shift = max(0, area.x0 + 4 - box.x0) - max(0, box.x1 - area.x1 + 4)
                label.set_position((shift * 72 / fig.dpi, 6))
            save_png(fig, output, "signal_quality_3models" + suffix, theme)


def render_pareto(data, output):
    markers = ("s", "s", "D", "o", "o", "^")
    subtitles = ("pre-registered primary\n(corrected labels)", "registered secondary\n",
                 "registered replication\n(no registered margin)")
    for suffix, theme in THEMES.items():
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
            fig, axes = plt.subplots(1, 3, figsize=(17.4, 7.2), sharex=True, sharey=True)
            fig.patch.set_facecolor(theme["background"])
            fig.subplots_adjust(left=.055, right=.985, bottom=.18, top=.755, wspace=.09)
            fig.text(.055, .965, "All 43 frozen configurations: accuracy–cost trade-off",
                     color=theme["primary"], fontsize=20, fontweight="bold", va="top")
            handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=8,
                              label=name.replace(":", " ") + (" (offline)" if name.startswith("ORACLE") else ""))
                       for name, color, marker in zip(HIGHLIGHTS, theme["policies"], markers)]
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.047, .915),
                       ncol=6, frameon=False, labelcolor=theme["primary"], fontsize=11,
                       columnspacing=2.1, handletextpad=.6)
            for i, (axis, (model, rows), subtitle) in enumerate(zip(axes, data.items(), subtitles)):
                theme_axis(axis, theme)
                for row in rows:
                    x, y = row["token_saving_percent"], row["pass_difference_pp"]
                    axis.plot(row["token_saving_percent_ci95"], [y, y], color=theme["faint"], lw=.7, zorder=1)
                    axis.plot([x, x], row["pass_difference_pp_ci95"], color=theme["faint"], lw=.7, zorder=1)
                    axis.scatter(x, y, s=20, color=theme["muted"], alpha=.7, zorder=2)
                for name, color, marker in zip(HIGHLIGHTS, theme["policies"], markers):
                    row = select(rows, candidate=name)
                    x, y = row["token_saving_percent"], row["pass_difference_pp"]
                    axis.plot(row["token_saving_percent_ci95"], [y, y], color=color, lw=1.3, zorder=3)
                    axis.plot([x, x], row["pass_difference_pp_ci95"], color=color, lw=1.3, zorder=3)
                    axis.scatter(x, y, s=60, color=color, marker=marker, edgecolors=theme["background"],
                                 linewidth=.6, zorder=4)
                axis.axhline(0, color=theme["zero"], lw=.9)
                axis.axvline(0, color=theme["zero"], lw=.9)
                if i < 2:
                    axis.axhline(-3, color=theme["policies"][2], lw=1, ls="--", zorder=2)
                axis.set(xlim=(-40, 80), ylim=(-17, 11), xticks=[-40, -20, 0, 20, 40, 60, 80],
                         yticks=[-15, -10, -5, 0, 5, 10])
                axis.set_xlabel("Generated-token saving vs N (%) →", color=theme["primary"], labelpad=11)
                axis.set_title(f"{model}\n{subtitle}", color=theme["primary"], fontsize=12.5, pad=15)
            axes[0].set_ylabel("Pass-rate difference vs N (percentage points) ↑", color=theme["primary"], labelpad=10)
            fig.text(.055, .045,
                     "Dashed line: −3 pp non-inferiority margin for HYB-B vs N; the 95% CI lower bound must exceed it.",
                     color=theme["secondary"], fontsize=10.5, va="top")
            save_png(fig, output, "pareto_43" + suffix, theme)


def render_public_check_risk(data, output):
    for suffix, theme in THEMES.items():
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
            fig, axes = plt.subplots(1, 2, figsize=(16.8, 7.6))
            fig.patch.set_facecolor(theme["background"])
            fig.subplots_adjust(left=.155, right=.975, bottom=.20, top=.72, wspace=.52)
            fig.text(.04, .965, "Public checks: accepted drafts and observed stopping risk",
                     color=theme["primary"], fontsize=20, fontweight="bold", va="top")
            for axis in axes:
                theme_axis(axis, theme)
                axis.grid(axis="y", visible=False)
                axis.set_ylim(-1.1, 7)
                axis.set_yticks((6, 3, 0))
            axes[0].set(xlim=(0, 104), xticks=[0, 25, 50, 75, 100])
            axes[0].set_xlabel("Question-weighted percentage (95% CI)", color=theme["primary"], labelpad=10)
            axes[1].set(xlim=(.42, 1.02), xticks=[.5, .6, .7, .8, .9, 1.0])
            axes[1].axvline(.5, color=theme["zero"], ls=":", lw=1.2)
            axes[1].set_xlabel("AUROC for harm on the target model (95% CI)", color=theme["primary"], labelpad=10)
            left_labels = []
            for center, row in zip((6, 3, 0), data["models"]):
                left_labels.append(f"{row['model']}\n{row['public_pass_drafts']:,} accepted · {row['public_pass_questions']} questions")
                for offset, key, color, marker in ((.42, "recall", theme["blue"], "o"),
                                                   (-.42, "wrong_among_accepted", theme["orange"], "s")):
                    point = row[key]
                    value, lo, hi = point["display_percent"]
                    y = center + offset
                    axes[0].plot([lo, hi], [y, y], color=color, lw=1.5, solid_capstyle="round")
                    axes[0].plot(value, y, marker, color=color, markersize=6)
                    axes[0].annotate(point["formatted"], (value, y), xytext=(0, 8), textcoords="offset points",
                                     ha="right" if value > 75 else "center", va="bottom", fontsize=10,
                                     color=theme["secondary"])
                for offset, key, color, marker in ((.66, "overall", theme["blue"], "o"),
                                                   (0, "leetcode", theme["orange"], "s")):
                    point, y = row[key], center + offset
                    axes[1].plot(point["ci95"], [y, y], color=color, lw=1.5, solid_capstyle="round")
                    axes[1].plot(point["value"], y, marker, color=color, markersize=6)
                    axes[1].text(1, y + .18, point["formatted"], ha="right", va="bottom", fontsize=10,
                                 color=theme["secondary"])
                point, y = row["collection_only"], center - .66
                axes[1].plot(point["value"], y, "D", color=theme["secondary"], markersize=5)
                axes[1].annotate(point["formatted"], (point["value"], y), xytext=(8, 0),
                                 textcoords="offset points", va="center", color=theme["secondary"], fontsize=10)
            axes[0].set_yticklabels(left_labels, color=theme["primary"], fontsize=10.5, linespacing=1.6)
            axes[1].set_yticklabels([row["model"] for row in data["models"]], color=theme["primary"], fontsize=10.5)
            titles = ("What does the public check accept?", "Can other models' miss rates distinguish harm?")
            subtitles = ("Independent hidden-test labels · Qwen/DS post hoc · Llama descriptive",
                         "Other two models' scores · exploratory, post hoc")
            legends = (
                [(theme["blue"], "o", "Correct drafts accepted"),
                 (theme["orange"], "s", "Incorrect among accepted drafts")],
                [(theme["blue"], "o", "All eligible questions"), (theme["orange"], "s", "LeetCode only"),
                 (theme["secondary"], "D", "Collection identity only (overall)")],
            )
            for x, axis, title, subtitle, items in zip((.04, .55), axes, titles, subtitles, legends):
                fig.text(x, .895, title, color=theme["primary"], fontsize=14, fontweight="bold", va="top")
                fig.text(x, .857, subtitle, color=theme["secondary"], fontsize=11, va="top")
                handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=6, label=label)
                           for c, m, label in items]
                fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x - .006, .827),
                           frameon=False, labelcolor=theme["secondary"], fontsize=10,
                           ncol=1, borderaxespad=0, labelspacing=.4, handletextpad=.4)
            fig.text(.04, .075,
                     "Left: question-weighted rates.\n"
                     "Right: labels show AUROC [95% CI] and harmed / eligible questions; collection-only markers have no saved interval.",
                     color=theme["secondary"], fontsize=10.5, linespacing=1.5, va="top")
            save_png(fig, output, "public_check_risk" + suffix, theme)


def load_report_evidence(pareto, pareto_sources):
    """Read the report additions and validate the agreed display precision."""
    data = {"c2_nested_comparison": [], "signals_after_public_pass": [], "safety_by_collection": []}
    nested_sources, conditional_sources, safety_sources = [], [], []
    expected_nested = (("0.9313", "0.1268", "0.0960", "0.1620"),
                       ("0.9590", "0.1433", "0.1039", "0.1887"),
                       ("0.9349", "0.1628", "0.1188", "0.2127"))
    expected_conditional = (("0.737", "0.612", "0.782", "0.046", "-0.124", "-0.206", "-0.041", "0.945"),
                            ("0.735", "0.665", "0.796", "0.062", "-0.070", "-0.151", "0.014", "0.980"),
                            ("0.664", "0.602", "0.836", "0.172", "-0.062", "-0.158", "0.035", "0.969"))
    original_counts = (((262, 706), (504, 1022), (96, 278)),
                       ((221, 563), (306, 672), (120, 218)))
    harm_sources = (QWEN_SOURCE, DS_SOURCE, LLAMA_STATS_SOURCE)
    for i, (key, model_source, harm_path) in enumerate(zip(("qwen", "ds", "llama"), C2_SOURCES, harm_sources)):
        name, _, _, _, path = model_source
        saved = read(path)
        common = dict(corpus="virtual_v1_pipeline", scope="complete_deliverable", label="PRIV-IND")
        pub = select(saved["comparisons"], population="pooled", comparison="PUB", signal_name="PUB", **common)
        nested = pub["paired_differences"]["baseline_plus_signal_minus_baseline"]["auroc"]
        values = (pub["baseline_plus_signal"]["auroc"], nested["difference"], nested["ci"]["lower"], nested["ci"]["upper"])
        for value, expected in zip(values, expected_nested[i]):
            check_display(value, expected, 4, f"{name} nested PUB comparison")
        data["c2_nested_comparison"].append(dict(model=name, auroc=values[0], gain=nested,
                                                 recall=pub["signal_metrics"]["recall"],
                                                 pub_minus_metadata=pub["paired_differences"]["signal_minus_baseline"]["auroc"]))
        nested_sources.append(source_record(path, "comparisons[pooled, PUB, PRIV-IND, virtual_v1_pipeline, complete_deliverable]",
                              ["baseline_plus_signal.auroc", "paired_differences.baseline_plus_signal_minus_baseline.auroc",
                               "signal_metrics.recall"]))
        conf = select(saved["comparisons"], population="pooled", comparison="PUB_PASS_CONF50", signal_name="CONF-50", **common)
        probe_path = f"runs/probe_execution/fit/{key}/heldout/summary.json"
        probes = read(probe_path)["comparisons"]
        probe = select(probes, population="pooled", position="draft_end", comparison="probe_minus_baseline_pub_pass", **common)
        combo = select(probes, population="pooled", position="draft_end", comparison="probe_pub_minus_pub", **common)
        if probe["status"] != "COMPLETE" or conf["status"] != "COMPLETE":
            raise ValueError(f"STOP: Incomplete conditional signal result for {name}")
        if probe["reference_auroc"] != conf["baseline"]["auroc"]:
            raise ValueError(f"STOP: Conditional metadata baselines differ for {name}")
        if (probe["drafts"], probe["questions"]) != (conf["matched_rows"], conf["matched_questions"]):
            raise ValueError(f"STOP: Conditional populations differ for {name}")
        diff = conf["paired_differences"]["signal_minus_baseline"]["auroc"]
        values = (conf["baseline"]["auroc"], conf["signal_metrics"]["auroc"], probe["candidate_auroc"],
                  probe["difference"], diff["difference"], diff["ci"]["lower"], diff["ci"]["upper"], combo["candidate_auroc"])
        for value, expected in zip(values, expected_conditional[i]):
            check_display(value, expected, 3, f"{name} conditional signals / combination")
        # The saved `ci` belongs to the paired contrast, not candidate_auroc.
        if any(k in probe for k in ("candidate_auroc_ci", "candidate_ci95", "candidate_ci")):
            raise ValueError("STOP: Probe absolute-interval schema changed; inspect the newly saved interval")
        data["signals_after_public_pass"].append(dict(
            model=name, metadata=saved_point(conf["baseline"]["auroc"], conf["baseline"]["auroc_ci"]),
            confidence=saved_point(conf["signal_metrics"]["auroc"], conf["signal_metrics"]["auroc_ci"]),
            probe=dict(value=probe["candidate_auroc"], ci95=None),
            confidence_minus_metadata=diff, probe_minus_metadata=dict(difference=probe["difference"], ci=probe["ci"]),
            probe_plus_pub=dict(auroc=combo["candidate_auroc"], difference=combo["difference"], ci=combo["ci"]),
            drafts=probe["drafts"], questions=probe["questions"]))
        conditional_sources.extend([
            source_record(path, "comparisons[population=pooled, comparison=PUB_PASS_CONF50, signal_name=CONF-50, PRIV-IND]",
                          ["baseline.auroc", "baseline.auroc_ci", "signal_metrics.auroc", "signal_metrics.auroc_ci",
                           "paired_differences.signal_minus_baseline.auroc", "matched_rows", "matched_questions"]),
            source_record(probe_path, "comparisons[population=pooled, position=draft_end, comparison=probe_minus_baseline_pub_pass or probe_pub_minus_pub]",
                          ["candidate_auroc", "reference_auroc", "difference", "ci", "drafts", "questions"]),
        ])
        harms = read(harm_path)
        for j, (collection, collection_name) in enumerate(COLLECTIONS):
            population = f"collection:{collection}"
            row = select(saved["comparisons"], population=population, comparison="PUB", signal_name="PUB", **common)
            metrics, coverage = row["signal_metrics"], row["population_coverage"]
            if metrics["weighting"] != WEIGHTING:
                raise ValueError(f"STOP: Wrong safety-figure weighting: {name} {population}")
            raw, accepted = metrics["raw_counts"], coverage["public_pass_drafts"]
            if raw["fp"] + raw["tp"] != accepted:
                raise ValueError("STOP: Saved accepted denominator differs from tp+fp")
            if i < 2 and (raw["fp"], accepted) != original_counts[i][j]:
                raise ValueError(f"STOP: PRIV-IND raw counts differ from original labels: {name} {population}")
            parent = select(harms, version="erratum", population=population) if i == 0 else select(harms["populations"], population=population)
            comparison = parent["comparisons"]["N"]
            if not comparison["complete"] or comparison["candidate"] != "HYB-B" or comparison["reference"] != "N":
                raise ValueError("STOP: Missing same-definition HYB-B harm result")
            strict = comparison["strict"]
            wrong = saved_point(1 - metrics["precision"], dict(lower=1 - metrics["precision_ci"]["upper"],
                                                               upper=1 - metrics["precision_ci"]["lower"]))
            data["safety_by_collection"].append(dict(model=name, key=key, population=population, collection=collection_name,
                harmed_questions=strict["harmed_questions"], questions=strict["questions"],
                pub_raw_counts=raw, pub_positive_drafts=accepted, pub_weighted_precision=metrics["precision"],
                wrong_among_accepted=wrong, display_error_percent=100 * wrong["value"],
                display_ci_percent=[100 * v for v in wrong["ci95"]], weighting=WEIGHTING, label="PRIV-IND"))
        safety_sources.extend([
            source_record(path, "comparisons[population=collection:*, comparison=PUB, signal_name=PUB, PRIV-IND]",
                          ["signal_metrics.precision", "signal_metrics.precision_ci", "signal_metrics.weighting",
                           "signal_metrics.raw_counts", "population_coverage.public_pass_drafts"]),
            source_record(harm_path, "[version=erratum, population=collection:*]" if i == 0 else "populations[population=collection:*]",
                          ["comparisons.N.strict.harmed_questions", "comparisons.N.strict.questions", "comparisons.N.strict.assumptions"]),
        ])
    data["llama_policy_results"] = [select(pareto["DS-R1-Llama-8B"], candidate=k) for k in ("HYB-B", "BF@4096", "CONF-50:k1:m0")]
    overall_harm = select(read(LLAMA_STATS_SOURCE)["populations"], population="pooled")["comparisons"]["N"]["strict"]
    data["llama_harm_overall"] = {k: overall_harm[k] for k in ("harmed_questions", "questions")}
    confidence_policy = data["llama_policy_results"][2]
    for value, expected in zip((confidence_policy["pass_difference_pp"], *confidence_policy["pass_difference_pp_ci95"]), ("-6.2", "-8.5", "-3.9")):
        check_display(value, expected, 1, "Llama CONF-50 policy")
    manifest = {
        "c2_nested_comparison": dict(sources=nested_sources, evidence_level="Qwen/DS post hoc; Llama descriptive frozen-code output, not R1"),
        "signals_after_public_pass": dict(sources=conditional_sources, outputs=["docs/execexit/figures/signals_after_public_pass.png"],
            evidence_level="H-P2: registered probe addendum; CONF-50 subset contrasts: Qwen/DS post hoc, Llama frozen-code output, not a registered criterion"),
        "safety_by_collection": dict(sources=safety_sources, outputs=["docs/execexit/figures/safety_by_collection.png"],
            evidence_level="Harm: Qwen/DS registered secondary, Llama descriptive; PRIV-IND error rates: Qwen/DS post hoc, Llama descriptive frozen-code output",
            fields={"error_rate": "1 - signal_metrics.precision", "error_ci": "[1 - precision_ci.upper, 1 - precision_ci.lower]",
                    "raw_label": "signal_metrics.raw_counts.fp / population_coverage.public_pass_drafts (=tp+fp)"}),
        "llama_policy_results": dict(sources=[source_record(PARETO_SOURCES[2][1],
            "populations[population=pooled].comparisons[reference=N, candidate=HYB-B/BF@4096/CONF-50:k1:m0]",
            ["pass_difference_pp", "pass_difference_pp_ci95", "token_saving_percent", "token_saving_percent_ci95"])],
            evidence_level="43 pointwise intervals output under Llama addendum; no registered HYB-B non-inferiority criterion"),
        "llama_harm_overall": dict(sources=[source_record(LLAMA_STATS_SOURCE, "populations[population=pooled].comparisons.N.strict",
            ["harmed_questions", "questions"])], evidence_level="Descriptive same-definition ANY-harm count; no registered Llama HYB-B NI test"),
    }
    manifest["safety_by_collection"]["copied_saved_source"] = dict(
        path=LLAMA_STATS_SOURCE, sha256=hashlib.sha256((ROOT / LLAMA_STATS_SOURCE).read_bytes()).hexdigest())
    data["main_results"] = dict(
        c2=[dict(model=r["model"], contrast=r["pub_minus_metadata"]) for r in data["c2_nested_comparison"]],
        c3=[dict(model=name, **{k: select(pareto[name], candidate="HYB-B")[k]
                               for k in ("pass_difference_pp", "pass_difference_pp_ci95")}) for name, _ in PARETO_SOURCES],
        c4_main=read(C4_SOURCE)["primary"], c4_sensitivity=read(SENSITIVITY_SOURCE)["sensitivity"],
        margin_pp=-3, margin_models=["Qwen3-8B", "DS-R1-Qwen-7B"])
    manifest["main_results"] = dict(
        outputs=["docs/execexit/figures/main_results.svg"],
        sources=[source_record(p, "comparisons[pooled PUB PRIV-IND]", ["paired_differences.signal_minus_baseline.auroc"])
                 for *_, p in C2_SOURCES]
                + [source_record(QWEN_SOURCE, "[version=erratum, population=pooled].comparisons.N",
                                 ["pass_difference_pp", "pass_difference_pp_ci95"]),
                   source_record(DS_SOURCE, "populations[population=pooled].comparisons.N",
                                 ["pass_difference_pp", "pass_difference_pp_ci95"]),
                   source_record(PARETO_SOURCES[2][1], "populations[population=pooled].comparisons[reference=N, candidate=HYB-B]",
                                 ["pass_difference_pp", "pass_difference_pp_ci95"])]
                + [source_record(C4_SOURCE, "primary", ["ratio_of_means", "ci95_percentile", "pairs"]),
                   source_record(SENSITIVITY_SOURCE, "sensitivity", ["ratio_of_means", "ci95_percentile", "pairs"]),
                   select(pareto_sources, path="runs/report_figures/source/stats.py")],
        evidence_level="C2: Qwen/DS post hoc, Llama registered R1 replication; C3: Qwen primary, DS registered secondary, Llama registered replication without NI margin; C4: registered secondary / post-freeze sensitivity")
    return data, manifest


def render_safety(rows, output):
    theme = THEMES[""]
    colors = (theme["blue"], theme["orange"], "#31816C")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.9), sharey=True)
        fig.subplots_adjust(left=.09, right=.98, top=.72, bottom=.22, wspace=.27)
        fig.text(.05, .96, "Policy harm and errors among public-check passes", fontsize=19, weight="bold", va="top")
        handles = [Line2D([], [], color=c, marker="o", ls="none", label=model[0]) for c, model in zip(colors, C2_SOURCES)]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.045, .91), ncol=3, frameon=False, fontsize=11)
        for ax in axes:
            theme_axis(ax, theme)
            ax.grid(axis="y", visible=False)
            ax.set_ylim(-1.05, 7.05)
            ax.set_yticks((6, 3, 0), [v for _, v in COLLECTIONS])
        axes[0].set(xlim=(0, 13), xticks=[0, 3, 6, 9, 12])
        axes[1].set(xlim=(0, 72), xticks=[0, 20, 40, 60])
        axes[0].set_title("HYB-B harmed questions\nQwen/DS registered · Llama descriptive", loc="left", fontsize=12, pad=16)
        axes[1].set_title("Incorrect drafts among PUB passes\nPRIV-IND · Qwen/DS post hoc · Llama descriptive", loc="left", fontsize=12, pad=16)
        axes[0].set_xlabel("Harmed questions (labels: harmed / all questions)", labelpad=10)
        axes[1].set_xlabel("Question-weighted incorrect fraction (%) · 95% CI", labelpad=10)
        for center, (collection, _) in zip((6, 3, 0), COLLECTIONS):
            for offset, model, color in zip((.65, 0, -.65), C2_SOURCES, colors):
                r = select(rows, model=model[0], population=f"collection:{collection}")
                y = center + offset
                axes[0].barh(y, r["harmed_questions"], height=.33, color=color)
                axes[0].text(r["harmed_questions"] + .18, y, f"{r['harmed_questions']}/{r['questions']}", va="center", fontsize=10, color=color)
                point = r["display_error_percent"]
                axes[1].plot(r["display_ci_percent"], [y, y], color=color, lw=1.5)
                axes[1].plot(point, y, "o", color=color, ms=5)
                axes[1].text(point, y + .17, f"{point:.1f}% · {r['pub_raw_counts']['fp']}/{r['pub_positive_drafts']}",
                             ha="center", va="bottom", fontsize=9.5, color=theme["secondary"])
        fig.text(.05, .075,
                 "Left: any paired N-pass / HYB-B-fail sample; right: question-weighted 1 − precision on PRIV-IND with 95% CIs.\n"
                 "Raw fp/(tp+fp) labels on the right are unchanged from the original-label counts for Qwen and DS.",
                 color=theme["secondary"], fontsize=10, linespacing=1.5, va="top")
        save_png(fig, output, "safety_by_collection", theme)


def render_conditional_signals(rows, output):
    theme = THEMES[""]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharex=True, sharey=True)
        fig.subplots_adjust(left=.13, right=.98, top=.69, bottom=.27, wspace=.12)
        fig.text(.05, .96, "Which signals distinguish correctness after a public-check pass?", fontsize=18, weight="bold", va="top")
        fig.text(.05, .885, "PUB-passing drafts · independent PRIV-IND labels", fontsize=12, color=theme["secondary"])
        for i, (ax, row) in enumerate(zip(axes, rows)):
            theme_axis(ax, theme)
            ax.grid(axis="y", visible=False)
            ax.set(xlim=(.50, .91), ylim=(-.42, 2.55), xticks=[.5, .6, .7, .8, .9], yticks=[2, 1, 0])
            ax.axvline(.5, color=theme["zero"], lw=1, ls=":")
            ax.set_yticklabels(["Metadata", "CONF-50", "Draft-end probe"])
            status = "CONF post hoc" if i < 2 else "CONF descriptive"
            ax.set_title(f"{row['model']}\n{status} · H-P2 registered", fontsize=11.5, pad=18)
            ax.set_xlabel("Absolute AUROC", labelpad=9)
            for y, key, color, marker in ((2, "metadata", theme["blue"], "o"), (1, "confidence", theme["orange"], "s"),
                                          (0, "probe", "#31816C", "D")):
                p = row[key]
                if p["ci95"] is not None:
                    ax.plot(p["ci95"], [y, y], color=color, lw=1.5)
                ax.plot(p["value"], y, marker, color=color, ms=6)
                text = f"{p['value']:.3f}" if p["ci95"] is None else f"{p['value']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]"
                ax.text(.895, y + .17, text, fontsize=10, ha="right", va="bottom", color=theme["secondary"])
        fig.text(.05, .115,
                 "Metadata and CONF-50 show saved 95% CIs; probe points have no saved absolute-AUROC interval.\n"
                 "Only Qwen's paired CONF-50 − metadata CI excludes zero; Llama's larger H-P2 contrast accompanies the lowest metadata baseline and similar probe points.",
                 color=theme["secondary"], fontsize=9.5, linespacing=1.5, va="top")
        save_png(fig, output, "signals_after_public_pass", theme)


def update_figure_metadata(pareto, pareto_sources, risk, risk_sources, report_data, report_sources, turning_points):
    sources_path = ROOT / "runs/report_figures/sources.json"
    data_path = ROOT / "runs/report_figures/raw/plot_data.json"
    sources, data = json.loads(sources_path.read_text()), json.loads(data_path.read_text())
    sources["pareto_43"] = dict(
        outputs=["docs/execexit/figures/pareto_43.png", "docs/execexit/figures/pareto_43_dark.png"],
        sources=pareto_sources,
        selector="pooled; reference=N; all 43 configurations in each of three models",
        fields=["pass_difference_pp", "pass_difference_pp_ci95", "token_saving_percent", "token_saving_percent_ci95"],
        evidence_level={"Qwen3-8B HYB-B vs N": "pre-registered primary, corrected labels",
                        "Qwen3-8B other configurations": "pre-registered secondary, pointwise nominal intervals",
                        "DS-R1-Qwen-7B": "registered secondary at original freeze",
                        "DS-R1-Llama-8B": "registered replication; registered secondary policy outputs; no registered HYB-B margin"},
        fields_for_annotations={"configurations": "Llama source: configurations (43), matched by both JSONL selections",
                                "questions_per_model": "Llama populations[population=pooled].questions; Qwen/DS saved statistics questions (199)",
                                "other_configurations": "43 configurations excluding HYB-B (42)",
                                "nominal_ci_level": "Saved *_ci95 fields (95%)",
                                "margin_pp": "runs/report_figures/source/stats.py::summarize.nominal_noninferiority; pass_difference_pp_ci95[0] > -3; Qwen/DS HYB-B vs N only",
                                "intervals": "Saved pass_difference_pp_ci95 and token_saving_percent_ci95"},
    )
    sources["public_check_risk"] = dict(
        outputs=["docs/execexit/figures/public_check_risk.png", "docs/execexit/figures/public_check_risk_dark.png"],
        sources=risk_sources,
        selector="Left: pooled PUB, known PRIV-IND, virtual_v1_pipeline, complete_deliverable; right: qwen/ds/llama M3 and saved LeetCode follow-up",
        fields={"recall": "signal_metrics.recall and recall_ci.lower/upper",
                "wrong_among_accepted": "1 - signal_metrics.precision; bounds = [1 - precision_ci.upper, 1 - precision_ci.lower]",
                "percentage_units": "100 times the selected saved rate or its complement; display transform only",
                "accepted_denominators": "population_coverage.public_pass_drafts and population_coverage.public_pass_questions (separate counts)",
                "unknown_excluded": "comparisons[pooled PUB].excluded_missing_rows",
                "overall_risk": "M3_harm_prediction.{model}.auroc, ci95.lower/upper, harmed, problems",
                "leetcode_risk": "results.{model}.leetcode_auroc, leetcode_ci95.lower/upper, leetcode_harmed, leetcode_problems",
                "collection_reference": "results.{model}.collection_only_auroc; no saved interval",
                "primary_harm_caption": "Qwen erratum pooled comparisons.N.strict.harmed_questions/questions",
                "chance_reference": "AUROC 0.5 by definition; reference line, not an estimated result"},
        evidence_level={"left Qwen/DS": "post hoc after completion of PRIV-IND labels",
                        "left Llama": "descriptive frozen-code output; registered R1 is AUROC, not precision/recall",
                        "right M3": "exploratory, post hoc on existing questions",
                        "right LeetCode and collection identity": "post-hoc addition after seeing M3"},
        display_text={"left_subtitle": "Independent hidden-test labels · Qwen/DS post hoc · Llama descriptive",
                      "accepted_counts": "{public_pass_drafts:,} accepted · {public_pass_questions} questions",
                      "left_caption": "Left: question-weighted rates."},
    )
    data["pareto_43"] = pareto
    data["pareto_43_display"] = dict(configurations=43, questions_per_model=199, highlights=list(HIGHLIGHTS),
                                    margin_pp=-3, margin_models=["Qwen3-8B", "DS-R1-Qwen-7B"],
                                    other_configurations=42, nominal_ci_percent=95,
                                    criterion="pass_difference_pp_ci95[0] > -3")
    data["public_check_risk"] = dict(risk, chance_reference=.5, nominal_ci_percent=95)
    sources.update(report_sources)
    data.update(report_data)
    for path, obj in ((sources_path, sources), (data_path, data)):
        serialized = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if path.read_text() != serialized:
            path.write_text(serialized)
    values_path = ROOT / README_FIGURE_SOURCE
    values = [point for point in read(README_FIGURE_SOURCE) if point.get("figure") != "turning_points"]
    values.append(dict(figure="turning_points",
                       outputs=["docs/execexit/figures/turning_points.png", "docs/execexit/figures/turning_points_dark.png"],
                       points=turning_points))
    serialized = "[\n" + ",\n".join(json.dumps(point, ensure_ascii=False, allow_nan=False) for point in values) + "\n]\n"
    if values_path.read_text() != serialized:
        values_path.write_text(serialized)


def render(output):
    # Validate all requested saved values before replacing any figure.
    turning_points = load_turning_points()
    pareto, pareto_sources = load_pareto()
    risk, risk_sources = load_public_check_risk()
    report_data, report_sources = load_report_evidence(pareto, pareto_sources)
    signal_quality = load_signal_quality()
    render_main(output, pareto)
    render_signal_quality(signal_quality, output)
    render_pareto(pareto, output)
    render_public_check_risk(risk, output)
    render_safety(report_data["safety_by_collection"], output)
    render_conditional_signals(report_data["signals_after_public_pass"], output)
    render_turning_points(turning_points, output)
    if output.resolve() == (ROOT / "docs/execexit/figures").resolve():
        update_figure_metadata(pareto, pareto_sources, risk, risk_sources, report_data, report_sources, turning_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "figures")
    render(parser.parse_args().output_dir)
