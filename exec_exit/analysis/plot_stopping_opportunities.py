"""One measured opportunity figure; absence/UNKNOWN never become zero values."""
from __future__ import annotations

from collections import Counter
import time
from pathlib import Path

IMPLEMENTED = True


def partition_rows(rows):
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Each planned task must appear exactly once")
    observed, absent = [], []
    for row in rows:
        position = row["first_full_pass_probe_position"]
        if position is None:
            absent.append(row)
        elif not 0 < position <= row["native_thinking_generated_tokens"]:
            raise ValueError("Observed qualifying position must lie inside its native thinking prefix")
        else:
            if row["opportunity_status"] != "OBSERVED_FULL_PASS":
                raise ValueError("A plotted success requires an observed complete pass")
            observed.append(row)
    return observed, absent


def render(rows, output_dir, *, measured_candidates):
    wall, cpu = time.perf_counter(), time.process_time()
    observed, absent = partition_rows(rows)
    if sum(row["measured_candidates"] for row in rows) != measured_candidates:
        raise ValueError("Figure candidate count differs from measured rows")
    # Rendering happens only after the replay's GPU-drain check. No illustrative
    # values, interpolation, or resampling enter either panel.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axis, missing) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True,
                                      gridspec_kw={"height_ratios": [5, 1], "hspace": .12})
    maximum = max([row["native_thinking_generated_tokens"] for row in rows] + [1])
    if observed:
        axis.scatter([row["native_thinking_generated_tokens"] for row in observed],
                     [row["first_full_pass_probe_position"] for row in observed],
                     s=38, color="#2468a2", alpha=.85, label=f"Observed full-pass probe ({len(observed)})")
    axis.plot([0, maximum*1.03], [0, maximum*1.03], color="#888888", linewidth=1, linestyle="--", label="Same token position")
    axis.set(xlim=(0, maximum*1.04), ylim=(0, maximum*1.04),
             ylabel="First full-test-pass probe position\n(generated tokens, prompt excluded)")
    axis.grid(alpha=.18)
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    axis.set_title(f"Measured stopping opportunities: {len(rows)} Dev tasks, {measured_candidates} candidates", loc="left", fontsize=12)
    types = ("NO_OBSERVED_FULL_PASS", "UNKNOWN_INCOMPLETE_TEST_COVERAGE", "NO_CANDIDATES")
    styles = {"NO_OBSERVED_FULL_PASS": ("x", "#444444", "No observed pass"),
              "UNKNOWN_INCOMPLETE_TEST_COVERAGE": ("?", "#a86b1b", "Incomplete test coverage"),
              "NO_CANDIDATES": ("|", "#888888", "No candidate")}
    active_types = [kind for kind in types if any(row["opportunity_status"] == kind for row in absent)]
    for y, kind in enumerate(active_types):
        subset = [row for row in absent if row["opportunity_status"] == kind]
        marker, color, label = styles[kind]
        marker = "$?$" if marker == "?" else marker
        missing.scatter([row["native_thinking_generated_tokens"] for row in subset], [y]*len(subset),
                        marker=marker, color=color, s=36)
    missing.set_yticks(range(len(active_types)), [styles[kind][2] for kind in active_types], fontsize=8)
    missing.set_ylim(-.6, max(len(active_types)-1, 0)+.6)
    missing.set_xlabel("Native thinking length before generated </think> (tokens)")
    missing.grid(axis="x", alpha=.18)
    if not absent:
        missing.text(.02, .5, "All tasks have an observed qualifying probe", transform=missing.transAxes, fontsize=9)
        missing.set_yticks([])
    fig.text(.09, .025,
             "Absent/UNKNOWN tasks are retained in the lower panel; they are not assigned a zero position.\n"
             "Only measured drafts and test suites are covered. Position gaps exclude draft/check costs; they are not net savings.",
             fontsize=8, color="#444444")
    fig.subplots_adjust(left=.20, right=.97, bottom=.15, top=.91)
    output_dir = Path(output_dir)
    png, pdf = output_dir / "first_full_pass_opportunities.png", output_dir / "first_full_pass_opportunities.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return dict(png=str(png), pdf=str(pdf), planned_tasks=len(rows), observed_positions=len(observed),
                matplotlib_version=matplotlib.__version__,
                nonposition_status_counts=dict(Counter(row["opportunity_status"] for row in absent)),
                measured_candidates=measured_candidates, wall_seconds=time.perf_counter()-wall,
                cpu_seconds=time.process_time()-cpu,
                interpretation="Raw position gap is not net benefit; draft/control/check costs are reported in replay records")
