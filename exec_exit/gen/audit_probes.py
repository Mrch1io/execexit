"""CPU-only integrity/throughput audit of a completed probe run; never generate.

Refuses a live or unsuccessful GPU run before opening token records. Inputs are
read-only; reports go to a fresh standard run directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def supervisor_validation_boundary(summary, supervision):
    policy = summary["supervisor_deadline_policy"]
    deadline = policy["startup_deadline_epoch"]
    terminal = policy["terminal_first_observed_epoch"]
    old_threshold = deadline + policy["phase_grace_s"]
    return dict(normal_gpu_process_exit=summary["worker_returncode"] == 0 and not summary["supervisor_timeout"],
        terminal_progress_observed_by_supervisor=terminal is not None,
        original_startup_deadline_epoch=deadline, original_timeout_threshold_epoch=old_threshold,
        terminal_first_observed_epoch=terminal,
        expired_startup_deadline_at_terminal_observation=(terminal > deadline) if terminal is not None else "NOT_OBSERVED",
        expired_original_timeout_at_terminal_observation=(terminal > old_threshold) if terminal is not None else "NOT_OBSERVED",
        supervisor_ended_before_original_startup_deadline=supervision["ended_epoch"] < deadline,
        cpu_expired_deadline_regression="runs/supervisor_fix/summary.json",
        boundary="A normal GPU exit validates this actual completion path. It validates the expired-startup-deadline failure precondition only if terminal progress was observed after that deadline (and the old timeout threshold). Otherwise that branch remains covered by the prior CPU regression; the GPU run was not lengthened to manufacture it.")


def audit(source, destination):
    source, destination = source.resolve(), destination.resolve()
    summary, supervision = read(source / "summary.json"), read(source / "supervisor.json")
    if not (summary["status"] == supervision["status"] == "COMPLETE"
            and summary["worker_returncode"] == supervision["worker_returncode"] == 0
            and summary["supervisor_timeout"] is False and summary["native_frontend_drained"] is True):
        raise RuntimeError("A completed, normally exited, drained GPU run is required before reading token records")
    destination.mkdir()
    (destination / "raw").mkdir()
    started, usage0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    config, manifest = read(source / "config.yaml"), read(source / "manifest.json")
    root = Path(__file__).resolve().parents[2]
    audit_config = dict(source_run=str(source), source_config_sha256=digest(source / "config.yaml"),
        source_summary_sha256=digest(source / "summary.json"), audit_source=str(Path(__file__).resolve()),
        audit_source_sha256=digest(__file__), gpu_used=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=["python", "-m", "exec_exit.gen.audit_probes", "--run-dir", str(source), "--output-dir", str(destination)])
    write(destination / "config.yaml", audit_config)
    (destination / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True))
    checks, records, call_records = 0, [], {}

    def require(ok, detail):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(detail)

    result = dict(status="FAILED", gpu_used=False, gpu_seconds=0, source_run_status=summary["status"])
    try:
        for path, expected in config["source_sha256"].items():
            require(digest(path) == expected, "Frozen source/input changed: " + path)
        require(config["concurrency"] == 8, "Expected concurrency 8")
        source_rows = list(rows(Path(config["trajectory_dir"]) / "raw/trajectories.jsonl"))
        annotations = list(rows(Path(config["annotations_dir"]) / "raw/annotations.jsonl"))
        expected = []
        for trajectory, annotation in zip(source_rows, annotations):
            require(trajectory["case_id"] == annotation["case_id"], "Source question ordering")
            thinking = next(call["token_ids"] for call in trajectory["calls"] if call["stage"] == 0)
            require(token_digest(thinking) == annotation["stage0_token_ids_sha256"], "Source thinking hash")
            for candidate in annotation["candidates"]:
                expected.append((trajectory, thinking, candidate))
        require(len(source_rows) == len(annotations) == 36 and len(expected) == 386, "Frozen denominator")
        require(len(manifest["probes"]) == 386, "Manifest denominator")
        submissions = list(rows(source / "raw/submissions.jsonl"))
        require(len(submissions) == 386, "Submission denominator")
        totals = Counter()
        ids = []
        for index, (row, entry, source_info) in enumerate(zip(rows(source / "raw/probes.jsonl"), manifest["probes"], expected)):
            trajectory, thinking, candidate = source_info
            identity, call = row["probe_id"], row["call"]
            position = candidate["position_generated_tokens"]
            expected_id = trajectory["case_id"].replace(":", "_") + f".p{position:05d}"
            require(identity == expected_id == entry["probe_id"] == submissions[index]["probe_id"], "Candidate/submission ordering")
            require(row["candidate_index"] == entry["candidate_index"] == submissions[index]["candidate_index"] == index, "Candidate index")
            require(row["case_id"] == trajectory["case_id"] and row["position_generated_tokens"] == position, "Candidate coordinates")
            require(row["status"] == entry["status"] == "COMPLETE", "Unfinished checkpoint")
            require(row["config_identity"] == submissions[index]["config_identity"] == config["config_identity"], "Config identity")
            checkpoint = source / entry["checkpoint"]
            require(digest(checkpoint) == entry["checkpoint_sha256"] and read(checkpoint) == row, "Checkpoint/aggregate mismatch")
            prompt = trajectory["prompt_token_ids"] + thinking[:position] + config["control_token_ids"]
            require(call["prompt_token_ids"] == prompt and row["control_token_ids"] == config["control_token_ids"], "Exact prompt/control IDs")
            require(token_digest(prompt) == row["prompt_token_ids_sha256"] and call["prompt_tokens"] == len(prompt), "Prompt identity/count")
            seed = int(hashlib.sha256(f"reasoning_capacity_v1:{row['case_id']}:0:1".encode()).hexdigest()[:8], 16)
            require(call["seed"] == entry["seed"] == seed, "Frozen request seed")
            params = call["sampling_parameters"]
            require(params["temperature"] == 0 and params["max_tokens"] == 2048 and params["logprobs"] == 1
                    and params["stop"] == ["```"] and params["include_stop_str_in_output"] is True, "Frozen generation protocol")
            token_ids, scores, probabilities, times = call["token_ids"], call["rank_one_logprobs"], call["logprobs"], call["token_observation_ns"]
            count = len(token_ids)
            require(0 < count <= 2048 and all(type(value) is int for value in token_ids), "Native token IDs/count")
            require(count == len(scores) == len(probabilities) == len(times) == call["token_count"] == entry["generated_tokens"], "Per-token alignment")
            rank_counts = []
            for token_id, score, probability in zip(token_ids, scores, probabilities):
                require(str(token_id) in probability, "Sampled token absent in logprobs")
                ranks = [item["logprob"] for item in probability.values() if item["rank"] == 1]
                require(bool(ranks) and all(value == score for value in ranks), "Rank-one values differ")
                require(all(math.isfinite(item["logprob"]) and item["logprob"] <= 0 for item in probability.values()), "Invalid native logprob")
                rank_counts.append(len(ranks))
            require(rank_counts == call["returned_rank_one_entry_counts"], "Returned tie counts")
            require([i for i, n in enumerate(rank_counts) if n > 1] == call["returned_rank_one_tie_positions_zero_based"], "Returned tie positions")
            require(all(a <= b for a, b in zip(times, times[1:])) and call["started_ns"] <= times[0] <= times[-1] <= call["ended_ns"], "Observed clock ordering")
            require(call["native_finished"] is True and call["error"] is None, "Native completion")
            require(call["finish_reason"] in ("stop", "length") and call["finish_reason"] == entry["finish_reason"] and call["stop_reason"] == entry["stop_reason"], "Finish reasons preserved")
            cached = call["num_cached_tokens"]
            require(type(cached) is int and 0 <= cached <= len(prompt), "Missing/invalid native cache count")
            totals.update(output=count, prompt=len(prompt), cached=cached, tie_positions=sum(n > 1 for n in rank_counts))
            totals["finish_" + call["finish_reason"]] += 1
            call_records[identity], ids = call, ids + [identity]
            records.append(dict(probe_id=identity, status="PASS", candidate_index=index,
                                prompt_tokens=len(prompt), cached_tokens=cached, output_tokens=count))
        require(len(ids) == len(set(ids)) == 386, "Aggregate denominator/uniqueness")
        require(sum(1 for _ in rows(source / "raw/probes.jsonl")) == 386, "No extra aggregate rows")
        # One streamed pass; compare native chunk content and times to checkpoints.
        offsets, text_parts, starts, ends = Counter(), {}, [], []
        stream_rows = 0
        for path in sorted((source / "raw").glob("stream_*.jsonl")):
            for event in rows(path):
                stream_rows += 1
                kind, identity = event["event"], event.get("probe_id")
                if kind == "probe_start":
                    starts.append(identity)
                    require(event["prompt_token_ids"] == call_records[identity]["prompt_token_ids"], "Stream prompt")
                elif kind == "probe_output":
                    call, offset, chunk = call_records[identity], offsets[identity], event["token_ids"]
                    span = slice(offset, offset + len(chunk))
                    require(event["token_offset"] == offset and chunk == call["token_ids"][span], "Stream token chunk")
                    require((event["logprobs"] or []) == call["logprobs"][span] and event["rank_one_logprobs"] == call["rank_one_logprobs"][span], "Stream logprobs")
                    require([event["native_frontend_receive_ns"]] * len(chunk) == call["token_observation_ns"][span], "Stream native receipt time")
                    text_parts.setdefault(identity, []).append(event["text"])
                    offsets[identity] += len(chunk)
                elif kind == "probe_end":
                    require(event["call"] == call_records[identity], "Stream final call")
                    ends.append(identity)
                elif kind != "worker_snapshot":
                    raise AssertionError("Unexpected stream event: " + kind)
        require(starts == ids and len(ends) == len(set(ends)) == 386 and set(ends) == set(ids), "Native start/end accounting")
        for identity, call in call_records.items():
            require(offsets[identity] == call["token_count"] and "".join(text_parts[identity]) == call["raw_text"], "Stream concatenation")
        require(totals["output"] == summary["generated_tokens_completed"] == summary["generated_tokens_observed_all_calls"], "Output ledger total")
        require(totals["prompt"] == summary["total_prompt_tokens_completed"] and totals["cached"] == summary["native_cached_tokens_completed"], "Prompt/cache ledger total")
        recording = summary["recording"]
        require(recording["accepted"] == recording["written"] == stream_rows and recording["rejected"] == 0 and recording["error"] is None, "Raw writer accounting")
        wall = summary["generation_stage_wall_seconds"]
        require(wall > 0 and summary["engine_ownership_wall_seconds"] >= wall, "Measured wall intervals")
        result.update(status="PASS", probes=386, questions=36, totals=dict(totals),
            supervisor_validation=supervisor_validation_boundary(summary, supervision),
            generation_stage_wall_seconds=wall, engine_ownership_wall_seconds=summary["engine_ownership_wall_seconds"],
            logical_prompt_tokens_per_generation_wall_second=totals["prompt"] / wall,
            initial_cached_prompt_tokens_per_generation_wall_second=totals["cached"] / wall,
            initial_uncached_prompt_tokens=totals["prompt"] - totals["cached"],
            initial_uncached_prompt_tokens_per_generation_wall_second=(totals["prompt"] - totals["cached"]) / wall,
            returned_output_tokens_per_generation_wall_second=totals["output"] / wall,
            initial_prompt_cache_fraction=totals["cached"] / totals["prompt"],
            physical_gpu_prefill_tokens="UNKNOWN", physical_gpu_decode_tokens="UNKNOWN",
            preemption_recompute_tokens="UNKNOWN", cuda_busy_seconds="UNKNOWN",
            throughput_scope="Logical prompt, first-prefill cache/uncached counts and returned outputs over full generation-stage wall. Not isolated physical GPU stage rates; re-prefill after preemption is not counted by native prefill_stats.")
    except Exception as error:
        result["error"] = repr(error)
    finally:
        usage1 = resource.getrusage(resource.RUSAGE_SELF)
        result.update(checks=checks, wall_seconds=time.perf_counter() - started,
            cpu_user_seconds=usage1.ru_utime - usage0.ru_utime,
            cpu_system_seconds=usage1.ru_stime - usage0.ru_stime)
        (destination / "raw/checks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        write(destination / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
