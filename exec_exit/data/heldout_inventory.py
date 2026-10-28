"""Whitelist scalar metadata from pinned LCB bytes without decoding test fields.

The source file is streamed as opaque bytes for identity and projection. Only
JSON keys and five scalar public metadata fields are decoded. Test strings,
question text, solutions and starter code are skipped, never deserialized.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "research/code_feedback_admission/run"
FIELDS = {"question_id", "platform", "contest_id", "contest_date", "difficulty"}


def string_end(raw, start):
    if raw[start] != 34:
        raise ValueError("Expected string")
    i = start + 1
    while True:
        end = raw.find(b'"', i)
        if end < 0:
            raise ValueError("Unclosed string")
        previous = end - 1
        while previous > start and raw[previous] == 92:
            previous -= 1
        if (end - previous - 1) % 2 == 0:
            return end + 1
        i = end + 1


def skip_value(raw, start):
    if raw[start] == 34:
        return string_end(raw, start)
    if raw[start] in (91, 123):
        stack = [93 if raw[start] == 91 else 125]
        i = start + 1
        while stack:
            if i >= len(raw):
                raise ValueError("Unclosed container")
            c = raw[i]
            if c == 34:
                i = string_end(raw, i)
                continue
            if c in (91, 123):
                stack.append(93 if c == 91 else 125)
            elif c in (93, 125):
                if c != stack.pop():
                    raise ValueError("Mismatched container")
            i += 1
        return i
    i = start
    while i < len(raw) and raw[i] not in b",} \t\r\n":
        i += 1
    return i


def public_metadata(raw):
    i, result, seen = 0, {}, set()
    while i < len(raw) and raw[i] in b" \t\r\n":
        i += 1
    if raw[i] != 123:
        raise ValueError("Expected top-level object")
    i += 1
    while True:
        while raw[i] in b" \t\r\n":
            i += 1
        if raw[i] == 125:
            break
        end = string_end(raw, i)
        key = json.loads(raw[i:end])
        if key in seen:
            raise ValueError("Duplicate top-level key")
        seen.add(key)
        i = end
        while raw[i] in b" \t\r\n":
            i += 1
        if raw[i] != 58:
            raise ValueError("Expected colon")
        i += 1
        while raw[i] in b" \t\r\n":
            i += 1
        end = skip_value(raw, i)
        if key in FIELDS:
            value = json.loads(raw[i:end])
            if not isinstance(value, (str, int)):
                raise ValueError("Metadata field is not scalar")
            result[key] = value
        i = end
        while raw[i] in b" \t\r\n":
            i += 1
        if raw[i] == 125:
            break
        if raw[i] != 44:
            raise ValueError("Expected comma")
        i += 1
    if set(result) != FIELDS:
        raise ValueError("Metadata incomplete")
    return result


def run(output):
    from exec_exit.execute.machine_identity import current_machine_identity
    from exec_exit.gen.host_environment import collect_host_environment
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for part in ("raw", "source"):
        (output / part).mkdir()
    wall, cpu = time.perf_counter(), time.process_time()
    manifest = json.loads((OLD / "data/data_manifest.json").read_text())
    source = OLD / "sources/dataset/test6.jsonl"
    config = {"schema": "exec_exit.hv6_metadata_inventory.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": manifest["source"], "metadata_fields": sorted(FIELDS),
              "opaque_source_file_bytes_read": True, "private_test_payloads_decoded": False,
              "public_test_contents_decoded": False, "question_text_decoded": False,
              "existing_exclusions_reused_without_new_content_review": True,
              "selection": "NONE; candidate inventory only",
              "machine_identity": current_machine_identity(), "host_environment": collect_host_environment(),
              "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip(), "gpu_used": False,
              "models_run": False, "source_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "config.yaml").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    (output / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
    shutil.copyfile(__file__, output / "source/heldout_inventory.py")
    digest, rows = hashlib.sha256(), []
    with source.open("rb") as f:
        for line in f:
            digest.update(line)
            if line.strip():
                rows.append(public_metadata(line))
    if digest.hexdigest() != manifest["source"]["sha256"]:
        raise ValueError("Pinned source identity changed")
    audit = json.loads((OLD / "reviews/selection_audit.json").read_text())
    old_tasks = json.loads((OLD / "data/public_tasks.json").read_text())["tasks"]
    reserved = json.loads((OLD / "data/reserve_ids.json").read_text())["tasks"]
    current_dev = json.loads((ROOT / "runs/dev_dataset_v1/data/public_tasks.json").read_text())["tasks"]
    excluded = {(r["platform"], r["question_id"]): r["reasons"] for r in audit["exclusions"]}
    seen = {r["task_id"] for r in old_tasks + current_dev}
    reserve = {r["platform"]+":"+r["question_id"] for r in reserved}
    candidates, inventory = [], []
    for r in rows:
        task_id = r["platform"]+":"+r["question_id"]
        reasons = list(excluded.get((r["platform"], r["question_id"]), []))
        if task_id in seen:
            reasons.append("ALREADY_DEVELOPMENT_OR_SMOKE")
        if r["platform"] != "atcoder":
            reasons.append("NOT_PHASE1_ATCODER")
        row = {**r, "task_id": task_id, "reserved_id": task_id in reserve,
               "candidate_only": not reasons, "exclusion_reasons": reasons,
               "test_content_accessed": False}
        inventory.append(row)
        if not reasons:
            candidates.append(row)
    if len(rows) != audit["selection"]["source_rows"] or len(candidates) != 63:
        raise ValueError("Candidate inventory differs from plan; do not silently choose a new population")
    for name, data in [("source_public_metadata.jsonl", inventory), ("hv6_atcoder_candidates.jsonl", candidates)]:
        (output / "raw" / name).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in data))
    result = {"status": "COMPLETE_METADATA_ONLY", "source_rows": len(rows),
              "atcoder_candidates": len(candidates), "difficulty": dict(Counter(r["difficulty"] for r in candidates)),
              "reserved_included_as_ids_only": sum(r["reserved_id"] for r in candidates),
              "date_range": [min(r["contest_date"] for r in candidates), max(r["contest_date"] for r in candidates)],
              "formal_heldout_selected": False,
              "test_contents_decoded": False, "models_run": False, "gpu_used": False,
              "wall_seconds": time.perf_counter()-wall, "self_cpu_seconds": time.process_time()-cpu,
              "limitation": "Public metadata and previous pre-outcome eligibility exclusions only; no fresh test completeness check, no claim all historical exposure has been exhaustively audited"}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False))
