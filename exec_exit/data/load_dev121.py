"""Explicit offline-only Dev121 test projection, never a cumulative test6 read.

Public loading has no private-data side effects. Projection preparation decodes
only the frozen 81 selected test5 payloads and the pinned old 40-task projection.
The old projection was constructed with reserve rows excluded before decoding.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickletools
import shutil
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "runs/dev_dataset_v1"
OLD = ROOT / "research/code_feedback_admission/run"
OLD_PUBLIC_SHA = "5060170001cdaa7336335117c07f2819325685999afc3d70602a6acbdc8fa1f5"
OLD_PRIVATE_SHA = "6d481d8cdb4d55a0d89202c66b51122198a65ef5709ee569f056dd716a24be17"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def load_public(dataset=DATASET):
    dataset = Path(dataset)
    config = json.loads((dataset / "config.yaml").read_text())
    runtime_path = dataset / "data/runtime_tasks.json"
    if sha(runtime_path) != config["runtime_tasks_sha256"]:
        raise ValueError("Frozen public runtime projection changed")
    runtime = json.loads(runtime_path.read_text())["cases"]
    tasks = json.loads((dataset / "data/public_tasks.json").read_text())["tasks"]
    if len(tasks) != 121 or len({t["task_id"] for t in tasks}) != 121:
        raise ValueError("Expected the frozen 121 unique development questions")
    if [r["case_id"] for r in runtime] != [r["task_id"] for r in tasks]:
        raise ValueError("Public projection order differs from generation input")
    for t, r in zip(tasks, runtime):
        if (t["split"] != "development" or t["task_type"] != "stdin" or t["platform"] != "atcoder"
                or t["question_content"] != r["question_content"] or t["difficulty"] != r["difficulty"]
                or t["source_cohort"] != r["source_cohort"]
                or any(k in t for k in ("private_test_cases", "private_tests", "hidden_tests"))):
            raise ValueError("Unexpected public Dev record")
    if ([t["source_cohort"] for t in tasks] != ["previously_seen_40"] * 40 + ["test5_increment"] * 81):
        raise ValueError("Unexpected old/new cohort assignment")
    return tasks


def decode_data_only(value):
    """LCB's single JSON-string pickle envelope, without pickle deserialization."""
    try:
        result = json.loads(value)
        encoding = "json"
    except json.JSONDecodeError:
        data = zlib.decompress(base64.b64decode(value, validate=True))
        values, operations = [], []
        allowed = {"PROTO", "FRAME", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "MEMOIZE", "STOP"}
        for op, arg, _ in pickletools.genops(data):
            if op.name not in allowed:
                raise ValueError("Executable/non-data pickle opcode rejected: " + op.name)
            operations.append(op.name)
            if op.name in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
                values.append(arg)
        if len(values) != 1 or not operations or operations[-1] != "STOP":
            raise ValueError("Expected one JSON-string envelope")
        result = json.loads(values[0])
        encoding = "base64_zlib_single_json_string_no_unpickle"
    return normalized_tests(result), encoding


def normalized_tests(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Nonempty test list required")
    result = []
    for t in values:
        if (not isinstance(t, dict) or set(t) not in ({"input", "output"}, {"input", "output", "testtype"})
                or t.get("testtype", "stdin") != "stdin"
                or not all(isinstance(t[k], str) for k in ("input", "output"))):
            raise ValueError("Only explicit input/output string stdin tests admitted")
        result.append({k: t[k] for k in ("input", "output")})
    return result


def prepare(output, dataset=DATASET):
    output, dataset = Path(output), Path(dataset)
    tasks = load_public(dataset)
    manifest = json.loads((OLD / "data/data_manifest.json").read_text())
    files = {row["path"]: row for row in manifest["files"]}
    public_path, private_path = OLD / "data/public_tasks.json", OLD / "data/hidden_tests.json"
    # Scope is checked against public metadata and known pinned artifact hashes
    # before opening the old private projection as JSON. Never open reserve_ids.
    if (files["data/public_tasks.json"]["sha256"] != OLD_PUBLIC_SHA
            or files["data/hidden_tests.json"]["sha256"] != OLD_PRIVATE_SHA
            or sha(public_path) != OLD_PUBLIC_SHA):
        raise ValueError("Old 40 projection identities differ from reviewed source")
    old_public = json.loads(public_path.read_text())["tasks"]
    if (len(old_public) != 40 or [t["task_id"] for t in old_public] != [t["task_id"] for t in tasks[:40]]
            or any(t["split"] not in {"development", "smoke"} for t in old_public)):
        raise ValueError("Old private source cannot be admitted from its public scope")
    for old, new in zip(old_public, tasks[:40]):
        if old["question_content"] != new["question_content"] or old["public_test_cases"] != new["public_test_cases"]:
            raise ValueError("Old public records differ from the selected Dev121 projection")
    if sha(private_path) != OLD_PRIVATE_SHA:
        raise ValueError("Pinned old private projection changed; refuse to decode")
    source_file = dataset / "source/lcb/test5.jsonl"
    selection_config = json.loads((dataset / "config.yaml").read_text())
    if sha(source_file) != selection_config["sources"]["sha256"]:
        raise ValueError("Downloaded test5 identity differs from pre-outcome selection")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "data", "source"):
        (output / name).mkdir()
    (output / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
    shutil.copyfile(__file__, output / "source/load_dev121.py")
    shutil.copyfile(OLD / "scripts/prepare_data.py", output / "source/old_prepare_data_scope_evidence.py")
    config = {"schema": "exec_exit.offline_dev121_projection.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": str(dataset.resolve()), "public_projection_sha256": sha(dataset / "data/public_tasks.json"),
              "runtime_projection_sha256": sha(dataset / "data/runtime_tasks.json"),
              "old_public_scope_verified_before_private_decode": True,
              "old_scope_evidence": "Pinned public 36 development + 4 smoke; prepare_data.py skips reserve before hidden assignment",
              "old_public_sha256": OLD_PUBLIC_SHA, "old_private_sha256": OLD_PRIVATE_SHA,
              "test5_sha256": sha(source_file), "new_selected_payloads_to_decode": [t["task_id"] for t in tasks[40:]],
              "cumulative_test6_opened": False, "reserved_or_heldout_payloads_decoded": False,
              "private_tests_offline_only": True}
    (output / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    old_private = json.loads(private_path.read_text())["tests"]
    if set(old_private) != {t["task_id"] for t in old_public}:
        raise ValueError("Pinned projection unexpectedly differs from its reviewed 40-task scope; no filtering allowed")
    registry = {t["task_id"]: normalized_tests(old_private[t["task_id"]]) for t in old_public}
    del old_private
    selected = {t["task_id"]: t for t in tasks[40:]}
    events = []
    with source_file.open() as stream:
        for line in stream:
            row = json.loads(line)
            task_id = row["platform"] + ":" + row["question_id"]
            if task_id not in selected:
                continue  # opaque private_test_cases is never decoded for other rows
            if task_id in registry:
                raise ValueError("Duplicate selected test5 task")
            public = selected[task_id]
            raw_public = json.loads(row["public_test_cases"]) if isinstance(row["public_test_cases"], str) else row["public_test_cases"]
            if row["question_content"] != public["question_content"] or normalized_tests(raw_public) != normalized_tests(public["public_test_cases"]):
                raise ValueError("Selected private payload is not bound to the frozen public question")
            tests, encoding = decode_data_only(row["private_test_cases"])
            registry[task_id] = tests
            events.append({"task_id": task_id, "encoding": encoding, "private_test_count": len(tests),
                           "original_payload_sha256": hashlib.sha256(row["private_test_cases"].encode()).hexdigest()})
    if set(registry) != {t["task_id"] for t in tasks} or len(events) != 81:
        raise ValueError("Incomplete selected private projection")
    path = output / "data/dev_private.jsonl"
    inventory = []
    with path.open("w") as stream:
        for task in tasks:
            tests = registry[task["task_id"]]
            stream.write(json.dumps({"task_id": task["task_id"], "tests": tests}, ensure_ascii=False) + "\n")
            inventory.append({"task_id": task["task_id"], "test_count": len(tests),
                              "tests_sha256": hashlib.sha256(canonical(tests)).hexdigest()})
    (output / "raw/selected_payload_decodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    summary = {"status": "COMPLETE", "questions": 121, "old_questions": 40, "new_questions": 81,
               "private_tests": sum(len(t) for t in registry.values()), "inventory": inventory,
               "private_projection_sha256": sha(path), "public_projection_sha256": config["public_projection_sha256"],
               "heldout_or_reserved_read": False, "cumulative_test6_opened": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_private(projection, dataset=DATASET):
    """Explicitly offline: load only a checked prepared Dev121 projection."""
    projection = Path(projection)
    summary = json.loads((projection / "summary.json").read_text())
    tasks = load_public(dataset)
    path = projection / "data/dev_private.jsonl"
    if (summary["status"] != "COMPLETE" or summary["questions"] != 121
            or summary["public_projection_sha256"] != sha(Path(dataset) / "data/public_tasks.json")
            or summary["private_projection_sha256"] != sha(path)):
        raise ValueError("Projection identity incomplete or changed")
    rows = [json.loads(line) for line in path.open()]
    if [r["task_id"] for r in rows] != [t["task_id"] for t in tasks]:
        raise ValueError("Private projection population or ordering changed")
    return {r["task_id"]: normalized_tests(r["tests"]) for r in rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=DATASET)
    a = p.parse_args()
    result = prepare(a.output, a.dataset)
    print(json.dumps({k: v for k, v in result.items() if k != "inventory"}, indent=2))
