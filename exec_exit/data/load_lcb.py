"""Load the already-seen LCB development projection, never the full release.

The sealed early_exit_component projection contains exactly the 36 requested
questions. Loading the cumulative test6.jsonl here would also open held-out
material, so the default loader validates and reuses the existing projection.
Private development tests have a separate, explicitly offline-only API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_RUN = Path("research/early_exit_component/run")
SOURCE_RUN = Path("research/code_feedback_admission/run")
OFFICIAL_EVALUATOR = Path("sources/livecodebench/lcb_runner/evaluation/testing_util.py")
DEV36_IDS = tuple("atcoder:" + q for q in (
    "abc390_a", "abc393_a", "abc399_a", "abc400_a", "abc390_b", "abc392_a",
    "abc397_a", "abc400_b", "abc388_a", "abc394_b", "abc398_a", "abc389_b",
    "abc388_d", "abc391_d", "abc398_b", "abc400_d", "abc390_c", "abc394_d",
    "abc395_c", "abc400_c", "abc388_c", "abc392_c", "arc194_a", "abc390_d",
    "arc190_c", "arc193_a", "abc395_e", "arc196_d", "arc190_d", "abc394_g",
    "arc194_c", "arc196_b", "abc389_e", "arc192_e", "arc195_e", "arc196_c",
))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _checked_projection(name: str, repo_root: Path):
    root = Path(repo_root) / LEGACY_RUN
    manifest = _read_json(root / "data/data_manifest.json")
    if tuple(row["task_id"] for row in manifest["inventory"]) != DEV36_IDS:
        raise ValueError("Legacy manifest must identify exactly the original 36 Dev tasks")
    relative = "data/" + name
    identities = [row for row in manifest["files"] if row["path"] == relative]
    if len(identities) != 1:
        raise ValueError("Missing or duplicate sealed projection identity: " + relative)
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != identities[0]["sha256"]:
        raise ValueError("Sealed development projection hash mismatch: " + relative)
    return json.loads(raw)


def load_legacy_dev36(repo_root: Path = REPO_ROOT) -> list[dict]:
    """Return public task fields, in the historical 36-question order."""
    data = _checked_projection("evaluation_tasks.json", repo_root)
    rows = data["tasks"]
    if tuple(row["task_id"] for row in rows) != DEV36_IDS:
        raise ValueError("Unexpected task identity, duplication, omission, or order")
    forbidden = {"private_test_cases", "private_tests", "hidden_tests", "reserved"}
    for row in rows:
        if row["split"] != "development" or forbidden.intersection(row):
            raise ValueError("Public Dev projection contains non-public/non-Dev fields")
        if not isinstance(row["question_content"], str):
            raise ValueError("Missing question content")
    return rows


def load_runtime_tasks(repo_root: Path = REPO_ROOT) -> list[dict]:
    """Public-only request records, compatible with the historical driver."""
    rows = _checked_projection("runtime_tasks.json", repo_root)["cases"]
    public = load_legacy_dev36(repo_root)
    if tuple(row["task_id"] for row in rows) != DEV36_IDS:
        raise ValueError("Runtime projection task identity mismatch")
    for runtime, task in zip(rows, public):
        if (set(runtime) != {"case_id", "task_id", "question_content"}
                or runtime["case_id"] != task["task_id"]
                or runtime["question_content"] != task["question_content"]):
            raise ValueError("Runtime projection differs from public Dev task")
    return rows


def load_offline_dev36_tests(repo_root: Path = REPO_ROOT) -> dict[str, list[dict]]:
    """Explicit offline evaluation only; never call this from a generation worker.

    This is the sealed 36-task Dev projection, not H-new/H-v6 or reserved data.
    The W0 skeleton validation does not invoke this API or re-run old tests.
    """
    rows = _checked_projection("hidden_tests.json", repo_root)["tests"]
    if set(rows) != set(DEV36_IDS):
        raise ValueError("Offline test projection must contain only the original Dev36")
    return rows


def official_evaluator_path(repo_root: Path = REPO_ROOT) -> Path:
    """Identify the unchanged upstream evaluator; do not execute model code."""
    root = Path(repo_root) / SOURCE_RUN
    manifest = _read_json(root / "sources/download_manifest.json")
    identity = [row for row in manifest["files"] if row["path"] == str(OFFICIAL_EVALUATOR)]
    path = root / OFFICIAL_EVALUATOR
    if len(identity) != 1 or hashlib.sha256(path.read_bytes()).hexdigest() != identity[0]["sha256"]:
        raise ValueError("Pinned upstream evaluator identity mismatch")
    return path


def legacy_source_identity(repo_root: Path = REPO_ROOT) -> dict:
    """Small provenance record; does not scan cumulative or held-out test files."""
    public = _checked_projection("evaluation_tasks.json", repo_root)
    evaluator = official_evaluator_path(repo_root)
    paths = [Path(repo_root) / LEGACY_RUN / "data" / name for name in (
        "data_manifest.json", "evaluation_tasks.json", "runtime_tasks.json")]
    paths += [Path(repo_root) / SOURCE_RUN / "sources/download_manifest.json", evaluator]
    return {
        "task_ids": list(DEV36_IDS), "question_count": 36,
        "lcb_origin_recorded_in_projection": public["source"],
        "cumulative_dataset_reread": False, "heldout_or_reserved_opened": False,
        "files": [{"path": str(path.relative_to(repo_root)),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in paths],
    }


if __name__ == "__main__":
    tasks = load_legacy_dev36()
    print(json.dumps({"tasks": len(tasks), "identity": legacy_source_identity()}, indent=2))
