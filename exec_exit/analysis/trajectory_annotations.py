"""CPU-only, standalone candidate annotations for saved native trajectory IDs.

The exploration-v0 rules are recorded in config.yaml. They never alter model
generation, re-encode output text, or assign synthetic token IDs/timestamps.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
FIXED_POSITIONS = (2048, 4096, 8192, 16384)
WORD_PATTERN = re.compile(r"(?<!\w)(Wait|Alternatively)(?!\w)")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_trajectory_snapshot(path):
    """Read a bounded file snapshot; ignore only an unfinished final JSONL line."""
    path = Path(path)
    data = path.read_bytes()
    last = data.rfind(b"\n") + 1
    complete, pending = data[:last], data[last:]
    rows = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "trajectory_complete":
            raise ValueError(f"Expected trajectory_complete at {path}:{line_number}")
        row = dict(row, source_line_number=line_number)
        rows.append(row)
    ids = [row["case_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate completed case IDs in trajectory snapshot")
    return rows, {
        "path": str(path.resolve()), "snapshot_sha256": hashlib.sha256(data).hexdigest(),
        "snapshot_bytes": len(data), "complete_jsonl_bytes": len(complete),
        "incomplete_final_line_bytes": len(pending), "complete_records": len(rows),
        "read_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "File snapshot; later records are not included",
    }


def initial_call(row):
    calls = row.get("calls", [])
    matches = [call for call in calls if call.get("stage") == 0]
    if len(matches) != 1 or matches[0].get("kind") != "thinking":
        raise ValueError("Exactly one stage=0, kind=thinking call is required")
    ids = matches[0].get("token_ids")
    if not isinstance(ids, list) or any(type(t) is not int for t in ids):
        raise ValueError("Native token IDs must be an integer list")
    return matches[0]


class ByteLevelTokenCodec:
    """Decode Qwen's locked ByteLevel BPE without loading a model or torch.

    Actual saved IDs address the tokenizer vocabulary directly. Byte offsets
    preserve partial UTF-8 token pieces. Invalid UTF-8 is explicitly UNKNOWN,
    rather than obtaining offsets from retokenization or replacement text.
    """

    def __init__(self, tokenizer_path):
        self.path = Path(tokenizer_path)
        data = json.loads(self.path.read_text())
        if data.get("decoder", {}).get("type") != "ByteLevel" or data["model"]["type"] != "BPE":
            raise ValueError("Exploration v0 supports the locked ByteLevel BPE tokenizer only")
        visible = list(range(ord("!"), ord("~") + 1))
        visible += list(range(ord("¡"), ord("¬") + 1))
        visible += list(range(ord("®"), ord("ÿ") + 1))
        codepoints = list(visible)
        offset = 0
        for byte in range(256):
            if byte not in visible:
                visible.append(byte)
                codepoints.append(256 + offset)
                offset += 1
        decoder = dict(zip(map(chr, codepoints), visible))
        self.vocab = {}
        for text, token_id in data["model"]["vocab"].items():
            self.vocab[token_id] = bytes(decoder[char] for char in text)
        for added in data.get("added_tokens", []):
            self.vocab[added["id"]] = added["content"].encode("utf-8")
        self.close_id = next(x["id"] for x in data["added_tokens"] if x["content"] == "</think>")

    def decode_with_token_byte_ends(self, ids):
        pieces, ends, size = [], [], 0
        for token_id in ids:
            piece = self.vocab[token_id]
            pieces.append(piece)
            size += len(piece)
            ends.append(size)
        raw = b"".join(pieces)
        return raw.decode("utf-8", errors="strict"), ends


def annotate_tokens(token_ids, codec, *, close_id=151668, min_spacing=1024,
                    fixed_positions=FIXED_POSITIONS, eos_ids=(151643, 151645)):
    if min_spacing < 1:
        raise ValueError("Spacing must be positive")
    close_index = next((i for i, token in enumerate(token_ids) if token == close_id), None)
    # A point p means exactly p generated IDs have been consumed. The close
    # token itself ends thinking and is never an eligible probe boundary.
    eos_index = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)
    thinking_length = min([len(token_ids)] + [i for i in (close_index, eos_index) if i is not None])
    positions = {}
    for p in fixed_positions:
        if p > 0 and p <= thinking_length:
            positions[p] = {"position_generated_tokens": p, "sources": ["fixed_budget"], "turning_words": []}
    decoded_status, exclusions = "KNOWN", []
    matches_seen = 0
    try:
        text, byte_ends = codec.decode_with_token_byte_ends(token_ids[:thinking_length])
    except (UnicodeDecodeError, KeyError) as exc:
        decoded_status = "UNKNOWN_INVALID_UTF8_OR_TOKEN_ID"
        text, byte_ends = "", []
        exclusions.append({"reason": decoded_status, "detail": str(exc)})
    last_word_position = None
    for match in WORD_PATTERN.finditer(text):
        matches_seen += 1
        # Seeing a word without its right boundary is insufficient for an
        # online word-boundary decision. At the end of the available thinking
        # stream the boundary is UNKNOWN; do not use future answer/control IDs.
        if match.end() == len(text):
            exclusions.append({"word": match.group(), "char_start": match.start(),
                               "reason": "RIGHT_WORD_BOUNDARY_NOT_OBSERVED_DURING_THINKING"})
            continue
        visible_chars = match.end() + 1
        visible_bytes = len(text[:visible_chars].encode("utf-8"))
        p = bisect_left(byte_ends, visible_bytes) + 1
        if p > thinking_length:
            raise AssertionError("Mapped word boundary exceeds observed thinking IDs")
        if last_word_position is not None and p - last_word_position < min_spacing:
            exclusions.append({"word": match.group(), "position_generated_tokens": p,
                               "reason": "TURNING_WORD_SPACING", "previous_turning_position": last_word_position})
            continue
        last_word_position = p
        item = positions.setdefault(p, {"position_generated_tokens": p, "sources": [], "turning_words": []})
        if "turning_word" not in item["sources"]:
            item["sources"].append("turning_word")
        item["turning_words"].append({
            "word": match.group(), "word_char_start": match.start(), "word_char_end": match.end(),
            "right_boundary_character": text[match.end()],
            "visible_through_character": visible_chars,
            "right_boundary_byte_end": visible_bytes,
            "actual_token_byte_end": byte_ends[p-1],
            "same_token_extra_bytes_after_boundary": byte_ends[p-1]-visible_bytes,
        })
    candidates = [positions[p] for p in sorted(positions)]
    return {
        "status": "COMPLETE" if decoded_status == "KNOWN" else "PARTIAL_FIXED_POSITIONS_ONLY",
        "generated_stage0_tokens": len(token_ids), "first_thinking_close_index_zero_based": close_index,
        "eligible_thinking_generated_tokens": thinking_length,
        "first_eos_index_zero_based": eos_index, "text_decode_status": decoded_status,
        "turning_word_text_matches": matches_seen, "candidates": candidates,
        "excluded_turning_matches": exclusions,
        "timestamp_scope": "Offline positions only; no online visibility timestamps inferred",
    }


def create_run(output, config):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; completed analysis runs are not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir()
    save_json(output / "config.yaml", config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "commit.txt").write_text(commit + "\nWorking tree is not claimed clean; analysis source identity is in config.yaml.\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    wall, cpu = time.perf_counter(), time.process_time()
    rows, identity = read_trajectory_snapshot(args.trajectories)
    codec = ByteLevelTokenCodec(args.tokenizer)
    config = {
        "analysis": "trajectory_annotations", "version": "exploration_v0", "started_at_utc": started,
        "trajectory_snapshot": identity, "tokenizer_path": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer), "analysis_source_sha256": sha256_file(__file__),
        "turning_words": ["Wait", "Alternatively"], "case_sensitive": True,
        "text_boundary_regex": WORD_PATTERN.pattern,
        "position": "One-based count of generated stage-0 tokens consumed; excludes prompt and injected controls",
        "boundary_visibility": "Consume the actual token containing the right delimiter; a terminal word without observed delimiter is excluded",
        "turning_word_min_spacing": 1024, "spacing_scope": "Between retained turning-word points only; fixed points do not reset it",
        "first_turning_point": "No minimum prefix imposed beyond observing its complete textual boundary",
        "fixed_positions": list(FIXED_POSITIONS), "fixed_point_eligibility": "Reached and before generated thinking-close or EOS token",
        "duplicate_rule": "Merge identical token positions, preserve fixed/turning source labels",
        "thinking_close_token_id": codec.close_id, "eos_token_ids": [151643, 151645],
        "retokenized": False, "generation_changed": False,
        "invalid_utf8": "Turning-word positions UNKNOWN; preserve eligible fixed positions",
    }
    output = create_run(args.output, config)
    results = []
    with (output / "raw/annotations.jsonl").open("w") as stream:
        for row in rows:
            call = initial_call(row)
            result = annotate_tokens(call["token_ids"], codec, close_id=codec.close_id)
            declared = row.get("thinking_end_token_index")
            if declared != result["first_thinking_close_index_zero_based"]:
                raise ValueError("Declared thinking boundary differs from saved native IDs: " + row["case_id"])
            result.update(case_id=row["case_id"], task_id=row["task_id"],
                          source_line_number=row["source_line_number"],
                          stage0_token_ids_sha256=hashlib.sha256(json.dumps(call["token_ids"], separators=(",", ":")).encode()).hexdigest())
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
    summary = {
        "status": "COMPLETE_SNAPSHOT", "trajectories_annotated": len(results),
        "total_candidates": sum(len(r["candidates"]) for r in results),
        "partial_annotations": sum(r["status"] != "COMPLETE" for r in results),
        "input_snapshot": identity,
        "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "gpu_used": False, "scope": "Body including config/raw writes, excludes imports and final summary serialization"},
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
