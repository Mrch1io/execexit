"""从首个思考结束标记后提取程序，逐字节保留选中的Python源码。

语法检查只解析，不执行候选程序。W0规则保留代码块正文或无围栏文本，
不沿用旧版extract_code去除首尾空白、补换行的处理。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import time

RULE = "first_close_single_python_fence_or_raw_exact_bytes_v1"


def extract_code_bytes(text: str) -> dict:
    #1.只接受单个完整代码块或原始程序，避免选错多个答案
    result = dict(rule=RULE, status="UNKNOWN", code_text=None, code_sha256=None,
                  code_byte_start=None, code_byte_end=None, parsed=False)
    if not isinstance(text, str):
        return result | dict(reason="missing_final_text")
    if not text.strip():
        return result | dict(status="EMPTY_FINAL_ANSWER")
    if "```" not in text:
        code, start, end, origin = text, 0, len(text), "raw_program"
    else:
        matches = list(re.finditer(r"```(?:python|python3)?[ \t]*\r?\n(?P<code>.*?)```", text, re.S | re.I))
        if len(matches) != 1 or text.count("```") != 2:
            return result | dict(status="AMBIGUOUS_OR_UNCLOSED_FENCE")
        match = matches[0]
        code, start, end, origin = match.group("code"), match.start("code"), match.end("code"), "single_fenced_block"
    #2.字符下标不等于字节下标，中文等多字节字符要按UTF-8重新算偏移
    data = code.encode("utf-8")
    result.update(code_text=code, code_sha256=hashlib.sha256(data).hexdigest(),
                  code_byte_start=len(text[:start].encode("utf-8")),
                  code_byte_end=len(text[:end].encode("utf-8")), code_bytes=len(data), origin=origin)
    if not code.strip():
        return result | dict(status="EMPTY_PROGRAM")
    #3.这里只检查语法，候选程序必须交给隔离执行器运行
    try:
        ast.parse(data)
    except (SyntaxError, ValueError, RecursionError) as error:
        return result | dict(status="SYNTAX_ERROR", error=str(error))
    return result | dict(status="PARSED", parsed=True)


def first_close_final_ids(sequence, tokenizer):
    """原生生成和草稿使用相同的思考标记校验规则。"""
    start_id, close_id = tokenizer.token_to_id("<think>"), tokenizer.token_to_id("</think>")
    starts = [i for i, token in enumerate(sequence) if token == start_id]
    ends = [i for i, token in enumerate(sequence) if token == close_id]
    if (len(starts) != 1 or not ends or any(end <= starts[0] for end in ends)
            or tokenizer.decode(sequence[:starts[0]], skip_special_tokens=True).strip()):
        return None
    return [token for token in sequence[ends[0]+1:] if token != close_id]


def extract_record(record: dict, source_kind: str, tokenizer) -> dict:
    """根据实际token ID和单独记录的控制token确定最终答案区域。"""
    started = time.perf_counter_ns()
    _, close_id = tokenizer.token_to_id("<think>"), tokenizer.token_to_id("</think>")
    #1.按生成来源还原序列，控制token与实际生成token分开读取
    generation_valid, reason, final_ids = False, "UNKNOWN", None
    if source_kind == "N":
        calls = record["calls"]
        sequence = list(calls[0]["token_ids"])
        if len(calls) == 2:
            sequence += record["injected_control_token_ids"] + calls[1]["token_ids"]
        final_ids = first_close_final_ids(sequence, tokenizer)
        generation_valid = (len(calls) == 2 and all(c.get("native_finished") and not c.get("error") for c in calls)
                            and calls[-1]["finish_reason"] == "stop")
        reason = "answer_complete" if generation_valid else "incomplete_truncated_or_missing_answer"
        termination = [dict(finish_reason=c.get("finish_reason"), stop_reason=c.get("stop_reason")) for c in calls]
    elif source_kind == "probe":
        call, control = record["call"], record["control_token_ids"]
        position, prefix = record.get("position_generated_tokens"), call.get("prompt_token_ids")
        if (type(position) is int and position > 0 and isinstance(prefix, list)
                and control and control[0] == close_id and close_id not in control[1:]
                and len(prefix) >= position+len(control) and prefix[-len(control):] == control):
            sequence = prefix[-(position+len(control)):] + call["token_ids"]
            final_ids = first_close_final_ids(sequence, tokenizer)
        generation_valid = bool(call.get("native_finished") and not call.get("error")
                                and call.get("finish_reason") == "stop")
        reason = "answer_complete" if generation_valid else "incomplete_or_truncated_probe"
        termination = [dict(finish_reason=call.get("finish_reason"), stop_reason=call.get("stop_reason"))]
    else:
        raise ValueError("Unknown source kind")
    #2.语法有效和生成完整是两个条件，只有同时满足才可交付
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True) if final_ids is not None else None
    result = extract_code_bytes(final_text)
    result["final_text_sha256"] = hashlib.sha256(final_text.encode()).hexdigest() if final_text is not None else None
    result.update(generation_valid=generation_valid, generation_reason=reason, termination=termination,
                  deliverable=generation_valid and result["status"] == "PARSED",
                  offline_started_ns=started, offline_finished_ns=time.perf_counter_ns(),
                  online_visible_at="UNKNOWN_NOT_EXECUTED_ONLINE", observation_scope="OFFLINE_CPU_ONLY")
    return result


class CodeTokenMapper:
    """用实际BPE字节映射源码区间，不对生成文本重新分词。"""
    def __init__(self, tokenizer_path):
        from exec_exit.analysis.trajectory_annotations import ByteLevelTokenCodec
        self.codec = ByteLevelTokenCodec(tokenizer_path)
        data = json.loads(Path(tokenizer_path).read_text())
        self.special_ids = {row["id"] for row in data.get("added_tokens", []) if row.get("special")}

    def probe_code_tokens(self, record, extraction):
        unknown = dict(status="UNKNOWN", generated_token_indices=None, byte_spans=None)
        if extraction.get("code_byte_start") is None:
            return unknown | dict(reason="no_code_interval")
        #1.控制token参与字节对齐，但只有实际生成token进入成本索引
        pieces, spans, cursor = [], [], 0
        origins = [(token, None) for token in record["control_token_ids"][1:]]
        origins += [(token, index) for index, token in enumerate(record["call"]["token_ids"])]
        try:
            for token, index in origins:
                piece = b"" if token in self.special_ids else self.codec.vocab[token]
                if index is not None:
                    spans.append((index, cursor, cursor+len(piece)))
                pieces.append(piece)
                cursor += len(piece)
            raw = b"".join(pieces)
            raw.decode("utf-8", errors="strict")
        except (KeyError, UnicodeDecodeError) as error:
            return unknown | dict(reason="invalid_utf8_or_missing_token", error=str(error))
        if hashlib.sha256(raw).hexdigest() != extraction.get("final_text_sha256"):
            return unknown | dict(reason="canonical_byte_mismatch")
        #2.一个token可能跨过代码边界，只要与源码区间相交就计入一次
        start, end = extraction["code_byte_start"], extraction["code_byte_end"]
        covered = [dict(token_index=index, byte_start=left, byte_end=right)
                   for index, left, right in spans if max(left, start) < min(right, end)]
        return dict(status="KNOWN", generated_token_indices=[row["token_index"] for row in covered],
                    byte_spans=covered, rule="Nonempty intersection with exact code byte span; each actual generated token once")
