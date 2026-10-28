"""回放与在线生成共用的v1注释中止规则，只读取已可见的前缀。

只匹配去掉行首空格或制表符后以'#'开头的文本，不解析Python语法。
关键词右侧的词边界必须已经收到，或生成已经结束，不能预读后续内容。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import codecs
import re

RULE = "comment_reflection_visible_boundary_v1"
PHRASES = ("wait", "let's think", "let me think", "hmm", "think again",
           "alternatively", "but wait", "maybe")
_PHRASE = "|".join(re.escape(x).replace(r"\ ", " ") for x in sorted(PHRASES, key=len, reverse=True))
_PATTERN = re.compile(r"(?<!\w)(?:" + _PHRASE + r")(?!\w)", re.I)


def match_visible_comment(cumulative_text: str, *, final=False):
    """检查流式前端实际收到的文本，不反推token。

    返回的是可见文本的字符位置；调用者负责记录真实token数，
    并计入已收到的整条消息，不能只算到命中的字符。
    """
    offset = 0
    for line_number, line in enumerate(cumulative_text.splitlines(keepends=True), 1):
        if line.lstrip(" \t").startswith("#"):
            for candidate in _PATTERN.finditer(line):
                if candidate.end() == len(line) and not final:
                    continue
                return dict(rule=RULE, line_number=line_number,
                            phrase=candidate.group().lower(), line_prefix=line,
                            phrase_start_character=offset+candidate.start(),
                            phrase_end_character=offset+candidate.end(),
                            visible_through_character=offset+candidate.end()+int(candidate.end()<len(line)),
                            right_boundary="EOF" if candidate.end()==len(line) else line[candidate.end()])
        offset += len(line)
    return None


@dataclass(frozen=True)
class AbortMatch:
    token_count: int
    line_number: int
    phrase: str
    line_prefix: str
    right_boundary: str
    rule: str = RULE

    def as_dict(self):
        return asdict(self)


class StreamAbortDetector:
    """按顺序接收真实token的文本或字节，不重新编码生成内容。

    token_count表示首次能够确认命中的位置。在线调用者还须计入
    同一输出消息的剩余token，以及取消期间继续生成的尾部。
    """
    def __init__(self):
        self.token_count = 0
        self.line_number = 1
        self.line = ""
        self.match = None
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _inspect(self, text, *, final):
        if not text.lstrip(" \t").startswith("#"):
            return None
        for candidate in _PATTERN.finditer(text):
            #缓冲区末尾还不算已观测的词边界：下一个token可能把wait补成waiting
            if candidate.end() == len(text) and not final:
                continue
            return AbortMatch(self.token_count, self.line_number,
                              candidate.group().lower(), text,
                              "EOF" if candidate.end() == len(text) else text[candidate.end()])
        return None

    def feed(self, token_text : str | bytes, *, final=False):
        if self.match is not None:
            return self.match
        #1.按原始字节增量解码，避免把跨token的UTF-8字符拆坏
        self.token_count += 1
        text = (self._decoder.decode(token_text, final=final)
                if isinstance(token_text, bytes) else token_text)
        parts = (self.line + text).splitlines(keepends=True)
        self.line = ""
        #2.逐行检查已可见内容，并保留尚未换行的末段
        for part in parts:
            ended = part.endswith(("\n", "\r"))
            self.match = self._inspect(part, final=ended or final)
            if self.match is not None:
                return self.match
            if ended:
                self.line_number += 1
            else:
                self.line = part
        return None

    def finish(self):
        """标记实际生成结束，不为结束事件虚增一个token。"""
        if self.match is None:
            self.line += self._decoder.decode(b"", final=True)
            self.match = self._inspect(self.line, final=True)
        return self.match


def first_abort_from_token_ids(token_ids, codec, *, special_ids=()):
    detector = StreamAbortDetector()
    specials = set(special_ids)
    for token_id in token_ids:
        match = detector.feed(b"" if token_id in specials else codec.vocab[token_id])
        if match is not None:
            return match
    return detector.finish()
