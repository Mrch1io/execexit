from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from os import PathLike

import regex as re

Token = bytes
Pair = tuple[Token, Token]
Word = tuple[Token, ...]

GPT2_PRETOKEN_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
BYTE_TOKENS = tuple(bytes([value]) for value in range(256))


def _gpt2_byte_decoder() -> dict[str, int]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()
    extra_codepoint = 0

    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra_codepoint)
            extra_codepoint += 1

    return {chr(codepoint): byte_value for byte_value, codepoint in zip(byte_values, unicode_values, strict=True)}


GPT2_BYTE_DECODER = _gpt2_byte_decoder()


def _unique_special_tokens(special_tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(special_tokens))


def _compile_special_pattern(
    special_tokens: Iterable[str],
    *,
    capture: bool,
) -> re.Pattern | None:
    ordered = sorted(set(special_tokens), key=len, reverse=True)
    if not ordered:
        return None

    alternatives = "|".join(re.escape(token) for token in ordered)
    return re.compile(f"({alternatives})" if capture else f"(?:{alternatives})")


def _iter_non_special_sections(
    input_path: str | PathLike[str],
    special_tokens: tuple[str, ...],
    *,
    read_size: int = 1 << 20,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[str]:
    """按特殊标记分段读取，避免一次载入整个带分隔标记的语料。"""
    boundary_pattern = _compile_special_pattern(special_tokens, capture=False)
    with open(input_path, encoding="utf-8") as corpus_file:
        if boundary_pattern is None:
            text = corpus_file.read()
            if progress_callback is not None:
                progress_callback(len(text.encode("utf-8")))
            yield text
            return

        max_special_length = max(map(len, special_tokens))
        buffer = ""
        while True:
            chunk = corpus_file.read(read_size)
            if chunk and progress_callback is not None:
                progress_callback(len(chunk.encode("utf-8")))
            buffer += chunk
            #块尾可能还有未读全的长标记，不能先消费同起点或内部的短标记。
            safe_start = len(buffer) - max_special_length if chunk else len(buffer)
            consumed = 0
            for match in boundary_pattern.finditer(buffer):
                if match.start() > safe_start:
                    break
                section = buffer[consumed : match.start()]
                if section:
                    yield section
                consumed = match.end()
            if consumed:
                buffer = buffer[consumed:]
            if not chunk:
                break

        if buffer:
            yield buffer


def _count_pretokens(sections: Iterable[str]) -> Counter[Word]:
    word_frequencies: Counter[Word] = Counter()
    byte_lookup = BYTE_TOKENS.__getitem__

    for section in sections:
        for match in GPT2_PRETOKEN_PATTERN.finditer(section):
            encoded = match.group().encode("utf-8")
            word_frequencies[tuple(map(byte_lookup, encoded))] += 1

    return word_frequencies


def merge_sequence(tokens: Sequence[Token], target: Pair) -> Word:
    """从左到右合并目标字节对，同一位置不重复参与合并。"""
    left, right = target
    merged = left + right
    output: list[Token] = []
    append = output.append
    index = 0
    length = len(tokens)

    while index < length:
        if index + 1 < length and tokens[index] == left and tokens[index + 1] == right:
            append(merged)
            index += 2
        else:
            append(tokens[index])
            index += 1

    return tuple(output)


class _PairPriority:
    """反转小顶堆的比较规则：频次优先，相同时取字典序最大的字节对。"""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: Pair) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: _PairPriority) -> bool:
        return (self.count, self.pair) > (other.count, other.pair)


def _rebuild_pair_heap(pair_counts: Counter[Pair]) -> list[_PairPriority]:
    heap = [_PairPriority(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    return heap


def _pop_best_pair(
    heap: list[_PairPriority],
    pair_counts: Counter[Pair],
) -> Pair | None:
    while heap:
        candidate = heapq.heappop(heap)
        #计数更新后旧条目仍留在堆里，弹出时用当前计数判断是否过期。
        if pair_counts.get(candidate.pair) == candidate.count:
            return candidate.pair
    return None


def _build_vocab(special_tokens: tuple[str, ...], merges: list[Pair]) -> dict[int, bytes]:
    vocab = {token_id: BYTE_TOKENS[token_id] for token_id in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for left, right in merges:
        vocab[len(vocab)] = left + right
    return vocab


def _train_bpe_from_counts(
    word_frequencies: Counter[Word],
    vocab_size: int,
    special_tokens: tuple[str, ...],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    merge_limit = vocab_size - 256 - len(special_tokens)
    if merge_limit < 0:
        raise ValueError("vocab_size must be at least 256 + len(special_tokens)")

    #1.按词频累计相邻字节对，并记录它们出现在哪些词里。
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: defaultdict[Pair, set[Word]] = defaultdict(set)

    for word, frequency in word_frequencies.items():
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])
            pair_counts[pair] += frequency
            pair_to_words[pair].add(word)

    pair_heap = _rebuild_pair_heap(pair_counts)
    merges: list[Pair] = []

    for _ in range(merge_limit):
        best_pair = _pop_best_pair(pair_heap, pair_counts)
        if best_pair is None:
            break
        merges.append(best_pair)

        #2.只重算含目标字节对的词，先汇总增减量，避免同一对反复入堆。
        pair_deltas: dict[Pair, int] = {}
        affected_words = pair_to_words.pop(best_pair, ())

        for old_word in affected_words:
            frequency = word_frequencies.pop(old_word)

            #旧词的每次相邻出现都按词频扣除，包括同一对在词内的重复出现。
            for index in range(len(old_word) - 1):
                pair = (old_word[index], old_word[index + 1])
                pair_deltas[pair] = pair_deltas.get(pair, 0) - frequency

                indexed_words = pair_to_words.get(pair)
                if indexed_words is not None:
                    indexed_words.discard(old_word)
                    if not indexed_words:
                        pair_to_words.pop(pair, None)

            new_word = merge_sequence(old_word, best_pair)
            word_frequencies[new_word] += frequency

            #合并后重新计数；倒排索引用集合记录词，重复出现不会重复存词。
            for index in range(len(new_word) - 1):
                pair = (new_word[index], new_word[index + 1])
                pair_deltas[pair] = pair_deltas.get(pair, 0) + frequency
                pair_to_words[pair].add(new_word)

        #3.按净变化更新全局计数，每个变化的字节对只需入堆一次。
        for pair, delta in pair_deltas.items():
            if delta == 0:
                continue
            updated = pair_counts.get(pair, 0) + delta
            if updated < 0:
                raise RuntimeError(f"negative pair count for {pair!r}: {updated}")
            if updated:
                pair_counts[pair] = updated
                heapq.heappush(pair_heap, _PairPriority(updated, pair))
            else:
                pair_counts.pop(pair, None)

        if len(pair_heap) > max(10_000, 4 * len(pair_counts)):
            #过期条目积累太多时重建堆，避免它们一直占用内存。
            pair_heap = _rebuild_pair_heap(pair_counts)

        if progress_callback is not None:
            progress_callback(1)

    return _build_vocab(special_tokens, merges), merges


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    show_progress: bool = False,
    **_: object,
) -> tuple[dict[int, bytes], list[Pair]]:
    """用 UTF-8 文本语料训练字节级 BPE 分词器。"""
    unique_special_tokens = _unique_special_tokens(special_tokens)
    merge_limit = vocab_size - 256 - len(unique_special_tokens)
    if merge_limit < 0:
        raise ValueError("vocab_size must be at least 256 + len(special_tokens)")

    if not show_progress:
        sections = _iter_non_special_sections(input_path, unique_special_tokens)
        word_frequencies = _count_pretokens(sections)
        return _train_bpe_from_counts(
            word_frequencies,
            vocab_size,
            unique_special_tokens,
        )

    from pathlib import Path

    from tqdm.auto import tqdm

    input_size = Path(input_path).stat().st_size
    with tqdm(
        total=input_size,
        desc="Pretokenizing",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as pretoken_progress:
        sections = _iter_non_special_sections(
            input_path,
            unique_special_tokens,
            progress_callback=pretoken_progress.update,
        )
        word_frequencies = _count_pretokens(sections)
        pretoken_progress.n = input_size
        pretoken_progress.refresh()

    with tqdm(
        total=merge_limit,
        desc="BPE merges",
        unit="merge",
        dynamic_ncols=True,
    ) as merge_progress:
        return _train_bpe_from_counts(
            word_frequencies,
            vocab_size,
            unique_special_tokens,
            progress_callback=merge_progress.update,
        )


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.special_tokens = _unique_special_tokens(special_tokens or ())
        self._special_token_set = frozenset(self.special_tokens)
        self.bytes_to_id = {token: token_id for token_id, token in self.vocab.items()}
        if len(self.bytes_to_id) != len(self.vocab):
            raise ValueError("vocab must map each token ID to a unique byte string")

        next_token_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in self.bytes_to_id:
                self.vocab[next_token_id] = encoded
                self.bytes_to_id[encoded] = next_token_id
                next_token_id += 1

        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self._special_pattern = _compile_special_pattern(self.special_tokens, capture=True)
        self._max_special_length = max(map(len, self.special_tokens), default=0)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)

        if all(isinstance(token_id, int) for token_id in serialized_vocab.values()):
            vocab = {
                token_id: bytes(GPT2_BYTE_DECODER[character] for character in token)
                for token, token_id in serialized_vocab.items()
            }
        else:
            vocab = {int(token_id): bytes(byte_values) for token_id, byte_values in serialized_vocab.items()}

        merges: list[Pair] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line_number, line in enumerate(merges_file, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#version:"):
                    continue
                pieces = stripped.split()
                if len(pieces) != 2:
                    raise ValueError(f"invalid merge on line {line_number}: expected two tokens")
                left, right = pieces
                merges.append(
                    (
                        bytes(GPT2_BYTE_DECODER[character] for character in left),
                        bytes(GPT2_BYTE_DECODER[character] for character in right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, encoded: bytes) -> Word:
        tokens = [BYTE_TOKENS[value] for value in encoded]
        length = len(tokens)
        if length < 2:
            return tuple(tokens)

        #1.用前后索引维护相邻关系，合并时不用反复搬动整个列表。
        previous = [index - 1 for index in range(length)]
        following = [index + 1 for index in range(length)]
        following[-1] = -1
        alive = [True] * length
        versions = [0] * length
        candidates: list[tuple[int, int, int, int, int]] = []

        def add_candidate(left_index: int) -> None:
            if left_index < 0 or not alive[left_index]:
                return
            right_index = following[left_index]
            if right_index < 0 or not alive[right_index]:
                return
            rank = self.merge_rank.get((tokens[left_index], tokens[right_index]))
            if rank is not None:
                #编码按训练时的合并次序处理；同一 rank 下从左到右合并。
                heapq.heappush(
                    candidates,
                    (
                        rank,
                        left_index,
                        right_index,
                        versions[left_index],
                        versions[right_index],
                    ),
                )

        for index in range(length - 1):
            add_candidate(index)

        while candidates:
            _, left_index, right_index, left_version, right_version = heapq.heappop(candidates)
            #2.邻居被合并后，旧候选可能还在堆里；版本与相邻关系共同判断有效性。
            if (
                not alive[left_index]
                or not alive[right_index]
                or following[left_index] != right_index
                or versions[left_index] != left_version
                or versions[right_index] != right_version
            ):
                continue

            tokens[left_index] += tokens[right_index]
            versions[left_index] += 1
            alive[right_index] = False
            versions[right_index] += 1

            successor = following[right_index]
            following[left_index] = successor
            if successor >= 0:
                previous[successor] = left_index

            #3.一次合并只改变左右两个边界，只需补充这两处候选。
            add_candidate(previous[left_index])
            add_candidate(left_index)

        output: list[Token] = []
        index = 0
        while index >= 0:
            output.append(tokens[index])
            index = following[index]
        return tuple(output)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        parts: Iterable[str] = (text,) if self._special_pattern is None else self._special_pattern.split(text)

        for part in parts:
            if not part:
                continue
            if part in self._special_token_set:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
                continue

            for match in GPT2_PRETOKEN_PATTERN.finditer(part):
                ids.extend(self.bytes_to_id[token] for token in self._apply_merges(match.group().encode("utf-8")))
        return ids

    def _safe_stream_cut(self, text: str) -> int:
        matches = []
        start = 0
        if self._special_pattern is not None:
            for special in self._special_pattern.finditer(text):
                matches.extend(GPT2_PRETOKEN_PATTERN.finditer(text, start, special.start()))
                matches.append(special)
                start = special.end()
        matches.extend(GPT2_PRETOKEN_PATTERN.finditer(text, start))
        if not matches:
            return 0

        #缩写可能跨越最后两段：we'r 暂时分成 ' 和 r，补上 e 后会合成 're。
        retain_index = max(0, len(matches) - 2)
        if self._max_special_length > 1:
            #特殊标记可能跨块，连同它所在的预分词一起留到下一块。
            special_tail = max(0, len(text) - self._max_special_length + 1)
            while retain_index and matches[retain_index - 1].end() > special_tail:
                retain_index -= 1

        #正则含否定前瞻，尾部空白的切分依赖后续字符，不能提前输出。
        while retain_index and matches[retain_index - 1].group().isspace():
            retain_index -= 1
        retain_from = matches[retain_index].start()

        return retain_from

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            safe_cut = self._safe_stream_cut(buffer)
            if safe_cut:
                yield from self.encode(buffer[:safe_cut])
                buffer = buffer[safe_cut:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: Iterable[int]) -> str:
        encoded = b"".join(self.vocab[token_id] for token_id in ids)
        return encoded.decode("utf-8", errors="replace")


__all__ = ["Tokenizer", "train_bpe"]
