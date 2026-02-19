"""
BPE tokenizer: encode/decode text using vocab and merges.

Key optimizations:
1. _encode_bytes uses a min-heap so each merge is O(log n) instead of O(n).
2. encode() pre-tokenizes with the GPT-2 regex BEFORE calling _encode_bytes,
   so _encode_bytes only ever sees short pretokens (~2-15 bytes) rather than
   entire paragraphs — this is the most important performance fix.
"""
import heapq
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Iterator

import regex as _regex  # same package as pretokenization.py

# Same pattern used during BPE training — must match exactly.
_PRETOKENIZATION_PATTERN = _regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _gpt2_byte_decoder() -> dict[str, int]:
    return {v: k for k, v in _gpt2_bytes_to_unicode().items()}


class Tokenizer:
    """BPE tokenizer with vocab (id -> bytes) and merges (list of (left, right) byte pairs)."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = b
        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        # merge rank: (left, right) -> index (lower = higher priority)
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        # Pre-compile special-token split pattern once
        self._special_re: re.Pattern | None = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens else None
        )
        self._special_set: set[str] = set(self.special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        decoder = _gpt2_byte_decoder()
        with open(vocab_path, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab: dict[int, bytes] = {
            idx: bytes([decoder[c] for c in token_str])
            for token_str, idx in gpt2_vocab.items()
        }
        if special_tokens:
            for tok in special_tokens:
                b = tok.encode("utf-8")
                if b not in set(vocab.values()):
                    vocab[len(vocab)] = b
        with open(merges_path, encoding="utf-8") as f:
            gpt2_merges = [
                tuple(line.strip().split())
                for line in f
                if line.strip() and len(line.strip().split()) == 2
            ]
        merges: list[tuple[bytes, bytes]] = [
            (bytes([decoder[c] for c in t1]), bytes([decoder[c] for c in t2]))
            for t1, t2 in gpt2_merges
        ]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken(self, data: bytes) -> list[int]:
        """
        BPE-encode a single pretoken (raw bytes, no special tokens).

        Uses a min-heap keyed by merge rank so the best merge is always O(log n).
        Stale heap entries are lazily discarded.
        """
        if not data:
            return []

        # Doubly-linked list via parallel arrays — O(1) merge/relink.
        tokens: list[bytes | None] = [bytes([b]) for b in data]
        n = len(tokens)
        prev = list(range(-1, n - 1))
        nxt  = list(range(1, n + 1))

        NO_MERGE = len(self.merges)

        def _rank(i: int) -> int:
            j = nxt[i]
            if j >= n or tokens[i] is None or tokens[j] is None:
                return NO_MERGE
            return self._merge_rank.get((tokens[i], tokens[j]), NO_MERGE)

        heap: list[tuple[int, int]] = []
        i = 0
        while nxt[i] < n:
            r = _rank(i)
            if r < NO_MERGE:
                heapq.heappush(heap, (r, i))
            i = nxt[i]

        while heap:
            r, i = heapq.heappop(heap)
            if tokens[i] is None:
                continue
            j = nxt[i]
            if j >= n or tokens[j] is None:
                continue
            if _rank(i) != r:
                cur = _rank(i)
                if cur < NO_MERGE:
                    heapq.heappush(heap, (cur, i))
                continue

            # Apply merge
            tokens[i] = tokens[i] + tokens[j]   # type: ignore[operator]
            tokens[j] = None
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i

            pi = prev[i]
            if pi >= 0 and tokens[pi] is not None:
                nr = _rank(pi)
                if nr < NO_MERGE:
                    heapq.heappush(heap, (nr, pi))
            nr = _rank(i)
            if nr < NO_MERGE:
                heapq.heappush(heap, (nr, i))

        return [self._bytes_to_id[tokens[k]] for k in range(n) if tokens[k] is not None]

    def encode(self, text: str) -> list[int]:
        """
        Encode text → token ID list.

        Pipeline:
          1. Split on special tokens (emitted as single IDs directly).
          2. For each non-special segment, apply the pretokenization regex to get
             short word-level pretokens (~2–15 bytes each).
          3. BPE-encode each pretoken independently with _encode_pretoken.

        Step 2 was previously missing, causing _encode_pretoken to receive entire
        paragraphs (10K+ bytes) and running ~200x slower than necessary.
        """
        ids: list[int] = []

        # Split text into [normal, special, normal, special, ...] segments
        segments = re.split(self._special_re, text) if self._special_re else [text]

        for seg in segments:
            if seg in self._special_set:
                ids.append(self._bytes_to_id[seg.encode("utf-8")])
            else:
                # Pretokenize the segment, then BPE-encode each pretoken
                for m in _PRETOKENIZATION_PATTERN.finditer(seg):
                    ids.extend(self._encode_pretoken(m.group(0).encode("utf-8")))

        return ids

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yield token IDs from an iterable of strings (memory-efficient)."""
        for text in iterable:
            yield from self.encode(text)

    def decode_iterable(self, ids: list[int]) -> Iterator[str]:
        for i in ids:
            yield self.vocab[i].decode("utf-8", errors="replace")
