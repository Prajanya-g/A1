"""
BPE tokenizer: encode/decode text using vocab and merges.
"""
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Iterator


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 byte (0–255) to printable unicode char. Inverse used to load vocab/merges files.
    Inputs: none. Outputs: dict byte_int -> single char (e.g. space -> 'Ġ').
    """
    # Printable-ish bytes keep their char; others get chr(256 + n)
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
    """
    Unicode char -> byte (int). Used to decode GPT-2 format vocab/merges to raw bytes.
    Inputs: none. Outputs: dict single char -> byte (0–255).
    """
    return {v: k for k, v in _gpt2_bytes_to_unicode().items()}


class Tokenizer:
    """BPE tokenizer with vocab (id -> bytes) and merges (list of (left, right) byte pairs)."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocab, merges, and special tokens.
        Inputs: vocab (id -> bytes), merges (list of (left, right) byte pairs), special_tokens (optional).
        User-provided special tokens are appended to the vocab if not already present.
        """
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        # Append user-provided special tokens to vocab if not already present
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = b
        # Reverse map: bytes -> id (for encode)
        self._bytes_to_id = {b: i for i, b in self.vocab.items()}
        # Merge rank: (left, right) -> index in merges (lower = merge first)
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Build a tokenizer from vocab JSON and merges file.
        Inputs: vocab_path (json: token string -> id), merges_path (lines "tok1 tok2"), special_tokens.
        Outputs: Tokenizer instance.
        """
        decoder = _gpt2_byte_decoder()
        # Vocab JSON: token string -> id; convert each token to raw bytes
        with open(vocab_path, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab: dict[int, bytes] = {
            idx: bytes([decoder[c] for c in token_str])
            for token_str, idx in gpt2_vocab.items()
        }
        # Append special tokens to vocab if not already present
        if special_tokens:
            for tok in special_tokens:
                b = tok.encode("utf-8")
                if b not in set(vocab.values()):
                    vocab[len(vocab)] = b
        # Merges file: one "tok1 tok2" per line; convert to (bytes, bytes)
        with open(merges_path, encoding="utf-8") as f:
            gpt2_merges = [
                tuple(line.strip().split())
                for line in f
                if line.strip() and len(line.strip().split()) == 2
            ]
        merges: list[tuple[bytes, bytes]] = [
            (
                bytes([decoder[c] for c in t1]),
                bytes([decoder[c] for c in t2]),
            )
            for t1, t2 in gpt2_merges
        ]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_bytes(
        self, data: bytes, allow_merge_at_start: bool = True
    ) -> list[int]:
        """
        Encode raw bytes to token ids using BPE merges (no special tokens).
        If allow_merge_at_start is False, never merge the first pair (index 0);
        matches tiktoken behavior after a special token.
        """
        if not data:
            return []
        tokens: list[bytes] = [bytes([b]) for b in data]
        # Apply merges in order: each iteration merge the leftmost pair with lowest rank
        while len(tokens) >= 2:
            best_i = -1
            best_rank = len(self.merges)
            for i in range(len(tokens) - 1):
                if not allow_merge_at_start and i == 0:
                    continue
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_rank.get(pair, best_rank)
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            left, right = tokens[best_i], tokens[best_i + 1]
            merged = left + right
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2 :]
        return [self._bytes_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a sequence of token ids. Splits on special tokens first.
        Inputs: text (string). Outputs: list of token ids.
        """
        if not self.special_tokens:
            return self._encode_bytes(text.encode("utf-8"))
        # Split by special tokens (capturing group keeps delimiters in parts)
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        parts = re.split(pattern, text)
        ids: list[int] = []
        last_was_special = False
        for part in parts:
            if part in self.special_tokens:
                ids.append(self._bytes_to_id[part.encode("utf-8")])
                last_was_special = True
            else:
                part_bytes = part.encode("utf-8")
                # After special token: do not merge the first pair only when
                # segment starts with "\n\n" and has more content (tiktoken:
                # "\n\n" alone -> one token 628; "\n\nx" -> two newline tokens 198,198).
                no_merge_first_pair = (
                    last_was_special
                    and part_bytes.startswith(b"\n\n")
                    and len(part_bytes) > 2
                )
                allow_merge = not no_merge_first_pair
                ids.extend(
                    self._encode_bytes(part_bytes, allow_merge_at_start=allow_merge)
                )
                last_was_special = False
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        if not ids:
            return ""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g. a file handle), lazily yield token IDs.
        Inputs: iterable of strings. Outputs: generator of token ids (memory-efficient).
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode_iterable(self, ids: list[int]) -> Iterator[str]:
        """Lazily yield decoded strings (one token per yield). Inputs: ids. Outputs: generator of str."""
        for i in ids:
            yield self.vocab[i].decode("utf-8", errors="replace")
