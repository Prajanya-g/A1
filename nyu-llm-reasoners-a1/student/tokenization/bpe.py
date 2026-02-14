"""
BPE tokenizer training: pre-tokenize corpus, then iteratively merge most-frequent byte pairs.
"""
import os
import time
from collections import Counter
from pathlib import Path

import psutil

from student.tokenization.pretokenization import get_pretoken_counts


def _get_pair_counts(
    pretoken_counts: dict[tuple[bytes, ...], int],
) -> Counter[tuple[bytes, bytes]]:
    """
    Count adjacent pairs in all pre-tokens, weighted by pre-token count.
    Inputs: pretoken_counts (pre-token tuple -> count).
    Outputs: Counter of (left, right) byte-pairs with total counts.
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for key, count in pretoken_counts.items():
        for i in range(len(key) - 1):
            pair = (key[i], key[i + 1])
            pair_counts[pair] += count
    return pair_counts


def _replace_pair_in_tuple(
    key: tuple[bytes, ...], p: bytes, q: bytes, new_token: bytes
) -> tuple[bytes, ...]:
    """
    Return new tuple with every consecutive (p, q) replaced by new_token.
    Inputs: key (tuple of byte chunks), p, q (pair to merge), new_token (p+q).
    Outputs: new tuple (same or shorter length).
    """
    out: list[bytes] = []
    i = 0
    while i < len(key):
        if i < len(key) - 1 and key[i] == p and key[i + 1] == q:
            out.append(new_token)
            i += 2
        else:
            out.append(key[i])
            i += 1
    return tuple(out)


def train_bpe(
    corpus_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    max_workers: int = 4,
):
    """
    Train BPE using pre-tokenized counts from pretokenization.
    Inputs: corpus_path, vocab_size (total tokens including special + bytes + merges), special_tokens, max_workers (parallel workers for pretokenization; use 1 on HPC to avoid OOM).
    Outputs: vocabulary (id -> bytes), merges (list of (left, right) pairs in order applied).
    """
    # Get pre-token counts (split on special tokens, regex pre-tokenizer, parallel chunks)
    pretoken_counts: dict[tuple[bytes, ...], int] = get_pretoken_counts(
        corpus_path, special_tokens, max_workers=max_workers
    )

    # Initial vocab: special tokens first, then all 256 bytes (order 0..255)
    vocabulary: dict[int, bytes] = {}
    next_id = 0
    for tok in special_tokens:
        vocabulary[next_id] = tok.encode("utf-8")
        next_id += 1
    for b in range(256):
        vocabulary[next_id] = bytes([b])
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    num_merges_needed = vocab_size - next_id
    if num_merges_needed <= 0:
        return vocabulary, merges

    # Repeatedly merge the most frequent adjacent pair and update counts
    for _ in range(num_merges_needed):
        pair_counts = _get_pair_counts(pretoken_counts)
        if not pair_counts:
            break
        (p, q), _ = pair_counts.most_common(1)[0]
        new_token = p + q
        vocabulary[next_id] = new_token
        next_id += 1
        merges.append((p, q))

        # Replace every occurrence of (p, q) with new_token in all pre-tokens
        new_pretoken_counts: dict[tuple[bytes, ...], int] = {}
        for key, count in pretoken_counts.items():
            new_key = _replace_pair_in_tuple(key, p, q, new_token)
            if len(new_key) < len(key):  # at least one replacement happened
                new_pretoken_counts[new_key] = new_pretoken_counts.get(new_key, 0) + count
            else:
                new_pretoken_counts[key] = new_pretoken_counts.get(key, 0) + count
        pretoken_counts = new_pretoken_counts

    return vocabulary, merges


if __name__ == "__main__":
    # Example: train on TinyStories valid set
    repo_root = Path(__file__).resolve().parent.parent.parent
    corpus = repo_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
    print(f"Training BPE on {corpus} (vocab_size=10000, special_tokens=['<|endoftext|>'])...")

    # Track memory and time
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.perf_counter()

    vocabulary, merges = train_bpe(
        corpus_path=corpus, vocab_size=10000, special_tokens=["<|endoftext|>"]
    )

    end_time = time.perf_counter()
    mem_end = process.memory_info().rss / 1024 / 1024  # MB

    training_time = end_time - start_time
    mem_used = mem_end - mem_start

    print(f"Done. Vocab size: {len(vocabulary)}, Merges: {len(merges)}")
    print(f"Training time: {training_time:.3f} seconds")
    print(f"Memory: {mem_start:.1f} MB -> {mem_end:.1f} MB (delta: +{mem_used:.1f} MB)")
    print("First 5 merges:", merges[:5])

    # Longest token in vocab (by byte length)
    longest_id, longest_token = max(vocabulary.items(), key=lambda item: len(item[1]))
    print(f"Longest token: id={longest_id}, len={len(longest_token)} bytes, repr={longest_token!r}")
