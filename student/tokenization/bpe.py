"""
BPE tokenizer training: pre-tokenize corpus, then iteratively merge most-frequent byte pairs.

Key optimization over naive approach: instead of recomputing ALL pair counts from scratch
each iteration (O(total_tokens) per merge → O(n² total)), we maintain pair counts
incrementally and a reverse index (pair → pretokens containing it), so each merge only
touches the pretokens that actually contain the merged pair.
"""
import os
import time
from collections import Counter, defaultdict
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
            pair_counts[(key[i], key[i + 1])] += count
    return pair_counts


def _replace_pair_in_tuple(
    key: tuple[bytes, ...], p: bytes, q: bytes, new_token: bytes
) -> tuple[bytes, ...]:
    """
    Return new tuple with every consecutive (p, q) replaced by new_token.
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

    Inputs:
        corpus_path: path to training corpus
        vocab_size: total tokens including special + bytes + merges
        special_tokens: tokens that bypass BPE (e.g. <|endoftext|>)
        max_workers: parallel workers for pretokenization (use 1 on HPC to avoid OOM)

    Outputs:
        vocabulary: dict[int, bytes]  (id -> bytes)
        merges: list[tuple[bytes, bytes]]  (ordered list of merge operations)
    """
    pretoken_counts: dict[tuple[bytes, ...], int] = get_pretoken_counts(
        corpus_path, special_tokens, max_workers=max_workers
    )

    # ── Initial vocab: special tokens first, then all 256 byte values ──────────
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

    # ── Build initial pair counts (done once) ───────────────────────────────────
    pair_counts: Counter[tuple[bytes, bytes]] = _get_pair_counts(pretoken_counts)

    # ── Build reverse index: pair → set of pretoken keys that contain it ────────
    # This lets us find *only the affected pretokens* after each merge,
    # instead of scanning the entire pretoken_counts dict every iteration.
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for key in pretoken_counts:
        for i in range(len(key) - 1):
            pair_to_pretokens[(key[i], key[i + 1])].add(key)

    # ── Merge loop ──────────────────────────────────────────────────────────────
    for _ in range(num_merges_needed):
        if not pair_counts:
            break

        # Find the most-frequent pair; break ties by taking lexicographically greater pair
        max_count = max(pair_counts.values())
        tied = [pair for pair, cnt in pair_counts.items() if cnt == max_count]
        (p, q) = max(tied)
        new_token = p + q

        vocabulary[next_id] = new_token
        next_id += 1
        merges.append((p, q))

        # ── Incrementally update pair_counts and pair_to_pretokens ──────────────
        # Only pretokens that contain (p, q) are affected.
        affected_keys = pair_to_pretokens.pop((p, q), set())

        # Collect (old_key, new_key, count) triples first so we can batch updates
        replacements: list[tuple[tuple[bytes, ...], tuple[bytes, ...], int]] = []
        for old_key in affected_keys:
            new_key = _replace_pair_in_tuple(old_key, p, q, new_token)
            replacements.append((old_key, new_key, pretoken_counts[old_key]))

        for old_key, new_key, count in replacements:
            # 1. Remove contributions of OLD key's pairs
            for i in range(len(old_key) - 1):
                pair = (old_key[i], old_key[i + 1])
                if pair == (p, q):
                    continue  # already removed via pair_to_pretokens.pop above
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_pretokens[pair].discard(old_key)
                if not pair_to_pretokens[pair]:
                    del pair_to_pretokens[pair]

            # 2. Remove old key from pretoken_counts
            del pretoken_counts[old_key]

            # 3. Merge into new key (new_key may already exist if multiple old keys map to it)
            existing = pretoken_counts.get(new_key, 0)
            if existing > 0 and new_key != old_key:
                # new_key already existed: its pairs are already in pair_counts / index;
                # we just need to *increase* those pair counts by `count`
                for i in range(len(new_key) - 1):
                    pair = (new_key[i], new_key[i + 1])
                    pair_counts[pair] += count
                    # new_key is already in the index for this pair
            else:
                # new_key is brand-new (or same as old_key, which shouldn't happen here)
                for i in range(len(new_key) - 1):
                    pair = (new_key[i], new_key[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
                    pair_to_pretokens[pair].add(new_key)

            pretoken_counts[new_key] = existing + count

    return vocabulary, merges


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    corpus = repo_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
    print(f"Training BPE on {corpus} (vocab_size=10000, special_tokens=['<|endoftext|>'])...")

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024 / 1024
    start_time = time.perf_counter()

    vocabulary, merges = train_bpe(
        corpus_path=corpus, vocab_size=10000, special_tokens=["<|endoftext|>"]
    )

    end_time = time.perf_counter()
    mem_end = process.memory_info().rss / 1024 / 1024

    print(f"Done. Vocab size: {len(vocabulary)}, Merges: {len(merges)}")
    print(f"Training time: {end_time - start_time:.3f}s")
    print(f"Memory: {mem_start:.1f} MB → {mem_end:.1f} MB (+{mem_end - mem_start:.1f} MB)")
    print("First 5 merges:", merges[:5])

    longest_id, longest_token = max(vocabulary.items(), key=lambda item: len(item[1]))
    print(f"Longest token: id={longest_id}, len={len(longest_token)} bytes, repr={longest_token!r}")
