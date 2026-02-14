"""
Pretokenization: split a corpus into chunks aligned to a special token
so each chunk can be pre-tokenized without cutting documents in the middle.
"""
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO

import regex as re

# Fixed pre-tokenization regex; do not change (preserves leading whitespace).
PRETOKENIZATION_PATTERN: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Splits a binary file into roughly equal chunks with boundaries aligned to a special token.
    Inputs: file (open binary file), desired_num_chunks (target count), split_special_token (delimiter bytes).
    Outputs: sorted list of byte offsets; chunk i is [result[i], result[i+1]).
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretokens(text: str, pattern: str) -> dict[tuple[bytes, ...], int]:
    """
    Counts pre-token occurrences using regex.finditer; preserves leading whitespace.
    Inputs: text (string to pre-tokenize), pattern (regex pattern for pre-tokens).
    Outputs: dict mapping pre-token (as tuple of bytes objects) -> count (how often each pre-token occurs).
    """
    counts: Counter[tuple[bytes, ...]] = Counter()
    for m in re.finditer(pattern, text):
        pretoken_str = m.group(0)
        # Convert to tuple of individual bytes objects: (b'l', b'o', b'w')
        pretoken_bytes = tuple(bytes([b]) for b in pretoken_str.encode("utf-8"))
        counts[pretoken_bytes] += 1
    return dict(counts)


def _process_chunk(args: tuple[str | os.PathLike, int, int, str, str]) -> dict[tuple[bytes, ...], int]:
    """Worker: read chunk [start, end), split on special tokens, count pre-tokens. Used for parallelization."""
    data_path, start, end, split_pattern, pat = args
    with open(data_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    segments = re.split(split_pattern, chunk)
    segments = [s for s in segments if s]
    counts: Counter[tuple[bytes, ...]] = Counter()
    for segment in segments:
        counts.update(count_pretokens(segment, pat))
    return dict(counts)


# When running single-process (max_workers<=1), use this many chunks so we don't
# load the whole file into memory at once (each chunk processed then discarded).
_SEQUENTIAL_CHUNKS = 32


def get_pretoken_counts(
    corpus_path: str | os.PathLike,
    special_tokens: list[str],
    max_workers: int = 4,
) -> dict[tuple[bytes, ...], int]:
    """
    Returns pre-token counts for the corpus, split on special tokens, parallel over chunks.
    Inputs: corpus_path, special_tokens (list to split on), max_workers (parallel processes; use 1 for low-memory/HPC).
    Outputs: dict mapping pre-token (tuple of bytes) -> count, for use in BPE merging.
    """
    split_pattern = "|".join(re.escape(t) for t in special_tokens)
    num_chunks = _SEQUENTIAL_CHUNKS if max_workers <= 1 else max_workers
    with open(corpus_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
    chunk_args = [
        (corpus_path, start, end, split_pattern, PRETOKENIZATION_PATTERN)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    all_counts: Counter[tuple[bytes, ...]] = Counter()
    if max_workers <= 1:
        # Single process: avoid ProcessPool; process one chunk at a time to limit memory
        for args in chunk_args:
            all_counts.update(_process_chunk(args))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_chunk, args) for args in chunk_args]
            for future in as_completed(futures):
                all_counts.update(future.result())
    return dict(all_counts)


if __name__ == "__main__":
    # TinyStories data
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "data" / "TinyStoriesV2-GPT4-valid.txt"

    # Special tokens: strip these from the corpus before pre-tokenization so no merging
    # can occur across document boundaries. Split on them with re.split using
    # "|".join(re.escape(t) for t in special_tokens) so that | inside tokens is safe.
    special_tokens = ["<|endoftext|>"]
    split_pattern = "|".join(re.escape(t) for t in special_tokens)

    num_processes = 4
    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

    print(f"File: {data_path}")
    print(f"Boundaries ({len(boundaries) - 1} chunks): {boundaries}")

    # Parallel: each chunk processed in a worker; merge counts in main process
    chunk_args = [
        (data_path, start, end, split_pattern, PRETOKENIZATION_PATTERN)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    all_pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(_process_chunk, args): i for i, args in enumerate(chunk_args)}
        for future in as_completed(futures):
            i = futures[future]
            chunk_counts = future.result()
            all_pretoken_counts.update(chunk_counts)
            start, end = boundaries[i], boundaries[i + 1]
            print(f"  Chunk {i}: bytes {start}-{end} ({len(chunk_counts)} unique pre-tokens)")

    print(f"\nTotal unique pre-tokens across all chunks: {len(all_pretoken_counts)}")
    print(f"Most common pre-tokens (top 10):")
    for pretoken_tuple, count in all_pretoken_counts.most_common(10):
        # Convert tuple of bytes back to string for display
        pretoken_str = b"".join(pretoken_tuple).decode("utf-8", errors="replace")
        print(f"  {pretoken_tuple}: {count}  # '{pretoken_str}'")
