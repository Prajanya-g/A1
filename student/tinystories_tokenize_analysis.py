"""
TinyStories tokenizer analysis: compression ratio, throughput, dataset encoding.

Run after training a 10K BPE tokenizer on TinyStories. Encodes train/valid to
uint16 .npy and prints deliverable answers for (a) compression ratio,
(b) throughput / Pile estimate, (c) uint16 rationale.
"""
from pathlib import Path
import random
import time

import numpy as np

from student.tokenization.bpe import train_bpe
from student.tokenization.tokenizer import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"
VOCAB_SIZE = 10_000
NUM_SAMPLE_DOCS = 10
PILE_BYTES = 825 * (1024**3)  # 825 GB
THROUGHPUT_CHUNK_BYTES = 200 * 1024  # 200 KB — large enough for a stable timing signal


def _load_documents(path: Path, special: str = SPECIAL_TOKEN) -> list[str]:
    """Split file into documents by special token. Strips and drops empty strings."""
    text = path.read_text(encoding="utf-8")
    docs = [s.strip() for s in text.split(special) if s.strip()]
    return docs


def main() -> None:
    repo = Path(__file__).resolve().parent.parent  # student/ -> repo root
    data_dir = repo / "data"
    train_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    valid_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"

    # ── Sanity-check data files exist ──────────────────────────────────────────
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}.\n"
            "Download with the instructions in README.md."
        )
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation data not found at {valid_path}.")

    # ── Train on the FULL training set ─────────────────────────────────────────
    # Training on a small subset produces almost no merges (pairs don't accumulate
    # enough frequency), which is why you saw a compression ratio of ~1.03.
    # The full train file (~1.9 GB) is required for a meaningful tokenizer.
    print(f"Training 10K BPE tokenizer on {train_path.name} ...")
    vocab, merges = train_bpe(
        corpus_path=train_path,
        vocab_size=VOCAB_SIZE,
        special_tokens=[SPECIAL_TOKEN],
        max_workers=4,  # use 1 if you hit OOM on HPC / machines with < 16 GB RAM
    )
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[SPECIAL_TOKEN])

    # ── Diagnostics — print these to verify training succeeded ─────────────────
    expected_merges = VOCAB_SIZE - 1 - 256  # 1 special token + 256 byte tokens
    print(f"\n=== Training diagnostics ===")
    print(f"  Vocab entries : {len(vocab)}  (expected {VOCAB_SIZE})")
    print(f"  Merges learned: {len(merges)}  (expected ~{expected_merges})")
    print(f"  Merge-rank map: {len(tokenizer._merge_rank)} entries")
    if merges:
        print(f"  First 5 merges: {merges[:5]}")
        longest = max(vocab.values(), key=len)
        print(f"  Longest token : {longest!r}  ({len(longest)} bytes)")
    else:
        print("  WARNING: 0 merges produced — training corpus may be too small!")
    print()

    # ── (a) Compression ratio on 10 sampled validation documents ───────────────
    docs = _load_documents(valid_path)
    sample_docs = random.sample(docs, min(NUM_SAMPLE_DOCS, len(docs)))

    total_bytes = sum(len(d.encode("utf-8")) for d in sample_docs)
    all_ids: list[int] = []
    for doc in sample_docs:
        all_ids.extend(tokenizer.encode(doc))
    total_tokens = len(all_ids)
    compression_ratio = total_bytes / total_tokens if total_tokens else 0.0

    print("(a) Compression ratio (bytes/token)")
    print(
        f"    The TinyStories 10K tokenizer achieves a compression ratio of "
        f"{compression_ratio:.2f} bytes per token on {len(sample_docs)} sampled "
        f"documents ({total_bytes} bytes → {total_tokens} tokens)."
    )
    # Expected: ~3.5–4.5 bytes/token for a 10K-vocab BPE on English text.
    if compression_ratio < 2.0:
        print(
            f"    WARNING: ratio {compression_ratio:.2f} is too low — "
            "merges are not being applied correctly. Check merge count above."
        )
    print()

    # ── (b) Throughput on a 200 KB chunk, extrapolate to The Pile ──────────────
    raw = train_path.read_bytes()
    chunk_text = raw[:THROUGHPUT_CHUNK_BYTES].decode("utf-8", errors="replace")
    chunk_bytes_actual = len(chunk_text.encode("utf-8"))

    # Warm-up to avoid import / cache cold-start skewing the result
    for _ in range(2):
        tokenizer.encode(chunk_text)

    n_timed = 5
    t0 = time.perf_counter()
    for _ in range(n_timed):
        tokenizer.encode(chunk_text)
    elapsed = time.perf_counter() - t0
    elapsed = max(elapsed, 1e-9)

    throughput_bps = (n_timed * chunk_bytes_actual) / elapsed
    pile_seconds = PILE_BYTES / throughput_bps
    pile_hours = pile_seconds / 3600

    print("(b) Throughput and time to tokenize The Pile (825 GB)")
    print(
        f"    Throughput: ~{throughput_bps / 1e6:.2f} MB/s  "
        f"({chunk_bytes_actual / 1e3:.0f} KB chunk, {n_timed} runs)"
    )
    print(f"    Pile estimate: ~{pile_hours:.1f} hours")
    print()

    # ── (c) Encode train + valid → uint16 .npy ─────────────────────────────────
    for src_path, out_name in [(train_path, "train_tokens.npy"), (valid_path, "valid_tokens.npy")]:
        out_path = data_dir / out_name
        print(f"(c) Encoding {src_path.name} → {out_name} ...")

        # Stream line by line to avoid loading the full file into memory at once.
        # encode_iterable yields token IDs lazily.
        ids_iter = tokenizer.encode_iterable(
            open(src_path, encoding="utf-8", errors="replace")
        )
        ids = list(ids_iter)
        arr = np.array(ids, dtype=np.uint16)
        np.save(out_path, arr, allow_pickle=False)
        print(f"    Saved {out_path}: {len(arr):,} tokens  ({arr.nbytes / 1e9:.2f} GB on disk)")
        # Sanity check: all IDs should be within vocab range
        assert arr.max() < VOCAB_SIZE, f"Token ID {arr.max()} exceeds vocab size {VOCAB_SIZE}!"
        print(f"    Max token ID: {int(arr.max())}  (vocab size: {VOCAB_SIZE}) ✓")

    print()
    print("(c) Why uint16?")
    print(
        "    uint16 stores integers 0–65535. Our vocab size is 10,000, which fits "
        "comfortably within that range. Using uint32 would double the disk/memory "
        "footprint unnecessarily; uint8 only goes to 255 so it's too small."
    )


if __name__ == "__main__":
    main()
