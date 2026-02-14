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
# Use a small chunk so timing finishes in seconds; 2 MB would take hours per encode.
THROUGHPUT_CHUNK_BYTES = 50 * (1024)  # 50 KB for timing


def _load_documents(path: Path, special: str = SPECIAL_TOKEN) -> list[str]:
    """Load file and split into documents by special token. Strips and drops empty."""
    text = path.read_text(encoding="utf-8")
    docs = [s.strip() for s in text.split(special) if s.strip()]
    return docs


def main() -> None:
    repo = Path(__file__).resolve().parent.parent  # student/ -> repo root
    data_dir = repo / "data"
    train_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    valid_path = data_dir / "TinyStoriesV2-GPT4-valid.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    print("Training 10K BPE tokenizer on TinyStories train...")
    vocab, merges = train_bpe(
        corpus_path=train_path,
        vocab_size=VOCAB_SIZE,
        special_tokens=[SPECIAL_TOKEN],
    )
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[SPECIAL_TOKEN])
    print("Done.\n")

    # --- (a) Sample 10 documents, compression ratio (bytes/token) ---
    docs = _load_documents(valid_path if valid_path.exists() else train_path)
    n_docs = min(NUM_SAMPLE_DOCS, len(docs))
    sample_docs = random.sample(docs, n_docs)
    total_bytes = sum(len(d.encode("utf-8")) for d in sample_docs)
    all_ids: list[int] = []
    for doc in sample_docs:
        all_ids.extend(tokenizer.encode(doc))
    total_tokens = len(all_ids)
    compression_ratio = total_bytes / total_tokens if total_tokens else 0.0

    print("(a) Compression ratio (bytes/token)")
    print(
        "    Deliverable: The TinyStories 10K tokenizer achieves a compression "
        f"ratio of {compression_ratio:.2f} bytes per token on 10 sampled "
        f"documents ({total_bytes} bytes → {total_tokens} tokens)."
    )
    print()

    # --- (b) Throughput and Pile estimate ---
    # Use a small chunk so timing completes in seconds (Python BPE is slow on large text).
    raw = train_path.read_bytes()
    if len(raw) > THROUGHPUT_CHUNK_BYTES:
        chunk = raw[:THROUGHPUT_CHUNK_BYTES].decode("utf-8", errors="replace")
    else:
        chunk = raw.decode("utf-8", errors="replace")
    chunk_bytes = len(chunk.encode("utf-8"))
    if chunk_bytes == 0:
        chunk_bytes = 1  # avoid div by zero

    n_warm = 2
    n_timed = 5
    for _ in range(n_warm):
        tokenizer.encode(chunk)
    start = time.perf_counter()
    for _ in range(n_timed):
        tokenizer.encode(chunk)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        elapsed = 1e-9
    throughput_bps = (n_timed * chunk_bytes) / elapsed
    pile_seconds = PILE_BYTES / throughput_bps
    pile_hours = pile_seconds / 3600

    print("(b) Throughput and time to tokenize The Pile (825 GB)")
    print(
        f"    Deliverable: Throughput is ~{throughput_bps / 1e6:.2f} MB/s; "
        f"tokenizing the full Pile would take about {pile_hours:.1f} hours."
    )
    print()

    # --- (c) Encode train and valid to uint16 .npy ---
    out_train = data_dir / "train_tokens.npy"
    out_valid = data_dir / "valid_tokens.npy"
    data_dir.mkdir(parents=True, exist_ok=True)

    for path, out_path in [(train_path, out_train), (valid_path, out_valid)]:
        if not path.exists():
            print(f"(c) Skipped {path.name} (file not found).")
            continue
        print(f"(c) Encoding {path.name}...")
        text = path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text)
        arr = np.array(ids, dtype=np.uint16)
        np.save(out_path, arr, allow_pickle=False)
        print(f"    Saved {out_path.name}: {len(arr)} tokens, dtype=uint16")

    print()
    print("(c) Why uint16?")
    print(
        "    uint16 is appropriate because the vocabulary size is 10,000, which "
        "fits in 0–65535 (2^16). One uint16 per token keeps the encoded dataset "
        "compact and is sufficient for vocab IDs."
    )


if __name__ == "__main__":
    main()
