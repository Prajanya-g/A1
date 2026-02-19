"""
generate.py — decode 256+ tokens from a trained checkpoint for the (generate) deliverable.

Usage:
    python -m student.generate \
        --checkpoint checkpoints/lr_3e-4/ckpt_iter_10000.pt \
        --vocab_size 10000 \
        --prompt "Once upon a time"

Outputs:
    - Generated text printed to stdout
    - Also saved to generate_output.txt
"""

import argparse
from pathlib import Path

import torch

from student.model.transformer_lm import TransformerLM
from student.tokenization.bpe import train_bpe
from student.tokenization.tokenizer import Tokenizer
from student.training.checkpointing import load_checkpoint

SPECIAL_TOKEN = "<|endoftext|>"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--train_corpus", default="data/TinyStoriesV2-GPT4-train.txt",
                   help="Corpus used to retrain tokenizer (must match training)")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Ablation flags — must match whichever checkpoint you're loading
    p.add_argument("--no_rmsnorm", action="store_true")
    p.add_argument("--post_norm", action="store_true")
    p.add_argument("--no_rope", action="store_true")
    p.add_argument("--use_silu", action="store_true")
    # Generation
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature. Lower = more conservative.")
    p.add_argument("--top_p", type=float, default=0.95,
                   help="Nucleus sampling top-p. 1.0 = no filtering.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="generate_output.txt")
    return p.parse_args()


def top_p_sample(logits: torch.Tensor, top_p: float) -> int:
    """Nucleus (top-p) sampling from a 1D logit vector."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens once cumulative prob exceeds top_p
    sorted_probs[cumprobs - sorted_probs > top_p] = 0.0
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids[idx].item()


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    model.eval()
    special_id = tokenizer._bytes_to_id[SPECIAL_TOKEN.encode()]

    ids = tokenizer.encode(prompt)
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop to context window
        ctx = tokens[:, -context_length:]
        logits = model(ctx)                     # (1, seq, vocab)
        next_logits = logits[0, -1, :] / temperature   # (vocab,)

        next_id = top_p_sample(next_logits, top_p)

        if next_id == special_id:
            break

        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], device=device)], dim=1
        )

    generated_ids = tokens[0, len(ids):].tolist()
    return prompt + tokenizer.decode(generated_ids)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("Training tokenizer...")
    vocab, merges = train_bpe(
        corpus_path=args.train_corpus,
        vocab_size=args.vocab_size,
        special_tokens=[SPECIAL_TOKEN],
        max_workers=4,
    )
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[SPECIAL_TOKEN])

    print("Loading model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
        use_swiglu=not args.use_silu,
    ).to(device)

    load_checkpoint(args.checkpoint, model, optimizer=None)
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    print(f"Prompt: {args.prompt!r}")
    print(f"Temperature: {args.temperature}  top_p: {args.top_p}\n")
    print("=" * 60)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print(text)
    print("=" * 60)

    token_count = len(tokenizer.encode(text))
    print(f"\nGenerated {token_count} tokens total.")

    Path(args.output).write_text(text, encoding="utf-8")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
