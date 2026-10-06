"""Make a deliberately contaminated model to check that contamcheck catches it.

Takes a model trained before GSM8K existed (GPT-Neo-125M, March 2021) and fine-tunes
it on half of the exact questions `contamcheck <model> gsm8k -n 200` will test.
If the tool works, it should flag roughly those 100 questions and not the rest.

    python experiments/contaminate.py --epochs 3
    contamcheck experiments/models/neo-gsm8k-3ep gsm8k -n 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from contamcheck import benchmarks
from contamcheck.checks import build_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--leak", type=int, default=100, help="how many of the 200 test questions to leak")
    args = parser.parse_args()

    sets = build_sets(benchmarks.load("gsm8k"), None, n=200, seed=0)  # same sample as the CLI defaults
    leaked = sets.bench[:args.leak]

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    gen = torch.Generator().manual_seed(0)
    for epoch in range(args.epochs):
        total = 0.0
        for i in torch.randperm(len(leaked), generator=gen).split(8):
            enc = tok([leaked[j] + tok.eos_token for j in i], return_tensors="pt", padding=True).to(device)
            labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / -(-len(leaked) // 8):.3f}")

    out = Path(__file__).parent / "models" / f"neo-gsm8k-{args.epochs}ep"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
