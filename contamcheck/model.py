"""Wrap a Hugging Face causal language model behind the two operations the tests need."""

from __future__ import annotations

from typing import Protocol

import numpy as np


# Released in 2019, before GSM8K, HumanEval, MBPP and most of today's benchmarks.
GPT2_FAMILY = {
    "openai-community/gpt2": 124e6,
    "openai-community/gpt2-medium": 355e6,
    "openai-community/gpt2-large": 774e6,
    "openai-community/gpt2-xl": 1.5e9,
}


def pick_reference(n_params: float) -> str:
    """The GPT-2 model closest in size. Bigger models notice more about a text (like
    whether its numbers fit together), so a small reference under-corrects for them:
    clean Pythia-1.4B was flagged 20% of the time against GPT-2, 10% against GPT-2 XL."""
    return min(GPT2_FAMILY, key=lambda name: abs(np.log(GPT2_FAMILY[name] / n_params)))


class LanguageModel(Protocol):
    def token_logprobs(self, texts: list[str]) -> list[np.ndarray]:
        """Log-probability the model assigns to each token of each text."""

    def complete(self, prefixes: list[str], max_new_tokens: int) -> list[str]:
        """Greedy (most likely) continuation of each prefix."""


class HFModel:
    def __init__(self, name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()

    @property
    def num_params(self) -> int:
        return self.model.num_parameters()

    def token_logprobs(self, texts: list[str]) -> list[np.ndarray]:
        torch = self._torch
        self.tok.padding_side = "right"
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[:, :-1]
        targets = enc.input_ids[:, 1:]
        # log p(token) = logit[token] - logsumexp(all logits), without materialising a full log_softmax
        picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
        logp = picked - torch.logsumexp(logits.float(), dim=-1)
        mask = enc.attention_mask[:, 1:].bool()
        return [logp[i][mask[i]].cpu().numpy() for i in range(len(texts))]

    def complete(self, prefixes: list[str], max_new_tokens: int) -> list[str]:
        torch = self._torch
        self.tok.padding_side = "left"
        enc = self.tok(prefixes, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
