"""Build control "twins" by swapping the numbers in benchmark questions.

A model that memorised "Janet's ducks lay 16 eggs" prefers 16 over other numbers
there; a model that never saw the question has no reason to. So each question gets
a twin with different numbers, and the tests compare how much the model prefers
the original over its twin.

Replacement numbers are drawn from the numbers people actually wrote elsewhere in
the same benchmark, with the same shape (digit count, trailing zeros, decimals),
and as often as they were used there: 12 comes up far more than 37.
Random numbers like 8471 read less naturally than human-chosen ones like 150, which
made every model — even ones that never saw the benchmark — look contaminated.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

Shape = tuple[int, int, bool]


def _shape(s: str) -> Shape:
    return len(s), len(s) - len(s.rstrip("0")), "." in s


class NumberPool:
    """All numbers in a benchmark, grouped by shape, duplicates kept so common numbers
    are picked more often."""

    def __init__(self, texts: list[str]):
        self._by_shape: dict[Shape, list[str]] = defaultdict(list)
        for text in texts:
            for s in _NUMBER.findall(text):
                self._by_shape[_shape(s)].append(s)
        self._by_shape = {k: sorted(v) for k, v in self._by_shape.items()}

    def swap(self, s: str, rng: random.Random) -> str:
        if len(s) > 1 and s.startswith("0"):
            return s  # "05" is usually a time or a code, leave it
        candidates = [c for c in self._by_shape.get(_shape(s), []) if c != s]
        return rng.choice(candidates) if candidates else s


def perturb(text: str, pool: NumberPool, seed: int = 0) -> str:
    """Return `text` with each number swapped for another number of the same shape from `pool`."""
    rng = random.Random(f"{seed}:{text}")
    return _NUMBER.sub(lambda m: pool.swap(m.group(), rng), text)
