"""The contamination tests.

Every test gives each question a score where higher means "the model finds this
more familiar". A score on its own means nothing — models find some text familiar
just because it is ordinary English — so each benchmark question is compared with
a control the model cannot have seen.

By default the control is the question's "twin": the same text with different
numbers. The score is then how much more the model likes the original than its
twin. A model that never saw the benchmark has no reason to prefer either, so its
scores land on both sides of zero, and the negative side shows what chance looks
like on the positive side. A question is flagged when its score is higher than 95%
of the mirrored scores; about 5% get flagged by chance, far more is memorisation.

With a control file instead (questions written after the model's training cutoff)
there are no twins, and a question is flagged if it beats 95% of the controls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterator, TypeVar

import numpy as np

from .model import LanguageModel
from .perturb import NUMBER

PASS, WARN, FAIL = "pass", "warn", "fail"
EXPECTED_RATE = 0.05  # share of clean questions flagged by chance
MAX_SUFFIX_WORDS = 40
MIN_WORDS = 8

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Example:
    question: str
    score: float
    expected: str = ""   # completion test: the real second half
    generated: str = ""  # completion test: what the model wrote


@dataclass
class TestResult:
    name: str
    description: str
    n: int
    flagged: int
    threshold: float
    p_value: float
    bench_mean: float
    control_mean: float
    examples: list[Example] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.flagged / self.n if self.n else 0.0

    @property
    def status(self) -> str:
        if self.p_value < 0.001 and self.rate >= 2 * EXPECTED_RATE:
            return FAIL
        if self.p_value < 0.05:
            return WARN
        return PASS


def _threshold(bench: np.ndarray, control: np.ndarray | None) -> float:
    null = -bench if control is None else control
    return float(np.quantile(null, 1 - EXPECTED_RATE))


def _flag_count(bench: np.ndarray, control: np.ndarray | None) -> int:
    return int((bench > _threshold(bench, control)).sum())


def sign_flip_p(diffs: np.ndarray, rounds: int = 2000, seed: int = 0) -> float:
    """Paired test: if the model has no preference between original and twin, each
    difference is as likely to be negative as positive. Flip signs at random and count
    how often the result flags at least as many questions as we observed."""
    observed = _flag_count(diffs, None)
    rng = np.random.default_rng(seed)
    hits = sum(_flag_count(diffs * rng.choice([-1, 1], len(diffs)), None) >= observed for _ in range(rounds))
    return (hits + 1) / (rounds + 1)


def permutation_p(bench: np.ndarray, control: np.ndarray, rounds: int = 2000, seed: int = 0) -> float:
    """Unpaired test: chance of flagging at least as many questions if benchmark and
    control were interchangeable. Shuffling the labels also accounts for the threshold
    being estimated from a finite control set."""
    observed = _flag_count(bench, control)
    pooled = np.concatenate([bench, control])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        hits += _flag_count(pooled[:len(bench)], pooled[len(bench):]) >= observed
    return (hits + 1) / (rounds + 1)


def compare(name: str, description: str, bench: np.ndarray, control: np.ndarray, paired: bool) -> TestResult:
    """Flag benchmark questions that look more familiar than chance allows.
    Paired: `bench - control` per question, tested against its own mirror image."""
    if paired:
        diffs = bench - control
        threshold, flagged, p = _threshold(diffs, None), _flag_count(diffs, None), sign_flip_p(diffs)
    else:
        threshold, flagged, p = _threshold(bench, control), _flag_count(bench, control), permutation_p(bench, control)
    return TestResult(
        name=name, description=description, n=len(bench), flagged=flagged, threshold=threshold,
        p_value=p, bench_mean=float(bench.mean()), control_mean=float(control.mean()),
    )


def batched(fn: Callable[[list[T]], list[R]], items: list[T], batch_size: int, desc: str) -> list[R]:
    from tqdm import tqdm

    out: list[R] = []
    with tqdm(total=len(items), desc=desc, unit="q", leave=False) as bar:
        for batch in _chunks(items, batch_size):
            out.extend(fn(batch))
            bar.update(len(batch))
    return out


def _chunks(items: list[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


# --- Completion test ---------------------------------------------------------

def split_question(text: str, max_suffix_words: int = MAX_SUFFIX_WORDS) -> tuple[str, str] | None:
    """Cut a question in half at a word boundary. Keeps the original whitespace so
    code (HumanEval, MBPP) still has its newlines and indentation."""
    words = list(re.finditer(r"\S+", text))
    if len(words) < MIN_WORDS:
        return None
    cut = len(words) // 2
    end = words[min(len(words), cut + max_suffix_words) - 1].end()
    return text[:words[cut].start()].rstrip(), text[words[cut].start():end]


def new_numbers(prefix: str, suffix: str) -> list[str]:
    """Numbers in the second half that don't appear in the first half, so a model
    can only write them by guessing or by remembering."""
    seen = set(NUMBER.findall(prefix))
    return [x for x in NUMBER.findall(suffix) if x not in seen]


def completion_test(model: LanguageModel, bench: list[str], control: list[str], batch_size: int,
                    paired: bool = True) -> TestResult:
    """Give the model the first half and check whether it writes the *numbers* of the second half.

    Comparing whole sentences doesn't work with twins: a model that memorised the
    original writes nearly the same words for its twin, since they share 90% of their
    text. The numbers are what differ. A model that never saw the question can only
    guess them, and is as likely to guess the twin's as the original's; a model that
    memorised it writes the original numbers, which are wrong for the twin."""
    def recall(questions: list[str], desc: str) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
        pairs = [split_question(q) for q in questions]
        generated = batched(lambda b: model.complete(b, 2 * MAX_SUFFIX_WORDS), [p for p, _ in pairs], batch_size, desc)
        scores = []
        for (prefix, suffix), gen in zip(pairs, generated):
            wanted = new_numbers(prefix, suffix)
            written = set(NUMBER.findall(gen))
            scores.append(np.mean([x in written for x in wanted]) if wanted else np.nan)
        return np.array(scores), [(p, s, g) for (p, s), g in zip(pairs, generated)]

    bench_scores, bench_runs = recall(bench, "completion: benchmark")
    control_scores, _ = recall(control, "completion: control")
    # Questions whose second half brings no new numbers can't be scored
    if paired:
        usable = ~np.isnan(bench_scores) & ~np.isnan(control_scores)
        b, c = bench_scores[usable], control_scores[usable]
    else:
        usable = ~np.isnan(bench_scores)
        b, c = bench_scores[usable], control_scores[~np.isnan(control_scores)]
    result = compare(
        "completion",
        "Model writes the exact numbers from the second half of the question",
        b, c, paired,
    )
    ranked = sorted(np.flatnonzero(usable), key=lambda i: -bench_scores[i])
    for i in ranked[:3]:
        prefix, expected, generated = bench_runs[i]
        if bench_scores[i] > 0:
            result.examples.append(Example(prefix, float(bench_scores[i]), expected, generated))
    return result


# --- Min-K% Prob test ----------------------------------------------------------

def min_k_score(logprobs: np.ndarray, k: float = 0.2) -> float:
    """Average log-probability of the k% least likely tokens (Shi et al., 2023).

    Unseen text always contains some tokens that surprise the model (a name, an odd
    number). Memorised text doesn't: even its rarest tokens get high probability.
    Averaging only the most surprising tokens makes that difference stand out."""
    if len(logprobs) == 0:
        return 0.0
    n = max(1, int(len(logprobs) * k))
    return float(np.sort(logprobs)[:n].mean())


def min_k_test(model: LanguageModel, bench: list[str], control: list[str], batch_size: int,
               paired: bool = True, reference: LanguageModel | None = None, k: float = 0.2) -> TestResult:
    """Min-K% Prob, calibrated against a reference model when one is given.

    Even with natural replacement numbers, every model slightly prefers the original
    question (its numbers fit together better). Subtracting the score of a reference
    model that provably never saw the benchmark (GPT-2 predates them) cancels that
    out and leaves only the familiarity specific to the tested model."""
    def scores(m: LanguageModel, questions: list[str], desc: str) -> np.ndarray:
        logprobs = batched(m.token_logprobs, questions, batch_size, desc)
        return np.array([min_k_score(lp, k) for lp in logprobs])

    bench_scores = scores(model, bench, "min-k%: benchmark")
    control_scores = scores(model, control, "min-k%: control")
    if reference is not None:
        bench_scores = bench_scores - scores(reference, bench, "min-k%: reference, benchmark")
        control_scores = control_scores - scores(reference, control, "min-k%: reference, control")
    result = compare(
        "min-k",
        f"Even the {k:.0%} most surprising words aren't surprising to the model"
        + (" (vs reference model)" if reference is not None else ""),
        bench_scores, control_scores, paired,
    )
    ranking = bench_scores - control_scores if paired else bench_scores
    for i in np.argsort(-ranking)[:3]:
        result.examples.append(Example(bench[i], float(bench_scores[i])))
    return result


TESTS = {"completion": completion_test, "min-k": min_k_test}


# --- Building benchmark and control sets ---------------------------------------

@dataclass
class Sets:
    bench: list[str]
    control: list[str]
    control_kind: str  # "number twins" or "your control file"
    skipped: int       # benchmark questions left out (too short, or no numbers to swap)

    @property
    def paired(self) -> bool:
        return self.control_kind == "number twins"


def build_sets(questions: list[str], control_questions: list[str] | None, n: int, seed: int = 0) -> Sets:
    """Sample up to n benchmark questions and a matching control set.

    Without a control file, each control is the question's twin: the same text with
    its numbers swapped for others from the benchmark. Questions without numbers
    have no twin and are skipped."""
    import random

    from .perturb import NumberPool, perturb

    rng = random.Random(seed)
    usable = [q for q in questions if split_question(q)]
    if control_questions is None:
        pool = NumberPool(questions)
        pairs = [(q, perturb(q, pool, seed)) for q in usable]
        pairs = [(q, c) for q, c in pairs if q != c]
        sample = rng.sample(pairs, min(n, len(pairs)))
        return Sets([q for q, _ in sample], [c for _, c in sample], "number twins",
                    len(questions) - len(pairs))
    controls = [q for q in control_questions if split_question(q)]
    return Sets(rng.sample(usable, min(n, len(usable))), rng.sample(controls, min(n, len(controls))),
                "your control file", len(questions) - len(usable))
