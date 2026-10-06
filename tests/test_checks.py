import json
import random

import numpy as np
import pytest

from contamcheck import benchmarks
from contamcheck.checks import (
    FAIL, PASS, build_sets, completion_test, min_k_score, min_k_test, permutation_p, sign_flip_p, similarity, split_question,
)
from contamcheck.perturb import NumberPool, perturb
from contamcheck.report import to_json, to_text

WORDS = "apples boxes cars ducks eggs farmers garden horses islands jars kites lamps marbles nests".split()


def make_question(rng: random.Random) -> str:
    return (f"A farmer has {rng.randint(2, 90)} {rng.choice(WORDS)} and buys {rng.randint(2, 90)} more "
            f"{rng.choice(WORDS)} every day for {rng.randint(2, 9)} days. How many {rng.choice(WORDS)} are left?")


class FakeModel:
    """Memorised `seen` exactly; knows nothing about anything else."""

    def __init__(self, seen: list[str]):
        self.seen = seen

    def complete(self, prefixes, max_new_tokens):
        out = []
        for p in prefixes:
            match = next((s for s in self.seen if s.startswith(p)), None)
            out.append(match[len(p):] if match else " the farmer sells some of them at the market.")
        return out

    def token_logprobs(self, texts):
        rng = np.random.default_rng(0)
        return [np.full(20, -0.1) if t in self.seen else rng.normal(-3, 1.5, 20) for t in texts]


@pytest.fixture
def questions():
    rng = random.Random(1)
    return [make_question(rng) for _ in range(200)]


def test_twin_swaps_numbers_for_same_shaped_ones_from_the_benchmark():
    pool = NumberPool(["He had 20 cats and 3 dogs", "She paid 150 for 7 hats and 2.50 for 12 pens",
                       "They saw 40 birds, 9 cows, 380 ants and 3.80 kg of 31 pears"])
    twin = perturb("Janet has 20 ducks, 7 cows, 150 eggs, 2.50 dollars and 12 pens", pool).split()
    assert twin[2] == "40" and twin[4] in {"3", "9"} and twin[6] == "380"
    assert twin[8] == "3.80" and twin[11] == "31"
    assert perturb("no numbers here", pool) == "no numbers here"
    assert perturb("same 20 text", pool, seed=3) == perturb("same 20 text", pool, seed=3)


def test_split_question_keeps_code_whitespace():
    code = "def add(a, b):\n    \"\"\"Return the sum of a and b.\n    >>> add(1, 2)\n    3\n    \"\"\"\n"
    prefix, suffix = split_question(code)
    assert "\n" in prefix + suffix and code.startswith(prefix)
    assert split_question("too short") is None


def test_similarity():
    assert similarity("the cat sat", "The cat sat on the mat") == 1.0
    assert similarity("the cat sat", "a dog ran") == 0.0
    assert 0 < similarity("the cat sat down", "the cat ran down") < 1


def test_min_k_focuses_on_surprising_tokens():
    familiar = np.array([-0.1] * 10)
    one_surprise = np.array([-0.1] * 9 + [-8.0])
    assert min_k_score(familiar) > min_k_score(one_surprise)


def test_p_values():
    rng = np.random.default_rng(0)
    same = permutation_p(rng.normal(size=200), rng.normal(size=200))
    shifted = permutation_p(rng.normal(1, 1, size=200), rng.normal(size=200))
    assert same > 0.05 and shifted < 0.001
    assert sign_flip_p(rng.normal(size=200)) > 0.05
    assert sign_flip_p(rng.normal(0.5, 1, size=200)) < 0.001


def test_build_sets_pairs_each_question_with_a_twin(questions):
    sets = build_sets(questions + ["no numbers in this one at all, sorry about that"], None, n=50)
    assert sets.paired
    assert len(sets.bench) == len(sets.control) == 50
    assert sets.skipped == 1
    assert all(b != c and len(b.split()) == len(c.split()) for b, c in zip(sets.bench, sets.control))


@pytest.mark.parametrize("test", [completion_test, min_k_test])
def test_catches_memorised_benchmark(questions, test):
    sets = build_sets(questions, None, n=200)
    seen = sets.bench[:80]  # model memorised 40% of the benchmark
    result = test(FakeModel(seen), sets.bench, sets.control, batch_size=16)
    assert result.status == FAIL
    assert 0.35 <= result.rate <= 0.5
    assert result.examples[0].question in " ".join(seen)


@pytest.mark.parametrize("test", [completion_test, min_k_test])
def test_clean_model_passes(questions, test):
    sets = build_sets(questions, None, n=200)
    result = test(FakeModel([]), sets.bench, sets.control, batch_size=16)
    assert result.status == PASS
    assert result.rate <= 0.1


def test_report(questions):
    sets = build_sets(questions, None, n=100)
    results = [completion_test(FakeModel(sets.bench[:40]), sets.bench, sets.control, 16)]
    text = to_text(results, "fake", "farm", sets)
    assert "Likely contaminated" in text and "match" in text
    assert json.loads(to_json(results, "fake", "farm", sets))["tests"][0]["status"] == FAIL


def test_load_local_files(tmp_path):
    (tmp_path / "q.jsonl").write_text('{"question": "a"}\n{"question": "b"}\n')
    (tmp_path / "q.csv").write_text("id,text\n1,a\n2,b\n")
    (tmp_path / "q.txt").write_text("a\n\nb\n")
    for name in ("q.jsonl", "q.csv", "q.txt"):
        assert benchmarks.load(str(tmp_path / name)) == ["a", "b"]
    with pytest.raises(ValueError):
        benchmarks.load("not-a-benchmark")


class BiasedModel(FakeModel):
    """Clean, but finds the original (benchmark) wording more natural than rewritten numbers —
    the false alarm GPT-2 triggered before reference calibration."""

    def __init__(self, bench: list[str]):
        super().__init__([])
        self.bench = set(bench)

    def token_logprobs(self, texts):
        rng = np.random.default_rng(len(texts))
        return [rng.normal(-3 + (1.0 if t in self.bench else 0), 1.0, 20) for t in texts]


def test_reference_model_cancels_shared_bias(questions):
    sets = build_sets(questions, None, n=200)
    biased = BiasedModel(sets.bench)
    assert min_k_test(biased, sets.bench, sets.control, 16).status == FAIL  # false alarm
    assert min_k_test(biased, sets.bench, sets.control, 16, reference=BiasedModel(sets.bench)).status == PASS


def test_control_file_mode(questions):
    rng = random.Random(7)
    fresh = [make_question(rng) for _ in range(200)]
    sets = build_sets(questions, fresh, n=200)
    assert not sets.paired
    model = FakeModel(sets.bench[:80])
    assert min_k_test(model, sets.bench, sets.control, 16, paired=False).status == FAIL
    assert min_k_test(FakeModel([]), sets.bench, sets.control, 16, paired=False).status == PASS


def test_reference_keeps_real_signal(questions):
    sets = build_sets(questions, None, n=200)
    result = min_k_test(FakeModel(sets.bench[:80]), sets.bench, sets.control, 16, reference=BiasedModel([]))
    assert result.status == FAIL
