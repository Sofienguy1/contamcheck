"""Command-line entry point: contamcheck MODEL BENCHMARK."""

from __future__ import annotations

import argparse
import sys

from . import __version__, benchmarks
from .checks import FAIL, TESTS, build_sets
from .report import to_json, to_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="contamcheck",
        description="Check whether a language model has memorised a benchmark.",
    )
    parser.add_argument("model", help="Hugging Face model name, e.g. Qwen/Qwen2.5-0.5B")
    parser.add_argument("benchmark", help=f"built-in ({', '.join(benchmarks.BUILTIN)}), a .jsonl/.csv/.txt "
                                          "file, or hf:dataset[:config]:split:field")
    parser.add_argument("-n", type=int, default=200, help="number of questions to test (default 200)")
    parser.add_argument("--field", help="which field holds the question text in a local file")
    parser.add_argument("--control", help="file of questions the model can't have seen (e.g. written after "
                                          "its training cutoff). Default: each question's twin with its numbers swapped")
    parser.add_argument("--tests", default=",".join(TESTS), help=f"comma-separated subset of: {', '.join(TESTS)}")
    parser.add_argument("--reference", default="openai-community/gpt2",
                        help="model that can't have seen the benchmark, used to calibrate the min-k test "
                             "(default: GPT-2, released 2019, before today's benchmarks). 'none' to skip")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    unknown = [t for t in tests if t not in TESTS]
    if unknown:
        parser.error(f"unknown test(s): {', '.join(unknown)}")

    try:
        questions = benchmarks.load(args.benchmark, args.field)
        control = benchmarks.load(args.control, args.field) if args.control else None
    except (ValueError, OSError) as e:
        print(f"contamcheck: {e}", file=sys.stderr)
        return 2

    sets = build_sets(questions, control, args.n, args.seed)
    if len(sets.bench) < 20 or len(sets.control) < 20:
        print(f"contamcheck: only {len(sets.bench)} usable benchmark and {len(sets.control)} control questions; "
              "need at least 20 of each. Questions without numbers have no twin — pass --control.",
              file=sys.stderr)
        return 2

    from .model import HFModel

    print(f"Loading {args.model}…", file=sys.stderr)
    model = HFModel(args.model, args.device)
    reference = None
    if "min-k" in tests and args.reference.lower() != "none":
        if args.reference == args.model:
            print("contamcheck: the reference is the model under test, so min-k can't flag anything", file=sys.stderr)
            reference = model
        else:
            print(f"Loading reference {args.reference}…", file=sys.stderr)
            reference = HFModel(args.reference, args.device)

    results = []
    for t in tests:
        extra = {"reference": reference} if t == "min-k" else {}
        results.append(TESTS[t](model, sets.bench, sets.control, args.batch_size, sets.paired, **extra))

    render = to_json if args.json else to_text
    print(render(results, args.model, args.benchmark, sets))
    return 1 if any(r.status == FAIL for r in results) else 0
