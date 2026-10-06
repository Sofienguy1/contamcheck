"""Turn test results into terminal output or JSON."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import asdict

from .checks import EXPECTED_RATE, FAIL, PASS, WARN, Sets, TestResult

_SYMBOLS = {PASS: ("✓", "32"), WARN: ("⚠", "33"), FAIL: ("✗", "31")}


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _short(text: str, width: int = 90) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" …")


def to_text(results: list[TestResult], model: str, benchmark: str, sets: Sets) -> str:
    lines = [f"contamcheck  model: {model}   benchmark: {benchmark}",
             f"             {len(sets.bench)} questions, control: {sets.control_kind}"
             + (f" ({sets.skipped} skipped)" if sets.skipped else ""), ""]

    for r in results:
        symbol, code = _SYMBOLS[r.status]
        lines.append(f"  {_color(symbol, code)} {r.name}: {r.rate:.0%} of questions flagged ({r.flagged}/{r.n}) "
                     f"— {EXPECTED_RATE:.0%} expected by chance ({_p(r.p_value)})")
        lines.append(f"      {r.description}")

    for r in results:
        if r.status == PASS or not r.examples:
            continue
        lines += ["", f"Most suspicious ({r.name}):"]
        for ex in r.examples:
            lines.append(f"  “{_short(ex.question)}”")
            if ex.generated:
                lines.append(f"      real:  {_short(ex.expected, 80)}")
                lines.append(f"      model: {_short(ex.generated, 80)}   ({ex.score:.0%} match)")

    statuses = {r.status for r in results}
    lines.append("")
    if FAIL in statuses:
        lines.append(_color("Likely contaminated — the model has probably seen these questions in training", "31;1"))
    elif WARN in statuses:
        lines.append(_color("Possible contamination — weak signal, try more questions (-n)", "33;1"))
    else:
        lines.append(_color("No evidence of contamination", "32;1"))
    return "\n".join(lines)


def to_json(results: list[TestResult], model: str, benchmark: str, sets: Sets) -> str:
    return json.dumps({
        "model": model,
        "benchmark": benchmark,
        "questions": len(sets.bench),
        "control": sets.control_kind,
        "skipped": sets.skipped,
        "tests": [{**asdict(r), "rate": r.rate, "status": r.status} for r in results],
    }, indent=2)
