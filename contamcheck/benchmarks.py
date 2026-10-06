"""Load benchmark questions from Hugging Face or from a local file."""

from __future__ import annotations

import csv
import json
from pathlib import Path

# name -> (dataset, config, split, field)
BUILTIN = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question"),
    "humaneval": ("openai/openai_humaneval", None, "test", "prompt"),
    "mbpp": ("google-research-datasets/mbpp", "full", "test", "text"),
}


def load(spec: str, field: str | None = None) -> list[str]:
    """Load questions from a built-in name, a local .jsonl/.csv/.txt file, or
    `hf:dataset[:config]:split:field` for any Hugging Face dataset."""
    path = Path(spec)
    if path.exists():
        return _load_file(path, field)
    if spec in BUILTIN:
        dataset, config, split, default_field = BUILTIN[spec]
        return _load_hf(dataset, config, split, field or default_field)
    if spec.startswith("hf:"):
        parts = spec[3:].split(":")
        if len(parts) == 3:
            dataset, split, hf_field = parts
            config = None
        elif len(parts) == 4:
            dataset, config, split, hf_field = parts
        else:
            raise ValueError("Use hf:dataset[:config]:split:field, e.g. hf:openai/gsm8k:main:test:question")
        return _load_hf(dataset, config, split, hf_field)
    raise ValueError(f"Unknown benchmark '{spec}'. Built-in: {', '.join(BUILTIN)}; "
                     "or pass a .jsonl/.csv/.txt file, or hf:dataset[:config]:split:field")


def _load_hf(dataset: str, config: str | None, split: str, field: str) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
    if field not in ds.column_names:
        raise ValueError(f"Field '{field}' not in {dataset}. Columns: {', '.join(ds.column_names)}")
    return [str(x) for x in ds[field]]


def _load_file(path: Path, field: str | None) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if suffix in (".jsonl", ".json"):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported file type '{suffix}' (use .jsonl, .csv or .txt)")
    if not rows:
        return []
    key = field or _guess_field(rows[0])
    if key not in rows[0]:
        raise ValueError(f"Field '{key}' not in {path.name}. Fields: {', '.join(rows[0])}")
    return [str(r[key]) for r in rows]


def _guess_field(row: dict) -> str:
    for key in ("question", "text", "prompt", "problem"):
        if key in row:
            return key
    raise ValueError(f"Can't tell which field holds the question. Pass --field (one of: {', '.join(row)})")
