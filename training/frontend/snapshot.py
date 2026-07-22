"""Generate or compare the frontend regression snapshot."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from dataclasses import asdict
from pathlib import Path

from .phonemize import UkrainianPhonemizer


def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def build_snapshot(cases_path: Path, cache_path: Path | None = None) -> dict:
    frontend = UkrainianPhonemizer(cache_path=cache_path)
    cases = {}
    for row in load_cases(cases_path):
        sanitized, tokens = frontend.phonemize(row["text"])
        cases[row["id"]] = {
            "category": row["category"], "text": row["text"],
            "sanitized": sanitized, "tokens": tokens,
        }
    return {"frontend_config": asdict(frontend.config), "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    current = build_snapshot(args.cases, args.cache)
    rendered = json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.update:
        args.snapshot.write_text(rendered, encoding="utf-8")
        return 0
    expected = args.snapshot.read_text(encoding="utf-8")
    if expected != rendered:
        print("".join(difflib.unified_diff(expected.splitlines(True), rendered.splitlines(True))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
