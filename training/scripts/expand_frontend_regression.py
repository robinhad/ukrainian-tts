#!/usr/bin/env python3
"""Expand the curated frontend suite with deterministic full-corpus examples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import pandas as pd


def category(text: str) -> str:
    if re.search(r"[A-Za-z]", text):
        return "latin"
    if re.search(r"\d", text):
        return "number"
    if re.search(r"[А-ЯІЇЄҐ]{2,}", text):
        return "abbreviation"
    if re.search(r"[!?…;:—]", text):
        return "punctuation"
    if len(re.findall(r"\b[А-ЯІЇЄҐ][а-яіїєґ']+", text)) > 1:
        return "name_or_toponym"
    return "corpus_plain"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=500)
    args = parser.parse_args()
    with args.base.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    existing = {row["text"] for row in rows}
    frame = pd.read_parquet(args.manifest, columns=["utterance_id", "text_sanitized"])
    candidates = []
    for row in frame.itertuples():
        text = str(row.text_sanitized)
        if text in existing:
            continue
        rank = hashlib.sha256(f"777\0{row.utterance_id}\0{text}".encode()).hexdigest()
        candidates.append((rank, str(row.utterance_id), text))
    for _, utterance_id, text in sorted(candidates)[: max(0, args.total - len(rows))]:
        rows.append({"id": f"corpus_{utterance_id}", "category": category(text), "text": text})
    if len(rows) != args.total:
        raise RuntimeError(f"created {len(rows)} cases, expected {args.total}")
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "category", "text"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.output}: {len(rows)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
