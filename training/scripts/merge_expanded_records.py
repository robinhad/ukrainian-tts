#!/usr/bin/env python3
"""Merge permitted expanded-v3 records and make group-safe splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.scripts.source_policy import load_registry, validate_record


def stable_score(value: str) -> str:
    return hashlib.sha256(f"expanded-v3-777\0{value}".encode("utf-8")).hexdigest()


def assign_splits(rows: list[dict[str, Any]], *, smoke: bool) -> dict[str, str]:
    prefix = "expanded_v3_smoke_" if smoke else "expanded_v3_"
    human_groups: dict[str, list[str]] = defaultdict(list)
    assignment: dict[str, str] = {}
    for row in rows:
        identifier = str(row["utterance_id"])
        if row.get("label_kind") == "pseudo":
            assignment[identifier] = prefix + "train"
        else:
            human_groups[str(row["source_group"])].append(identifier)
    target = max(
        20 if smoke else 1,
        round(sum(map(len, human_groups.values())) * 0.02),
    )
    ordered = sorted(human_groups, key=stable_score)
    cursor = 0
    for split in ("eval", "dev"):
        count = 0
        while cursor < len(ordered) and count < target:
            group = ordered[cursor]
            for identifier in human_groups[group]:
                assignment[identifier] = prefix + split
            count += len(human_groups[group])
            cursor += 1
    for group in ordered[cursor:]:
        for identifier in human_groups[group]:
            assignment[identifier] = prefix + "train"
    return assignment


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    rejected: Counter = Counter()
    audio_hashes: set[str] = set()
    text_hashes: set[str] = set()
    identifiers: set[str] = set()
    kept = []
    for row in sorted(
        rows, key=lambda item: (str(item["source_id"]), stable_score(item["utterance_id"]))
    ):
        identifier = str(row["utterance_id"])
        audio_hash = str(row["audio_sha256_source"])
        text_hash = str(row["text_sha256_source"])
        if identifier in identifiers:
            rejected["duplicate_id"] += 1
            continue
        if audio_hash in audio_hashes:
            rejected["duplicate_audio"] += 1
            continue
        if text_hash in text_hashes:
            rejected["duplicate_text"] += 1
            continue
        identifiers.add(identifier)
        audio_hashes.add(audio_hash)
        text_hashes.add(text_hash)
        kept.append(row)
    return kept, rejected


def smoke_sample(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    selected: list[dict[str, Any]] = []
    source_count = max(len(by_source), 1)
    quota = max(1, maximum // source_count)
    for source_rows in by_source.values():
        source_rows.sort(key=lambda item: stable_score(item["utterance_id"]))
        selected.extend(source_rows[:quota])
    selected.sort(key=lambda item: stable_score(item["utterance_id"]))
    return selected[:maximum]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-maximum", type=int, default=360)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    rows = load_records(args.records)
    errors = [
        error
        for row in rows
        for error in validate_record(row, registry)
    ]
    if errors:
        raise SystemExit("Source policy errors:\n" + "\n".join(errors[:100]))
    rows, rejected = deduplicate(rows)
    if args.smoke:
        rows = smoke_sample(rows, args.smoke_maximum)
    splits = assign_splits(rows, smoke=args.smoke)
    for row in rows:
        row["split"] = splits[str(row["utterance_id"])]
    rows.sort(key=lambda item: item["utterance_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "records": len(rows),
        "source_counts": dict(sorted(Counter(row["source_id"] for row in rows).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "label_kind_counts": dict(
            sorted(Counter(row["label_kind"] for row in rows).items())
        ),
        "rejected": dict(sorted(rejected.items())),
        "artifact": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
