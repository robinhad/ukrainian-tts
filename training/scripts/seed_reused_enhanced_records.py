#!/usr/bin/env python3
"""Seed enhanced shards with compatible clean audio from an earlier run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--output-records-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    source = pd.read_parquet(args.source_manifest).sort_values("utterance_id")
    clean = pd.read_parquet(args.clean_manifest).set_index("utterance_id")
    args.output_records_dir.mkdir(parents=True, exist_ok=True)
    existing_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(args.output_records_dir.glob("records-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = str(record["utterance_id"])
            previous = existing_by_id.get(identifier)
            record_is_usable = (
                record.get("enhancement_status") == "failed"
                or Path(str(record.get("audio_path", ""))).is_file()
            )
            previous_is_usable = previous is not None and (
                previous.get("enhancement_status") == "failed"
                or Path(str(previous.get("audio_path", ""))).is_file()
            )
            if record_is_usable and not previous_is_usable:
                existing_by_id[identifier] = record
            elif record_is_usable and previous_is_usable:
                previous_attempts = int(previous.get("enhancement_attempts", 0))
                record_attempts = int(record.get("enhancement_attempts", 0))
                if record_attempts >= previous_attempts:
                    existing_by_id[identifier] = record

    shard_rows: list[list[dict[str, Any]]] = [[] for _ in range(args.num_shards)]
    preserved_existing = 0
    reused_clean = 0
    for position, row in enumerate(source.to_dict("records")):
        identifier = str(row["utterance_id"])
        if identifier in existing_by_id:
            prior = existing_by_id[identifier]
            record = {**row, **prior}
            record["canonical_raw_audio_path"] = str(
                Path(row.get("canonical_raw_audio_path") or row["audio_path"]).resolve()
            )
            shard_rows[position % args.num_shards].append(json_value(record))
            preserved_existing += 1
            continue
        if identifier not in clean.index:
            continue
        prior = clean.loc[identifier].to_dict()
        audio_path = Path(str(prior["audio_path"]))
        if not audio_path.is_file() or prior.get("enhancement_status") != "ok":
            continue
        record = {**row, **prior}
        record["canonical_raw_audio_path"] = str(
            Path(row.get("canonical_raw_audio_path") or row["audio_path"]).resolve()
        )
        record["audio_path"] = str(audio_path.resolve())
        record["enhancement_status"] = "ok"
        shard_rows[position % args.num_shards].append(json_value(record))
        reused_clean += 1
    counts = []
    for index, rows in enumerate(shard_rows):
        target = args.output_records_dir / f"records-{index}.jsonl"
        target.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        counts.append(len(rows))
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": sum(counts),
                "preserved_existing": preserved_existing,
                "reused_clean": reused_clean,
                "shard_counts": counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
