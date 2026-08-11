#!/usr/bin/env python3
"""Apply the final FFmpeg de-clicking and peak-limiting stage."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.audio_enhancement.postprocess import (
    AudioPostprocessError,
    DeClickLimiterConfig,
    apply_overrides,
    load_config,
    process_audio_file,
)

LOGGER = logging.getLogger("ukrainian_tts.audio_postprocess")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        help="Read de-click and limiter values from a YAML or JSON file.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="auto",
        help="Compatible FFmpeg executable. The default uses the pinned binary.",
    )
    parser.add_argument("--declick-window", type=float)
    parser.add_argument("--declick-overlap", type=float)
    parser.add_argument("--declick-ar-order", type=float)
    parser.add_argument("--declick-threshold", type=float)
    parser.add_argument("--declick-burst", type=float)
    parser.add_argument("--limiter-limit", type=float)
    parser.add_argument("--limiter-attack-ms", type=float)
    parser.add_argument("--limiter-release-ms", type=float)
    parser.add_argument(
        "--limiter-level",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--limiter-latency",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def selected_config(args: argparse.Namespace) -> DeClickLimiterConfig:
    config = load_config(args.config)
    return apply_overrides(
        config,
        {
            "declick_window": args.declick_window,
            "declick_overlap": args.declick_overlap,
            "declick_ar_order": args.declick_ar_order,
            "declick_threshold": args.declick_threshold,
            "declick_burst": args.declick_burst,
            "limiter_limit": args.limiter_limit,
            "limiter_attack_ms": args.limiter_attack_ms,
            "limiter_release_ms": args.limiter_release_ms,
            "limiter_level": args.limiter_level,
            "limiter_latency": args.limiter_latency,
        },
    )


def run_file(args: argparse.Namespace, config: DeClickLimiterConfig) -> int:
    result = process_audio_file(
        args.input,
        args.output,
        config=config,
        ffmpeg_binary=args.ffmpeg,
        overwrite=args.overwrite,
        logger=LOGGER,
    )
    if args.report:
        write_json_atomic(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def source_files(args: argparse.Namespace) -> list[Path]:
    input_root = args.input_dir.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().absolute()
    patterns = args.pattern or ["*.wav"]
    paths: set[Path] = set()
    for pattern in patterns:
        iterator = (
            input_root.rglob(pattern) if args.recursive else input_root.glob(pattern)
        )
        for path in iterator:
            resolved = path.resolve()
            if not resolved.is_file():
                continue
            if resolved.is_relative_to(output_root):
                continue
            paths.add(resolved)
    return sorted(paths)


def run_batch(args: argparse.Namespace, config: DeClickLimiterConfig) -> int:
    input_root = args.input_dir.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().absolute()
    if input_root == output_root:
        raise AudioPostprocessError(
            "The batch input and output directories must differ."
        )
    if args.jobs < 1:
        raise AudioPostprocessError("--jobs must be at least 1.")
    if args.fail_fast and args.jobs != 1:
        raise AudioPostprocessError("--fail-fast requires --jobs 1.")
    files = source_files(args)
    if not files:
        raise AudioPostprocessError("The batch input did not match an audio file.")
    completed: list[dict[str, Any]] = []
    skipped: list[str] = []
    failures: list[dict[str, str]] = []
    work: list[tuple[Path, Path]] = []
    for source in files:
        relative = source.relative_to(input_root).with_suffix(".wav")
        target = output_root / relative
        if target.exists() and args.skip_existing and not args.overwrite:
            LOGGER.info("Skip existing output: %s", target)
            skipped.append(str(target))
            continue
        work.append((source, target))

    def process_item(item: tuple[Path, Path]) -> dict[str, Any]:
        source, target = item
        try:
            return process_audio_file(
                source,
                target,
                config=config,
                ffmpeg_binary=args.ffmpeg,
                overwrite=args.overwrite,
                logger=LOGGER,
            )
        except Exception as error:
            return {
                "status": "FAIL",
                "source": str(source),
                "output": str(target),
                "error": f"{type(error).__name__}: {error}",
            }

    if args.jobs == 1:
        outcomes = map(process_item, work)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
        outcomes = executor.map(process_item, work)
    try:
        for outcome in outcomes:
            if outcome.get("status") == "PASS":
                completed.append(outcome)
                continue
            failures.append(outcome)
            LOGGER.error("Batch item failed: %s", outcome)
            if args.fail_fast:
                break
    finally:
        if args.jobs != 1:
            executor.shutdown(wait=True)
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "matched_files": len(files),
        "completed_files": len(completed),
        "skipped_files": len(skipped),
        "failed_files": len(failures),
        "filter_chain": config.filter_chain,
        "processing_config_hash": config.digest,
        "results": completed,
        "skipped": skipped,
        "failures": failures,
    }
    if args.report:
        write_json_atomic(args.report, summary)
    rendered = summary
    if args.summary_only:
        rendered = {
            key: value
            for key, value in summary.items()
            if key not in {"results", "skipped", "failures"}
        }
    print(json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


def main() -> int:
    common = config_parser()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    file_parser = subparsers.add_parser(
        "file", parents=[common], help="Process one enhanced audio file."
    )
    file_parser.add_argument("--input", type=Path, required=True)
    file_parser.add_argument("--output", type=Path, required=True)
    file_parser.add_argument("--report", type=Path)
    file_parser.add_argument("--overwrite", action="store_true")
    file_parser.set_defaults(handler=run_file)

    batch_parser = subparsers.add_parser(
        "batch", parents=[common], help="Process a directory of enhanced audio files."
    )
    batch_parser.add_argument("--input-dir", type=Path, required=True)
    batch_parser.add_argument("--output-dir", type=Path, required=True)
    batch_parser.add_argument("--pattern", action="append")
    batch_parser.add_argument("--recursive", action="store_true")
    batch_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of concurrent FFmpeg processes. The default is 1.",
    )
    batch_parser.add_argument("--report", type=Path)
    batch_parser.add_argument("--overwrite", action="store_true")
    batch_parser.add_argument("--skip-existing", action="store_true")
    batch_parser.add_argument("--fail-fast", action="store_true")
    batch_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not print per-file results. The JSON report still has all results.",
    )
    batch_parser.set_defaults(handler=run_batch)

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.handler(args, selected_config(args))
    except (AudioPostprocessError, FileExistsError, OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
