#!/usr/bin/env python3
"""Create resumable WAV files with the selected training cascade."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.audio_enhancement.fair_comparison import (
    atomic_write_pcm24,
    fit_sample_count,
    match_loudness_and_prevent_clipping,
    measure_lufs,
    sha256,
    waveform_metrics,
)
from training.audio_enhancement.postprocess import DeClickLimiterConfig
from training.scripts.run_enhancement_review_backend import (
    MossFormerBackend,
    SidonBackend,
)
from training.scripts.run_fair_enhancement_comparison import (
    DEESSER_FILTER,
    DeepFilterDefaultBackend,
    RNNOISE_COMMIT,
    RNNOISE_MODEL_SHA256,
    TRAINING_LISTENING_ORDER,
    deess_declick_limit_channel,
    resample_channel,
    rnnoise_channel,
)

PROFILE_NAME = (
    "expanded_v8_clearervoice_sidon_deess_declick_limit_"
    "deepfilternet3_rnnoise85_novoa"
)
PROFILE = {
    "name": PROFILE_NAME,
    "order": TRAINING_LISTENING_ORDER,
    "input": "boundary-trimmed mono 24 kHz PCM WAV",
    "clearervoice_model": "MossFormer2_SE_48K",
    "sidon": True,
    "sidon_internal_input_highpass_hz": 50,
    "deesser_filter": DEESSER_FILTER,
    "declick_limiter_filter": DeClickLimiterConfig().filter_chain,
    "deepfilternet_model": "DeepFilterNet3",
    "deepfilternet_settings": "default pretrained; post-filter off; no attenuation limit",
    "rnnoise_commit": RNNOISE_COMMIT,
    "rnnoise_model_sha256": RNNOISE_MODEL_SHA256,
    "rnnoise_wet_mix": 0.85,
    "rnnoise_cascade_input_mix": 0.15,
    "loudness_match": "linear gain to source integrated loudness with -0.1 dBFS cap",
    "dynamic_loudness_normalization": False,
    "compression": False,
    "output_sample_rate": 24_000,
    "output_channels": 1,
    "output_format": "WAV/PCM_24",
}
PROFILE_HASH = hashlib.sha256(
    json.dumps(PROFILE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

_MALLOC_TRIM = getattr(ctypes.CDLL(None), "malloc_trim", None)
if _MALLOC_TRIM is not None:
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int


def release_host_memory() -> bool:
    """Return unused CPU allocations to the operating system when possible."""
    gc.collect()
    if _MALLOC_TRIM is None:
        return False
    return bool(_MALLOC_TRIM(0))


def read_prior(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        str(row["utterance_id"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def inspect_output(path: Path, expected_frames: int) -> dict[str, Any]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    info = sf.info(path)
    errors = []
    if path.is_symlink() or not path.is_file():
        errors.append("output is not a physical file")
    if sample_rate != 24_000 or info.channels != 1:
        errors.append("output is not mono 24 kHz")
    if info.frames != expected_frames:
        errors.append("output sample count changed")
    if info.subtype != "PCM_24":
        errors.append("output is not PCM 24-bit")
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        errors.append("output waveform is invalid")
    if audio.size and float(np.max(np.abs(audio))) >= 1.0:
        errors.append("output clips")
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": float(info.frames / info.samplerate),
        "format": f"{info.format}/{info.subtype}",
        "errors": errors,
    }


def output_is_valid(target: Path, source: Path, expected_frames: int) -> bool:
    metadata_path = target.with_suffix(".wav.json")
    if not target.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("status") == "PASS"
            and metadata.get("profile_hash") == PROFILE_HASH
            and metadata.get("source") == str(source.resolve())
            and metadata.get("source_sha256") == sha256(source)
            and metadata.get("output_sha256") == sha256(target)
            and not inspect_output(target, expected_frames)["errors"]
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def trusted_restart_output_is_valid(
    target: Path, source: Path, prior: dict[str, Any]
) -> bool:
    """Use recorded hashes after an in-process restart in the same run."""
    metadata_path = target.with_suffix(".wav.json")
    if target.is_symlink() or not target.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("status") == "PASS"
            and metadata.get("profile_hash") == PROFILE_HASH
            and metadata.get("source") == str(source.resolve())
            and metadata.get("output_sha256") == prior.get("audio_sha256")
            and target.stat().st_size > 44
        )
    except (OSError, TypeError, ValueError):
        return False


class Cascade:
    """Load the four neural stages once for one worker."""

    def __init__(self, device: str, cache: Path, rnnoise_binary: Path) -> None:
        torch.cuda.set_device(torch.device(device))
        self.clearervoice = MossFormerBackend(device, cache / "mossformer2")
        self.sidon = SidonBackend(device, cache / "sidon")
        self.deepfilter = DeepFilterDefaultBackend(cache / "deepfilternet")
        self.rnnoise_binary = rnnoise_binary.resolve(strict=True)
        self.identity = {
            "clearervoice": self.clearervoice.identity,
            "sidon": self.sidon.identity,
            "deepfilternet3": {
                "name": "DeepFilterNet3",
                "version": "0.5.6",
                "settings": PROFILE["deepfilternet_settings"],
            },
            "rnnoise85": {
                "commit": RNNOISE_COMMIT,
                "model_sha256": RNNOISE_MODEL_SHA256,
                "wet_mix": 0.85,
            },
        }

    @torch.inference_mode()
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        clear, clear_rate = self.clearervoice.process(audio, sample_rate)[
            "mossformer2_no_compression"
        ]
        sidon, sidon_rate = self.sidon.process(clear, clear_rate)[
            "sidon_no_compression"
        ]
        postprocessed = deess_declick_limit_channel(sidon, sidon_rate)
        deepfilter = self.deepfilter.channel(postprocessed, 24_000)
        rnnoise = rnnoise_channel(deepfilter, 24_000, self.rnnoise_binary)
        return resample_channel(rnnoise, 24_000, sample_rate)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-audio-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--rnnoise-binary", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    parser.add_argument(
        "--memory-trim-interval",
        type=int,
        default=8,
        help="Release unused host allocations after this many processed files.",
    )
    parser.add_argument(
        "--maximum-new-files-per-process",
        type=int,
        default=128,
        help="Reload models after this many new files to bound host memory.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("the shard index must be inside the shard count")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.memory_trim_interval < 1:
        parser.error("--memory-trim-interval must be positive")
    if args.maximum_new_files_per_process < 1:
        parser.error("--maximum-new-files-per-process must be positive")

    base = pd.read_parquet(args.manifest, columns=["utterance_id"])
    inputs = pd.read_parquet(
        args.input_audio_manifest, columns=["utterance_id", "audio_path"]
    )
    if base["utterance_id"].duplicated().any() or inputs["utterance_id"].duplicated().any():
        raise SystemExit("an input manifest has duplicate utterance IDs")
    frame = base.merge(inputs, on="utterance_id", validate="one_to_one").sort_values(
        "utterance_id"
    )
    if len(frame) != len(base):
        raise SystemExit("the clean input manifest does not cover the corpus")
    rows = frame.iloc[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows.head(args.limit)
    prior = read_prior(args.output_results) if args.resume else {}
    trusted_self_restart = os.environ.get("UKTTS_CASCADE_SELF_RESTART") == "1"
    terminal: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, Path, int, int]] = []
    for row in rows.itertuples(index=False):
        identifier = str(row.utterance_id)
        source = Path(str(row.audio_path))
        target = args.output_root / f"{identifier}.wav"
        old = prior.get(identifier)
        attempts = int(old.get("processing_attempts", 0)) if old else 0
        trusted = bool(
            trusted_self_restart
            and old
            and old.get("processing_status") == "ok"
            and trusted_restart_output_is_valid(target, source, old)
        )
        if trusted:
            terminal[identifier] = old
            continue
        info = sf.info(source)
        if old and old.get("processing_status") == "ok" and output_is_valid(
            target, source, info.frames
        ):
            terminal[identifier] = old
        elif (
            old
            and old.get("processing_status") == "failed"
            and attempts >= args.maximum_attempts
        ):
            terminal[identifier] = old
        else:
            pending.append((identifier, source, attempts + 1, int(info.frames)))

    cascade = Cascade(args.device, args.model_cache, args.rnnoise_binary) if pending else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = 0
    restart_required = False
    started_all = time.monotonic()
    with args.output_results.open("w", encoding="utf-8") as stream:
        for identifier in sorted(terminal):
            stream.write(json.dumps(terminal[identifier], sort_keys=True) + "\n")
        for identifier, source, attempts, expected_frames in pending:
            target = args.output_root / f"{identifier}.wav"
            started = time.monotonic()
            audio = output = matched = decoded = None
            try:
                audio, sample_rate = sf.read(source, always_2d=True, dtype="float32")
                if sample_rate != 24_000 or audio.shape[1] != 1:
                    raise RuntimeError("the boundary-trimmed input is not mono 24 kHz")
                if len(audio) != expected_frames or not np.isfinite(audio).all():
                    raise RuntimeError("the boundary-trimmed input waveform is invalid")
                input_lufs = measure_lufs(audio, sample_rate)
                output = cascade.process(audio[:, 0], sample_rate)  # type: ignore[union-attr]
                output = fit_sample_count(output[:, None], expected_frames)
                matched, loudness = match_loudness_and_prevent_clipping(
                    output, sample_rate, input_lufs
                )
                atomic_write_pcm24(target, matched, sample_rate)
                check = inspect_output(target, expected_frames)
                if check["errors"]:
                    raise RuntimeError(f"output validation failed: {check['errors']}")
                decoded, _ = sf.read(target, always_2d=True, dtype="float32")
                output_lufs = measure_lufs(decoded, sample_rate)
                metadata = {
                    "status": "PASS",
                    "utterance_id": identifier,
                    "profile": PROFILE,
                    "profile_hash": PROFILE_HASH,
                    "model_identity": cascade.identity,  # type: ignore[union-attr]
                    "source": str(source.resolve()),
                    "source_sha256": sha256(source),
                    "output": str(target.resolve()),
                    "output_sha256": sha256(target),
                    "input_metrics": waveform_metrics(audio, sample_rate, lufs=input_lufs),
                    "output_metrics": waveform_metrics(decoded, sample_rate, lufs=output_lufs),
                    "loudness_match": loudness,
                    "elapsed_seconds": time.monotonic() - started,
                }
                write_json_atomic(target.with_suffix(".wav.json"), metadata)
                result = {
                    "utterance_id": identifier,
                    "source_audio_path": str(source.resolve()),
                    "audio_path": str(target.resolve()),
                    "audio_sha256": metadata["output_sha256"],
                    "duration": check["duration"],
                    "sample_rate": check["sample_rate"],
                    "channels": check["channels"],
                    "format": check["format"],
                    "processing_config_hash": PROFILE_HASH,
                    "processing_status": "ok",
                    "processing_attempts": attempts,
                    "elapsed_seconds": metadata["elapsed_seconds"],
                }
                completed += 1
            except Exception as error:
                result = {
                    "utterance_id": identifier,
                    "source_audio_path": str(source.resolve()),
                    "audio_path": str(target.resolve()),
                    "processing_config_hash": PROFILE_HASH,
                    "processing_status": "failed",
                    "processing_attempts": attempts,
                    "processing_error": f"{type(error).__name__}: {error}",
                }
                failures += 1
            terminal[identifier] = result
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "completed": completed,
                        "failures": failures,
                        "utterance_id": identifier,
                    }
                ),
                flush=True,
            )
            audio = output = matched = decoded = None
            if (completed + failures) % args.memory_trim_interval == 0:
                release_host_memory()
            if completed + failures >= args.maximum_new_files_per_process:
                restart_required = completed + failures < len(pending)
                if restart_required:
                    print(
                        json.dumps(
                            {
                                "event": "worker_model_reload",
                                "processed_now": completed + failures,
                                "shard": args.shard_index,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    break
    if restart_required:
        release_host_memory()
        os.environ["UKTTS_CASCADE_SELF_RESTART"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    summary = {
        "status": "PASS" if failures == 0 else "FAIL",
        "shard_index": args.shard_index,
        "records": len(rows),
        "completed_now": completed,
        "failures_now": failures,
        "elapsed_seconds": time.monotonic() - started_all,
        "profile_hash": PROFILE_HASH,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if failures == 0 or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
