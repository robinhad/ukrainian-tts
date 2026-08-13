#!/usr/bin/env python3
"""Select, process, and validate a random fair enhancement comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

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
from training.scripts.run_enhancement_review_backend import (
    MOSS_CODE_COMMIT,
    MOSS_MODEL_REVISION,
    RESEMBLE_CODE_COMMIT,
    RESEMBLE_MODEL_REVISION,
    MossFormerBackend,
    ResembleBackend,
    SidonBackend,
)

SUFFIXES = {
    "rnnoise85": "_rnnoise85.wav",
    "deepfilternet3": "_deepfilternet3.wav",
    "resemble_denoise": "_resemble_denoise.wav",
    "resemble_enhance": "_resemble_enhance.wav",
    "clearervoice": "_clearervoice.wav",
    "clearervoice_resemble_enhance": "_clearervoice_resemble_enhance.wav",
    "clearervoice_sidon": "_clearervoice_sidon.wav",
    "clearervoice_sidon_rnnoise85": "_clearervoice_sidon_rnnoise85.wav",
    "clearervoice_sidon_deepfilternet3": (
        "_clearervoice_sidon_deepfilternet3.wav"
    ),
    "clearervoice_sidon_deepfilternet3_rnnoise85": (
        "_clearervoice_sidon_deepfilternet3_rnnoise85.wav"
    ),
}
RNNOISE_COMMIT = "70f1d256acd4b34a572f999a05c87bf00b67730d"
RNNOISE_MODEL_SHA256 = "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select(args: argparse.Namespace) -> int:
    frame = pd.read_parquet(args.manifest)
    training = frame.loc[frame["split"].astype(str).str.endswith("_train")].copy()
    if len(training) < args.count:
        raise ValueError("the training split has too few records")
    selected = training.sample(n=args.count, random_state=args.seed, replace=False)
    if not selected["source_id"].astype(str).str.contains("voa", case=False).any():
        voa = training.loc[
            training["source_id"].astype(str).str.contains("voa", case=False)
        ].sample(n=1, random_state=args.seed)
        non_voa = training.drop(index=voa.index).sample(
            n=args.count - 1, random_state=args.seed, replace=False
        )
        selected = pd.concat([voa, non_voa])
    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        name = f"{index:02d}_{row.utterance_id}"
        source = Path(row.audio_path)
        target = args.output / f"{name}_input.wav"
        shutil.copy2(source, target)
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f"failed to copy input: {target}")
        audio, sample_rate = sf.read(target, dtype="float32", always_2d=True)
        lufs = measure_lufs(audio, sample_rate)
        records.append(
            {
                "index": index,
                "name": name,
                "utterance_id": str(row.utterance_id),
                "dataset": str(row.source_id),
                "speaker_id": str(row.speaker_id),
                "text": str(row.text_sanitized),
                "source_audio": str(source.resolve()),
                "input_wav": str(target.resolve()),
                "input_sha256": sha256(target),
                "input_metrics": waveform_metrics(audio, sample_rate, lufs=lufs),
            }
        )
    selection = args.output / "selection.jsonl"
    with selection.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "records": len(records), "selection": str(selection)}))
    return 0


def resample_channel(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy()).view(1, -1)
    if source_rate != target_rate:
        tensor = AF.resample(tensor, source_rate, target_rate)
    return tensor.squeeze(0).cpu().numpy()


def rnnoise_channel(audio: np.ndarray, sample_rate: int, binary: Path) -> np.ndarray:
    dry_48k = resample_channel(audio, sample_rate, 48_000)
    frame = 480
    padded_count = ((len(dry_48k) + frame - 1) // frame) * frame
    padded = np.pad(dry_48k, (0, padded_count - len(dry_48k)))
    pcm = np.clip(np.round(padded * 32767.0), -32768, 32767).astype("<i2")
    with tempfile.TemporaryDirectory(prefix="uktts-rnnoise-") as temporary:
        input_raw = Path(temporary) / "input.pcm"
        output_raw = Path(temporary) / "output.pcm"
        pcm.tofile(input_raw)
        subprocess.run([str(binary), str(input_raw), str(output_raw)], check=True)
        wet = np.fromfile(output_raw, dtype="<i2").astype(np.float32) / 32768.0
    wet = fit_sample_count(wet[:, None], len(dry_48k))[:, 0]
    mixed = 0.85 * wet + 0.15 * dry_48k
    return resample_channel(mixed, 48_000, sample_rate)


class DeepFilterDefaultBackend:
    def __init__(self, cache: Path) -> None:
        from training.audio_enhancement.deepfilternet_compat import install

        install()
        os.environ["XDG_CACHE_HOME"] = str(cache.resolve())
        from df.enhance import enhance, init_df

        self.enhance = enhance
        self.model, self.state, _ = init_df(
            default_model="DeepFilterNet3",
            post_filter=False,
            log_file=None,
            log_level="WARNING",
        )

    def channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy()).view(1, -1)
        if sample_rate != 48_000:
            tensor = AF.resample(tensor, sample_rate, 48_000)
        output = self.enhance(self.model, self.state, tensor, pad=True)
        return resample_channel(output.squeeze(0).cpu().numpy(), 48_000, sample_rate)


def make_backend(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.backend == "deepfilternet3":
        backend = DeepFilterDefaultBackend(args.model_cache)
        return backend, {
            "name": "DeepFilterNet3",
            "version": "0.5.6",
            "settings": "default pretrained; post-filter off; no attenuation limit",
            "code_license": "MIT OR Apache-2.0",
            "model_license": "MIT OR Apache-2.0",
        }
    if args.backend in {"resemble_denoise", "resemble_enhance"}:
        backend = ResembleBackend(args.device, args.model_cache)
        identity = {
            **backend.identity,
            "code_license": "MIT",
            "model_license": "MIT",
            "mode": (
                "denoise only"
                if args.backend == "resemble_denoise"
                else "full enhancement"
            ),
        }
        if args.backend == "resemble_denoise":
            identity.pop("full_enhancement", None)
            identity["denoise_only"] = True
        return backend, identity
    if args.backend == "clearervoice":
        backend = MossFormerBackend(args.device, args.model_cache)
        return backend, {
            **backend.identity,
            "code_license": "Apache-2.0",
            "model_license": "Apache-2.0",
        }
    if args.backend == "clearervoice_resemble_enhance":
        clearervoice = MossFormerBackend(
            args.device,
            args.model_cache / "mossformer2",
        )
        resemble = ResembleBackend(
            args.device,
            args.model_cache / "resemble",
        )
        return (clearervoice, resemble), {
            "name": "ClearerVoice then Resemble Enhance",
            "order": ["MossFormer2_SE_48K", "Resemble Enhance full"],
            "intermediate_loudness_matching": False,
            "clearervoice": {
                **clearervoice.identity,
                "code_license": "Apache-2.0",
                "model_license": "Apache-2.0",
            },
            "resemble_enhance": {
                **resemble.identity,
                "code_license": "MIT",
                "model_license": "MIT",
                "mode": "full enhancement",
            },
        }
    if args.backend == "clearervoice_sidon":
        clearervoice = MossFormerBackend(
            args.device,
            args.model_cache / "mossformer2",
        )
        sidon = SidonBackend(
            args.device,
            args.model_cache / "sidon",
        )
        return (clearervoice, sidon), {
            "name": "ClearerVoice then Sidon",
            "order": ["MossFormer2_SE_48K", "Sidon"],
            "intermediate_loudness_matching": False,
            "clearervoice": {
                **clearervoice.identity,
                "code_license": "Apache-2.0",
                "model_license": "Apache-2.0",
            },
            "sidon": {
                **sidon.identity,
                "code_license": "MIT",
                "model_license": "MIT",
            },
        }
    if args.backend == "clearervoice_sidon_rnnoise85":
        if not args.rnnoise_binary.is_file():
            raise FileNotFoundError(args.rnnoise_binary)
        clearervoice = MossFormerBackend(
            args.device,
            args.model_cache / "mossformer2",
        )
        sidon = SidonBackend(
            args.device,
            args.model_cache / "sidon",
        )
        return (clearervoice, sidon, args.rnnoise_binary.resolve()), {
            "name": "ClearerVoice then Sidon then RNNoise85",
            "order": ["MossFormer2_SE_48K", "Sidon", "Xiph RNNoise85"],
            "intermediate_loudness_matching": False,
            "clearervoice": {
                **clearervoice.identity,
                "code_license": "Apache-2.0",
                "model_license": "Apache-2.0",
            },
            "sidon": {
                **sidon.identity,
                "code_license": "MIT",
                "model_license": "MIT",
            },
            "rnnoise85": {
                "name": "Xiph RNNoise",
                "code_commit": RNNOISE_COMMIT,
                "model_archive_sha256": RNNOISE_MODEL_SHA256,
                "code_license": "BSD-3-Clause",
                "model_license": "BSD-3-Clause",
                "wet_mix": 0.85,
                "cascade_input_mix": 0.15,
            },
        }
    if args.backend == "clearervoice_sidon_deepfilternet3":
        clearervoice = MossFormerBackend(
            args.device,
            args.model_cache / "mossformer2",
        )
        sidon = SidonBackend(
            args.device,
            args.model_cache / "sidon",
        )
        deepfilter = DeepFilterDefaultBackend(
            args.model_cache / "deepfilternet"
        )
        return (clearervoice, sidon, deepfilter), {
            "name": "ClearerVoice then Sidon then DeepFilterNet3",
            "order": ["MossFormer2_SE_48K", "Sidon", "DeepFilterNet3"],
            "intermediate_loudness_matching": False,
            "clearervoice": {
                **clearervoice.identity,
                "code_license": "Apache-2.0",
                "model_license": "Apache-2.0",
            },
            "sidon": {
                **sidon.identity,
                "code_license": "MIT",
                "model_license": "MIT",
            },
            "deepfilternet3": {
                "name": "DeepFilterNet3",
                "version": "0.5.6",
                "settings": (
                    "default pretrained; post-filter off; no attenuation limit"
                ),
                "code_license": "MIT OR Apache-2.0",
                "model_license": "MIT OR Apache-2.0",
            },
        }
    if args.backend == "clearervoice_sidon_deepfilternet3_rnnoise85":
        if not args.rnnoise_binary.is_file():
            raise FileNotFoundError(args.rnnoise_binary)
        clearervoice = MossFormerBackend(
            args.device,
            args.model_cache / "mossformer2",
        )
        sidon = SidonBackend(
            args.device,
            args.model_cache / "sidon",
        )
        deepfilter = DeepFilterDefaultBackend(
            args.model_cache / "deepfilternet"
        )
        return (
            clearervoice,
            sidon,
            deepfilter,
            args.rnnoise_binary.resolve(),
        ), {
            "name": (
                "ClearerVoice then Sidon then DeepFilterNet3 then RNNoise85"
            ),
            "order": [
                "MossFormer2_SE_48K",
                "Sidon",
                "DeepFilterNet3",
                "Xiph RNNoise85",
            ],
            "intermediate_loudness_matching": False,
            "clearervoice": {
                **clearervoice.identity,
                "code_license": "Apache-2.0",
                "model_license": "Apache-2.0",
            },
            "sidon": {
                **sidon.identity,
                "code_license": "MIT",
                "model_license": "MIT",
            },
            "deepfilternet3": {
                "name": "DeepFilterNet3",
                "version": "0.5.6",
                "settings": (
                    "default pretrained; post-filter off; no attenuation limit"
                ),
                "code_license": "MIT OR Apache-2.0",
                "model_license": "MIT OR Apache-2.0",
            },
            "rnnoise85": {
                "name": "Xiph RNNoise",
                "code_commit": RNNOISE_COMMIT,
                "model_archive_sha256": RNNOISE_MODEL_SHA256,
                "code_license": "BSD-3-Clause",
                "model_license": "BSD-3-Clause",
                "wet_mix": 0.85,
                "cascade_input_mix": 0.15,
            },
        }
    if args.backend == "rnnoise85":
        if not args.rnnoise_binary.is_file():
            raise FileNotFoundError(args.rnnoise_binary)
        return args.rnnoise_binary.resolve(), {
            "name": "Xiph RNNoise",
            "code_commit": RNNOISE_COMMIT,
            "model_archive_sha256": RNNOISE_MODEL_SHA256,
            "code_license": "BSD-3-Clause",
            "model_license": "BSD-3-Clause",
            "wet_mix": 0.85,
            "original_mix": 0.15,
        }
    raise ValueError(args.backend)


def backend_channel(
    backend_name: str,
    backend: Any,
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    if backend_name == "rnnoise85":
        return rnnoise_channel(audio, sample_rate, backend)
    if backend_name == "deepfilternet3":
        return backend.channel(audio, sample_rate)
    if backend_name in {"resemble_denoise", "resemble_enhance"}:
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
        model = (
            backend.model.denoiser
            if backend_name == "resemble_denoise"
            else backend.model
        )
        result, result_rate = backend.inference(
            model=model,
            dwav=waveform,
            sr=sample_rate,
            device=backend.device,
        )
        return resample_channel(result.cpu().numpy(), int(result_rate), sample_rate)
    if backend_name == "clearervoice_resemble_enhance":
        clearervoice, resemble = backend
        outputs = clearervoice.process(audio, sample_rate)
        intermediate, intermediate_rate = outputs["mossformer2_no_compression"]
        waveform = torch.from_numpy(
            np.asarray(intermediate, dtype=np.float32).copy()
        )
        result, result_rate = resemble.inference(
            model=resemble.model,
            dwav=waveform,
            sr=intermediate_rate,
            device=resemble.device,
        )
        return resample_channel(result.cpu().numpy(), int(result_rate), sample_rate)
    if backend_name == "clearervoice_sidon":
        clearervoice, sidon = backend
        outputs = clearervoice.process(audio, sample_rate)
        intermediate, intermediate_rate = outputs["mossformer2_no_compression"]
        sidon_outputs = sidon.process(intermediate, intermediate_rate)
        result, result_rate = sidon_outputs["sidon_no_compression"]
        return resample_channel(result, result_rate, sample_rate)
    if backend_name == "clearervoice_sidon_rnnoise85":
        clearervoice, sidon, rnnoise_binary = backend
        outputs = clearervoice.process(audio, sample_rate)
        intermediate, intermediate_rate = outputs["mossformer2_no_compression"]
        sidon_outputs = sidon.process(intermediate, intermediate_rate)
        sidon_result, sidon_rate = sidon_outputs["sidon_no_compression"]
        result = rnnoise_channel(sidon_result, sidon_rate, rnnoise_binary)
        return resample_channel(result, sidon_rate, sample_rate)
    if backend_name == "clearervoice_sidon_deepfilternet3":
        clearervoice, sidon, deepfilter = backend
        outputs = clearervoice.process(audio, sample_rate)
        intermediate, intermediate_rate = outputs["mossformer2_no_compression"]
        sidon_outputs = sidon.process(intermediate, intermediate_rate)
        sidon_result, sidon_rate = sidon_outputs["sidon_no_compression"]
        result = deepfilter.channel(sidon_result, sidon_rate)
        return resample_channel(result, sidon_rate, sample_rate)
    if backend_name == "clearervoice_sidon_deepfilternet3_rnnoise85":
        clearervoice, sidon, deepfilter, rnnoise_binary = backend
        outputs = clearervoice.process(audio, sample_rate)
        intermediate, intermediate_rate = outputs["mossformer2_no_compression"]
        sidon_outputs = sidon.process(intermediate, intermediate_rate)
        sidon_result, sidon_rate = sidon_outputs["sidon_no_compression"]
        deepfilter_result = deepfilter.channel(sidon_result, sidon_rate)
        result = rnnoise_channel(
            deepfilter_result,
            sidon_rate,
            rnnoise_binary,
        )
        return resample_channel(result, sidon_rate, sample_rate)
    outputs = backend.process(audio, sample_rate)
    result, result_rate = outputs["mossformer2_no_compression"]
    return resample_channel(result, result_rate, sample_rate)


def process(args: argparse.Namespace) -> int:
    records = read_jsonl(args.selection)
    backend, identity = make_backend(args)
    completed = []
    failures = []
    started_all = time.monotonic()
    for record in records:
        target = args.output / f"{record['name']}{SUFFIXES[args.backend]}"
        started = time.monotonic()
        try:
            source, sample_rate = sf.read(
                record["input_wav"], dtype="float32", always_2d=True
            )
            channels = []
            for channel in range(source.shape[1]):
                channels.append(
                    backend_channel(
                        args.backend,
                        backend,
                        source[:, channel],
                        sample_rate,
                    )
                )
            sample_count = len(source)
            shortest = min(len(channel) for channel in channels)
            combined = np.column_stack([channel[:shortest] for channel in channels])
            combined = fit_sample_count(combined, sample_count)
            matched, loudness = match_loudness_and_prevent_clipping(
                combined,
                sample_rate,
                record["input_metrics"]["integrated_loudness_lufs"],
            )
            atomic_write_pcm24(target, matched, sample_rate)
            decoded, output_rate = sf.read(target, dtype="float32", always_2d=True)
            output_lufs = measure_lufs(decoded, output_rate)
            metrics = waveform_metrics(decoded, output_rate, lufs=output_lufs)
            errors = []
            if output_rate != sample_rate:
                errors.append("sample rate changed")
            if metrics["sample_count"] != record["input_metrics"]["sample_count"]:
                errors.append("sample count changed")
            if metrics["channel_count"] != record["input_metrics"]["channel_count"]:
                errors.append("channel count changed")
            if sf.info(target).subtype != "PCM_24":
                errors.append("output is not PCM 24-bit")
            if metrics["peak_absolute"] >= 1.0:
                errors.append("output clips")
            if not metrics["finite"]:
                errors.append("output is not finite")
            metadata = {
                "status": "PASS" if not errors else "FAIL",
                "backend": args.backend,
                "backend_identity": identity,
                "dataset": record["dataset"],
                "utterance_id": record["utterance_id"],
                "input_wav": record["input_wav"],
                "output_wav": str(target.resolve()),
                "input_metrics": record["input_metrics"],
                "output_metrics": metrics,
                "loudness_match": loudness,
                "processing": {
                    "resample_only_for_model": True,
                    "returned_to_original_sample_rate": True,
                    "silence_trim": False,
                    "vad_removal": False,
                    "declick": False,
                    "normalization": "linear match to original integrated loudness",
                    "output_encoding": "PCM_24",
                },
                "elapsed_seconds": time.monotonic() - started,
                "output_sha256": sha256(target),
                "errors": errors,
            }
            write_json(target.with_suffix(".wav.json"), metadata)
            if errors:
                raise RuntimeError(str(errors))
            completed.append(metadata)
            print(json.dumps({"backend": args.backend, "completed": len(completed), "name": record["name"]}), flush=True)
        except Exception as error:
            failures.append({"name": record["name"], "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(failures[-1]), file=sys.stderr, flush=True)
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "backend": args.backend,
        "identity": identity,
        "completed": len(completed),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started_all,
    }
    write_json(args.output / f"backend_{args.backend}_summary.json", summary)
    return 0 if not failures else 1


def finalize(args: argparse.Namespace) -> int:
    records = read_jsonl(args.selection)
    rows = []
    metric_rows = []
    errors = []
    for record in records:
        row = {
            "dataset": record["dataset"],
            "utterance_id": record["utterance_id"],
            "text": record["text"],
            "input": Path(record["input_wav"]).name,
        }
        for backend, suffix in SUFFIXES.items():
            path = args.output / f"{record['name']}{suffix}"
            row[backend] = path.name
            metadata = path.with_suffix(".wav.json")
            if not path.is_file() or path.is_symlink() or not metadata.is_file():
                errors.append(f"missing physical output: {path}")
                continue
            details = json.loads(metadata.read_text(encoding="utf-8"))
            if details.get("status") != "PASS":
                errors.append(f"failed output: {path}")
                continue
            input_metrics = details["input_metrics"]
            output_metrics = details["output_metrics"]
            metric_rows.append(
                {
                    "dataset": record["dataset"],
                    "utterance_id": record["utterance_id"],
                    "backend": backend,
                    "input_hnr_db": input_metrics["hnr_db"],
                    "output_hnr_db": output_metrics["hnr_db"],
                    "input_noise_floor_dbfs": input_metrics["noise_floor_dbfs"],
                    "output_noise_floor_dbfs": output_metrics["noise_floor_dbfs"],
                    "input_peak_dbfs": input_metrics["peak_dbfs"],
                    "output_peak_dbfs": output_metrics["peak_dbfs"],
                    "input_lufs": input_metrics["integrated_loudness_lufs"],
                    "output_lufs": output_metrics["integrated_loudness_lufs"],
                    "input_duration_seconds": input_metrics["duration_seconds"],
                    "output_duration_seconds": output_metrics["duration_seconds"],
                    "input_sample_count": input_metrics["sample_count"],
                    "output_sample_count": output_metrics["sample_count"],
                    "input_channel_count": input_metrics["channel_count"],
                    "output_channel_count": output_metrics["channel_count"],
                }
            )
        rows.append(row)
    fields = ["dataset", "utterance_id", "text", "input", *SUFFIXES]
    with (args.output / "manifest.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    metric_fields = [
        "dataset",
        "utterance_id",
        "backend",
        "input_hnr_db",
        "output_hnr_db",
        "input_noise_floor_dbfs",
        "output_noise_floor_dbfs",
        "input_peak_dbfs",
        "output_peak_dbfs",
        "input_lufs",
        "output_lufs",
        "input_duration_seconds",
        "output_duration_seconds",
        "input_sample_count",
        "output_sample_count",
        "input_channel_count",
        "output_channel_count",
    ]
    with (args.output / "metrics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)
    identities = {}
    for backend in SUFFIXES:
        summary_path = args.output / f"backend_{backend}_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            identities[backend] = summary.get("identity")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "selection_seed": args.seed,
        "input_revision": "expanded_v4_trim_only",
        "reason_for_input_revision": "latest processed training manifest that includes VOA and has no prior denoiser",
        "utterance_count": len(records),
        "voa_count": sum("voa" in record["dataset"].lower() for record in records),
        "output_count": len(records) * len(SUFFIXES),
        "metrics_row_count": len(metric_rows),
        "metrics_log": str((args.output / "metrics.tsv").resolve()),
        "uses_symlinks": False,
        "backend_identities": identities,
        "errors": errors,
        "output": str(args.output.resolve()),
    }
    write_json(args.report, report)
    readme = """# Random Training-Audio Enhancement Comparison

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

This directory has 10 input WAV files and 10 matched outputs for each input.
The input selection includes VOA. The output file suffix identifies the model.
Listen to the input first. Then listen to all 10 outputs for the same item.

The process matches each output loudness to its input loudness. It keeps the
exact input sample rate, channel count, duration, and sample count. All outputs
use PCM 24-bit encoding. The files are physical copies and outputs. They are
not symbolic links.

Use `manifest.tsv` to identify each item. Use `metrics.tsv` to compare HNR,
noise floor, peak, loudness, duration, and sample count. Each output also has a
`.wav.json` file with complete processing metadata.

The process does not add de-clicking, silence trim, VAD removal, a high-pass
filter, de-essing, compression, or dynamic loudness normalization.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--manifest", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--count", type=int, default=10)
    select_parser.add_argument("--seed", type=int, default=20260813)
    select_parser.set_defaults(function=select)
    process_parser = subparsers.add_parser("process")
    process_parser.add_argument("--backend", choices=sorted(SUFFIXES), required=True)
    process_parser.add_argument("--selection", type=Path, required=True)
    process_parser.add_argument("--output", type=Path, required=True)
    process_parser.add_argument("--device", default="cuda:0")
    process_parser.add_argument("--model-cache", type=Path, required=True)
    process_parser.add_argument(
        "--rnnoise-binary",
        type=Path,
        default=REPOSITORY_ROOT / "training/vendor/rnnoise-src/examples/rnnoise_demo",
    )
    process_parser.set_defaults(function=process)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--selection", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.add_argument("--report", type=Path, required=True)
    finalize_parser.add_argument("--seed", type=int, default=20260813)
    finalize_parser.set_defaults(function=finalize)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
