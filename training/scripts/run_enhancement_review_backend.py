#!/usr/bin/env python3
"""Run one enhancement backend for the matched listening review."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.audio_enhancement.review_pipeline import (
    ReviewProcessingConfig,
    SidonDeessOnlyConfig,
    apply_deessing_only,
    inspect_wav,
    load_and_trim,
    master_without_compression,
    sha256,
    write_float_wav,
    write_json,
)
from training.audio_enhancement.postprocess import load_config

SIDON_CODE_COMMIT = "c8cde2b24e4c77c599ad43a9871140cdc9beeffa"
SIDON_MODEL_REVISION = "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
RESEMBLE_CODE_COMMIT = "8e978149bfe8abab3eb77d965d579a111afdb0ff"
RESEMBLE_MODEL_REVISION = "4e3510ce4a8391159f665903544c5150bee7b2cb"
MOSS_CODE_COMMIT = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
MOSS_MODEL_REVISION = "eff8c97925c8bec812af707814b3e5d777fd4503"
W2V_BERT_MODEL_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def to_tensor(audio: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(audio, dtype=np.float32).copy()).view(1, -1)


class DeepFilterBackend:
    profiles = ("dfn3_no_compression",)

    def __init__(self, device: str, cache: Path) -> None:
        del device
        from training.audio_enhancement.deepfilternet_compat import install

        install()
        os.environ["XDG_CACHE_HOME"] = str(cache.resolve())
        from df.enhance import enhance, init_df

        self.enhance = enhance
        self.model, self.state, _ = init_df(
            "DeepFilterNet3", post_filter=False, log_file=None, log_level="WARNING"
        )

    def process(
        self, audio: np.ndarray, sample_rate: int
    ) -> dict[str, tuple[np.ndarray, int]]:
        tensor = to_tensor(audio)
        if sample_rate != 48_000:
            tensor = AF.resample(tensor, sample_rate, 48_000)
        result = self.enhance(
            self.model, self.state, tensor, pad=True, atten_lim_db=18.0
        )
        return {self.profiles[0]: (result.squeeze(0).cpu().numpy(), 48_000)}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "DeepFilterNet3",
            "version": "0.5.6",
            "attenuation_limit_db": 18.0,
        }


class SidonBackend:
    profiles = ("sidon_no_compression",)

    def __init__(self, device: str, cache: Path) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import SeamlessM4TFeatureExtractor

        self.device = torch.device(device)
        kwargs = {
            "repo_id": "sarulab-speech/sidon-v0.1",
            "revision": SIDON_MODEL_REVISION,
            "cache_dir": str(cache),
        }
        feature_path = hf_hub_download(filename="feature_extractor_cuda.pt", **kwargs)
        decoder_path = hf_hub_download(filename="decoder_cuda.pt", **kwargs)
        self.feature = (
            torch.jit.load(feature_path, map_location=self.device)
            .to(self.device)
            .eval()
        )
        self.decoder = (
            torch.jit.load(decoder_path, map_location=self.device)
            .to(self.device)
            .eval()
        )
        self.preprocessor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0",
            revision=W2V_BERT_MODEL_REVISION,
            cache_dir=str(cache),
        )
        self.weight_hashes = {
            "feature_extractor_cuda.pt": sha256(Path(feature_path)),
            "decoder_cuda.pt": sha256(Path(decoder_path)),
        }

    @torch.inference_mode()
    def process(
        self, audio: np.ndarray, sample_rate: int
    ) -> dict[str, tuple[np.ndarray, int]]:
        waveform = to_tensor(audio)
        maximum = waveform.abs().max().clamp(min=1e-7)
        waveform = 0.9 * waveform / maximum
        target_samples = round(48_000 * waveform.shape[-1] / sample_rate)
        waveform = AF.highpass_biquad(waveform, sample_rate, 50)
        waveform = AF.resample(waveform, sample_rate, 16_000)
        waveform = torch.nn.functional.pad(waveform, (0, 24_000))
        restored = []
        feature_cache = None
        for chunk in waveform.view(-1).split(16_000 * 96):
            padded = torch.nn.functional.pad(chunk, (160, 160)).cpu().numpy()
            inputs = self.preprocessor(
                padded, sampling_rate=16_000, return_tensors="pt"
            )
            feature = self.feature(inputs["input_features"].to(self.device))[
                "last_hidden_state"
            ]
            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
            restored.append(self.decoder(feature.transpose(1, 2)).view(-1)[:-960])
            feature_cache = feature[:, -1:]
        output = torch.cat(restored).detach().cpu().numpy()[:target_samples]
        return {self.profiles[0]: (output, 48_000)}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "Sidon",
            "code_commit": SIDON_CODE_COMMIT,
            "model_revision": SIDON_MODEL_REVISION,
            "feature_preprocessor_revision": W2V_BERT_MODEL_REVISION,
            "license": "MIT",
            "internal_input_highpass_hz": 50,
            "weight_hashes": self.weight_hashes,
        }


class SidonDeessOnlyBackend(SidonBackend):
    profiles = ("sidon_deess_only",)


class ResembleBackend:
    profiles = (
        "resemble_denoise_no_compression",
        "resemble_full_no_compression",
    )

    def __init__(self, device: str, cache: Path) -> None:
        from huggingface_hub import snapshot_download
        from resemble_enhance.enhancer.enhancer import Enhancer
        from resemble_enhance.enhancer.hparams import HParams
        from resemble_enhance.inference import inference

        self.device = torch.device(device)
        self.inference = inference
        run_dir = (
            Path(
                snapshot_download(
                    "ResembleAI/resemble-enhance",
                    revision=RESEMBLE_MODEL_REVISION,
                    allow_patterns=["enhancer_stage2/*"],
                    cache_dir=str(cache),
                )
            )
            / "enhancer_stage2"
        )
        hp = HParams.load(run_dir)
        model = Enhancer(hp)
        weights = run_dir / "ds/G/default/mp_rank_00_model_states.pt"
        state = torch.load(weights, map_location="cpu", weights_only=False)["module"]
        model.load_state_dict(state)
        model.eval().to(self.device)
        model.configurate_(nfe=64, solver="midpoint", lambd=1.0, tau=0.5)
        self.model = model
        self.weight_hash = sha256(weights)

    @torch.inference_mode()
    def process(
        self, audio: np.ndarray, sample_rate: int
    ) -> dict[str, tuple[np.ndarray, int]]:
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
        denoised, denoised_sr = self.inference(
            model=self.model.denoiser,
            dwav=waveform,
            sr=sample_rate,
            device=self.device,
        )
        enhanced, enhanced_sr = self.inference(
            model=self.model,
            dwav=waveform,
            sr=sample_rate,
            device=self.device,
        )
        return {
            self.profiles[0]: (denoised.cpu().numpy(), int(denoised_sr)),
            self.profiles[1]: (enhanced.cpu().numpy(), int(enhanced_sr)),
        }

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "Resemble Enhance",
            "code_commit": RESEMBLE_CODE_COMMIT,
            "model_revision": RESEMBLE_MODEL_REVISION,
            "license": "MIT",
            "full_enhancement": {
                "nfe": 64,
                "solver": "midpoint",
                "lambda": 1.0,
                "tau": 0.5,
            },
            "weight_sha256": self.weight_hash,
        }


class MossFormerBackend:
    profiles = ("mossformer2_no_compression",)

    def __init__(self, device: str, cache: Path) -> None:
        del device
        os.environ["HF_HOME"] = str(cache.resolve())
        from clearvoice import ClearVoice

        previous_directory = Path.cwd()
        cache.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(cache)
            self.model = ClearVoice(
                task="speech_enhancement", model_names=["MossFormer2_SE_48K"]
            )
        finally:
            os.chdir(previous_directory)
        checkpoint = cache / "checkpoints/MossFormer2_SE_48K/last_best_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MossFormer2 checkpoint is missing: {checkpoint}")
        self.weight_hash = sha256(checkpoint)

    @torch.inference_mode()
    def process(
        self, audio: np.ndarray, sample_rate: int
    ) -> dict[str, tuple[np.ndarray, int]]:
        tensor = to_tensor(audio)
        if sample_rate != 48_000:
            tensor = AF.resample(tensor, sample_rate, 48_000)
        batch = tensor.cpu().numpy().astype(np.float32)
        result = np.asarray(self.model(batch, False), dtype=np.float32)
        while result.ndim > 1:
            result = result[0]
        return {self.profiles[0]: (result, 48_000)}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "MossFormer2_SE_48K",
            "code_commit": MOSS_CODE_COMMIT,
            "model_revision": MOSS_MODEL_REVISION,
            "license": "Apache-2.0",
            "weight_sha256": self.weight_hash,
        }


BACKENDS = {
    "deepfilternet3": DeepFilterBackend,
    "sidon": SidonBackend,
    "sidon_deess_only": SidonDeessOnlyBackend,
    "resemble": ResembleBackend,
    "mossformer2": MossFormerBackend,
}


def valid_existing(path: Path, config_hash: str) -> bool:
    metadata_path = path.with_suffix(".wav.json")
    if not path.is_file() or path.is_symlink() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("status") == "PASS"
            and metadata.get("processing_config_hash") == config_hash
            and not inspect_wav(path)["errors"]
        )
    except (OSError, ValueError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--postprocess-config", type=Path)
    parser.add_argument("--ffmpeg", default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N records. Use this option for a smoke test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace outputs that already passed validation.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Write the backend summary to this path.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in the selected shard range")
    config = replace(
        (
            SidonDeessOnlyConfig()
            if args.backend == "sidon_deess_only"
            else ReviewProcessingConfig()
        ),
        final_postprocess=load_config(args.postprocess_config),
    )
    args.model_cache.mkdir(parents=True, exist_ok=True)
    backend = BACKENDS[args.backend](args.device, args.model_cache)
    failures = []
    completed = 0
    skipped = 0
    started_all = time.monotonic()
    records = load_records(args.selection)
    records = records[args.shard_index :: args.num_shards]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        records = records[: args.limit]
    for record in records:
        name = f"{record['listening_index']:02d}_{record['utterance_id']}.wav"
        targets = {
            profile: args.output / record["dataset"] / profile / name
            for profile in backend.profiles
        }
        if not args.force and all(
            valid_existing(path, config.digest) for path in targets.values()
        ):
            skipped += len(targets)
            continue
        try:
            audio, sample_rate, trim = load_and_trim(
                Path(record["raw_audio_path"]), config
            )
            started = time.monotonic()
            outputs = backend.process(audio, sample_rate)
            backend_seconds = time.monotonic() - started
            if set(outputs) != set(targets):
                raise RuntimeError(f"backend profile mismatch: {sorted(outputs)}")
            for profile, (waveform, output_rate) in outputs.items():
                with tempfile.TemporaryDirectory(
                    prefix="uktts-review-backend-"
                ) as temporary:
                    intermediate = Path(temporary) / "backend.wav"
                    write_float_wav(intermediate, waveform, output_rate)
                    if isinstance(config, SidonDeessOnlyConfig):
                        postprocessing = apply_deessing_only(
                            intermediate,
                            targets[profile],
                            config,
                            ffmpeg_binary=args.ffmpeg,
                        )
                        loudness = {"applied": False}
                    else:
                        loudness = master_without_compression(
                            intermediate,
                            targets[profile],
                            config,
                            ffmpeg_binary=args.ffmpeg,
                        )
                        postprocessing = {
                            "post_highpass_applied": True,
                            "loudness_normalization_applied": True,
                            "compression_applied": False,
                            "declicking_applied": True,
                            "limiting_applied": True,
                        }
                check = inspect_wav(targets[profile])
                gate_errors = list(check["errors"])
                if isinstance(config, ReviewProcessingConfig):
                    output_i = float(loudness["second_pass"]["output_i"])
                    output_tp = float(loudness["second_pass"]["output_tp"])
                    if abs(output_i - config.target_lufs) > 1.0:
                        gate_errors.append(f"loudness={output_i}")
                    if output_tp > config.target_true_peak_db + 0.05:
                        gate_errors.append(f"true_peak={output_tp}")
                metadata = {
                    "status": "PASS" if not gate_errors else "FAIL",
                    "profile": profile,
                    "utterance_id": record["utterance_id"],
                    "dataset": record["dataset"],
                    "source": record["raw_audio_path"],
                    "source_sha256": sha256(Path(record["raw_audio_path"])),
                    "audio_sha256": sha256(targets[profile]),
                    "compression_applied": False,
                    "deessing_applied": True,
                    "post_highpass_applied": postprocessing["post_highpass_applied"],
                    "loudness_normalization_applied": postprocessing[
                        "loudness_normalization_applied"
                    ],
                    "declicking_applied": True,
                    "limiting_applied": True,
                    "processing_config_hash": config.digest,
                    "processing_config": asdict(config),
                    "backend": backend.identity,
                    "requested_device": args.device,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "backend_seconds_for_record": backend_seconds,
                    "real_time_factor": backend_seconds / max(check["duration"], 1e-9),
                    "trim": trim,
                    "loudness": loudness,
                    "postprocessing": postprocessing,
                    "waveform": check,
                    "errors": gate_errors,
                }
                write_json(targets[profile].with_suffix(".wav.json"), metadata)
                if gate_errors:
                    raise RuntimeError(f"{profile} failed validation: {gate_errors}")
                completed += 1
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "completed": completed,
                        "dataset": record["dataset"],
                        "utterance_id": record["utterance_id"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:
            failures.append(
                {
                    "dataset": record["dataset"],
                    "utterance_id": record["utterance_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(json.dumps(failures[-1], sort_keys=True), file=sys.stderr, flush=True)
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "backend": args.backend,
        "identity": backend.identity,
        "completed_outputs": completed,
        "skipped_outputs": skipped,
        "failed_records": failures,
        "elapsed_seconds": time.monotonic() - started_all,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "requested_device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    summary_path = args.summary or args.output / f"backend_{args.backend}_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "failed_records"},
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
