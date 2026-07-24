#!/usr/bin/env python3
"""Make a standalone release candidate from validated multispeaker artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pass_report(path: Path, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"{label} does not have PASS status: {path}")
    return report


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"a required file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.txt"
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in paths
    ]
    (root / "checksums.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_cards(
    output: Path,
    *,
    milestone: str,
    evaluation: dict,
    dataset: dict,
    source: dict,
    power: dict,
    metrics: dict,
    dataset_hash: str,
    training_commit: str,
    repository_commit: str,
) -> None:
    model_card = f"""# Ukrainian JETS multispeaker model

## Model

This release candidate contains a JETS model for Ukrainian speech synthesis.
ESPnet2 trained the model for {milestone} iterations. The model makes a raw
24 kHz mono waveform. It uses eSpeak-ng Ukrainian phonemes with stress marks.
It uses a 192-value ECAPA speaker embedding.

The model contains the waveform generator. It does not need a separate neural
vocoder. It does not use a separate verbalizer, stress model, or G2P model.

## Use

Use the same frontend, token list, statistics, and eSpeak-ng data that are in
this package. The Dmytro example is zero-shot. Dmytro data was not in the train
set.

Training commit: `{training_commit}`

Packaging commit: `{repository_commit}`

Dataset SHA-256: `{dataset_hash}`

## Limit

The public Common Voice source does not contain stable client IDs. The training
pipeline uses one utterance-level ECAPA embedding for each Common Voice file.
Do a listening test before you publish this model.
"""
    data_card = f"""# Data card

## Sources

The train data contains {source["source_counts"]["common_voice"]:,} Common Voice
records and {source["source_counts"]["lada"]:,} Lada records. It contains
{source["dmytro_training_rows"]} Dmytro records.

The complete prepared set contains {dataset["utterances"]:,} utterances and
{dataset["duration_hours"]:.3f} hours. The train split contains
{source["split_counts"]["multispeaker_train"]:,} utterances. The development
split contains {source["split_counts"]["multispeaker_dev"]:,} utterances. The
evaluation split contains {source["split_counts"]["multispeaker_eval"]:,}
utterances.

## Preparation

The pipeline makes 24 kHz mono PCM WAV copies. It removes leading and trailing
silence with a 40 dB relative frame-RMS threshold and 100 ms boundary padding.
It does not change source files. It does not use MFA, VAD, denoise, loudness
normalization, compression, de-essing, or mastering.

See `licenses/data.yaml` for the source license records.
"""
    gpu_lines = "\n".join(
        (
            f"- GPU {gpu['index']}: mean sampled power "
            f"{gpu['mean_sampled_power_w']:.2f} W; maximum sampled power "
            f"{gpu['maximum_sampled_power_w']:.2f} W; maximum temperature "
            f"{gpu['maximum_sampled_temperature_c']} C."
        )
        for gpu in power["gpus"]
    )
    validation_scalars = metrics.get("runs", {}).get("valid", {}).get(
        "scalar_tag_count", 0
    )
    evaluation_card = f"""# Evaluation report

## Automatic checks

The fixed evaluation set contains {evaluation["wav_count"]:,} generated WAV
files. The automatic WAV validation status is PASS. The median real-time factor
is {evaluation["rtf_median"]}. The report has
{evaluation["clipping_warnings"]} clipping warnings.

The TensorBoard export contains {validation_scalars} validation scalar tags.
See `reports/tensorboard_metrics.json` for the scalar values.

## GPU samples

{gpu_lines}

## Required human check

Listen to the balanced Common Voice and Lada set before release selection.
Automatic checks do not measure naturalness.
"""
    (output / "model-card.md").write_text(model_card, encoding="utf-8")
    (output / "data-card.md").write_text(data_card, encoding="utf-8")
    (output / "evaluation-report.md").write_text(
        evaluation_card,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestone", default="25k")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    evaluation = load_pass_report(args.evaluation_report, "evaluation report")
    if evaluation.get("wav_count") != 1677:
        raise RuntimeError(
            f"the evaluation report has {evaluation.get('wav_count')} WAV files; "
            "expected 1677"
        )
    dataset_path = root / "reports/multispeaker_full_dataset.json"
    source_path = root / "data/multispeaker_full/source_report.json"
    power_path = root / "reports/power_summary_multispeaker.json"
    metrics_path = root / "reports/tensorboard_metrics_multispeaker.json"
    dataset = load_pass_report(dataset_path, "dataset report")
    source = load_pass_report(source_path, "source report")
    power = load_pass_report(power_path, "power report")
    metrics = load_pass_report(metrics_path, "TensorBoard report")
    all_manifest = root / "data/multispeaker_full/manifests/all.parquet"
    dataset_hash = sha256(all_manifest)
    repository_commit = git_commit(root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}.",
        dir=args.output.parent,
    ))
    try:
        copy_required(args.checkpoint, temporary / "model.pth")
        tts_exp = (
            root
            / "exp_multispeaker_full"
            / "tts_jets_uk_24k_multispeaker"
        )
        training_commit_path = tts_exp / "training_git_commit.txt"
        copy_required(
            training_commit_path,
            temporary / "training-commit.txt",
        )
        training_commit = training_commit_path.read_text(
            encoding="utf-8"
        ).strip()
        copy_required(tts_exp / "config.yaml", temporary / "config.yaml")
        copy_required(
            root
            / "dump_multispeaker_full"
            / "token_list"
            / "phn_espeak_ng_ukrainian"
            / "tokens.txt",
            temporary / "tokens.txt",
        )
        stats = (
            root
            / "exp_multispeaker_full"
            / "tts_stats_raw_phn_espeak_ng_ukrainian"
            / "train"
        )
        for name in (
            "feats_stats.npz",
            "pitch_stats.npz",
            "energy_stats.npz",
        ):
            copy_required(stats / name, temporary / "stats" / name)
        for path in sorted((root / "frontend").glob("*.py")):
            copy_required(path, temporary / "frontend" / path.name)
        copy_required(
            root / "data/multispeaker_full/manifests/frontend.json",
            temporary / "frontend" / "frontend.json",
        )
        copy_required(
            root / "patches/espnet-espeak-ng-ukrainian.patch",
            temporary / "frontend" / "espnet-espeak-ng-ukrainian.patch",
        )
        for source_name, output_name in (
            ("ESPEAK_NG_VERSION", "espeak-version.txt"),
            ("ESPEAK_NG_DATA_HASH", "espeak-data-hash.txt"),
            ("ESPNET_COMMIT", "espnet-commit.txt"),
            ("SPEECHBRAIN_ECAPA_COMMIT", "speechbrain-ecapa-commit.txt"),
        ):
            copy_required(
                root / "vendor" / source_name,
                temporary / output_name,
            )
        (temporary / "repository-commit.txt").write_text(
            repository_commit + "\n",
            encoding="utf-8",
        )
        (temporary / "dataset-sha256.txt").write_text(
            dataset_hash + "\n",
            encoding="utf-8",
        )
        for speaker in ("lada", "dmytro_zero_shot"):
            name = f"multispeaker_{args.milestone}_{speaker}.wav"
            copy_required(
                root / "eval/generated" / name,
                temporary / "examples" / name,
            )
            copy_required(
                root / "eval/generated" / f"{name}.json",
                temporary / "examples" / f"{name}.json",
            )
        shutil.copytree(root / "licenses", temporary / "licenses")
        report_dir = temporary / "reports"
        copy_required(
            args.evaluation_report,
            report_dir / "automatic_evaluation.json",
        )
        copy_required(power_path, report_dir / "power_summary.json")
        copy_required(metrics_path, report_dir / "tensorboard_metrics.json")
        copy_required(dataset_path, report_dir / "dataset_validation.json")
        make_cards(
            temporary,
            milestone=args.milestone,
            evaluation=evaluation,
            dataset=dataset,
            source=source,
            power=power,
            metrics=metrics,
            dataset_hash=dataset_hash,
            training_commit=training_commit,
            repository_commit=repository_commit,
        )
        write_checksums(temporary)
        temporary.replace(args.output)
    except BaseException:
        shutil.rmtree(temporary)
        raise

    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "model_sha256": sha256(args.output / "model.pth"),
                "file_count": sum(
                    path.is_file() for path in args.output.rglob("*")
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
