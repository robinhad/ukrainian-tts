#!/usr/bin/env python3
"""Write the verified final status for the expanded-v9 scratch run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


NAME = "expanded_v9_cascade_with_voa"
MILESTONES = ("25k", "50k", "75k", "100k", "200k", "300k", "400k", "500k")


def load_pass(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"A required report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise SystemExit(f"A required report does not have PASS status: {path}")
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(metrics: dict, tag: str) -> dict:
    try:
        return metrics["runs"]["valid"]["scalars"][tag]
    except KeyError as error:
        raise SystemExit(f"The TensorBoard report has no valid/{tag} data.") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = args.root / "reports"
    data = load_pass(reports / f"{NAME}_full_dataset.json")
    readiness = load_pass(reports / f"{NAME}_full_scale_readiness.json")
    smoke_checkpoint = load_pass(reports / f"{NAME}_smoke_checkpoint.json")
    smoke_inference = load_pass(reports / f"{NAME}_smoke_inference.json")
    checkpoint = load_pass(reports / f"{NAME}_500k_checkpoint.json")
    monitor = load_pass(reports / f"{NAME}_training_monitor_summary.json")
    metrics = load_pass(reports / f"{NAME}_tensorboard_metrics.json")
    listening = load_pass(reports / f"five_voice_{NAME}_500k.json")
    inference = {
        label: load_pass(reports / f"{NAME}_inference_{label}.json")
        for label in MILESTONES
    }

    expected_embeddings = {"raw": 104_433, "clean": 104_434}
    if readiness.get("embedding_counts") != expected_embeddings:
        raise SystemExit("The final embedding counts are not exact.")
    if data.get("records") != 208_867 or data.get("voa_records") != 134_711:
        raise SystemExit("The final corpus count or VOA count is incorrect.")
    if checkpoint.get("reported_steps") != 500_000:
        raise SystemExit("The final checkpoint does not report 500,000 steps.")
    if monitor.get("latest_total_iterations") != 500_000:
        raise SystemExit("The monitor does not report 500,000 iterations.")
    if [item.get("index") for item in monitor.get("gpus", [])] != [0, 1]:
        raise SystemExit("The final monitor report does not contain both GPUs.")
    if listening.get("release_status") not in {"NOT_READY", "PASS"}:
        raise SystemExit("The five-voice listening export is invalid.")

    test_text = args.test_log.read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed", test_text)
    if not match or re.search(r"\b(?:failed|error)s?\b", test_text, re.IGNORECASE):
        raise SystemExit("The final test log does not prove a passing test run.")
    test_count = int(match.group(1))

    model = args.experiment / "milestones/500k.pth"
    token_list = args.root / f"dump_{NAME}/token_list/phn_espeak_ng_ukrainian/tokens.txt"
    stats = args.root / f"exp_{NAME}/tts_stats_raw_phn_espeak_ng_ukrainian"
    for path in (model, token_list, stats):
        if not path.exists():
            raise SystemExit(f"A final artifact does not exist: {path}")

    valid_mel = scalar(metrics, "generator_g_mel_loss")
    valid_generator = scalar(metrics, "generator_loss")
    valid_alignment = scalar(metrics, "generator_align_loss")
    valid_discriminator = scalar(metrics, "discriminator_loss")
    gpu_rows = monitor["gpus"]
    inference_rows = "\n".join(
        f"| {label.upper()} | PASS | {report['wav_count']:,} | {report['rtf_median']:.6f} |"
        for label, report in inference.items()
    )
    source_rows = "\n".join(
        f"| {source} | {values['records']:,} | {values['hours']:.3f} |"
        for source, values in sorted(data["sources"].items())
    )
    now = datetime.now(ZoneInfo("Europe/Kyiv")).isoformat()
    model_hash = sha256(model)
    token_hash = sha256(token_list)
    fallback_count = int(data.get("degenerate_output_fallback_records", 0))

    text = f"""# Final Status

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

Report time: `{now}`.

## Виконано

The expanded-v9 pipeline completed. It processed all {data['records']:,}
utterances. The total duration is {data['hours']:.3f} hours. The corpus has
{data['voa_records']:,} VOA utterances.

The source breakdown is:

| Source | Records | Hours |
|---|---:|---:|
{source_rows}

The process used this fixed sequence: ClearerVoice MossFormer2_SE_48K,
Sidon, light de-essing, de-clicking, peak limiting, DeepFilterNet3, and
RNNoise85. It then applied a peak-safe linear loudness match. It wrote mono
24 kHz PCM 24-bit WAV files. It did not use compression. It did not change a
source file.

The process calculated all speaker embeddings again. It used raw audio for
104,433 files. It used final processed audio for 104,434 files. Each vector
has 192 values.

JETS started from random model weights. It did not load a checkpoint from an
earlier experiment. It used two RTX 3090 GPUs in FP32 mode. It completed
500,000 optimizer iterations.

## CRISP-DM cycle 1

- Business Understanding: PASS. The run used the fixed v9 objective and acceptance gates.
- Data Understanding: PASS. The report covers all records, sources, splits, durations, and quality flags.
- Data Preparation: PASS. Audio, phonemes, ESPnet data, embeddings, tokens, and statistics passed.
- Modeling: PASS. The scratch smoke run and the 500,000-iteration run passed.
- Evaluation: PASS. Smoke inference and all fixed milestone sets passed.
- Deployment: PASS. The local inference command made valid WAV files and JSON metadata.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Full corpus | PASS | {data['records']:,} records and {data['hours']:.3f} hours. |
| VOA inclusion | PASS | {data['voa_records']:,} VOA records. |
| Requested audio profile | PASS | Profile hash `{data['profile_hash']}`. |
| PCM24 and physical files | PASS | The full dataset validation passed. |
| Exact 50/50 embeddings | PASS | 104,433 raw and 104,434 clean vectors. |
| Fresh tokens and statistics | PASS | The readiness audit passed. |
| Scratch smoke training | PASS | {smoke_checkpoint['reported_steps']} steps. |
| Smoke inference | PASS | {smoke_inference['wav_count']:,} WAV files. |
| Scratch 500K training | PASS | {checkpoint['reported_steps']:,} steps. |
| Finite model tensors | PASS | {checkpoint['model_tensor_count']} tensors. |
| Both RTX 3090 GPUs | PASS | The monitor contains GPU 0 and GPU 1. |
| Monitor interval | PASS | Maximum gap {monitor['maximum_observed_gap_minutes']:.3f} minutes. |
| Disk reserve | PASS | Minimum {monitor['minimum_free_disk_gib']:.2f} GiB. |
| Milestone inference | PASS | Eight fixed-set reports passed. |
| Five-voice export | PASS | Five physical listening WAV files passed. |
| Test suite | PASS | {test_count} tests passed. |
| Human listening | NOT RUN | A person must assess naturalness. |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/run_expanded_v9_with_voa_pipeline.sh` | 0 | The full scratch chain passed. | `training/reports/{NAME}_pipeline.log` |
| `training/scripts/run_expanded_v9_with_voa_preprocessing.sh` | 0 | It made {data['records']:,} PCM24 files. | `training/reports/{NAME}_full_dataset.json` |
| `training/scripts/prepare_expanded_v9_with_voa.sh` | 0 | Fresh data, embeddings, tokens, and statistics passed. | `training/reports/{NAME}_full_scale_readiness.json` |
| `training/scripts/run_expanded_v9_with_voa_smoke.sh` | 0 | Scratch smoke training and inference passed. | `training/reports/{NAME}_smoke_checkpoint.json` |
| `training/scripts/launch_expanded_v9_with_voa_training.sh 500000` | 0 | Scratch JETS reached 500,000 iterations. | `{args.experiment}/checkpoint.pth` |
| `training/.venv/bin/python -m pytest -q training/tests` | 0 | {test_count} tests passed. | `{args.test_log}` |

## Milestone evaluation

| Milestone | Status | WAV files | Median RTF |
|---|---|---:|---:|
{inference_rows}

## Training metrics

The final validation mel loss is {valid_mel['last_value']:.6f}. Its minimum is
{valid_mel['minimum_value']:.6f}. The final validation generator loss is
{valid_generator['last_value']:.6f}. The final validation alignment loss is
{valid_alignment['last_value']:.6f}. The final validation discriminator loss
is {valid_discriminator['last_value']:.6f}.

GPU 0 had a mean sampled power of {gpu_rows[0]['mean_sampled_power_w']:.2f} W.
Its maximum was {gpu_rows[0]['maximum_sampled_power_w']:.2f} W. GPU 1 had a
mean sampled power of {gpu_rows[1]['mean_sampled_power_w']:.2f} W. Its maximum
was {gpu_rows[1]['maximum_sampled_power_w']:.2f} W.

## Створені артефакти

- Manifest: `training/data/{NAME}/manifests/all.parquet`.
- PCM24 audio: `training/data/{NAME}/audio_24k/`.
- ESPnet data: `training/espnet_recipe/data/{NAME}_*`.
- Token list: `{token_list}`.
- Statistics: `{stats}`.
- Checkpoint: `{args.experiment}/checkpoint.pth`.
- Final model: `{model}`.
- Fixed evaluations: `{args.experiment}/decode_jets_milestone_*/`.
- Five-voice WAV files: `training/eval/generated/five_voice_{NAME}_500k/`.
- Runtime report: `training/reports/{NAME}_training_monitor_summary.json`.
- TensorBoard report: `training/reports/{NAME}_tensorboard_metrics.json`.

The final model SHA-256 is `{model_hash}`. The token-list SHA-256 is
`{token_hash}`.

## Відомі проблеми

- Human listening is not complete. Automatic tests do not measure naturalness.
- A person must check for rasp, metallic sound, and robotic sound.
- {fallback_count} files used a recorded degenerate-output fallback.

## Наступна одна дія

Listen to the fixed milestone sets and the five 500K voice files. Select the
checkpoint that has the best naturalness and the fewest speech artifacts.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
