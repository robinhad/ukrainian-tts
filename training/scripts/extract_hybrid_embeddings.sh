#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST=${MANIFEST:?Set MANIFEST to the clean expanded-v3 manifest.}
RAW_MANIFEST=${RAW_MANIFEST:?Set RAW_MANIFEST to the pre-enhancement manifest.}
KALDI_ROOT=${KALDI_ROOT:?Set KALDI_ROOT to the hybrid extraction data root.}
DUMP_DIR=${DUMP_DIR:?Set DUMP_DIR to the ESPnet dump root.}
REPORT=${REPORT:-"${ROOT}/reports/expanded_v3_hybrid_embeddings.json"}
OUTPUT_MANIFEST=${OUTPUT_MANIFEST:-"${KALDI_ROOT}/manifest/all.parquet"}
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
MODEL="${ROOT}/vendor/speechbrain-spkrec-ecapa-voxceleb"

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/prepare_hybrid_embeddings.py" \
    --manifest "$MANIFEST" \
    --raw-manifest "$RAW_MANIFEST" \
    --output-manifest "$OUTPUT_MANIFEST" \
    --kaldi-root "$KALDI_ROOT" \
    --report "$REPORT"

IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
if (( ${#GPUS[@]} < 1 )); then
    echo "This extraction requires at least one GPU UUID." >&2
    exit 2
fi

mapfile -t DATASETS < <(find "$KALDI_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if (( ${#DATASETS[@]} < 2 )); then
    echo "The hybrid input has fewer than two data sets." >&2
    exit 2
fi

pids=()
for index in "${!DATASETS[@]}"; do
    data_set=${DATASETS[$index]}
    gpu=${GPUS[$((index % ${#GPUS[@]}))]}
    output="${DUMP_DIR}/xvector/${data_set}"
    mkdir -p "$output"
    (
        cd "${ROOT}/vendor/espnet-src/egs2/TEMPLATE/tts1"
        CUDA_VISIBLE_DEVICES="$gpu" python pyscripts/utils/extract_spk_embed_parallel.py \
            --pretrained_model "$MODEL" \
            --toolkit speechbrain \
            --spk_embed_tag xvector \
            --device cuda \
            --num_workers "${SPK_EMBED_NUM_WORKERS:-8}" \
            --batch_size "${SPK_EMBED_BATCH_SIZE:-8}" \
            --prefetch 64 \
            "${KALDI_ROOT}/${data_set}" "$output" \
            > "${output}/spk_embed_extract.log" 2>&1
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo "Speaker embedding extraction failed." >&2
    exit 1
fi

for data_set in "${DATASETS[@]}"; do
    output="${DUMP_DIR}/xvector/${data_set}"
    for name in xvector.scp spk_xvector.scp; do
        LC_ALL=C sort "${output}/${name}" > "${output}/${name}.sorted"
        mv "${output}/${name}.sorted" "${output}/${name}"
    done
done

python - "$KALDI_ROOT" "$DUMP_DIR" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

from kaldiio import load_scp
import numpy as np

kaldi_root = Path(sys.argv[1])
dump_dir = Path(sys.argv[2])
report_path = Path(sys.argv[3])
sets = sorted(path.name for path in kaldi_root.iterdir() if path.is_dir())
set_reports = {}
for name in sets:
    expected = {
        line.split(maxsplit=1)[0]
        for line in (kaldi_root / name / "wav.scp").read_text().splitlines()
        if line.strip()
    }
    scp = dump_dir / "xvector" / name / "xvector.scp"
    vectors = load_scp(str(scp))
    actual = set(vectors.keys())
    dimensions = sorted({int(np.asarray(vectors[key]).squeeze().shape[0]) for key in actual})
    if expected != actual or dimensions != [192]:
        raise SystemExit(f"{name}: invalid hybrid embedding archive")
    set_reports[name] = {"records": len(actual), "dimensions": dimensions}
report = json.loads(report_path.read_text())
report["archives"] = set_reports
report["status"] = "PASS"
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
