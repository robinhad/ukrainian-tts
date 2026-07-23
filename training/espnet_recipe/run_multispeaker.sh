#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec "${ROOT}/espnet_recipe/run.sh" \
    --use_spk_embed true \
    --spk_embed_tag xvector \
    --spk_embed_tool speechbrain \
    --spk_embed_model \
        "${ROOT}/vendor/speechbrain-spkrec-ecapa-voxceleb" \
    --spk_embed_gpu_inference true \
    --spk_embed_parallel true \
    --spk_embed_num_workers "${SPK_EMBED_NUM_WORKERS:-8}" \
    --spk_embed_batch_size "${SPK_EMBED_BATCH_SIZE:-8}" \
    --train_config \
        "${ROOT}/espnet_recipe/conf/tuning/train_jets_uk_24k_multispeaker.yaml" \
    "$@"
