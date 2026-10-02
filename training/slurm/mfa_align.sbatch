#!/usr/bin/env bash
#SBATCH --job-name=uktts-mfa-align
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=training/quality_runs/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
run="$PWD/training/quality_runs/mfa_trim"
mfa_env="${MFA_ENV:-$run/tools/env}"
export PATH="$mfa_env/bin:$run/tools/sysroot/usr/bin:$PATH"
export LD_LIBRARY_PATH="$run/tools/sysroot/usr/lib/aarch64-linux-gnu:$mfa_env/lib:${LD_LIBRARY_PATH:-}"
export MFA_ROOT_DIR="$run/work"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
date -u +'%Y-%m-%dT%H:%M:%SZ MFA G2P started'
mfa g2p "$run/oovs.txt" "$run/models/ukrainian_mfa_g2p.zip" "$run/generated_oovs.dict" \
  --num_jobs 8 --num_pronunciations 1 --clean
cat "$run/models/ukrainian_mfa_dictionary.dict" "$run/generated_oovs.dict" > "$run/expanded.dict"
date -u +'%Y-%m-%dT%H:%M:%SZ MFA alignment started'
mfa align "$run/corpus" "$run/expanded.dict" "$run/models/ukrainian_mfa_acoustic.zip" "$run/aligned" \
  --config_path training/conf/quality_mfa_alignment.yaml --output_format json --num_jobs 8 --no_textgrid_cleanup --clean
date -u +'%Y-%m-%dT%H:%M:%SZ MFA alignment complete'
