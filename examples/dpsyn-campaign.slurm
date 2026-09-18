#!/bin/bash
#SBATCH --job-name=graphsmith-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:05:00
set -euo pipefail
: "${GRAPHSMITH_ROOT:?Set the GraphSmith checkout}"
: "${DPSYN_REPO_ROOT:?Set the dpsyn checkout}"
: "${DPSYN_DATA_ROOT:?Set the prepared data root}"
: "${DPSYN_WORK_ROOT:?Set private work storage}"
: "${DPSYN_PYTHON:?Set the dpsyn environment Python}"
export PYTHONPATH="$GRAPHSMITH_ROOT:$DPSYN_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$DPSYN_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "GPU required for this launcher"'
exec "$DPSYN_PYTHON" -u -m graphsmith dpsyn-campaign "${1:?Supply a campaign manifest}" \
  --dpsyn-repo "$DPSYN_REPO_ROOT" --output "$DPSYN_WORK_ROOT/campaign-${SLURM_JOB_ID:?}"
