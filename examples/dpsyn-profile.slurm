#!/bin/bash
#SBATCH --job-name=graphsmith-profile
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
: "${GRAPHSMITH_ROOT:?Set GRAPHSMITH_ROOT to the GraphSmith checkout}"
: "${DPSYN_REPO_ROOT:?Set DPSYN_REPO_ROOT to the training checkout}"
: "${DPSYN_DATA_ROOT:?Set DPSYN_DATA_ROOT to prepared inputs}"
: "${DPSYN_WORK_ROOT:?Set DPSYN_WORK_ROOT to private output storage}"
: "${DPSYN_PYTHON:?Set DPSYN_PYTHON to the dpsyn environment Python}"
export PYTHONPATH="$GRAPHSMITH_ROOT:$DPSYN_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$DPSYN_PYTHON" -m graphsmith dpsyn-profile \
    --dpsyn-repo "$DPSYN_REPO_ROOT" \
    --output "$DPSYN_WORK_ROOT/profile-${SLURM_JOB_ID:?Submit this script with sbatch}" "$@"
