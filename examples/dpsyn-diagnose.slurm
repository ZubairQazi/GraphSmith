#!/bin/bash
#SBATCH --job-name=graphsmith-diagnose
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
: "${GRAPHSMITH_ROOT:?}" "${DPSYN_REPO_ROOT:?}" "${DPSYN_DATA_ROOT:?}" "${DPSYN_WORK_ROOT:?}" "${DPSYN_PYTHON:?}"
export PYTHONPATH="$GRAPHSMITH_ROOT:$DPSYN_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$DPSYN_PYTHON" -u -m graphsmith dpsyn-diagnose \
  --dpsyn-repo "$DPSYN_REPO_ROOT" \
  --results-root /nfs/turbo/coe-dkoutra/zqazi/multigraph_gnn_results/dpsyn \
  --campaign "${1:?Supply a validated campaign directory}" \
  --output "$DPSYN_WORK_ROOT/diagnosis-${SLURM_JOB_ID:?}"
