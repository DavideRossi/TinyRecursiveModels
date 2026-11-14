#!/bin/bash
# Perturbed restarts for the puzzles the halt head has not accepted (experiments/restarts.py).
# Submit from the repo root:
#   CKPT=checkpoints/<project>/<run>/step_<N> OUT=<output directory> \
#     sbatch slurm/restarts_sudoku.sbatch [options of experiments/restarts.py run]
# e.g. a quick trial on the first 5000 puzzles:
#   ... sbatch --time=01:00:00 slurm/restarts_sudoku.sbatch --max-puzzles 5000
# or the full run, checked against the plain 128-step capture:
#   ... sbatch slurm/restarts_sudoku.sbatch --reference <dir of the 128-step capture> <N>
#
# The architecture comes from the all_config.yaml next to the checkpoint. With the defaults
# (N1 = N2 = 32, 3 seeds, 4 + 3 noise scales) the full test set takes about 6 h on an L40S:
# 2 h for phase 1 and about 4 h for the 22 phase-2 branches of the ~9% retried puzzles.
#SBATCH --job-name=trm_sudoku_restarts
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

SCRATCH="/scratch.hpc/${USER}"
SIF="${SIF:-${SCRATCH}/trm.sif}"
: "${CKPT:?Set CKPT=<path to a step_<N> checkpoint file>}"
: "${OUT:?Set OUT=<output directory>}"

# torch.compile / Triton need writable caches (the image is read-only).
export APPTAINERENV_TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}_inductor_${SLURM_JOB_ID}"
trap 'rm -rf /tmp/${USER}_triton_${SLURM_JOB_ID} /tmp/${USER}_inductor_${SLURM_JOB_ID}' EXIT

cd "${SLURM_SUBMIT_DIR}"
echo "Job ${SLURM_JOB_ID} on $(hostname), start $(date)"
nvidia-smi

apptainer exec --nv --bind "${SCRATCH}" "${SIF}" \
    python experiments/restarts.py run "${CKPT}" "${OUT}" "$@"

echo "End $(date)"
