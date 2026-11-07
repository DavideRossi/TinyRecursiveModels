#!/bin/bash
# Build the Sudoku-Extreme dataset (downloads from HuggingFace, so it needs
# internet on the compute node). Submit from the repo root:
#   sbatch slurm/build_dataset.sbatch
#SBATCH --job-name=trm_build_sudoku
#SBATCH --partition=sbuild
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

SCRATCH="/scratch.hpc/${USER}"
SIF="${SIF:-${SCRATCH}/trm.sif}"

# Keep the HF download cache off the home quota.
export APPTAINERENV_HF_HOME="${SCRATCH}/.hf_cache"
mkdir -p "${SCRATCH}/.hf_cache"

cd "${SLURM_SUBMIT_DIR}"
apptainer exec --bind "${SCRATCH}" "${SIF}" \
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-1k-aug-1000 \
        --subsample-size 1000 \
        --num-aug 1000

ls -lR data/sudoku-extreme-1k-aug-1000 | head -n 30
