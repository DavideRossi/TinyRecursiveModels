#!/bin/bash
# Build the Apptainer image. Submit from the repo root:
#   sbatch slurm/build_image.sbatch
#SBATCH --job-name=trm_build_image
#SBATCH --partition=sbuild
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

SCRATCH="/scratch.hpc/${USER}"
SIF="${SIF:-${SCRATCH}/trm.sif}"

export APPTAINER_CACHEDIR="${SCRATCH}/.apptainer_cache"
export APPTAINER_TMPDIR="${SCRATCH}/.apptainer_tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

cd "${SLURM_SUBMIT_DIR}"
echo "Building ${SIF} on $(hostname) at $(date)"
apptainer build --force "${SIF}" slurm/trm.def
echo "Done at $(date)"
ls -lh "${SIF}"
