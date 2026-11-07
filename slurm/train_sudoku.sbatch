#!/bin/bash
# Sudoku-Extreme, TRM with MLP token mixer (README: "pretrain_mlp_t_sudoku",
# ~18h on 1 L40S, expected ~87% exact accuracy). Submit from the repo root:
#   sbatch slurm/train_sudoku.sbatch
# Extra Hydra overrides can be appended, e.g. for a smoke test:
#   sbatch --time=00:30:00 slurm/train_sudoku.sbatch epochs=100 eval_interval=50
#SBATCH --job-name=trm_sudoku_mlp
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

SCRATCH="/scratch.hpc/${USER}"
SIF="${SIF:-${SCRATCH}/trm.sif}"
RUN_NAME="${RUN_NAME:-pretrain_mlp_t_sudoku}"

# wandb: online if an API key is available, otherwise log offline and run
# `wandb sync` on the offline-run directory later from a node with internet.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
    export APPTAINERENV_WANDB_MODE=online
else
    export APPTAINERENV_WANDB_MODE=offline
fi
export APPTAINERENV_WANDB_DIR="${SLURM_SUBMIT_DIR}"

# torch.compile / Triton need writable caches (the image is read-only).
export APPTAINERENV_TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}_inductor_${SLURM_JOB_ID}"
trap 'rm -rf /tmp/${USER}_triton_${SLURM_JOB_ID} /tmp/${USER}_inductor_${SLURM_JOB_ID}' EXIT

cd "${SLURM_SUBMIT_DIR}"
echo "Job ${SLURM_JOB_ID} on $(hostname), start $(date)"
nvidia-smi

apptainer exec --nv --bind "${SCRATCH}" "${SIF}" \
    python pretrain.py \
        arch=trm \
        data_paths="[data/sudoku-extreme-1k-aug-1000]" \
        evaluators="[]" \
        epochs=50000 eval_interval=5000 \
        lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
        arch.mlp_t=True arch.pos_encodings=none \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        +run_name="${RUN_NAME}" ema=True \
        "$@"

echo "End $(date)"
