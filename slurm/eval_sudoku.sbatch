#!/bin/bash
# Evaluate an existing Sudoku-Extreme checkpoint (no training), with optional
# evaluation overrides. Submit from the repo root:
#   CKPT=checkpoints/<project>/<run>/step_<N> RUN_NAME=<new name> \
#     sbatch slurm/eval_sudoku.sbatch [overrides]
# e.g. 32 steps with the per-step capture:
#   ... sbatch slurm/eval_sudoku.sbatch eval_halt_max_steps=32 eval_capture_steps=True
# or halting with q_head at threshold 0:
#   ... sbatch slurm/eval_sudoku.sbatch eval_halt_enable=True eval_halt_threshold=0
#
# The arch settings below are those of train_sudoku.sbatch. Repeat any arch
# override the training run used (e.g. 'arch.aux_halt_heads=[{type:detached_linear}]');
# the run stops if they differ from the all_config.yaml next to the checkpoint.
# RUN_NAME must differ from the training run's name: results go to
# checkpoints/<project>/<RUN_NAME>/, and the checkpoint's own folder is refused.
# One pass over the test set takes about 1 h per 16 steps on an L40S.
#SBATCH --job-name=trm_sudoku_eval
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

SCRATCH="/scratch.hpc/${USER}"
SIF="${SIF:-${SCRATCH}/trm.sif}"
: "${CKPT:?Set CKPT=<path to a step_<N> checkpoint file>}"
: "${RUN_NAME:?Set RUN_NAME=<name for this evaluation run>}"

# wandb: online if an API key is available, otherwise offline.
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
        arch.mlp_t=True arch.pos_encodings=none \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        +run_name="${RUN_NAME}" \
        eval_only=True ++load_checkpoint="${CKPT}" \
        "$@"

echo "End $(date)"
