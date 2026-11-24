#!/bin/bash
# Looped LM, fixed-depth baseline (looped_lm/README.md). Submit from the repo root:
#   RUN_NAME=pilot_L8_w256_s0 DEPTH=8 WIDTH=256 SEED=0 sbatch slurm/train_looped_lm.sbatch
# Extra options of looped_lm.train can be appended, e.g. --steps 50000 --eval_interval 1000.
# Width W gives W/64 heads of 64 and a SwiGLU hidden size of 11W/4 (704 at 256, 1408 at 512).
# At the time limit Slurm sends USR1 (5 minutes before); the run saves ckpt_latest.pt and stops.
# Submitting the same command again resumes it from that checkpoint.
#SBATCH --job-name=looped_lm
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail

SCRATCH="/scratch.hpc/${USER}"
SIF="${SIF:-${SCRATCH}/trm.sif}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
DEPTH="${DEPTH:-8}"
WIDTH="${WIDTH:-256}"
SEED="${SEED:-0}"
HEADS=$((WIDTH / 64))
MLP=$((WIDTH * 11 / 4))

# wandb: online if an API key is available, otherwise offline (sync later with `wandb sync`).
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
    export APPTAINERENV_WANDB_MODE=online
else
    export APPTAINERENV_WANDB_MODE=offline
fi
export APPTAINERENV_TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}_inductor_${SLURM_JOB_ID}"
trap 'rm -rf /tmp/${USER}_triton_${SLURM_JOB_ID} /tmp/${USER}_inductor_${SLURM_JOB_ID}' EXIT

cd "${SLURM_SUBMIT_DIR}"
echo "Job ${SLURM_JOB_ID} on $(hostname), start $(date): ${RUN_NAME} depth ${DEPTH} width ${WIDTH} seed ${SEED}"
nvidia-smi

apptainer exec --nv --bind "${SCRATCH}" "${SIF}" \
    python -m looped_lm.train \
        --run_name "${RUN_NAME}" --core_layers "${DEPTH}" --width "${WIDTH}" \
        --num_heads "${HEADS}" --head_dim 64 --mlp_hidden "${MLP}" --seed "${SEED}" \
        --num_workers 7 \
        "$@" &
PID=$!
# Forward the time-limit signal to the training process, then wait for it to save and exit
trap 'echo "USR1 at $(date): asking the run to checkpoint"; kill -USR1 ${PID}' USR1
wait ${PID}
status=$?
while kill -0 ${PID} 2>/dev/null; do wait ${PID}; status=$?; done
echo "End $(date), exit status ${status}"
exit ${status}
