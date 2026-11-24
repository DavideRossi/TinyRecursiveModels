#!/bin/bash
# Step time of the looped LM on one GPU (looped_lm/README.md, "Throughput"): the smallest (L = 2,
# width 256) and largest (L = 16, width 256; L = 8, width 512) models of the sweep, with and
# without torch.compile, STEPS steps each (the first 30 are not timed). Submit from the repo root:
#   sbatch slurm/bench_looped_lm.sbatch
# One JSON line per configuration in the log: median step time, data wait, peak memory.
#SBATCH --job-name=looped_lm_bench
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

SCRATCH="/scratch.hpc/${USER}"
SIF="${SIF:-${SCRATCH}/trm.sif}"
STEPS="${STEPS:-300}"
export APPTAINERENV_TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID}"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}_inductor_${SLURM_JOB_ID}"
trap 'rm -rf /tmp/${USER}_triton_${SLURM_JOB_ID} /tmp/${USER}_inductor_${SLURM_JOB_ID}' EXIT

cd "${SLURM_SUBMIT_DIR}"
echo "Job ${SLURM_JOB_ID} on $(hostname), start $(date)"
nvidia-smi

for spec in "2 256" "16 256" "8 512"; do
    set -- ${spec}
    for compile in false true; do
        echo "== depth $1 width $2 compile ${compile}"
        apptainer exec --nv --bind "${SCRATCH}" "${SIF}" \
            python -m looped_lm.train --benchmark true --steps "${STEPS}" --warmup_steps 100 \
                --run_name "bench_L$1_w$2_c${compile}" --out_dir "/tmp/${USER}_bench_${SLURM_JOB_ID}" \
                --core_layers "$1" --width "$2" --num_heads $(($2 / 64)) --head_dim 64 \
                --mlp_hidden $(($2 * 11 / 4)) --compile "${compile}" --num_workers 7 --wandb off
    done
done
rm -rf "/tmp/${USER}_bench_${SLURM_JOB_ID}"
echo "End $(date)"
