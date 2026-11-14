# Running TRM on the DISI Slurm cluster

Scripts to run the Sudoku-Extreme experiment (`pretrain_mlp_t_sudoku` from the
main README) on the DISI HPC cluster, inside an Apptainer container on an L40S
GPU. They should work on other Slurm + Apptainer clusters with small changes to
partition names and paths.

| File | What it does |
|---|---|
| `trm.def` | Apptainer definition: Ubuntu 24.04 + Python 3.12 + PyTorch + the repo's dependencies (exact versions in the file). Software only, no code or data. |
| `build_image.sbatch` | Builds `trm.sif` on the `sbuild` partition. |
| `build_dataset.sbatch` | Downloads Sudoku-Extreme from HuggingFace and builds `data/sudoku-extreme-1k-aug-1000`. |
| `train_sudoku.sbatch` | Trains on one L40S (`l40s` partition). |
| `eval_sudoku.sbatch` | Evaluates an existing checkpoint, without training. |
| `restarts_sudoku.sbatch` | Perturbed-restart experiment on an existing checkpoint (`experiments/restarts.py`). |
| `read_wandb_offline.py` | Prints eval metrics from an offline wandb run. |
| `logs/` | Slurm output of all jobs (`<job-name>_<job-id>.out`). |

## How it fits together

- The **image** contains only software. Code, data, checkpoints and logs stay
  in a checkout of this repo on `/scratch.hpc/$USER`, which is bind-mounted into
  the container. Code or config changes therefore never need a rebuild.
- `apptainer` is **not available on the login node**. Every Apptainer command,
  including image builds and one-off utilities, must run through Slurm (`sbatch`
  or `srun`). Use `sbuild` for anything that needs no GPU.
- **Submit every job from the repo root.** The scripts use the submit directory
  as their working directory, and Slurm writes logs to `slurm/logs/` relative to
  it.
- The image path defaults to `/scratch.hpc/$USER/trm.sif`. Override it with
  `SIF=...` in front of `sbatch`; Slurm passes your environment to the job.

## 1. Get the code onto scratch

On the login node (`slogin.disi.unibo.it`):

```bash
cd /scratch.hpc/$USER
git clone https://github.com/DavideRossi/TinyRecursiveModels.git
cd TinyRecursiveModels
```

## 2. Build the image

```bash
sbatch slurm/build_image.sbatch
tail -f slurm/logs/trm_build_image_*.out
```

A successful build ends with the import check's output, e.g.
`torch 2.x.y+cuNNN cuda NN.N ...`, followed by the size of the `.sif`. If the
build fails, the error is usually in the last 30 lines of the log.

To build a second image without touching one that running jobs use:

```bash
SIF=/scratch.hpc/$USER/trm-new.sif sbatch slurm/build_image.sbatch
```

The build overwrites the target `.sif` (`--force`). Never rebuild an image that
a running job is using.

## 3. Build the dataset

```bash
sbatch slurm/build_dataset.sbatch
```

This needs internet access, which `sbuild` nodes have. It produces
`data/sudoku-extreme-1k-aug-1000/{train,test}`: 1000 training puzzles × 1001
augmentations, and the full test set (~423k puzzles). Another checkout of the
repo can reuse it with a symlink:
`ln -s /scratch.hpc/$USER/TinyRecursiveModels/data data`.

## 4. Train

Full run (about 18–20 h on one L40S; the job requests 2 days):

```bash
sbatch slurm/train_sudoku.sbatch
```

- **Run name:** defaults to `pretrain_mlp_t_sudoku`. Change it with
  `RUN_NAME=... sbatch ...`. Do **not** pass `+run_name=...`: the script already
  sets it, and Hydra rejects the duplicate.
- **Extra settings:** anything after the script name is passed to `pretrain.py`
  as a Hydra override, e.g. `sbatch slurm/train_sudoku.sbatch global_batch_size=384`.
- **Email notification:** `#SBATCH` lines cannot use `$USER`, so add it at
  submit time: `sbatch --mail-type=END,FAIL --mail-user=you@unibo.it ...`.

### Quick checks before a full run

Each evaluation pass over the full test set takes about **1 hour**, and
`pretrain.py` evaluates after every `eval_interval` epochs. Size test runs
around that.

Training only, no evaluation (~1300 steps, well under an hour). This checks the
image, `torch.compile`, speed (`it/s` on the progress bar) and that the loss
goes down:

```bash
RUN_NAME=check sbatch --time=02:00:00 slurm/train_sudoku.sbatch \
    epochs=1000 eval_interval=1000 min_eval_interval=1
```

Up to the first evaluation and checkpoint (6510 steps + ~1 h of evaluation).
For reference, a full reference run scored about 20% `exact_accuracy` at this
point (63% at the second evaluation, 82% at the fifth):

```bash
RUN_NAME=eval1 sbatch --time=06:00:00 slurm/train_sudoku.sbatch \
    epochs=5000 eval_interval=5000
```

`epochs` must be a multiple of `eval_interval`.

## 5. Monitor

```bash
squeue -u $USER                                  # pending / running jobs
tail -f slurm/logs/trm_sudoku_mlp_<jobid>.out    # live log
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
scancel <jobid>
```

During evaluation the log shows `Processing batch N: all`, about 551 batches per
pass. Accuracy is **not** printed in the log; it only goes to wandb (next
section).

## 6. Results

**Checkpoints:** after each evaluation the job writes the model weights
(`step_<N>`) and the resolved config (`all_config.yaml`) to
`checkpoints/<project>/<run name>/`. This is weights only: the optimizer state
is not saved, so a killed run cannot be resumed exactly.

**Metrics:** by default the job logs to wandb **offline**, in `wandb/offline-run-*`.
The number to look at is `all/exact_accuracy`, the fraction of test puzzles
solved completely; expect ~87% ± 2% at the end. There are two ways to read it:

- **On the cluster** (works while the job is still running):

  ```bash
  srun -p sbuild --cpus-per-task=1 --mem=2G \
    apptainer exec --bind /scratch.hpc/$USER /scratch.hpc/$USER/trm.sif \
    python slurm/read_wandb_offline.py            # newest run, or pass a run dir
  ```

  This prints the evaluation metrics only. Add `--train` for the training
  metrics (`train/*`, logged every step), shown as one row per 1000 steps
  (`--window N`) with the mean over each window. `--last K` keeps the last K
  rows, and `--keys` keeps only metrics whose names contain one of the given
  strings. For example, to follow the halting heads of a running job:

  ```bash
  ... python slurm/read_wandb_offline.py --train --keys halt steps --last 10
  ```

  Training accuracies and `steps` are measured on the sequences that halted in
  each step, so they show how halting behaves during training. In evaluation
  every sequence runs all `halt_max_steps` steps, so the `all/` halting metrics
  only describe the last step.

- **On the wandb website.** Sync from an `sbuild` node, passing your API key
  through the environment:

  ```bash
  read -s WANDB_API_KEY && export WANDB_API_KEY     # paste the key from wandb.ai/authorize
  srun -p sbuild --cpus-per-task=1 --mem=2G \
    apptainer exec --bind /scratch.hpc/$USER /scratch.hpc/$USER/trm.sif \
    wandb sync wandb/offline-run-*
  ```

  If `WANDB_API_KEY` is exported when you submit `train_sudoku.sbatch`, the job
  logs online directly. This only works if the GPU nodes have internet access;
  check with `srun -p l40s --gres=gpu:1 curl -sI https://api.wandb.ai`.

## 7. Run from a different branch

Jobs run the code in the directory they are submitted from, so use **one
checkout per branch**. Switching branches in a checkout that a job is using
changes that job's code under its feet.

### Make a checkout for the branch

Push the branch from your machine first (`git push -u origin <branch>`), then on
the login node:

```bash
cd /scratch.hpc/$USER
git clone -b <branch> https://github.com/DavideRossi/TinyRecursiveModels.git TRM-<branch>
cd TRM-<branch>
mkdir -p slurm/logs                                       # skip if git already created it
ln -s /scratch.hpc/$USER/TinyRecursiveModels/data data    # reuse the dataset
```

To update an existing checkout instead, check that no job is running from it,
then `git fetch origin && git checkout <branch> && git pull`.

### Check whether the branch needs another image

The image only has to match `slurm/trm.def`. If the branch doesn't change that
file, any image built from the same `trm.def` works. Compare the definition
files of the two checkouts:

```bash
diff /scratch.hpc/$USER/TinyRecursiveModels/slurm/trm.def slurm/trm.def
```

Or ask an image which PyTorch it contains, and compare it with the version
pinned in `slurm/trm.def`:

```bash
srun -p sbuild apptainer exec --bind /scratch.hpc/$USER /scratch.hpc/$USER/trm.sif \
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Build an image for the branch (only if needed)

Give it its own name, so the image other checkouts use stays as it is:

```bash
SIF=/scratch.hpc/$USER/trm-<branch>.sif sbatch slurm/build_image.sbatch
```

Monitor the build:

```bash
squeue -u $USER                                         # ST: PD = queued, R = running; gone = finished
tail -f slurm/logs/trm_build_image_<jobid>.out          # live log; Ctrl-C stops watching, not the build
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode  # COMPLETED and 0:0 = success
scancel <jobid>                                         # cancel
```

Most of the time goes to downloading and installing PyTorch with its CUDA
libraries (several GB). The last step, packing the `.sif`, can print nothing for
several minutes. A successful build ends with the import check's version line.
Check the image afterwards:

```bash
ls -lh /scratch.hpc/$USER/trm-<branch>.sif
srun -p sbuild apptainer exec --bind /scratch.hpc/$USER /scratch.hpc/$USER/trm-<branch>.sif \
  python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
```

### Train

Pass the image with `SIF=...` on every submission from this checkout (and in
the `srun ... apptainer exec` commands of section 6). Give each run a distinct
`RUN_NAME`, so its checkpoints don't mix with other runs'. For example, on the
`aux-halt-heads` branch:

```bash
SIF=/scratch.hpc/$USER/trm-<branch>.sif RUN_NAME=sudoku_aux_linear \
  sbatch slurm/train_sudoku.sbatch \
  'arch.aux_halt_heads=[{type:detached_linear}]' eval_capture_steps=True
```

The first run of a branch that changes model code is a good time for one of the
quick checks of section 4.

## 8. Evaluate an existing checkpoint

`eval_sudoku.sbatch` runs only the evaluation of a saved checkpoint, with any
evaluation settings. Checkpoints hold the weights the in-training evaluation
used (the EMA weights), so the same settings reproduce the logged metrics.

```bash
CKPT=checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/<run>/step_<N> RUN_NAME=<new name> \
  sbatch slurm/eval_sudoku.sbatch [overrides]
```

- `RUN_NAME` must differ from the training run's name. Results go to
  `checkpoints/<project>/<RUN_NAME>/`, with the checkpoint's step number in the
  file names.
- Repeat every `arch.*` override of the training run (for example
  `'arch.aux_halt_heads=[{type:detached_linear}]'`). The job stops if the
  settings differ from the `all_config.yaml` next to the checkpoint.
- The metrics are printed at the end of the job log (`EVAL RESULTS`) and logged
  to wandb.
- One pass over the test set takes about 1 h per 16 steps.

| Override | Effect |
|---|---|
| `eval_halt_max_steps=32` | Step limit in evaluation (default: the training limit, `arch.halt_max_steps`). |
| `eval_halt_enable=True` | Each puzzle halts at the first step where its halt logit is ≥ the threshold, or at the step limit. Metrics use its outputs at that step; `steps` is the mean halting step and `halted_early` the fraction that halted before the limit. |
| `eval_halt_threshold=0` | The threshold (default 0). |
| `eval_halt_head=q_head` | The head that decides: `q_head` or an auxiliary head name. |
| `eval_capture_steps=True` | Save the per-step capture (`step_<N>_eval_steps.<rank>`). It always holds every step up to the limit, whatever the halting settings. |

These options also work in a training run, where they apply to its
evaluations.

## 9. Perturbed restarts

`restarts_sudoku.sbatch` runs `experiments/restarts.py`: every test puzzle runs
N1 steps with halting by a head; the puzzles the head has not accepted get N2
more steps, continuing (the control), continuing from a perturbed state, or
restarting from a noisy initial state, for several noise scales and seeds. The
script's docstring describes the procedure and outputs.

```bash
CKPT=checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/<run>/step_<N> \
OUT=checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/<run>_restarts SIF=/scratch.hpc/$USER/trm-modern.sif \
  sbatch slurm/restarts_sudoku.sbatch [options]
```

| Option | Effect |
|---|---|
| `--max-puzzles 5000` | Only the first 5000 test puzzles (quick trials). |
| `--n1 32 --n2 32` | Phase-1 steps and extra steps per attempt (defaults). |
| `--head q_head --threshold 5.27` | The halting rule, in both phases (defaults). |
| `--perturb-scales ... --restart-scales ... --seeds ...` | Noise scales of each variant and seeds (defaults 0.001 0.1 0.3 1.0; 0.1 0.3 1.0; 0 1 2). |
| `--reference DIR STEP` | A plain capture of the same checkpoint with at least N1 + N2 steps. The summary checks phase 1 and the control against it and adds plain continuation to the same budgets. |

The summary can be recomputed anywhere (no GPU), e.g. on a laptop after
copying the output directory (about 0.7 GB, most of it the per-branch captures
under `captures/`; the summary needs only `captures/continue/`):
`python experiments/restarts.py summarize OUT --reference DIR STEP`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Build fails in `%post` with `GLIBC_2.38 not found (required by libfakeroot.so)` | The cluster's Apptainer injects a fakeroot library that needs glibc ≥ 2.38. Keep the base image at Ubuntu 24.04 or newer. |
| Build warns `'nodev' mount option set on /scratch.hpc` | Usually harmless. If a build step fails with device or permission errors, point `APPTAINER_TMPDIR` in `build_image.sbatch` at node-local storage (e.g. `/tmp/$USER`, ~20 GB free). |
| `Could not append to config. An item is already at 'run_name'` | You passed `+run_name=...`. Use `RUN_NAME=...` instead. |
| No log file appears | `slurm/logs/` is missing (Slurm does not create it), or you submitted from outside the repo root. |
| Job killed at the time limit | Check `--time` against the partition limit: `sinfo -p l40s -o "%P %l"` (3 days at the time of writing). |
| `WARNING: Error changing the container working directory. Using '<home>' instead` | The current directory is under scratch (possibly through a symlink such as `~/scratch`), but scratch isn't mounted in the container. Add `--bind /scratch.hpc/$USER` to `apptainer exec`. Harmless for commands that read no files, but anything using relative paths would run in your home directory. |
| `torch.cuda.is_available()` is False | `--nv` is missing from `apptainer exec`, or the job didn't request `--gres=gpu:1`. |
