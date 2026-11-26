# Looped LM: model, training and evaluation

A decoder-only causal transformer family trained on the graph-reachability task of
[`dataset_generation/`](dataset_generation/README.md), with a training and evaluation pipeline.
The first use is the **fixed-depth baseline** (n_loops = 1), to calibrate the difficulty of the
task against depth and width. `looped_lm` depends on numpy and torch only (wandb optionally) and
has no link to the TRM code in this repository.

```
looped_lm/
  dataset_generation/   the data (see its README)
  model/                ModelConfig, LoopedLM, layers
  objective.py          the training loss (cross-entropy at loss_mask positions)
  evaluation.py         teacher-forced scoring of final answers on stored sets
  train.py              training loop: python -m looped_lm.train
  evaluate.py           evaluation of a checkpoint: python -m looped_lm.evaluate
  data/                 written evaluation suites (ignored by git)
slurm/train_looped_lm.sbatch, slurm/bench_looped_lm.sbatch
tests/looped_lm/test_model.py
```

## 1. Model (`model/`)

```
tokens -> embedding -> prelude (P layers) -> core (C layers) x n_loops -> coda (K layers)
       -> final RMSNorm -> head (tied to the embedding) -> logits
```

- **Layer** (the same everywhere): pre-norm RMSNorm → causal multi-head attention → residual;
  RMSNorm → SwiGLU MLP → residual. No biases, no dropout.
  - **Attention:** `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True`, and
    rotary position embeddings on queries and keys in every layer. Batches are right-padded,
    so real tokens never attend to padding and no padding mask is needed.
  - **RMSNorm** has a learnable gain (initialized to 1), computed in float32.
  - **SwiGLU:** `down(silu(gate(x)) * up(x))` with hidden size m.
- **Loops:** the core is a `ModuleList` applied `n_loops` times, so weight sharing can be added
  without restructuring. Only `n_loops = 1` is implemented (`NotImplementedError` otherwise); the
  model is then P + C + K distinct layers.
- **Head:** tied to the input embedding (`logits = h @ E^T`). Training applies it only at the
  positions that predict loss tokens; `model.logits(tokens)` gives the full logits.
- **Initialization:** the embedding normal with std 1/√d (`embed_init_std`; 0.0625 at width 256,
  0.044 at 512), so each row starts with norm about 1; every other weight normal with std 0.02;
  the output projections of attention (`o_proj`) and MLP (`down_proj`) with std
  0.02 / √(2 × (P + C + K)); norm gains 1.
  - **Why the embedding is larger than 0.02:** with the tied head, a logit is the final-normed
    hidden state (norm about √d) times an embedding row, so the largest logit the model can give
    a token is about √d × that row's norm. At std 0.02 and width 256 that is about 5, too little
    to concentrate the probability on one of 2,048 names, and the rows can only grow when their
    names occur, which is rarely: every node name is a rare token. With 0.02 the model stayed at
    the uniform loss for thousands of steps, even on a task that only asks to copy the queried
    name; with 1/√d a small model learned that copy in about 400 steps (CPU probe, width 128).
    The mean and maximum row norms (all tokens, and node names only) are logged at step 0 and at
    every evaluation as `model/embed_norm_*`.
- **Vocabulary:** from the dataset package (2,059 tokens: 11 specials and 2,048 node names).
- **Rotary embeddings:** `rotate_half`, `apply_rotary_pos_emb` and `RotaryEmbedding` are copied
  from the TRM code (`models/layers.py`), so this package doesn't import it. Everything else is
  written here.

**`ModelConfig`** (saved with every run):

| field | default | meaning |
|---|---|---|
| `vocab_size` | 2059 | from `dataset_generation.vocab` |
| `width` | 256 | residual stream d |
| `num_heads`, `head_dim` | 4, 64 | attention heads |
| `mlp_hidden` | 704 | SwiGLU hidden size m (1,408 at width 512) |
| `prelude_layers`, `core_layers`, `coda_layers` | 0, 8, 0 | P, C, K; the baseline uses P = K = 0, C = L |
| `n_loops` | 1 | only 1 is implemented |
| `max_positions` | 8192 | longest sequence (rotary table); the companion sets reach 4,623 tokens |
| `rope_base` | 10000 | |
| `norm_eps` | 1e-5 | |
| `init_std` | 0.02 | every weight except the embedding |
| `embed_init_std` | 1/√width | 0 in the config means 1/√width; the saved config holds the value |

**Parameters:** V·d + d + L·(4·d·(heads × head_dim) + 3·d·m + 2·d), checked by a test:

| model | parameters |
|---|---:|
| L = 2, width 256 | 2,134,016 |
| L = 8, width 256 | 6,953,984 |
| L = 16, width 256 | 13,380,608 |
| L = 8, width 512 (8 heads, m = 1,408) | 26,753,024 |

The embedding is 527k of these at width 256.

## 2. Training (`train.py`)

```bash
python -m looped_lm.train --run_name NAME --core_layers 8 --width 256 --seed 0 [--steps 50000 ...]
```

Every field of `TrainConfig` and `ModelConfig` is an option (`--field value`; booleans take
`true`/`false`).

- **Data:** `TrainStream` of `dataset_generation` on `configs/train.json` (`--data_config`),
  with the config's answer format unless `--answer_format` overrides it, batches of 64 built in the stream (one edge count per
  batch), a `DataLoader` with `batch_size=None` and `num_workers` workers (default 7). The stream
  seed is `data_seed`, or `seed` when `data_seed` is −1 (the default), so two seeds differ in both
  initialization and data.
- **Loss:** next-token cross-entropy on the positions where `loss_mask` is true: the logits at
  t − 1 predict the token at t. These are the answer tokens and the end token (`;` / `<eos>`)
  after each answer, about 5 per sample in the direct format. The loss is the mean over the loss
  tokens of the batch.
- **Optimization:** bf16 autocast on GPU; AdamW, betas (0.9, 0.95), weight decay 0.1 on the
  2-D weights of the layers (not on norm gains or the embedding); peak learning rate 1e-3,
  linear warm-up over 1,000 steps, cosine decay to 10% of the peak at the last step; gradient
  clipping at 1.0.
- **`--compile true`:** `torch.compile` of the model; batches are then padded to a multiple of
  128 tokens (`pad_multiple`) to limit recompilation over varying lengths. Padding after the
  real tokens changes nothing under causal attention.

**Run directory** `out_dir/run_name` (default `checkpoints/looped_lm/<run_name>`):

| file | content |
|---|---|
| `config.json` | model, training and task configs |
| `metrics.jsonl` | every logged metric, one JSON line per log, with `step` |
| `ckpt_latest.pt` | model, optimizer, step and configs: everything to resume (every `ckpt_interval` = 2,000 steps, at the end, and on a signal) |
| `model_step_N.pt` | weights and configs only, every `keep_interval` = 10,000 steps and at the end, for evaluation along training |
| `wandb/`, `wandb_id.txt` | the wandb run (offline unless `WANDB_MODE` says otherwise) |

**Resuming:** starting a run whose directory has `ckpt_latest.pt` resumes it: model, optimizer,
schedule step, and the data position. Batch j of the stream is a pure function of
(data seed, j), so the stream simply restarts at the saved step; nothing is skipped or repeated.
A resumed run must have the same model config, task config, batch size, seeds and answer
format, or it refuses to start. `steps` may change (e.g. to extend a run; the cosine then
follows the new length). **SIGUSR1 or SIGTERM** saves `ckpt_latest.pt` at the end of the
current step and stops; the Slurm script forwards Slurm's time-limit signal.

**Logged every `log_interval` = 50 steps** (averages over the interval): `train/loss`,
`train/final_loss` (final answers only), `train/final_accuracy` (role 1, argmax),
`train/end_accuracy` (role 5), `train/grad_norm` (before clipping), `train/lr`,
`train/tokens_per_s` (real tokens; `padded_tokens_per_s` with padding), `train/step_time`,
`train/data_wait` (time per step spent waiting for the DataLoader; if it approaches the step
time, add workers).

**Periodic evaluation** every `eval_interval` = 1,000 steps (and at the end) on the first
`eval_samples` = 200 samples of each set in `eval_sets`, from the stored suite
(`looped_lm/data/eval_suite`): `grid/d1_b1`, `grid/d2_b2`, `grid/d4_b2`, `grid/d8_b1`,
`grid/d8_b4`, `grid/d16_b4` (d = 16 is outside the training depths 1–8) and
`mixed/q4_in_distribution`. Per set it logs `eval/<set>/accuracy`, `eval/<set>/loss` (mean
cross-entropy on final answers), `eval/<set>/chance` and `eval/<set>/n`. **Watch the loss as well
as the accuracy:** on these tasks accuracy can sit at chance and then jump, and the loss usually
moves first. The run fails at start if a set is missing (write the suite first, §4). At step 0
and at every evaluation it also logs the mean and maximum L2 norm of the embedding rows, over
the whole vocabulary (`model/embed_norm_mean`, `model/embed_norm_max`) and over the node names
(`model/embed_norm_names_mean`, `model/embed_norm_names_max`), to follow the logit scale (§1).

**Benchmark mode** (`--benchmark true --steps 300`): no evaluation, logs or checkpoints; prints
one JSON line with the median step time (after 30 untimed steps), the median data wait and the
peak GPU memory.

## 3. Evaluation (`evaluate.py`)

```bash
python -m looped_lm.evaluate --checkpoint checkpoints/looped_lm/RUN [--skip companion/ diag/]
python -m looped_lm.evaluate --checkpoint checkpoints/looped_lm/RUN/model_step_20000.pt --only grid/ mixed/
```

- **Teacher-forced, one forward pass per batch:** a final answer (role 1) is correct when the
  argmax of the logits at its `pred_pos` (the position before it) is the answer token. Every
  query of every sample is scored in the same pass.
- **Per set:** accuracy with a 95% Wilson interval, the number of answers, chance (the mean of
  1 / #targets over the answers), the mean loss on final answers, the number of samples and
  their mean length. For sets with several queries per sample (the mixed sets), the same by the
  query's (d, b).
- **Output:** `<run>/eval/<checkpoint>_step<N>.json` (all results, configs and paths) and `.csv`
  (one row per set, plus one per (d, b) with the `depth` and `branching` columns filled), for
  comparing runs in a notebook. `--out` changes the directory.
- **Options:** `--suite` (default `looped_lm/data/eval_suite`), `--only` / `--skip` by name
  prefix, `--num N` (first N samples of each set), `--max_tokens` (padded tokens per batch;
  batches are grouped by length), `--device`.
- **Lengths:** the grid, diagnostic and mixed sets (≤ 1,566 tokens) are within the training
  lengths (588–1,570 tokens). The companion pairs (2,319–4,623 tokens) are not: their results
  include rotary-embedding extrapolation to unseen positions, so read them as out of distribution
  in length.

## 4. On the cluster

1. **Get the evaluation suite to the cluster** once (it is not in git; 162 MB). Generation is
   deterministic, so either copy a suite written elsewhere:
   ```bash
   ssh <cluster> "mkdir -p <repo>/looped_lm/data"
   rsync -a looped_lm/data/eval_suite/ <cluster>:<repo>/looped_lm/data/eval_suite/
   ```
   or write it on a compute node (apptainer isn't available on the login node), from the repo
   root:
   ```bash
   srun --partition=l40s --cpus-per-task=4 --mem=8G --time=00:30:00 \
       apptainer exec /scratch.hpc/$USER/trm.sif python -m looped_lm.dataset_generation.cli write \
           --suite looped_lm/dataset_generation/configs/eval_suite.json --check
   ```
   Writing takes a few minutes on the CPU and creates `looped_lm/data/eval_suite/`.
2. **Benchmark:** `sbatch slurm/bench_looped_lm.sbatch` times L = 2 and L = 16 at width 256 and
   L = 8 at width 512, each with and without `torch.compile`, 300 steps each.
3. **Pilot:** L = 8, width 256, 50,000 steps, evaluated every 1,000 steps:
   ```bash
   RUN_NAME=pilot_L8_w256_s0 DEPTH=8 WIDTH=256 SEED=0 sbatch slurm/train_looped_lm.sbatch \
       --steps 50000 --compile true
   ```
   About 1.6 h (§5). The step count of the sweep follows from where the pilot's evaluation loss
   and accuracy flatten.
4. **Sweep** (STEPS from the pilot):
   ```bash
   for L in 2 4 6 8 12 16; do for S in 0 1; do
       C=$([ $L -ge 4 ] && echo true || echo false)    # compile is slower at L = 2 (§5)
       RUN_NAME=base_L${L}_w256_s${S} DEPTH=$L WIDTH=256 SEED=$S sbatch slurm/train_looped_lm.sbatch \
           --steps $STEPS --compile $C
   done; done
   for S in 0 1; do
       RUN_NAME=base_L8_w512_s${S} DEPTH=8 WIDTH=512 SEED=$S sbatch slurm/train_looped_lm.sbatch \
           --steps $STEPS --compile true
   done
   ```
5. **Evaluate** each run: `apptainer exec --nv ... python -m looped_lm.evaluate --checkpoint
   checkpoints/looped_lm/<run>`.

`slurm/train_looped_lm.sbatch` takes `RUN_NAME`, `DEPTH`, `WIDTH` and `SEED` from the environment
(width W gives W/64 heads of 64 and m = 11W/4), passes any further arguments to `looped_lm.train`,
and logs to wandb offline unless `WANDB_API_KEY` is set (`wandb sync <run>/wandb/offline-run-*`
later). Slurm signals it 5 minutes before the time limit (12 h by default); the run saves its
checkpoint and stops, and **submitting the same command again resumes it**.

## 5. Throughput (measured)

`slurm/bench_looped_lm.sbatch` on one **L40** (the L40S nodes were busy), batch 64 (about 70k
tokens per step), bf16 autocast, median over steps 31–300:

| model | parameters | eager | `--compile true` | peak memory (eager / compiled) | 50k steps, faster mode |
|---|---:|---:|---:|---:|---:|
| L = 2, width 256 | 2.13 M | **52 ms** | 68 ms | 3.2 / 2.6 GB | 0.7 h |
| L = 16, width 256 | 13.4 M | 344 ms | **175 ms** | 21.5 / 17.2 GB | 2.4 h |
| L = 8, width 512 | 26.8 M | 388 ms | **187 ms** | 22.2 / 17.8 GB | 2.6 h |

- **Compile halves the step time of the larger models** and is slower at L = 2, where the step is
  dominated by fixed per-step overhead (kernel launches) rather than arithmetic; it also pads to a
  multiple of 128 tokens. Use it for L ≥ 4. L = 4 was not measured; by linear interpolation both
  modes are close there (about 83 ms compiled, 94 ms eager).
- **Data generation is not a bottleneck:** the median wait for a batch is under 0.5 ms with 7
  workers, for every model.
- **Memory** peaks at 22 GB of the 48 GB (the same on the L40 and the L40S), so batch 64 fits
  every model of the sweep.
- **Against the FLOP count** (0.93, 7.5 and 12.9 TFLOP per step, forward and backward), compiled
  L = 16 runs at about 43 TFLOPS and compiled L = 8 at width 512 at about 69 TFLOPS, 24% and 38%
  of the L40's 181 TFLOPS dense bf16 peak.
- **On an L40S** (the same chip, memory and bandwidth; 362 TFLOPS dense bf16, twice the L40's)
  the matrix products can run up to twice as fast, while the memory-bound operations (norms,
  activations, softmax) and the per-step overhead do not change. So the times above are an
  upper bound: the larger models should be faster by somewhat less than 2×, L = 2 about the same,
  and the advantage of compile is likely larger, since what it fuses is the part that doesn't
  speed up. Runs may land on either GPU; this changes their speed, not their results.

Interpolating linearly in L between the compiled measurements (an estimate, not measured):

| L at width 256 | 2 | 4 | 6 | 8 | 12 | 16 | L = 8, width 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| step time | 52 ms | ~83 ms | ~98 ms | ~114 ms | ~144 ms | 175 ms | 187 ms |
| 50k steps | 0.7 h | 1.2 h | 1.4 h | 1.6 h | 2.0 h | 2.4 h | 2.6 h |

At 50k steps the whole sweep (7 models × 2 seeds) is about 24 L40 GPU-hours, less on L40S nodes,
and every run fits in the 12 h limit. The periodic evaluation adds little (7 sets × 200 samples every 1,000 steps).

## 6. Tests

```bash
python tests/looped_lm/test_model.py      # 8 tests, CPU, under a minute
```

- **Causality:** the logits at position t don't change when the tokens after t change.
- **Loss:** equal to the cross-entropy of the full logits at the masked positions (so the head
  at loss positions only is exact); unmasking one answer removes exactly its term; padding never
  contributes (arbitrary tokens in place of `<pad>`, or each sample alone, give the same loss).
- **Evaluation:** with an oracle stub whose hidden state at position p encodes the token at
  p + 1, every answer is correct, chance and the Wilson interval are right, and the (d, b)
  breakdown adds up; the same oracle shifted to the answer's own position scores 0, so
  `pred_pos` is what is used.
- **Parameter counts** match the formula (also with prelude and coda, and heads × head_dim ≠
  width) and the figures above.
- **Structure and initialization:** `n_loops = 2` raises; layer counts; standard deviations of
  the embedding (1/√width by default, or the given value, saved in the config), the projections
  and the scaled output projections; norm gains 1; no biases; tied head; the logged embedding
  norms.
- **Learning-rate schedule** values at the warm-up, the middle and the end.
- **Overfitting:** a tiny model reaches a loss below 0.05 on one fixed batch.
- **Resuming:** a run stopped at step 6 and resumed gives the same losses on steps 7–12 as an
  uninterrupted run; resuming with another seed is refused.
