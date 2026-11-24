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
- **Initialization:** every weight normal with std 0.02; the output projections of attention
  (`o_proj`) and MLP (`down_proj`) with std 0.02 / √(2 × (P + C + K)); norm gains 1.
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
| `init_std` | 0.02 | |

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

- **Data:** `TrainStream` of `dataset_generation` on `configs/train.json` with
  `answer_format` (default `direct`), batches of 64 built in the stream (one edge count per
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
moves first. The run fails at start if a set is missing (write the suite first, §4).

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

1. **Write the evaluation suite** once (it is not in git), from the repo root:
   ```bash
   apptainer exec /scratch.hpc/$USER/trm.sif python -m looped_lm.dataset_generation.cli write \
       --suite looped_lm/dataset_generation/configs/eval_suite.json --check
   ```
   It takes a few minutes on the CPU and writes `looped_lm/data/eval_suite/` (162 MB).
2. **Benchmark:** `sbatch slurm/bench_looped_lm.sbatch` times L = 2 and L = 16 at width 256 and
   L = 8 at width 512, each with and without `torch.compile`, 300 steps each.
3. **Pilot:** L = 8, width 256, 50,000 steps, evaluated every 1,000 steps:
   ```bash
   RUN_NAME=pilot_L8_w256_s0 DEPTH=8 WIDTH=256 SEED=0 sbatch slurm/train_looped_lm.sbatch --steps 50000
   ```
   Add `--compile true` if the benchmark shows it pays. The step count of the sweep follows from
   where the pilot's evaluation loss and accuracy flatten.
4. **Sweep** (STEPS from the pilot):
   ```bash
   for L in 2 4 6 8 12 16; do for S in 0 1; do
       RUN_NAME=base_L${L}_w256_s${S} DEPTH=$L WIDTH=256 SEED=$S sbatch slurm/train_looped_lm.sbatch --steps $STEPS
   done; done
   for S in 0 1; do
       RUN_NAME=base_L8_w512_s${S} DEPTH=8 WIDTH=512 SEED=$S sbatch slurm/train_looped_lm.sbatch --steps $STEPS
   done
   ```
5. **Evaluate** each run: `apptainer exec --nv ... python -m looped_lm.evaluate --checkpoint
   checkpoints/looped_lm/<run>`.

`slurm/train_looped_lm.sbatch` takes `RUN_NAME`, `DEPTH`, `WIDTH` and `SEED` from the environment
(width W gives W/64 heads of 64 and m = 11W/4), passes any further arguments to `looped_lm.train`,
and logs to wandb offline unless `WANDB_API_KEY` is set (`wandb sync <run>/wandb/offline-run-*`
later). Slurm signals it 5 minutes before the time limit (12 h by default); the run saves its
checkpoint and stops, and **submitting the same command again resumes it**.

## 5. Throughput (estimate, to be replaced by the benchmark)

A step is about 64 × 1,087 ≈ 70k tokens. Counting matrix products and causal attention (mean
context about 600 tokens), forward and backward:

| model | TFLOP per step | step time at 15–30% of the L40S's 362 TFLOPS | 30k steps |
|---|---:|---:|---:|
| L = 2, width 256 | 0.93 | 10–20 ms | ~10 min |
| L = 16, width 256 | 7.5 | 70–140 ms | 0.6–1.2 h |
| L = 8, width 512 | 12.9 | 120–240 ms | 1–2 h |

Data generation costs about 1.1 ms per sample per worker, so about 10 ms per batch with 7
workers: the smallest model may be data-bound (watch `train/data_wait`).

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
  the embedding, the projections and the scaled output projections; norm gains 1; no biases;
  tied head.
- **Learning-rate schedule** values at the warm-up, the middle and the end.
- **Overfitting:** a tiny model reaches a loss below 0.05 on one fixed batch.
- **Resuming:** a run stopped at step 6 and resumed gives the same losses on steps 7–12 as an
  uninterrupted run; resuming with another seed is refused.
