# Graph-reachability data for the looped causal transformer

This package generates a synthetic reasoning task for a looped causal language model, written as
plain token sequences. A sample describes a directed graph as a list of edges, lists the target
nodes, and asks queries: *from this source, which target can you reach?* Answering requires
following the graph from the queried source for d hops through b competing branches.
Optionally, the model writes the search out (path, frontier or depth-first answers).

The package depends only on numpy and torch and has no link to the TRM code in this repository.

```
looped_lm/dataset_generation/
  vocab.py      the shared vocabulary (special tokens + 2048 node names)
  config.py     TaskConfig: every knob; Dist: integer distributions
  generate.py   make_sample(): one sample, a pure function of (config, seed, index, split, stream)
  formats.py    the four answer formats and the token roles
  dataset.py    TrainStream (infinite, on the fly), GeneratedDataset, StoredDataset, collate
  storage.py    fixed sets on disk (flat .npy arrays + meta.json + readable samples.jsonl)
  suite.py      evaluation suites: a (d, b) grid, companion pairs, a T-versus-b diagnostic and mixed sets
  checker.py    independent validation from token ids alone
  audit.py      shortcut audit and length statistics
  cli.py        command line
  configs/      train.json (training stream), eval_suite.json (evaluation sets)
looped_lm/data/            default output of `write` (ignored by git, like every data/ directory)
```

Contents: [1. The task](#1-the-task) · [2. Quick start](#2-quick-start) ·
[3. How a sample is built](#3-how-a-sample-is-built) · [4. Component shapes](#4-component-shapes-and-trimming) ·
[5. Backward search](#5-backward-search-and-the-number-of-targets) · [6. Tokens and layout](#6-tokens-and-layout) ·
[7. Answer formats](#7-answer-formats) · [8. What a sample contains](#8-what-a-sample-contains) ·
[9. Configuration reference](#9-configuration-reference) · [10. Determinism and seeds](#10-determinism-and-seeds) ·
[11. PyTorch access](#11-pytorch-access) · [12. Evaluation suites and files on disk](#12-evaluation-suites-and-files-on-disk) ·
[13. Command line](#13-command-line) · [14. Validation](#14-validation-checker-audit-statistics) ·
[15. Known properties](#15-known-properties-of-the-data) · [16. Tests](#16-tests)

## 1. The task

A real sample with depth d = 3 and branching b = 3, one query, one distractor and no padding
(T = 2 targets). The answer format changes only the text after `?`:

```
<bos>
n16 > n22 n32 > n14 n23 > n32 n28 > n38 n1 > n4 n24 > n39 n4 > n25 n26 > n21 n2 > n28 n11 > n7 n16 > n2 n22 > n26 n11 > n23 n8 > n24 n16 > n1 n27 > n19 n11 > n8 n7 > n27
T: n25 n14
Q: n11 ? n14 <eos>                                                   direct
Q: n11 ? n23 n32 n14 <eos>                                           path
Q: n11 ? [ n7 n23 n8 ] [ n27 n32 n24 ] [ n19 n14 n39 ] n14 <eos>     frontier
Q: n11 ? n7 n27 n19 <bt> n23 n32 n14 <eos>                           dfs
```

The queried source `n11` has three branches of three edges each:
- `n7 > n27 > n19` ends at a dead end;
- `n23 > n32 > n14` ends at the target `n14`;
- `n8 > n24 > n39` ends at a dead end.

The distractor `n16` has the same shape, and its branch `n1 > n4 > n25` ends at the other target.

Seen from the target list, `n14` and `n25` look alike. Only following the edges from `n11`
tells them apart.

**Guarantees, verified by the checker on every sample:**
- **No cycles and no merges.** No node has two incoming edges, and no node is shared between
  components.
- **Every queried source reaches exactly one target,** and every target is reached from exactly
  one source.
- **Every branch has the same length d,** and no branch has a fork after the source.
- **Every target has an incoming path.** A target can't be identified without following the
  graph.
- **Every sample has at least 2 targets,** exactly T in `fixed` mode (§3). Components without a
  target (padding) are allowed; all their branches end at dead ends.

## 2. Quick start

Everything runs from the repository root.

```bash
# Look at a few samples of a configuration (any field can be overridden with JSON values)
python -m looped_lm.dataset_generation.cli show --config looped_lm/dataset_generation/configs/train.json \
    --override depth=3 branching=3 total_edges=60 answer_format='"dfs"' --num 2

# Length statistics of the training stream, in batches of 64 as training makes them (with the
# padding fraction per batch), and of every evaluation set
python -m looped_lm.dataset_generation.cli stats --config looped_lm/dataset_generation/configs/train.json --batch-size 64
python -m looped_lm.dataset_generation.cli stats --suite looped_lm/dataset_generation/configs/eval_suite.json

# Shortcut audit of every evaluation set (per grid cell, with its number of targets, and pooled)
python -m looped_lm.dataset_generation.cli audit --suite looped_lm/dataset_generation/configs/eval_suite.json

# Write the evaluation sets (vocabulary included) to looped_lm/data/eval_suite/ and validate
# every written sample
python -m looped_lm.dataset_generation.cli write --suite looped_lm/dataset_generation/configs/eval_suite.json --check
```

In training code:

```python
from torch.utils.data import DataLoader
from looped_lm.dataset_generation import TaskConfig, TrainStream, StoredDataset, collate

cfg = TaskConfig.load("looped_lm/dataset_generation/configs/train.json")
train = DataLoader(TrainStream(cfg, seed=0, batch_size=64), batch_size=None, num_workers=4)
for batch in train:
    tokens, mask = batch["tokens"], batch["loss_mask"]   # [B, L] each
    # logits = model(tokens[:, :-1]); loss on mask[:, 1:] against tokens[:, 1:]
    ...

grid_cell = StoredDataset("looped_lm/data/eval_suite/grid/d16_b4")
eval_loader = DataLoader(grid_cell, batch_size=64, collate_fn=collate)
```

`write` writes to `looped_lm/data/<suite file name>` unless `--out` says otherwise. Every `data/`
directory is ignored by git, so generated sets stay out of the repository.

## 3. How a sample is built

`generate.make_sample(config, seed, index, split, stream)` works in five steps.

**1. Components.** A component is a source with b non-branching paths (branches) of d edges
each. Every component has at most one target. There are two modes, set by `targets_mode`; the
shipped configs use `fixed`.

**`targets_mode: "fixed"`** (§4 explains why):
- **Queried components:** `num_queries` of them. Each draws its own d and b from `depth` and
  `branching`. One branch, chosen at random, ends at a target; the other b − 1 end at dead ends.
- **Distractors:** T − `num_queries` of them, with T drawn from `num_targets`. Their d and b
  come from `distractor_depth` / `distractor_branching`, which default to the query
  distributions, so queried and distractor components are drawn identically. Each has exactly
  one target.
- **The edge count** E comes from `total_edges`:
  - **in the training stream, one E per batch,** drawn by the batch's own generator and shared
    by all its samples (§11);
  - **in map-style sets** (evaluation sets, `show`, `stats` without `--batch-size`, `audit`), E is
    the sample's own draw.
- **Trimming,** if the queried components and the distractors need more than E edges (or more
  than `trim_edges`, when set): one dead-end branch is removed from a randomly chosen distractor
  with b > 1, again and again, until they fit. A queried component is never trimmed, a depth is
  never changed, and no distractor goes below one branch. If even single-chain distractors don't
  fit, E is raised to that minimum in map-style sets; the training stream refuses such configs
  (§11).
- **Padding:** target-less components of distractor shapes fill the remaining edges while they
  fit (until 16 consecutive draws don't); a last **filler** takes the exact remainder r, with a
  shape d′ × b′ = r as close to the distractor distributions as possible (a chain is the
  fallback). Padding and filler carry no target, so every sample has **exactly T targets** and
  exactly E edges.
- **Separate generators.** Shapes, trimming, padding, names, edge order and the query/target
  permutations each use their own generator, derived from the sample's seed. Changing a later
  step leaves the earlier ones untouched: the queried shapes and the query order don't depend on
  E, and the companion pairs (§12) share their queried component.

**`targets_mode: "per_component"`** (the original mode, the default of `TaskConfig`, kept so that
old configs reproduce exactly the same samples):
- queried components as above, plus `num_distractors` distractors with a target each;
- padding components up to E **each carry a target too**, and only the filler has none; so the
  number of targets grows with E (in a grid cell, T = floor(E / (d·b)));
- no trimming: if the components need more than E edges, E is raised.

**2. Node names.** The names are drawn without replacement from the first `pool_size` names of
the vocabulary, at random for every sample. A name means nothing across samples, so the model
can't memorize roles. `TaskConfig` checks at construction that the worst-case node count of the
configuration fits in the pool (number of nodes = edges + components). The generator asserts it
again per sample.

**3. Edges.** Every branch contributes its d edges, the first from the source. `edge_order` sets
the order:
- **`random`** (default): all edges are shuffled.
- **`reverse`:** within each branch, every edge appears *after* its successor (the target end
  first). This allows composition while reading: by the time `s > a` is read, the rest of the
  path from `a` is already known.
- **`forward`:** within each branch, every edge appears before its successor.

For `reverse` and `forward`, the branches are interleaved by a uniformly random merge, which
keeps each branch's internal order. Components are therefore never written as contiguous blocks.

**4. Canonical branch order.** The branches of a source are ordered by where their **first
edge** (from the source) appears in the edge list. The `frontier` and `dfs` formats use this
order, so the model can determine it from the input.

**5. Targets and queries.**
- **The target list** holds the target of every component that has one (T in fixed mode), shuffled.
- **The queries** come in a random order, independent of the components' positions in the edge
  list, and each asks about a different source.

## 4. Component shapes and trimming

In `fixed` mode, queried and distractor components are drawn from the same shape distribution,
and only distractors are trimmed when the edge budget is short.

**Why only distractors.** A question's difficulty is set by the queried component's d and b. Those
are drawn before the edge count and never changed, so the difficulty of a question never depends
on the edge count E: a batch with few edges asks the same questions as one with many, and the
correlation between E and the queried d, b, d·b is zero (a test checks it on 12,000 samples). When
the budget is short, the distractors pay for it, one dead-end branch at a time.

**Why trimmed distractors are not a shortcut.** Branching occurs only at roots, so identifying a
component's shape requires tracing back to its root, which already reveals whether it is the
queried source. A model that finds "the target whose component has b branches" has done the
backward search of §5. The audit's `branching_root` heuristic measures exactly this, and is
reported but not counted as a shortcut.

**How often trimming happens** (share of samples with a trimmed distractor, and share of
distractors trimmed):

| set | samples trimmed | distractors trimmed | note |
|---|---:|---:|---|
| `configs/train.json`, edges 192–512 | 0.017% | ≈ 0 | untrimmed samples need 96 edges on average, 182 at the 99th percentile |
| same, edges 224–512 | 0.0006% | ≈ 0 | |
| same, edges 256–512 | < 0.0001% | 0 | |
| grid cells with d·b ≤ 64 (26 of 32) | 0 | 0 | 8 × d·b ≤ 512 |
| `grid/d12_b8`, `d16_b8`, `d24_b4`, `d24_b8`, `d32_b4`, `d32_b8` | 100% | 84–100% | d and b fixed, so every sample needs 8 × d·b > 512 edges |
| `mixed/q4_in_distribution`, `mixed/q4_deep` | 0 | 0 | 8 × 32 = 256 and 8 × 64 = 512 fit exactly |
| `diag/*` (512 edges) | 0, except `b8_t9` and `b8_t12`: 100% | 66% and 95% in those two | §12 |

In the six always-trimmed grid cells, the untrimmed subset of the audit is empty. The **companion
pairs** (§12) compare trimmed and untrimmed distractors at the same length instead.

## 5. Backward search and the number of targets

Every node has in-degree ≤ 1, so from a target the graph is a single chain back to its source. A
model can therefore answer a query **backwards**: trace every target back to its root (d steps
each, T × d in all) and pick the target whose root is the queried source. Its cost grows with
the number of targets T, not with the branching b. **Forward** search from the queried source
explores its b branches (b × d steps) and is indifferent to T.

So b measures how many alternatives a model must consider only while T ≥ b. With fewer targets
than branches, backward search is cheaper and b no longer measures plurality.

- **In `per_component` mode** the grid cells had T = floor(E / (d·b)) targets: 2 at d32_b8, where
  b = 8 then meant nothing, and 512 at d1_b1, with sequences longer than any training sample.
- **In `fixed` mode** every grid cell has T = 8, so T ≥ b for every b ≤ 8, chance is 1/8 in
  every cell, and every cell has exactly 1,551 tokens. `TaskConfig` warns when `num_targets`
  can be smaller than the configuration's maximum branching.

**The T-versus-b diagnostic** (`diag/bB_tT` in the suite, §12) separates the two strategies: at
d = 8 and one query, b ∈ {1, 2, 4, 8} is crossed with T ∈ {6, 9, 12}, without the cells where
T < b. **A model that searches forward should get worse with b and be indifferent to T; one that
searches backward, the opposite.** Distractor trimming in these cells changes neither cost:
forward search never enters a distractor, and backward search walks each target's chain of d
edges whatever its root's branching.

`depth_match` (does the target's chain have the query's d?) and `branching_root` (does the
target's root branch?) are backward-search heuristics: they need a trace from every target to
its root. The audit reports them and doesn't count them as shortcuts (§14).

## 6. Tokens and layout

Vocabulary (`vocab.json`, the same for every set): 11 special tokens, then the node names
`n0` … `n2047`, 2059 tokens in total.

| id | token | use |
|---:|---|---|
| 0 | `<pad>` | padding (collate) |
| 1 | `<bos>` | start of sample |
| 2 | `<eos>` | after the last answer |
| 3 | `>` | edge: `src > dst` |
| 4 | `T:` | starts the target list |
| 5 | `Q:` | starts a query |
| 6 | `?` | ends the query prompt; the answer follows |
| 7, 8 | `[`, `]` | frontier levels |
| 9 | `<bt>` | backtrack marker (dfs) |
| 10 | `;` | query separator: after every answer except the last |
| 11 … 2058 | `n0` … `n2047` | node names |

Layout with `targets_position="after"` (the default):

```
<bos> s1 > d1 s2 > d2 ... T: t1 t2 ... Q: x ? <answer> ; Q: y ? <answer> ; ... Q: z ? <answer> <eos>
```

With `targets_position="before"`, the `T: …` block comes right after `<bos>`.

## 7. Answer formats

Throughout, `b_1 … b_b` are the source's branches in canonical order, and `b_k[l]` is the node
at level l of branch k (level 1 = first node after the source; level d = the end of the branch).

| format | answer | length |
|---|---|---|
| `direct` | `t` | 1 |
| `path` | the target branch's nodes, levels 1 … d (the last is `t`) | d |
| `frontier` | for each level l = 1 … d: `[ b_1[l] … b_b[l] ]`; then `t` | d·(b + 2) + 1 |
| `dfs` | branches in canonical order. A dead-end branch is written out in full (levels 1 … d), then `<bt>`. The target branch is written out (its last node is `t`), and the answer stops there. | k·(d + 1) + d, where k = dead-end branches before the target branch |

**Frontier.** Every hop is written out explicitly. The last bracket holds the target and the
b − 1 dead ends, so the final token is a pure selection among the last bracket. Nodes keep their
branch's rank at every level.

**DFS.** Branches after the target branch are never visited. With the target branch first, the
dfs answer is the same as the path answer.

After every answer, the model must produce `;`, or `<eos>` after the last query. That token is
in the loss, so the model learns to signal that an answer is complete (variable length in
`path`, `frontier` and `dfs`).

## 8. What a sample contains

`make_sample` returns a `Sample`. `StoredSet.get(i)` returns a dict with the same fields, and
the datasets return them as tensors.

| field | shape / type | meaning |
|---|---|---|
| `tokens` | int [L] | token ids |
| `loss_mask` | bool [L] | true on answer tokens (including intermediate nodes, brackets and `<bt>`) and on the `;` / `<eos>` ending each answer; false on `<bos>`, edges, the target list and query prompts (`Q: x ?`) |
| `query_index` | int [L] | the token's query (from its `Q:` to its end token), −1 elsewhere |
| `depth`, `branching` | int [L] | d and b of the token's query, 0 elsewhere |
| `role` | int [L] | 0 other, 1 final answer, 2 intermediate node, 3 bracket, 4 backtrack marker, 5 end (`;` / `<eos>` after an answer) |
| `pred_pos` | int [L] | at a final answer: the position whose output predicts it (its position − 1); −1 elsewhere |
| `queries` | list of dicts | per query, in query order: `source`, `target`, `path` (levels 1 … d, ending at the target), `depth`, `branching`, `branch_order` (first node of each branch, canonical order), `prompt_pos` (position of `Q:`), `answer_start`, `answer_end` (position of the end token; the answer is `tokens[answer_start:answer_end]`) |
| `num_edges`, `num_components`, `num_targets` | int | sample-level counts |
| `trimmed` | bool | some distractor lost branches to fit the edge budget (`fixed` mode, §4) |
| `branches_removed` | int | dead-end branches removed from the distractors |
| `distractors` | list of dicts | per unqueried component with a target, in generation order: `depth`, `drawn_branching`, `branching` (after trimming). In `per_component` mode the target-carrying padding components are included, untrimmed. `None` for sets written before format version 3 |

`role`, `depth`, `branching` and `query_index` let you compute metrics per role (e.g. final-answer
accuracy, or intermediate-step accuracy) and per (d, b) within mixed batches. Every node token
in an answer that isn't the final one has role "intermediate", including the dead ends in
frontier and dfs.

**Next-token convention:** the logits at position t − 1 predict token t. To score final answers,
compare the argmax at `pred_pos[t]` with `tokens[t]` wherever `role == 1`.

## 9. Configuration reference

`TaskConfig` (in `config.py`) is a dataclass, read from and written to JSON. Its fields:

| field | default | sampled | meaning |
|---|---|---|---|
| `name` | `"default"` | | label (set names in suites) |
| `depth` | 4 | per query | d: edges from the source to the end of each branch (d − 1 intermediate nodes) |
| `branching` | 2 | per query | b: branches leaving the source |
| `targets_mode` | `"per_component"` | | `fixed`: T targets per sample, distractors trimmed when short of edges, target-less padding (§3, §4). `per_component`: the original mode, every padding component with a target. The shipped configs use `fixed` |
| `num_queries` | 1 | per sample | queried sources |
| `num_targets` | `null` | per sample | `fixed` only, required: T, the targets per sample (queried + distractors); T ≥ `num_queries` + 1 |
| `num_distractors` | `null` (= 1) | per sample | `per_component` only: unqueried components with a target, before padding. An error in `fixed` mode |
| `distractor_depth` | `null` | per distractor | `null`: same as `depth` |
| `distractor_branching` | `null` | per distractor | `null`: same as `branching` |
| `total_edges` | `null` | per sample; per batch in `TrainStream` | `null`: no padding (and no trimming). Otherwise the edge count to reach with padding components; the larger of this and what the sample needs (in `fixed` mode: with distractors trimmed to chains). `TrainStream` requires it, with a minimum that fits the worst case (§11). |
| `trim_edges` | `null` | | `fixed` only: the budget the distractors are trimmed to, if smaller than the edge count; the rest is padding. Used by the companion sets B (§12) |
| `edge_order` | `"random"` | | `random`, `reverse` or `forward` (§3) |
| `targets_position` | `"after"` | | `after` or `before` the edges |
| `answer_format` | `"direct"` | | `direct`, `path`, `frontier` or `dfs` (§7) |
| `pool_size` | 2048 | | names are drawn from `n0 … n{pool_size−1}` |

**Distributions** (`Dist`), for every field marked "sampled":
- `4`: always 4;
- `[1, 2, 4]`: uniform over the list;
- `{"range": [1, 8]}`: uniform over 1 … 8 inclusive;
- `{"values": [1, 2, 4], "probs": [0.5, 0.25, 0.25]}`: explicit probabilities.

**Validation** at construction raises `ValueError` when:
- a field has an unknown value;
- d, b or the number of queries is < 1;
- `per_component`: `num_queries + num_distractors` < 2 (at least two targets);
- `fixed`: `num_targets` missing, `num_targets` < `num_queries` + 1 in some sample,
  `num_distractors` set, or `trim_edges` without `total_edges`, above its minimum or below the
  worst case;
- `num_targets` or `trim_edges` given in `per_component` mode;
- `pool_size` exceeds the vocabulary;
- the worst-case node count exceeds `pool_size`.

It **warns** (Python `UserWarning`) when `num_targets` can be smaller than the largest branching:
backward search is then cheaper than forward search (§5).

`TaskConfig.check_batch_edges()` is the extra check of the training stream (§11); `TrainStream`
calls it at construction.

## 10. Determinism and seeds

A sample's random generator is seeded with `SeedSequence([split, seed, stream, index])`:

- **`split`** is `train` (0), `eval` (1) or `debug` (2):
  - the training stream always uses `train`;
  - written evaluation sets use `eval`;
  - `show`, `stats` and `audit` use `debug`.

  **The same seed number therefore gives unrelated samples in different splits,** so
  evaluation sets never replay the training stream. Duplicates between splits are not removed,
  but with random names over 2048 nodes, an exact duplicate is astronomically unlikely.
- **`stream`** separates the sets of a suite. Each set uses the CRC32 of its name, e.g.
  `grid/d8_b4`.
- **`index`** is the sample's number within the stream.

**In `fixed` mode** the sample's `SeedSequence` is split (`spawn`) into six independent
generators: shapes (queried and distractor d and b, T, the edge count), trimming (and each
distractor's target branch), padding, names, edge order, and the target-list and query
permutations. A step that changes (a different E, trimming or not) leaves the others' draws
unchanged. `per_component` mode keeps its single generator, so it reproduces the samples of the
code before `fixed` mode bit for bit (a test checks this against stored fingerprints).

**Batches** have their own generator, `SeedSequence([1000 + split, seed, stream, batch])`, used
only to draw the batch's edge count. The 1000 keeps it apart from every sample's generator.

`make_sample` is a pure function of these arguments and the config, and `make_batch` of
(config, seed, batch index, batch size, split, stream). Nothing depends on global
state, on the number of DataLoader workers, or on iteration order.

**Training must draw names from the full pool.** If training used a smaller pool than
evaluation, the unused names' embeddings would never be trained, and the evaluation would
measure that rather than reasoning. `TrainStream` therefore refuses a config with
`pool_size != 2048` unless `allow_small_pool=True`, which is meant for debugging only.

## 11. PyTorch access

**`TrainStream(config, seed=0, batch_size=None, start=0, rank=0, world_size=1)`** is an
`IterableDataset` that generates the infinite training stream on the fly (split `train`).
- With `batch_size=None`, it yields samples `start, start + 1, …`.
- With `batch_size=B`, it yields collated batches; batch j holds samples j·B … j·B + B − 1.
- **One edge count per batch.** Batch j's edge count E_j is drawn from `total_edges` by the
  batch's own generator (seeded from the seed and j, §10), and every sample of the batch is padded
  to it. All the other draws (depths, branchings, queries, names, orders) are the sample's own,
  from (seed, sample index). Without `batch_size`, every sample is a batch of one.
  - **Why:** with an edge count per sample, the samples of a batch have very different lengths.
    With `configs/train.json` and B = 64, the mean padding is 30.6% of the batch with
    per-sample counts and 1.1% with one count per batch. The rest comes from the target list and
    the answers, whose lengths still vary; `collate` pads it.
  - **E_j is independent of the samples' depths and branchings,** otherwise sequence length
    would be a cue for difficulty. So nothing is resampled or rejected. Instead, the config must
    guarantee that every E in the range fits every sample. `TaskConfig.max_required_edges()`
    is the worst case:
    - `fixed`: the queried components at maximum size plus all distractors as single chains of
      maximum depth (trimming handles everything above that): max over the draws of q × max(d·b)
      + (T − q) × max(distractor d);
    - `per_component`: `num_queries.max` × max(d·b) + `num_distractors.max` × max(d·b of
      distractors).

    `TrainStream` checks `total_edges.min` against it at construction
    (`TaskConfig.check_batch_edges`) and fails with the numbers otherwise; it also refuses
    `total_edges: null`.
  - **`configs/train.json`** (T from 5 to 12, up to 4 queries, d ≤ 8, b ≤ 4): 4 queries × 32 + 8
    chains × 8 = **192**, its minimum edge count. Untrimmed, a sample needs at most 12 × 32 = 384
    edges, 96 on average; trimming is rare (§4).
- Use it with `DataLoader(stream, batch_size=None, num_workers=W)`:
  - worker w makes items w, w + W, w + 2W, …;
  - the DataLoader returns workers' items in turn;
  - so the order is 0, 1, 2, … whatever W is.
- **Batch inside the stream (`batch_size=B`), not in the DataLoader.** A DataLoader with
  `batch_size` set would build each batch from one worker's samples, and the batch composition
  would then depend on W.
- **Several ranks:** rank r of R takes items r, r + R, … of the global stream, i.e. batches
  r, r + R, … with `batch_size`, so the ranks together see the same global batch sequence.
- **Generation speed:** about 1.1 ms per sample of `configs/train.json` per process (about 900
  samples/s), so a few workers keep up with a small model.

**`GeneratedDataset(config, size, seed, split="eval", stream=0, offset=0)`** is a map-style
dataset. Item i is sample offset + i, generated on access.

**`StoredDataset(path)`** is map-style access to a set written by the CLI.

**`collate(items)`** right-pads with `<pad>` to the longest sample in the batch. The padding
values are: `loss_mask` false, `query_index` −1, `pred_pos` −1, `role` 0, `depth` and
`branching` 0. It returns `[B, L]` tensors plus:
- `length` [B];
- `index` [B], the sample indices;
- `queries`, a list per sample of its query dicts;
- `trimmed` [B] (bool) and `branches_removed` [B];
- `distractors`, a list per sample of its distractor dicts.

## 12. Evaluation suites and files on disk

A suite JSON describes every fixed set, as in `configs/eval_suite.json`:

```json
{
 "seed": 1,
 "base": {"targets_mode": "fixed", "num_targets": 8, "edge_order": "random",
          "targets_position": "after", "answer_format": "direct", "pool_size": 2048},
 "grid": {"depths": [1, 2, 4, 8, 12, 16, 24, 32], "branchings": [1, 2, 4, 8],
          "num_samples": 1000, "total_edges": 512},
 "companions": {"num_samples": 200, "cells": "trimmed"},
 "diag": {"depth": 8, "branchings": [1, 2, 4, 8], "num_targets": [6, 9, 12],
          "num_samples": 1000, "total_edges": null},
 "mixed": [{"name": "q4_in_distribution", "num_samples": 2000,
            "overrides": {"depth": {"range": [1, 8]}, "branching": {"range": [1, 4]},
                          "num_queries": 4, "total_edges": 256}},
           {"name": "q4_deep", "num_samples": 2000,
            "overrides": {"depth": {"range": [9, 16]}, "branching": {"range": [1, 4]},
                          "num_queries": 4, "total_edges": 512}}],
 "readable_samples": 5
}
```

Token lengths below are with the `direct` format; the other formats add their answers.

**Grid cells (`grid/d{d}_b{b}`), 32 sets of 1,000:**
- `base` and the grid's `overrides`, with one query per sample, d and b fixed, and distractors
  of the same shape (the `distractor_*` fields are reset).
- **T = 8 and 512 edges in every cell,** so chance is 1/8 everywhere and every sample has
  exactly 1,551 tokens, within the training lengths. (`total_edges: null` would take the
  smallest count that fits every cell.)
- **Six cells always trim** (d12_b8, d16_b8, d24_b4, d24_b8, d32_b4, d32_b8): 8 × d·b > 512, so
  their distractors lose branches in every sample (§4). In the other 26 cells, nothing is
  trimmed and the rest of the 512 edges is target-less padding.

**Companion pairs (`companion/d{d}_b{b}_A` and `_B`), 200 samples each,** for the grid cells that
trim (`"cells": "trimmed"`, or a list of `[d, b]`). Both sets of a pair have the edge count that
fits every component untrimmed, 8 × d·b, T = 8 and one query:
- **A ("untrimmed"):** distractors at their drawn shapes, no trimming, no padding;
- **B ("trimmed_matched"):** distractors trimmed exactly as in the 512-edge grid cell
  (`trim_edges: 512`, the same procedure), the freed edges filled with target-less padding up to
  the same edge count as A.

A and B share their seed stream (the pair's name), and each generation step has its own
generator (§10), so **sample i of A and sample i of B have the same queried component, with the
same names, the same answer and token positions, and (in `random` edge order) the same edge
positions; they differ only in the distractors.** A test checks this. The pairs are longer than
any training sample:

| pair | edges | tokens (A and B) |
|---|---:|---:|
| d12_b8 | 768 | 2,319 |
| d24_b4 | 768 | 2,319 |
| d16_b8 | 1,024 | 3,087 |
| d32_b4 | 1,024 | 3,087 |
| d24_b8 | 1,536 | 4,623 |
| d32_b8 | 2,048 | not generated |

**d32_b8 has no pair; it was dropped.** Untrimmed, its 8 components need 8 × (1 + 256) = 2,056
node names, more than the 2,048 the pool holds. The other five pairs cover that corner of the
grid (d ≥ 12 with b = 8, and d ≥ 24 with b = 4). The suite skips the pair with a warning.

**T-versus-b diagnostic (`diag/b{b}_t{T}`), 11 sets of 1,000** (§5): d = 8, one query, every
branching in `branchings` crossed with every T in `num_targets`, without the cells where T < b
(here `b8_t6`). All cells have **512 edges**, like the grid, so their lengths match the grid's
and lie inside the training range:

| cell | b1_t6 | b1_t9 | b1_t12 | b2_t6 | b2_t9 | b2_t12 | b4_t6 | b4_t9 | b4_t12 | b8_t9 | b8_t12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tokens | 1,549 | 1,552 | 1,555 | 1,549 | 1,552 | 1,555 | 1,549 | 1,552 | 1,555 | 1,552 | 1,555 |
| samples trimmed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100% | 100% |
| distractors trimmed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66% | 95% |

Only the b = 8 cells trim (T × 64 > 512); their distractors keep enough branches that
`branching_root` stays at chance. With `total_edges: null` the section would instead take the
smallest count that fits every cell, 8 × 8 + 11 chains × 8 = 152 (469–475 tokens, shorter than any
training sample, with up to 100% of distractors trimmed); 768 edges would avoid trimming
altogether but exceed the training lengths (2,317–2,323 tokens).

**Mixed sets (`mixed/<name>`):** `base` plus the set's `overrides`, here 4 queries and T = 8 (4
distractors). They never trim: 8 × 32 = 256 and 8 × 64 = 512 edges fit exactly. 798 and 1,566
tokens.

**Skipping sections:** `write`, `stats` and `audit` take `--skip companion/ diag/` (any name
prefixes) to leave sets out; `--only` selects instead.

**The answer format** belongs to the suite: write one suite per format you train with, e.g.
a copy `eval_suite_dfs.json` with `"answer_format": "dfs"`, which `write` puts in
`looped_lm/data/eval_suite_dfs/`.

**Layout written by `write --suite S [--out DIR]`** (DIR defaults to `looped_lm/data/<S without .json>`):

```
DIR/vocab.json          the shared vocabulary
DIR/suite.json          a copy of the suite
DIR/grid/d8_b4/         one directory per set:
    meta.json           config, split, seed, stream, sample count, vocabulary fingerprint, length
                        stats, share of trimmed samples
    samples.jsonl       the first few samples, readable (text, queries with names, answers)
    tokens.npy          uint16, all samples' tokens concatenated
    offsets.npy         int64 [N + 1]: sample i is tokens[offsets[i]:offsets[i+1]]
    num_edges.npy, num_components.npy, num_targets.npy        int32 [N]
    trimmed.npy (uint8 [N]), branches_removed.npy (int32 [N])
    d_offsets.npy       int64 [N + 1]: the distractors of sample i are d_offsets[i] ... d_offsets[i+1] - 1
    d_depth.npy, d_drawn_branching.npy, d_branching.npy       int16 [D]
    q_offsets.npy       int64 [N + 1]: the queries of sample i are q_offsets[i] ... q_offsets[i+1] - 1
    q_source.npy, q_target.npy                                uint16 [Q]
    q_depth.npy, q_branching.npy                              int16 [Q]
    q_prompt_pos.npy, q_answer_start.npy, q_answer_end.npy    int32 [Q]
    q_path.npy, q_path_offsets.npy                            path tokens of every query, flattened
    q_branch_order.npy, q_branch_order_offsets.npy            branch orders, flattened
```

- **The per-token metadata is not stored.** It follows from the tokens and the query positions,
  and `StoredSet.get` rebuilds it, so a stored sample is identical to the generated one (a test
  checks this).
- **Size:** the default suite (55 sets, 49,000 samples) takes 162 MB.
- **Loading:** arrays open with `mmap`.
- **Version checks:** `meta.json` records a format version (3) and a vocabulary fingerprint, and
  loading fails on a mismatch. Version 2 sets (before the trimming fields) are still readable:
  their samples read as untrimmed, with `distractors` `None`.

## 13. Command line

`python -m looped_lm.dataset_generation.cli <command>`. Every command that generates samples takes `--seed`
(default 0).

| command | what it does |
|---|---|
| `vocab OUT_DIR` | writes `OUT_DIR/vocab.json` |
| `show --config C [--override K=JSON …] [--num 3]` | prints readable samples and their queries |
| `stats (--config C \| --suite S [--only/--skip PREFIX …]) [--num 2000] [--batch-size B]` | tokens (mean, min, max), edges, targets, nodes and answer lengths per configuration, and the trimming shares: samples with a trimmed distractor, and distractors trimmed. With `--batch-size`, the samples come in batches as `TrainStream` makes them (one edge count per batch), and a last column gives the mean padding fraction per batch, 1 − Σ lengths / (B × longest) |
| `audit (--config C \| --suite S [--only/--skip PREFIX …]) [--num 2000]` | shortcut audit per configuration, on all samples and on the untrimmed ones, with the number of targets and the share of trimmed samples of each set; for a suite, also the grid cells pooled, and a closing table of the exclusion heuristics, one row per set (§14) |
| `write --suite S [--out DIR] [--num N] [--check] [--skip PREFIX …]` | writes every set of the suite, the vocabulary and the suite to DIR (default `looped_lm/data/<S without .json>`); `--num` overrides the sample counts; `--check` runs the checker on everything written (including the exact number of targets in `fixed` mode); `--skip companion/ diag/` leaves those sections out |
| `check DIR [--num N]` | runs the checker on every set under DIR (all samples, or the first N per set); exits with status 1 on any invalid sample |

`--override` takes JSON values, e.g. `depth='{"range": [2, 6]}'` or `answer_format='"path"'`.
`--only grid/d32` restricts a suite to the sets whose name starts with the prefix; `--skip`
leaves them out. Building a suite with a companion pair that doesn't fit the name pool prints a
warning (d32_b8, §12).

## 14. Validation: checker, audit, statistics

### Checker (`checker.py`)

The checker validates a sample from its **token ids alone**. It imports only the vocabulary,
never the generator or `formats.py`, so a generator bug can't hide itself. It:

1. **Parses** `<bos>`, the edges, the target list (before or after the edges), and the queries
   with their answers and end tokens. Anything else is an error, e.g. tokens after `<eos>` other
   than padding, or a last query not ended by `<eos>`.
2. **Checks the graph structure:**
   - no self-loops and no duplicate edges;
   - no merges (in-degree ≤ 1) and no cycles (topological sort);
   - every component is a root whose branches all have the same depth (the component's d) and
     never branch: **branching occurs only at roots**;
   - at most one target per component; components without a target are allowed;
   - every target is the end of a path with one incoming edge, and is reached from exactly one
     source;
   - at least 2 targets, no duplicates in the list, and **exactly T** when given
     (`num_targets`; `check` and `write --check` pass each sample's own count and require it to
     be in the config's `num_targets`).
3. **Checks every query by its own graph search** (breadth-first from the source):
   - the source is a root;
   - exactly one target is reachable;
   - the path is rebuilt from parent pointers;
   - d = its length, b = the source's out-degree;
   - the canonical branch order is rebuilt from the edge positions;
   - the expected answer in the configured format is rebuilt and compared token by token.
4. **Checks the metadata** against what it derived, when given a full sample:
   - loss mask, query index, d, b, roles and predicting positions per token;
   - source, target, path, d, b, branch order and positions per query;
   - the distractors: their (depth, branching) must be exactly the shapes of the unqueried
     target-carrying components in the graph, with 1 ≤ branching ≤ drawn branching, and
     `branches_removed` and `trimmed` must agree with them.

   Optionally it also checks the edge order (each edge before or after its successor), the
   target-list position, and the edge count.

`check_sample(sample, fmt, edge_order, targets_position, num_edges, num_targets)` raises
`CheckError` with the first problem found.

### Audit (`audit.py`)

The audit uses samples of the debug split. For each query, each heuristic picks one target
without following the graph from the source; ties are broken at random. Every heuristic sees
the same parsed tokens. Every heuristic is reported **on all samples and on the untrimmed
subset** (samples where no distractor was trimmed; the subset is empty in the six grid cells
that always trim, and equal to all samples where nothing trims).

| heuristic | picks | chance |
|---|---|---|
| `recent` | the target mentioned last in the graph description (edges and target list) | 1 / #targets |
| `first_listed` | the first target of the list | 1 / #targets |
| `in_degree` | the target with the most incoming edges | 1 / #targets |
| `depth_match` | a target whose path length (walked backwards to its source) equals the query's d; **backward search, not counted as a shortcut** | 1 / #targets |
| `branching_root` | a target whose root (walked backwards) has out-degree > 1; any target if none; **backward search, not counted as a shortcut** | 1 / #targets |
| `proximity` | the target whose incoming edge is nearest, in the edge list, to one of the source's edges; **queries with d ≥ 2 only** | 1 / #targets |
| `elimination` | a target not already answered by an earlier query of the sample | 1 / (#targets − earlier queries) |
| `random` | any target (calibration: must be at chance) | 1 / #targets |
| `not_nearest` | uniform among all targets except the one `proximity` picks; **queries with d ≥ 2 only** | 1 / #targets (ceiling 1 / (#targets − 1)) |
| `not_recent` | uniform among all targets except the one `recent` picks | 1 / #targets (ceiling 1 / (#targets − 1)) |

**`depth_match` and `branching_root` are backward search** (§5): each needs a trace from every
target to its root, which costs as much as answering backwards and already reveals which root
is the queried source. They are above chance wherever the components' depths differ
(`depth_match`) or where only some roots branch, e.g. when distractors are trimmed or chains
(`branching_root`). The audit reports them, marks them "backward search, not a shortcut", and
leaves them out of its count of flags.

**The exclusion heuristics** (`not_nearest`, `not_recent`) ask whether ruling out one candidate
helps, which matters most when there are few targets: with T targets, excluding one can raise
the accuracy from 1/T to at most 1/(T − 1), i.e. from 1/2 to 1 when T = 2.
- **Their accuracy is computed exactly,** not sampled: per query, 1/(T − 1) if the excluded
  target isn't the answer, else 0.
- **Chance is 1/T:** that's what they score when the excluded target is the answer as often as any
  other target, i.e. when the base heuristic is at chance. The **ceiling** 1/(T − 1) is reached
  when the excluded target is never the answer. The audit prints both.
- An exclusion heuristic is above chance exactly when its base heuristic is below chance; the
  exclusion turns that into the accuracy a model could gain.

**Why `proximity` excludes d = 1:** at d = 1 the target's incoming edge *is* an edge of the
source, so "nearest" means following the one edge, the base case of the task, not a shortcut.
With d = 1 included, `proximity` scores 0.18 against a chance of 0.05 in a mixed set. Without it,
it's at chance.

**The z-score** of each heuristic is (hits − Σ chance) / √Σ var, with var the variance of a
query's hit at chance: chance·(1 − chance) for a heuristic that picks one target, and
1/(T²(T − 1)) for the exact hits of an exclusion heuristic (1/(T − 1) with probability
(T − 1)/T, else 0). That variance is small, so even a small advantage of an exclusion heuristic
gets a large z: read its accuracy against the chance and ceiling columns.
- The CLI flags it when |z| exceeds a Bonferroni threshold over all the tests of the run (5%
  family error, never below 3). For the default suite, 2 subsets × (55 sets + the pooled grid)
  × 10 heuristics = 1,120 tests and a threshold of 3.92.
- **Per grid cell and pooled:** every set is reported with its number of targets (fixed in a grid
  cell, a range in the other sets). With a suite, the grid cells are also pooled (sums of hits,
  chance and variance over all cells), and a closing table lists the two exclusion heuristics
  with accuracy, chance, ceiling and z, and the share of trimmed samples, one row per set.
- `ABOVE CHANCE` is a shortcut.
- `below chance` means the heuristic is anti-correlated with the answer. Its inverse could then
  be a shortcut, so check it (see §15).

**Results with 2,000 samples per configuration** (`fixed` mode):
- **The default evaluation suite** (`random` order, all 55 sets): no shortcut flag, above or
  below chance, on all samples or on the untrimmed subset. The largest |z| of a shortcut
  heuristic is 3.0 (`elimination` in `grid/d12_b4`). The backward-search heuristics behave as
  expected:
  - `depth_match` is at chance in every grid cell (all components have depth d) and at 0.65–0.67
    against 0.125 in the mixed sets (mixed depths);
  - `branching_root` is at chance where every root branches alike, and above it where distractors
    are trimmed or chains: 0.52 against 0.125 in `grid/d32_b8`, 0.24–0.27 in `d24_b8` and
    `d32_b4` and their B companions; it is at chance in every diagnostic cell at 512 edges.
- **`configs/train.json`** in `random` order: nothing flagged but `depth_match` (0.68 against
  0.12); the largest other |z| is 1.1. In `reverse` and `forward`, `proximity` is below chance and
  `not_nearest` above it (§15). `not_recent` is at chance in every order (|z| ≤ 0.7).
- **The grid and mixed sets in `reverse` or `forward` order:** `proximity` is above chance in the
  d = 2 cells with b ≤ 2 and below chance in `q4_in_distribution` (§15). `not_recent` is at
  chance everywhere (|z| ≤ 2.1).

### Statistics

`stats` gives the mean, maximum and minimum of the token count, edges, components, targets, nodes
and answer length, and the trimming shares. Use it to choose `total_edges` and the model's
context length. With `--batch-size B`, it makes the batches as `TrainStream` does and also
reports the mean padding fraction per batch: 1.1% for `configs/train.json` at B = 64 (30.6% if
each sample had its own edge count). Current values (`direct` format):

| set | tokens | edges | targets | trimmed samples |
|---|---|---|---|---:|
| `configs/train.json` (batches of 64) | 588–1,570, mean 1,087 | 192–512, mean 355 | 5–12, mean 8.5 | 0.017% |
| grid cells | 1,551 | 512 | 8 | 0 or 100% (§12) |
| companion pairs | 2,319–4,623 | 768–1,536 | 8 | A 0, B 100% |
| `diag/*` | 1,549–1,555 | 512 | 6, 9, 12 | 0, or 100% in `b8_t9` and `b8_t12` |
| `mixed/q4_in_distribution` | 798 | 256 | 8 | 0 |
| `mixed/q4_deep` | 1,566 | 512 | 8 | 0 |

## 15. Known properties of the data

Measured in `fixed` mode, 2,000 samples per set.

- **`depth_match` beats chance whenever depths are mixed within a sample:** 0.68 against 0.12 in
  `configs/train.json`, 0.65–0.67 against 0.125 in the mixed sets. Knowing the query's d narrows
  the candidates to the targets at that path length. It is inherent to mixing depths, and it is
  backward search (§5): a model can only use it by tracing every target back to its root. In the
  grid cells all components have depth d (trimming never changes a depth, and padding carries no
  target), so `depth_match` is at chance there.
- **`branching_root` beats chance where only some roots branch:** in the six always-trimmed grid
  cells (most at d·b ≥ 96; 0.52 against 0.125 at d32_b8, where 6 of the 7 distractors are chains),
  and their B companions. (With all distractors trimmed to chains, e.g. `diag/b8_t12` at 152
  edges, it reaches 1.0: the only branching root is the query's.)
  Also backward search: the shape of a component is only visible from its root. Where the
  distractors are untrimmed (`train.json`, the untrimmed grid cells, the A companions, the mixed
  sets), it is at chance.
- **Proximity in path orders (`reverse`, `forward`).** In `random` order, `proximity` and
  `not_nearest` are at chance everywhere. In the path orders they aren't, in two ways (accuracy
  against chance; the ceiling for the exclusion heuristic; the target position doesn't change
  these numbers):

  | set | targets | `proximity` | `not_nearest` (ceiling) | advantage of `not_nearest` |
  |---|---|---|---|---|
  | `configs/train.json`, reverse | 5–12 | 0.088 vs 0.127 (z −7.9) | 0.1335 vs 0.1271 (0.1478), z 7.6 | +0.006 |
  | `configs/train.json`, forward | 5–12 | 0.092 vs 0.127 (z −7.0) | 0.1328 vs 0.1271 (0.1478), z 6.7 | +0.006 |
  | `mixed/q4_in_distribution`, reverse | 8 | 0.085 vs 0.125 (z −10.2) | 0.1307 vs 0.1250 (0.1429), z 10.2 | +0.006 |
  | `mixed/q4_in_distribution`, forward | 8 | 0.089 vs 0.125 (z −9.0) | 0.1301 vs 0.1250 (0.1429), z 9.0 | +0.005 |
  | `mixed/q4_deep`, either | 8 | at chance | at chance (\|z\| ≤ 0.5) | none |
  | `grid/d2_b1`, reverse / forward | 8 | **0.193** vs 0.125 (z 9.2 / 9.1) | below chance | none: `proximity` itself is the cue |
  | `grid/d2_b2`, reverse / forward | 8 | **0.166 / 0.162** vs 0.125 (z 5.5 / 4.9) | below chance | none: `proximity` itself is the cue |
  | `grid/d2_b4`, `d2_b8` | 8 | 0.126–0.138 (\|z\| ≤ 1.8) | at chance | none |
  | every other grid cell, and the pooled grid | 8 | no flag (pooled 0.128, z ≤ 2.6) | no flag | none |

  - **Mixed shapes (the training config, `q4_in_distribution`):** the nearest target is the
    answer about 70% as often as chance. Excluding it gains about +0.006 in absolute accuracy,
    about 5% of chance and a third of the most that excluding one target can give (1/(T − 1) −
    1/T). With T ≈ 8 this is larger than it was with 25–30 targets in `per_component` mode.
  - **Short chains (`d2_b1`, `d2_b2`):** `proximity` itself is above chance; see the next item.
  - `not_recent` is at chance in every set and order (|z| ≤ 2.1).
- **A proximity shortcut at d = 2, in the path orders only.** In `reverse` and `forward` edge
  order, the target whose incoming edge is nearest to the queried source's edges is the answer
  more often than chance in the d = 2 grid cells with few branches:

  | cell (T = 8, chance 0.125) | `reverse` | `forward` |
  |---|---|---|
  | `grid/d2_b1` | 0.193 (z 9.2) | 0.193 (z 9.1) |
  | `grid/d2_b2` | 0.166 (z 5.5) | 0.162 (z 4.9) |
  | `grid/d2_b4` | 0.130 (z 0.6) | 0.138 (z 1.8) |
  | `grid/d2_b8` | 0.138 (z 1.8) | 0.126 (z 0.1) |

  The same with the target list before or after the edges. It is the one sizeable shortcut
  found, 1.5× chance at `d2_b1`. **It only affects the path orders, which neither
  `configs/train.json` nor the default suite uses** (both are `random`, where `proximity` is at
  chance everywhere); the generator is unchanged for now. A likely mechanism: within a branch the
  target's incoming edge comes first in `reverse` (last in `forward`), so target edges crowd one
  end of the list and source edges the other, and near a source edge there are fewer competing
  target edges than average. At larger d the edges of a branch spread apart, and the effect
  turns into the below-chance results of the mixed-shape sets. Check it before training with a
  path order.
- **Trimmed distractors are smaller than queried components,** by construction, in the cells
  that trim. Telling them apart takes a trace to the root (`branching_root` above). Padding and
  filler carry no target in `fixed` mode, so their shapes don't bias any target heuristic.
- **Grid lengths:** every grid cell has 512 edges, 8 targets and 1,551 tokens.
- **Training lengths:** `total_edges` is drawn once per batch, independently of the samples' d
  and b, over 192–512 edges, up to the evaluation grid's 512. So the model sees long sequences
  with easy queries as well as shorter ones, covering positions up to the grid's lengths. The
  minimum, 192, is the worst case of the per-sample draws (§11), so no sample ever needs more
  edges than its batch's count, and the count never depends on what the samples drew. Trimming
  is negligible at these counts (0.017% of samples, §4). The companion pairs (2,319–4,623 tokens)
  are longer than any training sample; the grid and the diagnostic (1,549–1,555 tokens) are
  inside the training range.

## 16. Tests

```bash
python tests/looped_lm/test_checker.py      # the checker on hand-written graphs (18 tests)
python tests/looped_lm/test_generator.py    # generator, datasets, storage, suites, audit (26 tests)
```

`tests/looped_lm/per_component_reference.json` holds fingerprints of `per_component` samples
written by the code before `fixed` mode existed (`per_component_reference.py write`); don't
rewrite it unless the old mode is meant to change.

They run standalone (no pytest needed) and also under pytest. They cover:

- **The checker:** valid samples in all formats. Each of these is rejected:
  - cycles;
  - merges;
  - two reachable targets;
  - a target without an incoming path, or not at a path end;
  - unequal branches, or a fork inside a branch;
  - wrong answers in every format;
  - malformed sequences;
  - wrong metadata, of each kind;
  - wrong edge order and wrong target position.
- **Canonical order:** in the checker, from hand-written edge orders; in the generator, against
  the edge positions, for every format and order.
- **Determinism:**
  - identical samples for identical arguments, and different ones for a different index, seed,
    split or stream;
  - the same stream and batches with 0, 1, 2 and 3 DataLoader workers;
  - ranks partition the stream, and with batches each rank takes its own batches of the same
    global sequence;
  - map-style access is independent of the order.
- **The loss mask:** exactly the answers plus their end tokens; never the edges, the targets or
  the prompts; roles and `pred_pos` consistent with it.
- **Edge counts, targets per component, query order independent of the graph, pool checks**
  (including the refusal of a small pool in training).
- **Storage:** a written set is identical to the generated samples and passes the checker;
  collate pads correctly.
- **One edge count per batch** (these tests use `per_component` configs; the `fixed`-mode
  versions are listed below):
  - every sample of a batch has the batch's edge count, for any number of workers, and without
    `batch_size` every sample is a batch of one;
  - the rest of a sample is its own draw (same queries as the map-style sample of that index);
  - over 3,000 batches of 4, the whole range of counts occurs, and the count is uncorrelated
    with each sample's mean d, mean b, total d·b of its queries and number of queries
    (|r| < 4/√12,000 ≈ 0.037), while it is correlated with the sample's length (r > 0.5), which
    shows the test can see a dependence;
  - the configuration check accepts a minimum that fits the worst case exactly, rejects one edge
    less, one distractor more, and `total_edges: null`, and accepts `configs/train.json`.
- **The audit:** at chance on a clean configuration (all 10 heuristics), and it flags a planted
  shortcut (the answer moved to the front of the target list).
- **The old mode reproduces previous samples exactly:** tokens, per-token metadata, queries and
  counts of `per_component` samples in all formats, orders and target positions, map-style and
  in batches, against the stored fingerprints.
- **`fixed` mode:**
  - the checker accepts hand-written graphs with a trimmed distractor (a single chain) and
    target-less components (with b > 1, and a chain), and rejects: the wrong number of targets
    (both ways), unequal depths in a distractor and in a target-less component, branching below
    a root, and a merge;
  - in every format, order and target position, every sample passes the checker with exactly
    T targets and E edges; T, the queried shapes and the drawn distractor shapes are identical to
    those of the same sample with a budget that never trims; trimming leaves every distractor at
    ≥ 1 branch, fits the budget, and stops at the first fit (the last branch removed would not
    fit back); between 20% and 100% of the samples of a tight config are trimmed;
  - validation: T < `num_queries` + 1, `num_distractors` set, `num_targets` missing, and
    `num_targets` in `per_component` mode are refused; T below the maximum branching warns;
  - the batch check accepts the new worst case exactly (3 × 24 + 4 chains × 6 = 96), refuses one
    edge less with the expected explanation, and `configs/train.json` has 4 × 32 + 8 × 8 = 192;
  - in batches of a config that trims 5–95% of samples, the edge count is uncorrelated with each
    sample's queried d, b and d·b (|r| < 4/√12,000), and a sample's queried shapes are the same
    whatever the edge count;
  - the trimming metadata is checked (a wrong `branches_removed`, `trimmed`, branching or depth
    is rejected), stored and read back, and collated;
  - `branching_root` on a hand-built sample where only the answer's root branches;
  - the suite: 32 grid cells at T = 8 and 512 edges; the trimming cells are exactly the six with
    d·b > 64; companion pairs for five of them (d32_b8 skipped with a warning); 11 diagnostic
    cells at 512 edges (152 with `total_edges: null`), without b8_t6; paired companion samples with the same queries, edge
    count, T and query-edge positions, A untrimmed and B trimmed to 512; `depth_match` at chance
    (1/8) in an untrimmed and a trimmed grid cell.
- **The exclusion heuristics on a hand-built sample** with 3 targets: when the nearest and most
  recent target is the answer, both score 0; when they're another target, both score 1/2. And
  over the T possible excluded targets, the mean is exactly 1/T.
