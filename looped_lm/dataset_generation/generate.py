"""Sample generation: a pure function of (config, split, seed, stream, index).

A sample is a set of disjoint components. A component is a source with b non-branching paths
("branches") of d edges each; at most one branch ends at a target, the others end at dead ends.
Queried components and distractor components are drawn from the same shape distributions and
have exactly one target each. Extra components pad the sample to the configured edge count; the
last one ("filler") takes the exact remainder and has no target.

targets_mode "per_component": num_distractors distractors, and the padding components carry a
target each. targets_mode "fixed": num_targets targets (queries + distractors); if the components
exceed the edge budget, distractors lose dead-end branches ("trimming"); padding carries no target.
See README.md.
"""
import dataclasses
from typing import List, Optional

import numpy as np

from .config import TaskConfig
from .formats import ROLE_END, ROLE_FINAL, ROLE_OTHER, answer_tokens
from .vocab import ARROW, ASK, BOS, EOS, QUERY, SEP, TARGETS, VOCAB, Vocab

# Seed namespaces: the same seed number gives unrelated samples in different splits
SPLITS = {"train": 0, "eval": 1, "debug": 2}
PADDING_ATTEMPTS = 16  # draws of a padding component that doesn't fit before using the filler


@dataclasses.dataclass
class Component:
    kind: str                 # query | distractor | padding | filler
    depth: int
    branching: int
    target_branch: Optional[int]  # index into branches of the branch ending at the target; None: no target
    drawn_branching: int = 0  # the branching before trimming (fixed mode); 0: same as branching
    source: int = -1          # token ids, assigned after the shape is fixed
    branches: List[List[int]] = dataclasses.field(default_factory=list)

    @property
    def num_edges(self) -> int:
        return self.depth * self.branching

    @property
    def num_nodes(self) -> int:
        return 1 + self.depth * self.branching


@dataclasses.dataclass
class Sample:
    tokens: np.ndarray        # int32 [L]
    loss_mask: np.ndarray     # bool [L]: answer tokens and the end token after each answer
    query_index: np.ndarray   # int16 [L]: query of the token (Q: x ? answer end), -1 elsewhere
    depth: np.ndarray         # int16 [L]: d of the token's query, 0 elsewhere
    branching: np.ndarray     # int16 [L]: b of the token's query, 0 elsewhere
    role: np.ndarray          # int8 [L]: formats.ROLE_*
    pred_pos: np.ndarray      # int32 [L]: for final answers, the position predicting them (pos - 1); -1 elsewhere
    queries: List[dict]       # per query: source, target, path, depth, branching, branch_order, positions
    num_edges: int
    num_components: int
    num_targets: int
    key: tuple                # (split, seed, stream, index)
    trimmed: bool = False     # fixed mode: some distractor lost branches to fit the edge budget
    branches_removed: int = 0
    distractors: List[dict] = dataclasses.field(default_factory=list)
    # per unqueried component with a target, in generation order: depth, drawn_branching, branching

    def __len__(self):
        return len(self.tokens)

    def text(self, vocab: Vocab = VOCAB, pretty: bool = True) -> str:
        return vocab.to_text(self.tokens, pretty=pretty)


def sample_rng(split: str, seed: int, stream: int, index: int) -> np.random.Generator:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    if min(seed, stream, index) < 0:
        raise ValueError("seed, stream and index must be non-negative")
    return np.random.default_rng(np.random.SeedSequence([SPLITS[split], int(seed), int(stream), int(index)]))


# Namespace of the per-batch generators (first word of their seed; the samples' is the split's 0-2)
BATCH_NAMESPACE = 1000


def batch_rng(split: str, seed: int, stream: int, batch: int) -> np.random.Generator:
    """Generator of batch number `batch` of (split, seed, stream): independent of every sample's."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    if min(seed, stream, batch) < 0:
        raise ValueError("seed, stream and batch must be non-negative")
    return np.random.default_rng(np.random.SeedSequence(
        [BATCH_NAMESPACE + SPLITS[split], int(seed), int(stream), int(batch)]))


def batch_edges(config: TaskConfig, seed: int, batch: int, split: str = "train", stream: int = 0) -> int:
    """Edge count shared by the samples of batch `batch`, drawn from total_edges with the batch's own
    generator: it doesn't depend on anything the samples draw (depths, branchings, ...)."""
    config.check_batch_edges()
    return config.total_edges.sample(batch_rng(split, seed, stream, batch))


def make_batch(config: TaskConfig, seed: int, batch: int, batch_size: int, split: str = "train",
               stream: int = 0, vocab: Vocab = VOCAB) -> List["Sample"]:
    """Batch number `batch`: samples batch*B ... batch*B + B - 1, all with the batch's edge count."""
    edges = batch_edges(config, seed, batch, split, stream)
    return [make_sample(config, seed, k, split, stream, vocab, total_edges=edges)
            for k in range(batch * batch_size, (batch + 1) * batch_size)]


def _filler_shape(r: int, config: TaskConfig, rng: np.random.Generator):
    """(depth, branching) with depth * branching == r, as close to the distractor shapes as possible."""
    shapes = [(r // b, b) for b in range(1, r + 1) if r % b == 0]
    depths, branchings = set(config.d_depth.support), set(config.d_branching.support)
    for keep in (lambda d, b: d in depths and b in branchings,
                 lambda d, b: b in branchings and config.d_depth.min <= d <= config.d_depth.max,
                 lambda d, b: b in branchings,
                 lambda d, b: config.d_depth.min <= d <= config.d_depth.max,
                 lambda d, b: True):
        good = [s for s in shapes if keep(*s)]
        if good:
            return good[int(rng.integers(len(good)))]


def _components(config: TaskConfig, rng: np.random.Generator, total_edges: Optional[int] = None) -> List[Component]:
    comps = []
    for _ in range(config.num_queries.sample(rng)):
        d, b = config.depth.sample(rng), config.branching.sample(rng)
        comps.append(Component("query", d, b, int(rng.integers(b))))
    for _ in range(config.num_distractors.sample(rng)):
        d, b = config.d_depth.sample(rng), config.d_branching.sample(rng)
        comps.append(Component("distractor", d, b, int(rng.integers(b))))
    edges = sum(c.num_edges for c in comps)
    if config.total_edges is not None:
        target = config.total_edges.sample(rng)  # drawn even when overridden: the later draws stay the same
        if total_edges is not None:
            assert edges <= total_edges, (f"{edges} edges needed, more than the batch's {total_edges}; "
                                          f"TaskConfig.check_batch_edges should have refused this config")
            target = total_edges
        remaining = max(target, edges) - edges
        failures = 0
        while remaining > 0:
            d, b = config.d_depth.sample(rng), config.d_branching.sample(rng)
            if d * b <= remaining:
                comps.append(Component("padding", d, b, int(rng.integers(b))))
                remaining -= d * b
                failures = 0
                continue
            failures += 1
            if failures >= PADDING_ATTEMPTS:
                d, b = _filler_shape(remaining, config, rng)
                comps.append(Component("filler", d, b, None))
                remaining = 0
    return comps


def _pad(comps: List[Component], remaining: int, config: TaskConfig, rng: np.random.Generator) -> None:
    """Fixed mode: target-less padding components of distractor shapes while they fit, then the filler."""
    failures = 0
    while remaining > 0:
        d, b = config.d_depth.sample(rng), config.d_branching.sample(rng)
        if d * b <= remaining:
            comps.append(Component("padding", d, b, None))
            remaining -= d * b
            failures = 0
            continue
        failures += 1
        if failures >= PADDING_ATTEMPTS:
            d, b = _filler_shape(remaining, config, rng)
            comps.append(Component("filler", d, b, None))
            remaining = 0


def _components_fixed(config: TaskConfig, rngs: dict, total_edges: Optional[int]) -> List[Component]:
    """Fixed mode: num_queries queried components and num_targets - num_queries distractors, all
    shapes drawn first; then the distractors are trimmed to the budget (trim_edges, else the edge
    count), never the queried components and never a depth; then target-less padding up to the
    edge count. Each step has its own generator, so two configs that differ only in a later step
    (the companion pairs) share everything drawn before it."""
    rng = rngs["shapes"]
    comps = []
    for _ in range(config.num_queries.sample(rng)):
        d, b = config.depth.sample(rng), config.branching.sample(rng)
        comps.append(Component("query", d, b, int(rng.integers(b)), drawn_branching=b))
    num_targets = config.num_targets.sample(rng)
    distractors = []
    for _ in range(num_targets - len(comps)):
        d, b = config.d_depth.sample(rng), config.d_branching.sample(rng)
        distractors.append(Component("distractor", d, b, None, drawn_branching=b))
    target = config.total_edges.sample(rng) if config.total_edges is not None else None
    if total_edges is not None:
        target = total_edges
    query_edges = sum(c.num_edges for c in comps)
    if target is not None:
        floor = query_edges + sum(c.depth for c in distractors)  # distractors as single chains
        if total_edges is not None:
            assert floor <= total_edges, (f"{floor} edges needed, more than the batch's {total_edges}; "
                                          f"TaskConfig.check_batch_edges should have refused this config")
        target = max(target, floor)
        budget = max(min(config.trim_edges, target), floor) if config.trim_edges is not None else target
        trim = rngs["trim"]
        size = query_edges + sum(c.num_edges for c in distractors)
        while size > budget:  # remove one dead-end branch of a random distractor with b > 1
            cands = [c for c in distractors if c.branching > 1]
            c = cands[int(trim.integers(len(cands)))]
            c.branching -= 1
            size -= c.depth
    for c in distractors:  # the target's branch, among the branches that remain
        c.target_branch = int(rngs["trim"].integers(c.branching))
    comps += distractors
    if target is not None:
        _pad(comps, target - sum(c.num_edges for c in comps), config, rngs["pad"])
    return comps


def make_sample(config: TaskConfig, seed: int, index: int, split: str = "train", stream: int = 0,
                vocab: Vocab = VOCAB, total_edges: Optional[int] = None) -> Sample:
    """Sample number `index` of the stream (split, seed, stream): a pure function of its arguments.
    total_edges: the edge count to use instead of the sample's own draw (the batch's, see make_batch)."""
    if config.fixed:
        root = np.random.SeedSequence([SPLITS[split], int(seed), int(stream), int(index)])
        sample_rng(split, seed, stream, index)  # argument checks
        rngs = dict(zip(("shapes", "trim", "pad", "names", "order", "queries"),
                        (np.random.default_rng(c) for c in root.spawn(6))))
        comps = _components_fixed(config, rngs, total_edges)
        rng = rngs["order"]
        names_rng, lists_rng = rngs["names"], rngs["queries"]
    else:
        rng = sample_rng(split, seed, stream, index)
        comps = _components(config, rng, total_edges)
        names_rng = lists_rng = rng

    # Node names, drawn without replacement from the pool
    num_nodes = sum(c.num_nodes for c in comps)
    assert num_nodes <= config.pool_size, f"{num_nodes} nodes > pool of {config.pool_size}"
    names = [vocab.node(int(i)) for i in names_rng.permutation(config.pool_size)[:num_nodes]]
    it = iter(names)
    for c in comps:
        c.source = next(it)
        c.branches = [[next(it) for _ in range(c.depth)] for _ in range(c.branching)]

    # Edges: one sequence per branch, ordered as configured
    branch_edges = []
    for c in comps:
        for br in c.branches:
            seq = list(zip([c.source] + br[:-1], br))
            branch_edges.append(seq[::-1] if config.edge_order == "reverse" else seq)
    if config.edge_order == "random":
        flat = [e for seq in branch_edges for e in seq]
        edges = [flat[i] for i in rng.permutation(len(flat))]
    else:
        # Random interleaving that keeps each branch's own order
        labels = np.repeat(np.arange(len(branch_edges)), [len(s) for s in branch_edges])
        rng.shuffle(labels)
        cursor = [0] * len(branch_edges)
        edges = []
        for lab in labels:
            edges.append(branch_edges[lab][cursor[lab]])
            cursor[lab] += 1
    edge_pos = {e: i for i, e in enumerate(edges)}

    # Canonical branch order: by the position of each branch's first edge
    for c in comps:
        order = sorted(range(c.branching), key=lambda k: edge_pos[(c.source, c.branches[k][0])])
        if c.target_branch is not None:
            c.target_branch = order.index(c.target_branch)
        c.branches = [c.branches[k] for k in order]

    targets = [c.branches[c.target_branch][-1] for c in comps if c.target_branch is not None]
    targets = [targets[i] for i in lists_rng.permutation(len(targets))]
    queried = [c for c in comps if c.kind == "query"]
    queried = [queried[i] for i in lists_rng.permutation(len(queried))]  # query order independent of the graph

    # Tokens
    toks, roles, qidx, qd, qb, pred, mask = [], [], [], [], [], [], []

    def emit(ids, role=ROLE_OTHER, q=-1, d=0, b=0, m=False, role_list=None):
        for k, t in enumerate(ids):
            r = role_list[k] if role_list is not None else role
            pred.append(len(toks) - 1 if r == ROLE_FINAL else -1)
            toks.append(t)
            roles.append(r)
            qidx.append(q)
            qd.append(d)
            qb.append(b)
            mask.append(m)

    emit([BOS])
    target_block = [TARGETS] + targets
    if config.targets_position == "before":
        emit(target_block)
    for s, t in edges:
        emit([s, ARROW, t])
    if config.targets_position == "after":
        emit(target_block)
    queries = []
    for qi, c in enumerate(queried):
        prompt_pos = len(toks)
        emit([QUERY, c.source, ASK], q=qi, d=c.depth, b=c.branching)
        ids, rls = answer_tokens(config.answer_format, c.branches, c.target_branch, c.source)
        answer_start = len(toks)
        emit(ids, q=qi, d=c.depth, b=c.branching, m=True, role_list=rls)
        answer_end = len(toks)
        emit([SEP if qi < len(queried) - 1 else EOS], role=ROLE_END, q=qi, d=c.depth, b=c.branching, m=True)
        path = c.branches[c.target_branch]
        queries.append({"source": c.source, "target": path[-1], "path": list(path), "depth": c.depth,
                        "branching": c.branching, "branch_order": [br[0] for br in c.branches],
                        "prompt_pos": prompt_pos, "answer_start": answer_start, "answer_end": answer_end})

    distractors = [{"depth": c.depth, "drawn_branching": c.drawn_branching or c.branching, "branching": c.branching}
                   for c in comps if c.kind != "query" and c.target_branch is not None]
    removed = sum(x["drawn_branching"] - x["branching"] for x in distractors)
    return Sample(
        tokens=np.asarray(toks, dtype=np.int32), loss_mask=np.asarray(mask, dtype=bool),
        query_index=np.asarray(qidx, dtype=np.int16), depth=np.asarray(qd, dtype=np.int16),
        branching=np.asarray(qb, dtype=np.int16), role=np.asarray(roles, dtype=np.int8),
        pred_pos=np.asarray(pred, dtype=np.int32), queries=queries, num_edges=len(edges),
        num_components=len(comps), num_targets=len(targets), key=(split, seed, stream, index),
        trimmed=removed > 0, branches_removed=removed, distractors=distractors)
