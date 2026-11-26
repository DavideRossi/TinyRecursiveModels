"""Task configuration: the knobs of one data stream or evaluation set, saved as JSON."""
import dataclasses
import itertools
import json
import warnings
from typing import Any, List, Optional

import numpy as np

from .vocab import NODE_POOL

EDGE_ORDERS = ("random", "reverse", "forward")
TARGET_POSITIONS = ("after", "before")
ANSWER_FORMATS = ("direct", "path", "frontier", "dfs", "copy")
TARGETS_MODES = ("per_component", "fixed")


class Dist:
    """A distribution over positive integers.

    Spec forms (JSON): an int (constant), a list of ints (uniform over them),
    {"range": [lo, hi]} (uniform over lo..hi inclusive), or {"values": [...], "probs": [...]}.
    """

    def __init__(self, values: List[int], probs: Optional[List[float]] = None, spec: Any = None):
        if not values:
            raise ValueError("a distribution needs at least one value")
        self.values = np.asarray([int(v) for v in values], dtype=np.int64)
        if probs is None:
            self.probs = None
        else:
            p = np.asarray(probs, dtype=np.float64)
            if p.shape != self.values.shape or (p < 0).any() or p.sum() <= 0:
                raise ValueError(f"invalid probabilities {probs} for values {values}")
            self.probs = p / p.sum()
        self.spec = spec if spec is not None else self._default_spec()

    def _default_spec(self):
        if self.probs is None:
            return [int(v) for v in self.values]
        return {"values": [int(v) for v in self.values], "probs": [float(p) for p in self.probs]}

    @classmethod
    def from_spec(cls, spec: Any) -> "Dist":
        if isinstance(spec, Dist):
            return spec
        if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
            return cls([int(spec)], spec=int(spec))
        if isinstance(spec, (list, tuple)):
            return cls(list(spec), spec=list(spec))
        if isinstance(spec, dict) and "range" in spec:
            lo, hi = spec["range"]
            if lo > hi:
                raise ValueError(f"empty range {spec}")
            return cls(list(range(int(lo), int(hi) + 1)), spec={"range": [int(lo), int(hi)]})
        if isinstance(spec, dict) and "values" in spec:
            return cls(spec["values"], spec.get("probs"), spec=spec)
        raise ValueError(f"cannot read a distribution from {spec!r}")

    def sample(self, rng: np.random.Generator) -> int:
        if len(self.values) == 1:
            return int(self.values[0])
        return int(rng.choice(self.values, p=self.probs))

    @property
    def support(self) -> List[int]:
        if self.probs is None:
            return sorted(set(int(v) for v in self.values))
        return sorted(set(int(v) for v, p in zip(self.values, self.probs) if p > 0))

    @property
    def min(self) -> int:
        return min(self.support)

    @property
    def max(self) -> int:
        return max(self.support)

    def __repr__(self):
        return f"Dist({self.spec})"


def _dist_or_none(x):
    return None if x is None else Dist.from_spec(x)


@dataclasses.dataclass
class TaskConfig:
    """Knobs of the task. Distributions are sampled per sample (num_queries, num_targets,
    num_distractors, total_edges) or per component (depth, branching; distractor_* for the
    distractor components).

    targets_mode: how many targets a sample has (README, "Component shapes"):
        per_component: num_distractors distractors, and every padding component carries a target
                       too, so the number of targets grows with the edge count (the original mode);
        fixed: num_targets targets per sample (num_queries queried components and the rest
               distractors); distractors are trimmed (dead-end branches removed) when the edge
               count is short, and padding components carry no target.
    depth: edges from a source to the end of each of its paths (a path has depth - 1 intermediate nodes)
    branching: paths leaving a source (one ends at a target, the others at dead ends)
    num_queries: queried sources per sample
    num_targets: (fixed) targets per sample, >= num_queries + 1
    num_distractors: (per_component) unqueried components with a target, before padding
    distractor_depth, distractor_branching: their shape distributions (None: same as the queries)
    total_edges: None, or the edge count to reach with padding components
                 (the larger of this and the edges the sample needs anyway)
    trim_edges: (fixed, optional) the edge budget the distractors are trimmed to, when it should be
                smaller than total_edges (the rest is padding); None: total_edges
    edge_order: random | reverse (each edge after its successor) | forward (each edge before it)
    targets_position: after | before the edges
    answer_format: direct | path | frontier | dfs | copy (a pipeline check: the answer is the
                   queried name itself, the token right after "Q:")
    pool_size: node names are drawn from the first pool_size names of the vocabulary
    """
    name: str = "default"
    depth: Any = 4
    branching: Any = 2
    num_queries: Any = 1
    num_distractors: Any = None  # per_component: None means 1
    targets_mode: str = "per_component"
    num_targets: Any = None
    distractor_depth: Any = None
    distractor_branching: Any = None
    total_edges: Any = None
    trim_edges: Optional[int] = None
    edge_order: str = "random"
    targets_position: str = "after"
    answer_format: str = "direct"
    pool_size: int = NODE_POOL

    def __post_init__(self):
        self.depth = Dist.from_spec(self.depth)
        self.branching = Dist.from_spec(self.branching)
        self.num_queries = Dist.from_spec(self.num_queries)
        if self.targets_mode == "per_component" and self.num_distractors is None:
            self.num_distractors = 1
        self.num_distractors = _dist_or_none(self.num_distractors)
        self.num_targets = _dist_or_none(self.num_targets)
        self.distractor_depth = _dist_or_none(self.distractor_depth)
        self.distractor_branching = _dist_or_none(self.distractor_branching)
        self.total_edges = _dist_or_none(self.total_edges)
        self.validate()

    # ---------------------------------------------------------------- derived quantities
    @property
    def d_depth(self) -> Dist:
        return self.distractor_depth if self.distractor_depth is not None else self.depth

    @property
    def d_branching(self) -> Dist:
        return self.distractor_branching if self.distractor_branching is not None else self.branching

    @property
    def fixed(self) -> bool:
        return self.targets_mode == "fixed"

    def _query_distractor_counts(self):
        """(queries, distractors) pairs the sample can draw."""
        if self.fixed:
            return [(q, t - q) for q, t in itertools.product(self.num_queries.support, self.num_targets.support)]
        return [(q, k) for q, k in itertools.product(self.num_queries.support, self.num_distractors.support)]

    def max_required_edges(self) -> int:
        """The most edges a sample can need, before padding, in the worst case.
        per_component: every queried and distractor component at maximum size.
        fixed: queried components at maximum size, distractors trimmed to single chains of maximum
        depth (trimming takes care of everything above that)."""
        q_size = self.depth.max * self.branching.max
        d_size = self.d_depth.max if self.fixed else self.d_depth.max * self.d_branching.max
        return max(q * q_size + k * d_size for q, k in self._query_distractor_counts())

    def max_untrimmed_edges(self) -> int:
        """The most edges of the queried and distractor components at their drawn shapes."""
        return max(q * self.depth.max * self.branching.max + k * self.d_depth.max * self.d_branching.max
                   for q, k in self._query_distractor_counts())

    def worst_case_text(self) -> str:
        q, k = max(self._query_distractor_counts(),
                   key=lambda qk: qk[0] * self.depth.max * self.branching.max
                   + qk[1] * (self.d_depth.max if self.fixed else self.d_depth.max * self.d_branching.max))
        if self.fixed:
            return (f"{q} queries x {self.depth.max * self.branching.max} edges at maximum depth and branching "
                    f"+ {k} distractors as single chains of depth {self.d_depth.max}")
        return (f"{q} queries x {self.depth.max * self.branching.max} + {k} distractors x "
                f"{self.d_depth.max * self.d_branching.max}, at maximum depth and branching")

    def check_batch_edges(self) -> None:
        """Required for one edge count per batch (TrainStream): every edge count in total_edges must fit
        the worst case of the per-sample draws, so the count never has to depend on them."""
        if self.total_edges is None:
            raise ValueError(f"config {self.name!r}: the training stream needs total_edges (one edge count "
                             f"per batch); without it, a sample's length would follow its depths and branchings")
        need = self.max_required_edges()
        if self.total_edges.min < need:
            counts = "num_targets" if self.fixed else "num_distractors"
            raise ValueError(
                f"config {self.name!r}: total_edges starts at {self.total_edges.min}, but a sample can need "
                f"{need} edges ({self.worst_case_text()}). Raise the minimum of total_edges to {need} or more, "
                f"or lower num_queries, {counts}, depth or branching.")

    def max_edges(self) -> int:
        if self.total_edges is None:
            return self.max_untrimmed_edges() if self.fixed else self.max_required_edges()
        return max(self.total_edges.max, self.max_required_edges())

    def max_nodes(self) -> int:
        """Upper bound on the nodes of a sample: edges + components (one source each)."""
        components = self.num_targets.max if self.fixed else self.num_queries.max + self.num_distractors.max
        if self.total_edges is not None:
            min_size = self.d_depth.min * self.d_branching.min
            components += self.total_edges.max // min_size + 1
        return self.max_edges() + components

    def validate(self) -> None:
        if self.edge_order not in EDGE_ORDERS:
            raise ValueError(f"edge_order must be one of {EDGE_ORDERS}, got {self.edge_order!r}")
        if self.targets_position not in TARGET_POSITIONS:
            raise ValueError(f"targets_position must be one of {TARGET_POSITIONS}, got {self.targets_position!r}")
        if self.answer_format not in ANSWER_FORMATS:
            raise ValueError(f"answer_format must be one of {ANSWER_FORMATS}, got {self.answer_format!r}")
        if self.targets_mode not in TARGETS_MODES:
            raise ValueError(f"targets_mode must be one of {TARGETS_MODES}, got {self.targets_mode!r}")
        for name in ("depth", "branching", "num_queries"):
            if getattr(self, name).min < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.d_depth.min < 1 or self.d_branching.min < 1:
            raise ValueError("distractor depth and branching must be >= 1")
        if self.fixed:
            if self.num_targets is None:
                raise ValueError(f"config {self.name!r}: targets_mode 'fixed' needs num_targets")
            if self.num_distractors is not None:
                raise ValueError(f"config {self.name!r}: num_distractors is not used with targets_mode 'fixed' "
                                 f"(the distractors are num_targets - num_queries)")
            if self.num_targets.min < self.num_queries.max + 1:
                raise ValueError(f"config {self.name!r}: num_targets must be >= num_queries + 1 in every sample "
                                 f"(num_targets from {self.num_targets.min}, num_queries up to {self.num_queries.max})")
            if self.trim_edges is not None:
                if self.total_edges is None or self.trim_edges > self.total_edges.min:
                    raise ValueError("trim_edges needs total_edges, and must not exceed its minimum")
                if self.trim_edges < self.max_required_edges():
                    raise ValueError(f"trim_edges {self.trim_edges} is below what a sample can need "
                                     f"({self.max_required_edges()}: {self.worst_case_text()})")
            b_max = max(self.branching.max, self.d_branching.max)
            if self.num_targets.min < b_max:
                warnings.warn(f"config {self.name!r}: num_targets can be {self.num_targets.min}, fewer than the "
                              f"maximum branching {b_max}: backward search from the targets is then cheaper "
                              f"than forward search (README, \"Backward search\")", stacklevel=3)
        else:
            if self.num_targets is not None or self.trim_edges is not None:
                raise ValueError(f"config {self.name!r}: num_targets and trim_edges need targets_mode 'fixed'")
            if self.num_distractors.min < 0:
                raise ValueError("num_distractors must be >= 0")
            if self.num_queries.min + self.num_distractors.min < 2:
                raise ValueError("every sample needs at least 2 targets: num_queries + num_distractors >= 2")
        if self.total_edges is not None and self.total_edges.min < 1:
            raise ValueError("total_edges must be >= 1")
        if not 1 <= self.pool_size <= NODE_POOL:
            raise ValueError(f"pool_size must be in 1..{NODE_POOL} (the vocabulary), got {self.pool_size}")
        if self.max_nodes() > self.pool_size:
            raise ValueError(f"config {self.name!r}: up to {self.max_nodes()} nodes per sample, "
                             f"more than the pool of {self.pool_size} names")

    # ---------------------------------------------------------------- (de)serialization
    def to_dict(self) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.spec if isinstance(v, Dist) else v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "TaskConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def replace(self, **changes) -> "TaskConfig":
        d = self.to_dict()
        d.update(changes)
        return TaskConfig.from_dict(d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def load(cls, path: str) -> "TaskConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
