"""Shortcut audit and length statistics.

The heuristics guess a query's answer without following the graph from the queried source. They
see the parsed tokens (edges, target list, query order) and, for depth_match, the query's d.
Each picks one target; ties are broken at random. Every heuristic is also reported on the
queries of untrimmed samples only (fixed mode trims distractors when the edge count is short).
depth_match and branching_root trace every target back to its root: they cost as much as a
backward search, so they are reported but not counted as shortcuts. The audit compares their accuracy with their
chance level and flags a significant excess.

The exclusion heuristics (not_nearest, not_recent) pick uniformly among all targets except the one
proximity (resp. recent) picks. Their accuracy is computed exactly rather than sampled: per query,
1 / (T - 1) if the excluded target isn't the answer, else 0 (T = #targets). Chance is 1 / T, what
they score when the excluded target is the answer as often as any other; 1 / (T - 1) is their
ceiling, reached when it never is.
"""
import math
from typing import Callable, Dict, List, Optional

import numpy as np

from .checker import Graph, parse
from .config import TaskConfig
from .generate import make_batch, make_sample
from .vocab import VOCAB

Z_FLAG = 3.0  # |z| above this: flagged (at least; see z_threshold)
HEURISTICS = ("recent", "first_listed", "in_degree", "depth_match", "branching_root", "proximity", "random",
              "elimination", "not_nearest", "not_recent")
# Reported but not counted as shortcuts: they cost a backward trace from every target to its root,
# which is graph following itself (README, "Backward search")
BACKWARD = {"depth_match", "branching_root"}
EXCLUSION = {"not_nearest": "proximity", "not_recent": "recent"}  # heuristic -> the one whose pick it excludes


def z_threshold(num_tests: int, family_error: float = 0.05) -> float:
    """Bonferroni: the z above which one of num_tests null heuristics is flagged with probability
    family_error at most (one-sided), and never below Z_FLAG."""
    from statistics import NormalDist
    return max(Z_FLAG, NormalDist().inv_cdf(1 - family_error / max(1, num_tests)))
# Heuristics counted on queries of depth >= 2 only: at d = 1 the answer edge starts at the source,
# so "the target nearest the source's edges" is a one-hop lookup, the base case of the task.
DEPTH2_ONLY = {"proximity", "not_nearest"}


def _pick(cands: List[int], rng: np.random.Generator) -> int:
    return cands[int(rng.integers(len(cands)))]


def _argbest(targets: List[int], score: Callable[[int], float], rng) -> int:
    best = max(score(t) for t in targets)
    return _pick([t for t in targets if score(t) == best], rng)


def heuristic_guesses(tokens, depths: List[int], rng: np.random.Generator) -> Dict[str, List[int]]:
    """For each heuristic except elimination, its guess for every query of the sample (in query order)."""
    p = parse(tokens)
    g = Graph(p.edges)
    targets = p.targets
    # Last mention of each target in the graph description (edges and target list)
    last = {t: -1 for t in targets}
    for k, (u, v) in enumerate(p.edges):
        if v in last:
            last[v] = max(last[v], p.edge_dest_pos[k])
    for k, t in enumerate(targets):
        last[t] = max(last[t], p.targets_pos + 1 + k)
    # Length of each target's path and its root, walking back to its source (backward, not from the query)
    length, root = {}, {}
    for t in targets:
        n, u = 0, t
        while g.indeg.get(u, 0) == 1:
            u = g.parent[u]
            n += 1
        length[t], root[t] = n, u
    branching_roots = [t for t in targets if len(g.out[root[t]]) > 1]
    in_edge = {t: g.edge_index[(g.parent[t], t)] for t in targets}
    out = {k: [] for k in ("recent", "first_listed", "in_degree", "depth_match", "branching_root", "proximity",
                           "random")}
    for qi, q in enumerate(p.queries):
        s = q["source"]
        out["recent"].append(_argbest(targets, lambda t: last[t], rng))
        out["first_listed"].append(targets[0])
        out["in_degree"].append(_argbest(targets, lambda t: g.indeg[t], rng))
        matching = [t for t in targets if length[t] == depths[qi]]
        out["depth_match"].append(_pick(matching or targets, rng))
        out["branching_root"].append(_pick(branching_roots or targets, rng))
        src_edges = [g.edge_index[(s, v)] for v in g.out[s]]
        out["proximity"].append(_argbest(targets, lambda t: -min(abs(in_edge[t] - e) for e in src_edges), rng))
        out["random"].append(_pick(targets, rng))
    return out


def exclusion_hits(excluded: List[int], truth: List[int], num_targets: int) -> List[float]:
    """Exact expected hit of "uniform among the targets except `excluded`", per query."""
    return [0.0 if e == t else 1.0 / (num_targets - 1) for e, t in zip(excluded, truth)]


def summarize(sums: dict, z_flag: float = Z_FLAG) -> dict:
    """Accuracy, chance, ceiling and z of one heuristic from its sums (see audit's "sums")."""
    n = sums["queries"]
    if n == 0:
        return {"accuracy": float("nan"), "chance": float("nan"), "ceiling": float("nan"), "z": 0.0,
                "queries": 0, "flag": "no queries"}
    sd = math.sqrt(sums["var"]) or 1.0
    z = (sums["hits"] - sums["chance"]) / sd
    return {"accuracy": sums["hits"] / n, "chance": sums["chance"] / n, "ceiling": sums["ceiling"] / n,
            "z": z, "queries": n, "flag": "ABOVE CHANCE" if z > z_flag else ("below chance" if z < -z_flag else "")}


def audit(config: TaskConfig, num_samples: int = 2000, seed: int = 0, stream: int = 0,
          z_flag: float = Z_FLAG, return_sums: bool = False):
    """Accuracy of each heuristic vs chance (1 / #targets per query), over samples of the debug split.
    A heuristic is flagged when its z-score exceeds z_flag.

    z = (hits - sum of chance) / sd under chance. For a heuristic that picks one target, a query's hit
    is Bernoulli(p) with p = 1/T (1/(T - qi) for elimination); for an exclusion heuristic it is the
    exact expectation, 1/(T-1) with probability (T-1)/T and 0 otherwise, of variance 1/(T^2 (T-1)).
    Every result also has "untrimmed": the same on the queries of untrimmed samples only.
    With return_sums, also returns the sums (hits, chance, ceiling, var, queries) per heuristic and
    subset ("all", "untrimmed"), to pool sets, and info: the number of targets (min, mean, max) and
    the share of trimmed samples."""
    rng = np.random.default_rng([seed, stream, 12345])
    hits = {}
    num_targets, elim_chance, depth, untrimmed, trimmed_samples = [], [], [], [], []
    for i in range(num_samples):
        s = make_sample(config, seed, i, split="debug", stream=stream)
        truth = [q["target"] for q in s.queries]
        guesses = _guesses_with_truth(s.tokens, [q["depth"] for q in s.queries], truth, rng)
        for name, gs in guesses.items():
            hits.setdefault(name, []).extend(float(gq == t) for gq, t in zip(gs, truth))
        for name, base in EXCLUSION.items():
            hits.setdefault(name, []).extend(exclusion_hits(guesses[base], truth, s.num_targets))
        num_targets.extend(s.num_targets for _ in truth)
        elim_chance.extend(1.0 / (s.num_targets - qi) for qi in range(len(truth)))
        depth.extend(q["depth"] for q in s.queries)
        untrimmed.extend(not s.trimmed for _ in truth)
        trimmed_samples.append(s.trimmed)
    T, elim_chance, depth = np.asarray(num_targets, dtype=float), np.asarray(elim_chance), np.asarray(depth)
    untrimmed = np.asarray(untrimmed, dtype=bool)
    sums = {}
    for name, h in hits.items():
        h = np.asarray(h)
        if name == "elimination":
            p, var, ceil = elim_chance, elim_chance * (1 - elim_chance), np.ones_like(T)
        elif name in EXCLUSION:
            p, var, ceil = 1 / T, 1 / (T ** 2 * (T - 1)), 1 / (T - 1)
        else:
            p, var, ceil = 1 / T, (1 / T) * (1 - 1 / T), np.ones_like(T)
        keep = depth >= 2 if name in DEPTH2_ONLY else np.ones(len(h), dtype=bool)
        sums[name] = {}
        for subset, m in (("all", keep), ("untrimmed", keep & untrimmed)):
            sums[name][subset] = {"hits": float(h[m].sum()), "chance": float(p[m].sum()),
                                  "ceiling": float(ceil[m].sum()), "var": float(var[m].sum()), "queries": int(m.sum())}
    res = {}
    for name in HEURISTICS:
        res[name] = summarize(sums[name]["all"], z_flag)
        res[name]["untrimmed"] = summarize(sums[name]["untrimmed"], z_flag)
    if return_sums:
        info = {"min": int(T.min()), "mean": float(T.mean()), "max": int(T.max()),
                "trimmed_share": float(np.mean(trimmed_samples))}
        return res, {name: sums[name] for name in HEURISTICS}, info
    return res


def _guesses_with_truth(tokens, depths, truth, rng):
    """heuristic_guesses, with the elimination heuristic told the earlier queries' true answers."""
    p = parse(tokens)
    out = heuristic_guesses(tokens, depths, rng)
    targets = p.targets
    out["elimination"] = [_pick([t for t in targets if t not in truth[:qi]], rng) for qi in range(len(truth))]
    return out


def stats(config: TaskConfig, num_samples: int = 2000, seed: int = 0, stream: int = 0,
          batch_size: Optional[int] = None) -> dict:
    """Length statistics (tokens, edges, components, targets, answer tokens) over the debug split,
    plus the trimming shares: "trimmed" (samples), "trimmed_distractors" (distractors that lost a
    branch) and "branches_removed" (per sample). With batch_size B: ceil(num_samples / B) batches as the training stream makes them (one edge count
    per batch, generate.make_batch), plus "padding": the padding fraction of each batch,
    1 - sum(lengths) / (B * longest)."""
    keys = ("tokens", "edges", "components", "targets", "nodes", "answer_tokens", "trimmed", "trimmed_distractors",
            "branches_removed")
    vals = {k: [] for k in keys}
    if batch_size is None:
        samples = (make_sample(config, seed, i, split="debug", stream=stream) for i in range(num_samples))
    else:
        batches = [make_batch(config, seed, j, batch_size, split="debug", stream=stream)
                   for j in range(-(-num_samples // batch_size))]
        vals["padding"] = [1 - sum(len(s) for s in b) / (batch_size * max(len(s) for s in b)) for b in batches]
        samples = (s for b in batches for s in b)
    for s in samples:
        vals["tokens"].append(len(s))
        vals["edges"].append(s.num_edges)
        vals["components"].append(s.num_components)
        vals["targets"].append(s.num_targets)
        vals["nodes"].append(len({int(t) for t in s.tokens if VOCAB.is_node(t)}))
        vals["answer_tokens"].extend(q["answer_end"] - q["answer_start"] for q in s.queries)
        vals["trimmed"].append(float(s.trimmed))
        vals["trimmed_distractors"].extend(float(x["branching"] < x["drawn_branching"]) for x in s.distractors)
        vals["branches_removed"].append(s.branches_removed)
    vals = {k: v if v else [0.0] for k, v in vals.items()}
    return {k: {"mean": float(np.mean(v)), "max": np.max(v).item(), "min": np.min(v).item(),
                "p99": float(np.percentile(v, 99))} for k, v in vals.items()}
