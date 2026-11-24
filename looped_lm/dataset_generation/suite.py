"""Evaluation suites: one JSON file describing every fixed set.

{
  "seed": 1,
  "base": { ...TaskConfig fields shared by every set... },
  "grid": {"depths": [...], "branchings": [...], "num_samples": 1000,
           "total_edges": null,          # null: the smallest count that fits every cell
           "overrides": {...}},          # optional, applied to every cell
  "companions": {"num_samples": 200,     # optional, fixed mode: pairs for the grid cells that trim
                 "cells": "trimmed"},    # or a list of [d, b]
  "diag": {"depth": 8, "branchings": [1, 2, 4, 8], "num_targets": [6, 9, 12],
           "num_samples": 1000, "total_edges": null, "overrides": {...}},   # optional, fixed mode
  "mixed": [{"name": "...", "num_samples": 1000, "overrides": {...}}, ...],
  "readable_samples": 5
}

grid/dD_bB: one query, d and b fixed, distractors of the same shape, the same edge count in every
cell.

companion/dD_bB_A and _B (fixed mode), for the grid cells whose distractors get trimmed: both at
the edge count that fits every component untrimmed (num_targets x d x b), with the same T and the
same queried shape. A: distractors at their drawn shapes. B: distractors trimmed as in the grid
cell (trim_edges = the grid's edge count), the freed edges filled with target-less padding. A and
B share their seed stream, so sample i of A and of B have the same queried component, with the same
names and (in random edge order) the same edge positions; they differ only in distractor shapes.

A pair whose sets need more node names than the pool (d32_b8 at T = 8: 2,048 edges and 8 sources)
is skipped with a warning.

diag/bB_tT (fixed mode): d fixed, one query, every branching crossed with every T, cells with
T < b omitted, one edge count for all (null: the smallest that fits every cell).

Each set has its own seed stream (from its name), in the "eval" split.
"""
import json
import warnings
from typing import List, NamedTuple

from .config import TaskConfig
from .storage import stream_id


class SetSpec(NamedTuple):
    name: str
    config: TaskConfig
    num_samples: int
    seed: int
    stream: int


def grid_cell_name(d: int, b: int) -> str:
    return f"grid/d{d}_b{b}"


def _grid_cell(base: dict, grid: dict, d: int, b: int, edges) -> dict:
    cell = dict(base, **grid.get("overrides", {}))
    cell.update(num_queries=1, distractor_depth=None, distractor_branching=None, depth=d, branching=b,
                total_edges=edges)
    return cell


def _common_edges(cells: List[dict]) -> int:
    """The smallest edge count that fits every cell (each at its worst case)."""
    return max(TaskConfig.from_dict(dict(c, total_edges=None)).max_required_edges() for c in cells)


def expand(suite: dict) -> List[SetSpec]:
    seed = int(suite.get("seed", 1))
    base = dict(suite.get("base", {}))
    out = []
    grid = suite.get("grid")
    if grid:
        cells = [(d, b) for d in grid["depths"] for b in grid["branchings"]]
        edges = grid.get("total_edges")
        if edges is None:
            edges = _common_edges([_grid_cell(base, grid, d, b, None) for d, b in cells])
        grid_cfgs = {}
        for d, b in cells:
            name = grid_cell_name(d, b)
            cfg = TaskConfig.from_dict(dict(_grid_cell(base, grid, d, b, edges), name=name))
            grid_cfgs[(d, b)] = cfg
            out.append(SetSpec(name, cfg, int(grid["num_samples"]), seed, stream_id(name)))
        comp = suite.get("companions")
        if comp:
            if not all(c.fixed for c in grid_cfgs.values()):
                raise ValueError("companions need targets_mode 'fixed'")
            which = comp.get("cells", "trimmed")
            if which == "trimmed":
                which = [db for db, c in grid_cfgs.items() if c.max_untrimmed_edges() > edges]
            for d, b in which:
                cfg = grid_cfgs[(d, b)]
                full = cfg.max_untrimmed_edges()
                pair = f"companion/d{d}_b{b}"
                try:
                    a = cfg.replace(name=pair + "_A", total_edges=full)
                    bb = cfg.replace(name=pair + "_B", total_edges=full, trim_edges=edges)
                except ValueError as e:
                    if "pool" not in str(e):
                        raise
                    warnings.warn(f"{pair} skipped: {full} edges need more node names than the pool ({e})")
                    continue
                for c in (a, bb):  # both from the pair's stream: paired samples
                    out.append(SetSpec(c.name, c, int(comp["num_samples"]), seed, stream_id(pair)))
    diag = suite.get("diag")
    if diag:
        cells = [(b, t) for b in diag["branchings"] for t in diag["num_targets"] if t >= b]
        dicts = {(b, t): dict(_grid_cell(base, diag, diag["depth"], b, None), num_targets=t) for b, t in cells}
        edges = diag.get("total_edges")
        if edges is None:
            edges = _common_edges(list(dicts.values()))
        for (b, t), c in dicts.items():
            name = f"diag/b{b}_t{t}"
            cfg = TaskConfig.from_dict(dict(c, name=name, total_edges=edges))
            out.append(SetSpec(name, cfg, int(diag["num_samples"]), seed, stream_id(name)))
    for m in suite.get("mixed", []):
        name = "mixed/" + m["name"]
        cfg = TaskConfig.from_dict(dict(base, **m.get("overrides", {}), name=name))
        out.append(SetSpec(name, cfg, int(m["num_samples"]), seed, stream_id(name)))
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate set names in the suite")
    return out


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
