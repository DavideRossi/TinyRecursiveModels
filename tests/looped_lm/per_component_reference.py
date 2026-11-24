"""Reference fingerprints of targets_mode="per_component" (tests/looped_lm/test_generator.py).

Written from the code before targets_mode="fixed" existed; the test recomputes them with the
current code and requires them to be identical, so the old mode keeps producing exactly the same
samples.
    python tests/looped_lm/per_component_reference.py write
"""
import hashlib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from looped_lm.dataset_generation.config import TaskConfig  # noqa: E402
from looped_lm.dataset_generation.generate import make_batch, make_sample  # noqa: E402

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_component_reference.json")
MIXED = dict(depth={"range": [1, 6]}, branching={"range": [1, 4]}, num_queries={"range": [1, 3]},
             num_distractors={"range": [1, 3]}, total_edges={"range": [8, 150]})
# configs/train.json and two cells of the evaluation grid as they were before the fixed mode
OLD_TRAIN = dict(name="train", depth={"range": [1, 8]}, branching={"range": [1, 4]}, num_queries={"range": [1, 4]},
                 num_distractors=1, total_edges={"range": [160, 512]})
OLD_CELLS = {"d4_b2": dict(depth=4, branching=2, num_queries=1, num_distractors=1, total_edges=512),
             "d32_b8": dict(depth=32, branching=8, num_queries=1, num_distractors=1, total_edges=512)}


def digest(s) -> str:
    h = hashlib.sha256()
    for a in (s.tokens, s.loss_mask, s.query_index, s.depth, s.branching, s.role, s.pred_pos):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps([s.queries, s.num_edges, s.num_components, s.num_targets], default=int).encode())
    return h.hexdigest()


def compute() -> dict:
    out = {}
    for fmt, order, pos in itertools.product(("direct", "path", "frontier", "dfs"), ("random", "reverse", "forward"),
                                             ("after", "before")):
        cfg = TaskConfig(**MIXED, answer_format=fmt, edge_order=order, targets_position=pos)
        out[f"mixed/{fmt}/{order}/{pos}"] = [digest(make_sample(cfg, 3, i, split, 5))
                                             for split in ("train", "eval") for i in range(6)]
    train = TaskConfig(**OLD_TRAIN)
    out["train/map"] = [digest(make_sample(train, 0, i)) for i in range(10)]
    out["train/batches"] = [[digest(s) for s in make_batch(train, 0, j, 4)] for j in range(4)]
    for name, c in OLD_CELLS.items():
        out[f"grid/{name}"] = [digest(make_sample(TaskConfig(**c), 1, i, "eval", 77)) for i in range(5)]
    return out


if __name__ == "__main__":
    assert sys.argv[1:] == ["write"], "usage: python tests/looped_lm/per_component_reference.py write"
    with open(PATH, "w") as f:
        json.dump(compute(), f, indent=1, sort_keys=True)
    print("wrote", PATH)
