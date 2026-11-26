"""Multi-query associative recall (MQAR, Arora et al. 2023, "Zoology"), as a check of the training
stack, and a ladder of single changes from it towards the graph task.

Original format (all flags off): keys from the first half of the node names, values from the
second half, all distinct within a sample:

    <bos> k1 v1 k2 v2 ... kN vN   then the query region:   _ _ k7 v7 _ k2 v2 _ _ k9 v9 ...

The query region holds every key once, in random order, each followed by its value, at random
non-overlapping slots; the rest is filler (token 0, <pad>, as the original's 0). The loss is on
each queried value, predicted at its key's position. No markers, no end tokens.

Flags, applied cumulatively in the experiments (a, then a + b, ...), each one difference:
    shared_vocab   (a) keys and values drawn together from all node names
    arrow          (b) pairs written "k > v"
    query_markers  (c) queries written "Q: k ? v", the value predicted at "?"
    target_list    (d) "T: " + the values in random order, after the pairs

A sample has exactly seq_len tokens. Batches carry the fields the training loop uses (tokens,
loss_mask, role, pred_pos, query_index, depth, branching, length); the stream is deterministic
per batch index and the same for any number of DataLoader workers, like TrainStream.

Chance: copying a random value from the context is right 1 / num_pairs of the time.
"""
import dataclasses
import json
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..dataset_generation.vocab import ARROW, ASK, BOS, NODE_POOL, PAD, QUERY, TARGETS, VOCAB

ROLE_FINAL = 1
KIND = "mqar"


@dataclasses.dataclass
class MqarConfig:
    name: str = "mqar"
    num_pairs: int = 16
    seq_len: int = 160
    shared_vocab: bool = False
    arrow: bool = False
    query_markers: bool = False
    target_list: bool = False

    def __post_init__(self):
        if self.num_pairs < 1:
            raise ValueError("num_pairs must be >= 1")
        half = NODE_POOL // 2
        if self.num_pairs > half:
            raise ValueError(f"num_pairs must be <= {half}")
        if self.query_region < self.num_pairs * self.query_len:
            raise ValueError(f"config {self.name!r}: seq_len {self.seq_len} leaves {self.query_region} tokens for "
                             f"{self.num_pairs} queries of {self.query_len} tokens; raise seq_len to at least "
                             f"{self.seq_len - self.query_region + self.num_pairs * self.query_len}")

    @property
    def pair_len(self) -> int:
        return 3 if self.arrow else 2

    @property
    def query_len(self) -> int:
        return 4 if self.query_markers else 2

    @property
    def context_len(self) -> int:
        """<bos>, the pairs and the optional target list."""
        return 1 + self.num_pairs * self.pair_len + (1 + self.num_pairs if self.target_list else 0)

    @property
    def query_region(self) -> int:
        return self.seq_len - self.context_len

    def to_dict(self) -> dict:
        return {"kind": KIND, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "MqarConfig":
        d = dict(d)
        if d.pop("kind", KIND) != KIND:
            raise ValueError("not an MQAR config")
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "MqarConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def is_mqar(d: dict) -> bool:
    return d.get("kind") == KIND


def make_sample(cfg: MqarConfig, rng: np.random.Generator) -> dict:
    N, L = cfg.num_pairs, cfg.seq_len
    half = NODE_POOL // 2
    if cfg.shared_vocab:
        names = rng.choice(NODE_POOL, size=2 * N, replace=False)
        keys, values = names[:N], names[N:]
    else:
        keys = rng.choice(half, size=N, replace=False)
        values = half + rng.choice(half, size=N, replace=False)
    keys = np.array([VOCAB.node(int(k)) for k in keys])
    values = np.array([VOCAB.node(int(v)) for v in values])

    toks = [BOS]
    for k, v in zip(keys, values):
        toks += [k, ARROW, v] if cfg.arrow else [k, v]
    if cfg.target_list:
        toks += [TARGETS] + [int(values[i]) for i in rng.permutation(N)]
    assert len(toks) == cfg.context_len

    tokens = np.full(L, PAD, dtype=np.int64)
    tokens[:len(toks)] = toks
    role = np.zeros(L, dtype=np.int64)
    query_index = np.full(L, -1, dtype=np.int64)
    order = rng.permutation(N)  # which pair each query asks, in query order
    slots = np.sort(rng.choice(cfg.query_region // cfg.query_len, size=N, replace=False))
    for qi, (pair, slot) in enumerate(zip(order, slots)):
        pos = cfg.context_len + slot * cfg.query_len
        block = [QUERY, int(keys[pair]), ASK, int(values[pair])] if cfg.query_markers else [int(keys[pair]), int(values[pair])]
        tokens[pos:pos + len(block)] = block
        ans = pos + len(block) - 1
        role[ans] = ROLE_FINAL
        query_index[pos:ans + 1] = qi
    loss_mask = role == ROLE_FINAL
    pred_pos = np.where(loss_mask, np.arange(L) - 1, -1)
    depth = loss_mask.astype(np.int64)
    return {"tokens": tokens, "loss_mask": loss_mask, "role": role, "pred_pos": pred_pos,
            "query_index": query_index, "depth": depth, "branching": depth.copy(),
            "keys": keys, "values": values, "order": order}


def make_batch(cfg: MqarConfig, seed: int, j: int, batch_size: int) -> dict:
    """Batch j of the stream: deterministic in (seed, j)."""
    rng = np.random.default_rng(np.random.SeedSequence([2000, seed, j]))
    samples = [make_sample(cfg, rng) for _ in range(batch_size)]
    out = {k: torch.from_numpy(np.stack([s[k] for s in samples]))
           for k in ("tokens", "loss_mask", "role", "pred_pos", "query_index", "depth", "branching")}
    out["role"] = out["role"].to(torch.int8)
    out["length"] = torch.full((batch_size,), cfg.seq_len)
    return out


class MqarStream(IterableDataset):
    """Infinite stream of batches start, start + 1, ...; worker w of W makes batches w, w + W, ...,
    so the loader returns the same stream for any number of workers."""

    def __init__(self, cfg: MqarConfig, seed: int = 0, batch_size: int = 64, start: int = 0):
        self.cfg, self.seed, self.batch_size, self.start = cfg, seed, batch_size, start

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        k = wid
        while True:
            yield make_batch(self.cfg, self.seed, self.start + k, self.batch_size)
            k += nw
