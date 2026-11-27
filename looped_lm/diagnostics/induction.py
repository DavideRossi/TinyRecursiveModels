"""Induction pretraining (phase 1 of the two-phase diagnostic, looped_lm/README.md section 6):
copying repeated spans, in the vocabulary of the graph task, so the weights carry over to it.

A sample is <bos> followed by segments until its length (drawn from [min_len, max_len]):

    fresh segment   seg_min..seg_max tokens: node names, unique within the sample, with special
                    tokens placed between names (never first, last, or next to another special)
    repeat          a copy of an earlier fresh segment, whole (1 - fragment_prob) or a fragment:
                    the segment from a random offset (on a name) to its end, at least min_fragment
                    tokens. Fragments run to the end so that every copy of a segment ends where the
                    segment does: a copy ending mid-segment would give its last name a second,
                    different continuation (the next segment) in the context.

The length is drawn per sample, or (length_per_batch) once per batch, so that a batch has no
padding: the graph stream does the same with its edge counts.

A segment is a repeat with probability repeat_prob (once a fresh segment exists). The loss is on
every token of a repeat except its first (role 1, so train/final_accuracy is the accuracy on the
supervised tokens), predicted at the previous position as everywhere.

Every supervised token is determined by the context: its predecessor is a name, which occurs in
one fresh segment and its copies only, always followed by the same token (one-token induction
suffices), or a special token, which recurs all over
the sample; then only the name before the special identifies the span (two-token matching, the
pattern of "x > t" in the graph task). The batch field after_special marks the latter, and
training logs their accuracy separately (train/after_special_accuracy, train/other_accuracy).

    python -m looped_lm.diagnostics.induction looped_lm/diagnostics/configs/induction.json [--num 2000]

prints the supervised fraction per sample, the share of after-special tokens and an example.
"""
import argparse
import dataclasses
import json
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..dataset_generation.vocab import ARROW, ASK, BACKTRACK, BOS, CLOSE, NODE_POOL, OPEN, PAD, QUERY, SEP, TARGETS, VOCAB

ROLE_FINAL = 1
KIND = "induction"
SPECIAL_TOKENS = (ARROW, TARGETS, QUERY, ASK, OPEN, CLOSE, BACKTRACK, SEP)  # not <pad>, <bos>, <eos>
FIELDS = ("tokens", "loss_mask", "role", "pred_pos", "query_index", "depth", "branching", "after_special")


@dataclasses.dataclass
class InductionConfig:
    name: str = "induction"
    min_len: int = 100
    max_len: int = 300
    seg_min: int = 4
    seg_max: int = 16
    repeat_prob: float = 0.65
    fragment_prob: float = 0.5
    min_fragment: int = 3
    special_prob: float = 0.33     # per interior slot of a fresh segment, if the previous token is a name
    length_per_batch: bool = False  # one length per batch (no padding) instead of one per sample

    def __post_init__(self):
        if not 2 <= self.min_len <= self.max_len:
            raise ValueError("need 2 <= min_len <= max_len")
        if not 3 <= self.seg_min <= self.seg_max:
            raise ValueError("need 3 <= seg_min <= seg_max")
        if not 2 <= self.min_fragment <= self.seg_min:
            raise ValueError("need 2 <= min_fragment <= seg_min")
        for k in ("repeat_prob", "fragment_prob", "special_prob"):
            if not 0.0 <= getattr(self, k) < 1.0:
                raise ValueError(f"{k} must be in [0, 1)")
        if self.max_len > NODE_POOL:
            raise ValueError(f"max_len must be <= {NODE_POOL} (names are unique within a sample)")

    def to_dict(self) -> dict:
        return {"kind": KIND, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "InductionConfig":
        d = dict(d)
        if d.pop("kind", KIND) != KIND:
            raise ValueError("not an induction config")
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "InductionConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def is_induction(d: dict) -> bool:
    return d.get("kind") == KIND


def _fresh_segment(cfg: InductionConfig, rng: np.random.Generator, names: Iterator[int]) -> list:
    n = int(rng.integers(cfg.seg_min, cfg.seg_max + 1))
    seg = [next(names)]
    for i in range(1, n):
        interior = i < n - 1
        if interior and VOCAB.is_node(seg[-1]) and rng.random() < cfg.special_prob:
            seg.append(int(SPECIAL_TOKENS[rng.integers(len(SPECIAL_TOKENS))]))
        else:
            seg.append(next(names))
    return seg


def make_sample(cfg: InductionConfig, rng: np.random.Generator, length: Optional[int] = None) -> dict:
    """One sample; its length is drawn from [min_len, max_len] unless given."""
    if length is None:
        length = int(rng.integers(cfg.min_len, cfg.max_len + 1))
    perm = rng.permutation(NODE_POOL)
    names = iter(VOCAB.node(int(i)) for i in perm)
    toks, sup, seg_id, offset = [BOS], [False], [-1], [-1]
    fresh = []  # the fresh segments so far
    while len(toks) < length:
        if fresh and rng.random() < cfg.repeat_prob:
            k = int(rng.integers(len(fresh)))
            src = fresh[k]
            a = 0
            if rng.random() < cfg.fragment_prob:
                starts = [i for i in range(len(src) - cfg.min_fragment + 1) if VOCAB.is_node(src[i])]
                a = starts[int(rng.integers(len(starts)))]
            seg = src[a:]
            sup += [False] + [True] * (len(seg) - 1)
        else:
            k, a = len(fresh), 0
            seg = _fresh_segment(cfg, rng, names)
            fresh.append(seg)
            sup += [False] * len(seg)
        toks += seg
        seg_id += [k] * len(seg)
        offset += list(range(a, a + len(seg)))
    tokens = np.array(toks[:length], dtype=np.int64)
    loss_mask = np.array(sup[:length], dtype=bool)
    role = np.where(loss_mask, ROLE_FINAL, 0).astype(np.int64)
    pred_pos = np.where(loss_mask, np.arange(length) - 1, -1)
    prev = np.concatenate([[PAD], tokens[:-1]])
    after_special = loss_mask & (prev < VOCAB.first_node)
    zeros = np.zeros(length, dtype=np.int64)
    return {"tokens": tokens, "loss_mask": loss_mask, "role": role, "pred_pos": pred_pos,
            "query_index": np.full(length, -1, dtype=np.int64), "depth": zeros, "branching": zeros.copy(),
            "after_special": after_special,
            # which fresh-segment token each token is (or copies): segment number and offset (-1: <bos>)
            "segment": np.array(seg_id[:length], dtype=np.int64), "offset": np.array(offset[:length], dtype=np.int64)}


def collate(samples: list) -> dict:
    L = max(len(s["tokens"]) for s in samples)
    fill = {"tokens": PAD, "loss_mask": False, "role": 0, "pred_pos": -1, "query_index": -1, "depth": 0,
            "branching": 0, "after_special": False}
    out = {}
    for k in FIELDS:
        rows = [np.concatenate([s[k], np.full(L - len(s[k]), fill[k], dtype=s[k].dtype)]) for s in samples]
        out[k] = torch.from_numpy(np.stack(rows))
    out["role"] = out["role"].to(torch.int8)
    out["length"] = torch.tensor([len(s["tokens"]) for s in samples])
    return out


def make_batch(cfg: InductionConfig, seed: int, j: int, batch_size: int) -> dict:
    """Batch j of the stream: deterministic in (seed, j)."""
    rng = np.random.default_rng(np.random.SeedSequence([3000, seed, j]))
    length = int(rng.integers(cfg.min_len, cfg.max_len + 1)) if cfg.length_per_batch else None
    return collate([make_sample(cfg, rng, length) for _ in range(batch_size)])


class InductionStream(IterableDataset):
    """Infinite stream of batches start, start + 1, ...; worker w of W makes batches w, w + W, ...,
    so the loader returns the same stream for any number of workers."""

    def __init__(self, cfg: InductionConfig, seed: int = 0, batch_size: int = 64, start: int = 0):
        self.cfg, self.seed, self.batch_size, self.start = cfg, seed, batch_size, start

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        k = wid
        while True:
            yield make_batch(self.cfg, self.seed, self.start + k, self.batch_size)
            k += nw


def describe(cfg: InductionConfig, num: int = 2000, seed: int = 0) -> dict:
    """Statistics over num samples: supervised fraction per sample (supervised / tokens), share of
    after-special tokens among the supervised ones, special-token density, lengths."""
    rng = np.random.default_rng(np.random.SeedSequence([3001, seed]))
    frac, n_sup, n_as, n_tok, n_spec = [], 0, 0, 0, 0
    for _ in range(num):
        s = make_sample(cfg, rng)
        m = s["loss_mask"]
        frac.append(m.mean())
        n_sup += int(m.sum())
        n_as += int(s["after_special"].sum())
        n_tok += len(m)
        n_spec += int(((s["tokens"] < VOCAB.first_node) & (s["tokens"] != BOS)).sum())
    frac = np.array(frac)
    return {"samples": num, "supervised_fraction_mean": float(frac.mean()),
            "supervised_fraction_p05": float(np.quantile(frac, 0.05)),
            "supervised_fraction_p95": float(np.quantile(frac, 0.95)),
            "supervised_fraction_min": float(frac.min()), "supervised_fraction_max": float(frac.max()),
            "after_special_share": n_as / max(1, n_sup), "special_density": n_spec / n_tok,
            "mean_length": n_tok / num}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m looped_lm.diagnostics.induction", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    cfg = InductionConfig.load(a.config)
    print(json.dumps(describe(cfg, a.num, a.seed), indent=1))
    s = make_sample(cfg, np.random.default_rng(a.seed))
    words = VOCAB.decode(s["tokens"][:80])
    print(" ".join(f"[{w}]" if m else w for w, m in zip(words, s["loss_mask"][:80])), "...")
    print("([x]: supervised)")


if __name__ == "__main__":
    main()
