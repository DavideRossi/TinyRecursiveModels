"""PyTorch access: an infinite on-the-fly training stream and map-style datasets for fixed sets.

Every sample is make_sample(config, seed, index, split, stream), a pure function, so the data
depend only on those arguments, never on the number of workers or on the iteration order.
"""
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .config import TaskConfig
from .generate import Sample, make_batch, make_sample
from .storage import StoredSet
from .vocab import NODE_POOL, PAD

PAD_VALUES = {"tokens": PAD, "loss_mask": False, "query_index": -1, "depth": 0, "branching": 0, "role": 0,
              "pred_pos": -1}
TENSOR_DTYPES = {"tokens": torch.long, "loss_mask": torch.bool, "query_index": torch.long, "depth": torch.long,
                 "branching": torch.long, "role": torch.long, "pred_pos": torch.long}


def to_item(sample, index: int) -> dict:
    """A sample (generate.Sample or StoredSet record) as a dict of 1-D tensors plus its queries."""
    get = (lambda k: sample[k]) if isinstance(sample, dict) else (lambda k: getattr(sample, k))
    item = {k: torch.as_tensor(np.asarray(get(k)), dtype=dt) for k, dt in TENSOR_DTYPES.items()}
    item["queries"] = get("queries")
    item["index"] = index
    item["trimmed"] = bool(get("trimmed"))
    item["branches_removed"] = int(get("branches_removed"))
    item["distractors"] = get("distractors")
    return item


def collate(items: List[dict]) -> dict:
    """Pad to the longest sample (right padding). Returns [B, L] tensors, lengths, indices, the
    trimming fields ([B] tensors) and the per-sample query and distractor lists. The loss is on positions where loss_mask is true: the logits at
    position t - 1 predict the token at t."""
    L = max(len(it["tokens"]) for it in items)
    out = {}
    for k, pad in PAD_VALUES.items():
        out[k] = torch.full((len(items), L), pad, dtype=TENSOR_DTYPES[k])
        for i, it in enumerate(items):
            out[k][i, :len(it[k])] = it[k]
    out["length"] = torch.tensor([len(it["tokens"]) for it in items])
    out["index"] = torch.tensor([it["index"] for it in items])
    out["queries"] = [it["queries"] for it in items]
    out["trimmed"] = torch.tensor([it["trimmed"] for it in items], dtype=torch.bool)
    out["branches_removed"] = torch.tensor([it["branches_removed"] for it in items])
    out["distractors"] = [it["distractors"] for it in items]
    return out


class GeneratedDataset(Dataset):
    """Map-style: item i is sample offset + i of the stream (split, seed, stream), generated on access."""

    def __init__(self, config: TaskConfig, size: int, seed: int = 0, split: str = "eval", stream: int = 0,
                 offset: int = 0):
        self.config, self.size, self.seed, self.split, self.stream, self.offset = config, size, seed, split, stream, offset

    def __len__(self):
        return self.size

    def __getitem__(self, i: int) -> dict:
        if not 0 <= i < self.size:
            raise IndexError(i)
        return to_item(make_sample(self.config, self.seed, self.offset + i, self.split, self.stream), self.offset + i)


class StoredDataset(Dataset):
    """Map-style access to a set written by the CLI (storage.write_set)."""

    def __init__(self, path: str):
        self.set = StoredSet(path)
        self.config = self.set.config

    def __len__(self):
        return len(self.set)

    def __getitem__(self, i: int) -> dict:
        return to_item(self.set.get(i), i)


class TrainStream(IterableDataset):
    """Infinite on-the-fly training stream (split "train").

    batch_size=None: yields samples start, start + 1, ... (of this rank).
    batch_size=B: yields collated batches; batch j holds samples j*B ... j*B + B - 1.
    Edge count: one per batch (generate.make_batch), drawn from total_edges with the batch's own
    generator; without batch_size every sample is a batch of one. The config must satisfy
    TaskConfig.check_batch_edges (checked here).
    Use DataLoader(stream, batch_size=None, num_workers=W): workers take turns (worker w makes
    items w, w + W, ...) and the loader returns them in order, so the stream is the same for any W.
    With several ranks, rank r of R takes every R-th sample (or batch) starting at r.
    """

    def __init__(self, config: TaskConfig, seed: int = 0, batch_size: Optional[int] = None, start: int = 0,
                 rank: int = 0, world_size: int = 1, allow_small_pool: bool = False):
        if config.pool_size != NODE_POOL and not allow_small_pool:
            raise ValueError(f"training must draw names from the full pool of {NODE_POOL} "
                             f"(got pool_size={config.pool_size}); allow_small_pool=True for debugging only")
        config.check_batch_edges()
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside world size {world_size}")
        self.config, self.seed, self.batch_size, self.start = config, seed, batch_size, start
        self.rank, self.world_size = rank, world_size

    def item(self, j: int):
        """Item j of the global stream (a sample, or a batch with batch_size)."""
        if self.batch_size is None:
            return to_item(make_batch(self.config, self.seed, j, 1, "train")[0], j)
        samples = make_batch(self.config, self.seed, j, self.batch_size, "train")
        return collate([to_item(s, j * self.batch_size + k) for k, s in enumerate(samples)])

    def __iter__(self) -> Iterator:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        k = wid
        while True:
            yield self.item(self.start + self.rank + self.world_size * k)
            k += nw
