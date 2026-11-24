"""Fixed sets on disk: compact flat arrays (.npy) plus meta.json.

The tokens of all samples are concatenated (uint16), with offsets[i]:offsets[i+1] the tokens of
sample i. Per-query arrays are concatenated likewise, with q_offsets; the query paths and branch
orders are flattened once more with their own offsets. The per-token metadata (loss mask, query
index, d, b, role, predicting position) is not stored: it follows from the tokens and the query
positions, and StoredSet.get rebuilds it. Per sample, the trimming fields (trimmed,
branches_removed) and the distractor shapes (flattened with d_offsets) are stored too. Everything
loads with mmap. Version 2 sets (before the trimming fields) remain readable: their samples read
as untrimmed, with distractors None.
"""
import json
import os
import zlib
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import TaskConfig
from .formats import ROLE_BACKTRACK, ROLE_BRACKET, ROLE_END, ROLE_FINAL, ROLE_INTERMEDIATE
from .generate import Sample, make_sample
from .vocab import BACKTRACK, CLOSE, OPEN, VOCAB, Vocab

FORMAT_VERSION = 3
READABLE_VERSIONS = (2, 3)
QUERY_SCALARS = {"source": np.uint16, "target": np.uint16, "depth": np.int16, "branching": np.int16,
                 "prompt_pos": np.int32, "answer_start": np.int32, "answer_end": np.int32}
QUERY_LISTS = ("path", "branch_order")
SAMPLE_SCALARS = ("num_edges", "num_components", "num_targets", "branches_removed")
DISTRACTOR_FIELDS = ("depth", "drawn_branching", "branching")


def stream_id(name: str) -> int:
    """Seed stream of a named set: different sets never share samples."""
    return zlib.crc32(name.encode())


def write_set(out_dir: str, config: TaskConfig, num_samples: int, seed: int, stream: int,
              split: str = "eval", readable: int = 5, vocab: Vocab = VOCAB) -> dict:
    assert vocab.size <= np.iinfo(np.uint16).max + 1
    os.makedirs(out_dir, exist_ok=True)
    tok = []
    qs = {k: [] for k in QUERY_SCALARS}
    ql = {k: [] for k in QUERY_LISTS}
    ss = {k: [] for k in SAMPLE_SCALARS}
    offsets, q_offsets, lengths = [0], [0], []
    trimmed, dist = [], {k: [] for k in DISTRACTOR_FIELDS}
    d_offsets = [0]
    with open(os.path.join(out_dir, "samples.jsonl"), "w") as jf:
        for i in range(num_samples):
            s = make_sample(config, seed, i, split=split, stream=stream, vocab=vocab)
            tok.append(s.tokens)
            for q in s.queries:
                for k in QUERY_SCALARS:
                    qs[k].append(q[k])
                for k in QUERY_LISTS:
                    ql[k].append(q[k])
            for k in SAMPLE_SCALARS:
                ss[k].append(getattr(s, k))
            trimmed.append(s.trimmed)
            for x in s.distractors:
                for k in DISTRACTOR_FIELDS:
                    dist[k].append(x[k])
            d_offsets.append(d_offsets[-1] + len(s.distractors))
            offsets.append(offsets[-1] + len(s))
            q_offsets.append(q_offsets[-1] + len(s.queries))
            lengths.append(len(s))
            if i < readable:
                jf.write(json.dumps(readable_record(s, i, vocab)) + "\n")
    arrays = {"offsets": np.asarray(offsets, np.int64), "q_offsets": np.asarray(q_offsets, np.int64)}
    arrays["tokens"] = np.concatenate(tok).astype(np.uint16)
    for k, dt in QUERY_SCALARS.items():
        arrays["q_" + k] = np.asarray(qs[k], dt)
    for k in QUERY_LISTS:
        arrays[f"q_{k}_offsets"] = np.cumsum([0] + [len(x) for x in ql[k]]).astype(np.int64)
        arrays[f"q_{k}"] = np.asarray([t for x in ql[k] for t in x], np.uint16)
    for k in SAMPLE_SCALARS:
        arrays[k] = np.asarray(ss[k], np.int32)
    arrays["trimmed"] = np.asarray(trimmed, np.uint8)
    arrays["d_offsets"] = np.asarray(d_offsets, np.int64)
    for k in DISTRACTOR_FIELDS:
        arrays["d_" + k] = np.asarray(dist[k], np.int16)
    for k, v in arrays.items():
        np.save(os.path.join(out_dir, k + ".npy"), v)
    meta = {"format_version": FORMAT_VERSION, "config": config.to_dict(), "split": split, "seed": seed,
            "stream": stream, "num_samples": num_samples, "vocab_fingerprint": vocab.fingerprint(),
            "vocab_size": vocab.size, "tokens_mean": float(np.mean(lengths)), "tokens_max": int(np.max(lengths)),
            "tokens_min": int(np.min(lengths)), "trimmed_share": float(np.mean(trimmed))}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def readable_record(s: Sample, index: int, vocab: Vocab = VOCAB) -> dict:
    return {"index": index, "text": s.text(vocab), "num_edges": s.num_edges, "num_targets": s.num_targets,
            "trimmed": s.trimmed, "branches_removed": s.branches_removed, "distractors": s.distractors,
            "queries": [{"source": vocab.tokens[q["source"]], "target": vocab.tokens[q["target"]],
                         "path": vocab.decode(q["path"]), "depth": q["depth"], "branching": q["branching"],
                         "branch_order": vocab.decode(q["branch_order"]),
                         "answer": vocab.to_text(s.tokens[q["answer_start"]:q["answer_end"]], pretty=False)}
                        for q in s.queries]}


class StoredSet:
    """Read access to a set written by write_set. get(i) returns a dict with the fields of
    generate.Sample (arrays, and the list of query records)."""

    def __init__(self, path: str, vocab: Vocab = VOCAB):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        if self.meta["format_version"] not in READABLE_VERSIONS:
            raise ValueError(f"{path}: format version {self.meta['format_version']}, readable: {READABLE_VERSIONS}")
        if self.meta["vocab_fingerprint"] != vocab.fingerprint():
            raise ValueError(f"{path} was written with a different vocabulary")
        self.config = TaskConfig.from_dict(self.meta["config"])
        self.a = {}
        for name in os.listdir(path):
            if name.endswith(".npy"):
                self.a[name[:-4]] = np.load(os.path.join(path, name), mmap_mode="r")

    def __len__(self):
        return self.meta["num_samples"]

    def get(self, i: int) -> dict:
        if not 0 <= i < len(self):
            raise IndexError(i)
        a = self.a
        lo, hi = int(a["offsets"][i]), int(a["offsets"][i + 1])
        out = {"tokens": np.asarray(a["tokens"][lo:hi], dtype=np.int32)}  # a copy: memmaps are read-only
        queries = []
        for j in range(int(a["q_offsets"][i]), int(a["q_offsets"][i + 1])):
            q = {k: int(a["q_" + k][j]) for k in QUERY_SCALARS}
            for k in QUERY_LISTS:
                o = a[f"q_{k}_offsets"]
                q[k] = [int(t) for t in a["q_" + k][int(o[j]):int(o[j + 1])]]
            queries.append(q)
        out["queries"] = queries
        out.update(token_metadata(out["tokens"], queries))
        for k in SAMPLE_SCALARS:
            out[k] = int(a[k][i]) if k in a else 0
        if "d_offsets" in a:
            lo, hi = int(a["d_offsets"][i]), int(a["d_offsets"][i + 1])
            out["distractors"] = [{k: int(a["d_" + k][j]) for k in DISTRACTOR_FIELDS} for j in range(lo, hi)]
            out["trimmed"] = bool(a["trimmed"][i])
        else:  # version 2
            out["distractors"], out["trimmed"] = None, False
        out["index"] = i
        return out


def token_metadata(tokens: np.ndarray, queries: List[dict]) -> Dict[str, np.ndarray]:
    """Per-token metadata of a sample from its tokens and query records (as generate.make_sample)."""
    L = len(tokens)
    out = {"loss_mask": np.zeros(L, bool), "query_index": np.full(L, -1, np.int16),
           "depth": np.zeros(L, np.int16), "branching": np.zeros(L, np.int16), "role": np.zeros(L, np.int8),
           "pred_pos": np.full(L, -1, np.int32)}
    for qi, q in enumerate(queries):
        lo, a0, a1 = q["prompt_pos"], q["answer_start"], q["answer_end"]
        out["query_index"][lo:a1 + 1] = qi
        out["depth"][lo:a1 + 1] = q["depth"]
        out["branching"][lo:a1 + 1] = q["branching"]
        out["loss_mask"][a0:a1 + 1] = True
        ans = tokens[a0:a1]
        role = np.full(len(ans), ROLE_INTERMEDIATE, np.int8)
        role[(ans == OPEN) | (ans == CLOSE)] = ROLE_BRACKET
        role[ans == BACKTRACK] = ROLE_BACKTRACK
        role[-1] = ROLE_FINAL
        out["role"][a0:a1] = role
        out["role"][a1] = ROLE_END
        out["pred_pos"][a1 - 1] = a1 - 2
    return out
