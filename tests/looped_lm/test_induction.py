"""Tests of the induction diagnostic (looped_lm/diagnostics/induction.py), its metrics in
training, and --init_from.

    python tests/looped_lm/test_induction.py
"""
import dataclasses
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from looped_lm.dataset_generation.vocab import BOS, PAD, VOCAB  # noqa: E402
from looped_lm.diagnostics.induction import (InductionConfig, InductionStream, describe, make_batch,  # noqa: E402
                                             make_sample)
from looped_lm.diagnostics.special_swap import make_probe, matcher_predictor, score, swap_sample  # noqa: E402
from looped_lm.model import LoopedLM, ModelConfig  # noqa: E402
from looped_lm.objective import loss_and_stats  # noqa: E402
from looped_lm.train import TrainConfig, load_task, train  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CONFIG = os.path.join(ROOT, "looped_lm", "diagnostics", "configs", "induction.json")
LONG = os.path.join(ROOT, "looped_lm", "diagnostics", "configs", "induction_long.json")
LOOKUP = os.path.join(ROOT, "looped_lm", "dataset_generation", "configs", "diag_lookup_q16.json")
TINY = dict(width=32, num_heads=2, head_dim=16, mlp_hidden=64, core_layers=1)


def _spans(mask):
    """(start, end) of the maximal runs of supervised tokens."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def test_sample_structure():
    cfg = InductionConfig.load(CONFIG)
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = make_sample(cfg, rng)
        t, m = s["tokens"], s["loss_mask"]
        assert t[0] == BOS and cfg.min_len <= len(t) <= cfg.max_len and not m[0]
        assert ((s["role"] == 1) == m).all() and (s["pred_pos"][m] == np.nonzero(m)[0] - 1).all()
        assert (s["pred_pos"][~m] == -1).all() and (s["query_index"] == -1).all()
        special = (t < VOCAB.first_node) & (t != BOS)
        assert not (special[1:] & special[:-1]).any()            # never two specials in a row
        assert (s["after_special"] == (m & np.concatenate([[False], special[:-1]]))).all()
        # Every supervised token is determined by the context: it continues an earlier occurrence of
        # its (one or two) preceding tokens, which ends in a name; and every such occurrence agrees.
        for p in np.nonzero(m)[0]:
            k = 2 if special[p - 1] else 1
            assert not special[p - k], "the matched context must end in a name"
            ctx = tuple(t[p - k:p])
            nxt = {int(t[q + k]) for q in range(1, p - k) if tuple(t[q:q + k]) == ctx}
            assert nxt == {int(t[p])}, (p, ctx, nxt)
        # The repeat's first token is unsupervised: it occurred earlier (a name), but nothing announces it
        for a, b in _spans(m):
            assert not m[a - 1] and VOCAB.is_node(t[a - 1]) and (t[:a - 1] == t[a - 1]).sum() >= 1
        # Names outside repeats are unique: an unsupervised name seen before starts a repeat (the next
        # token is supervised), or is a repeat cut to one token by the end of the sample
        seen = set()
        for p in range(1, len(t)):
            if not m[p] and VOCAB.is_node(t[p]):
                if int(t[p]) in seen:
                    assert p == len(t) - 1 or m[p + 1], p
                seen.add(int(t[p]))


def test_segment_offset_and_special_swap():
    """segment/offset identify the copied token; the special-swap probe separates a matcher of the
    name alone (unaffected) from a matcher of the name and the special (no match after the swap)."""
    cfg = InductionConfig.load(CONFIG)
    rng = np.random.default_rng(1)
    for _ in range(100):
        s = make_sample(cfg, rng)
        same = {}
        for t, g, o in zip(s["tokens"], s["segment"], s["offset"]):
            assert same.setdefault((int(g), int(o)), int(t)) == int(t)
        swapped, pos = swap_sample(s, rng)
        assert ((swapped != s["tokens"]) <= s["loss_mask"]).all()          # only repeat tokens change
        changed = np.nonzero(swapped != s["tokens"])[0]
        assert all(VOCAB.first_node > swapped[q] != BOS for q in changed)
        assert set(pos.tolist()) == {q + 1 for q in changed if q + 1 < len(swapped) and s["loss_mask"][q + 1]}
    probe = make_probe(cfg, 200)
    name, pair = score(matcher_predictor(False), probe), score(matcher_predictor(True), probe)
    assert name["swapped_accuracy"] == name["control_accuracy"] == 1.0 and name["n"] > 1000
    assert pair["control_accuracy"] == 1.0 and pair["swapped_accuracy"] == 0.0


def test_length_per_batch_and_long_config():
    """induction_long: one length per batch (no padding), lengths over 100-1600 across batches, the
    same supervised fraction and after-special share as the short config."""
    cfg = InductionConfig.load(LONG)
    assert cfg.length_per_batch and (cfg.min_len, cfg.max_len) == (100, 1600)
    lengths = []
    for j in range(40):
        b = make_batch(cfg, 0, j, 8)
        assert (b["length"] == b["length"][0]).all() and b["tokens"].shape[1] == int(b["length"][0])
        assert (b["tokens"] != PAD).all()
        lengths.append(int(b["length"][0]))
    assert min(lengths) < 500 and max(lengths) > 1100
    d = describe(cfg, num=200)
    assert 0.45 <= d["supervised_fraction_mean"] <= 0.58 and d["after_special_share"] >= 0.2, d
    short = InductionConfig.load(CONFIG)
    assert not short.length_per_batch
    assert len({int(n) for n in make_batch(short, 0, 0, 16)["length"]}) > 1   # per sample, as before


def test_statistics():
    d = describe(InductionConfig.load(CONFIG), num=500)
    assert 0.45 <= d["supervised_fraction_mean"] <= 0.55, d
    assert d["after_special_share"] >= 0.2, d
    assert 100 <= d["mean_length"] <= 300


def test_stream_deterministic_and_worker_independent():
    cfg = InductionConfig.load(CONFIG)
    a, b = make_batch(cfg, 3, 5, 8), make_batch(cfg, 3, 5, 8)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(make_batch(cfg, 4, 5, 8)["tokens"][:, :50], a["tokens"][:, :50])
    L = a["tokens"].shape[1]
    assert L == int(a["length"].max())
    for i, n in enumerate(a["length"].tolist()):
        assert (a["tokens"][i, n:] == PAD).all() and not a["loss_mask"][i, n:].any()
    one = [x["tokens"] for _, x in zip(range(6), DataLoader(InductionStream(cfg, 3, 8, start=2), batch_size=None))]
    two = [x["tokens"] for _, x in zip(range(6), DataLoader(InductionStream(cfg, 3, 8, start=2), batch_size=None,
                                                             num_workers=2))]
    assert all(torch.equal(x, y) for x, y in zip(one, two))
    assert torch.equal(one[0], make_batch(cfg, 3, 2, 8)["tokens"])


class NextTokenOracle(torch.nn.Module):
    """Stub: the hidden state at position p is the one-hot of the token at p + 1, the head the identity."""

    def forward(self, tokens):
        return 30.0 * F.one_hot(torch.roll(tokens, -1, dims=1), VOCAB.size).float()

    def head(self, h):
        return h


def test_objective_alignment_and_split():
    batch = make_batch(InductionConfig.load(CONFIG), 0, 0, 6)
    loss, st = loss_and_stats(NextTokenOracle(), batch)
    n = int(batch["loss_mask"][:, 1:].sum())
    assert int(st["final_n"]) == int(st["final_correct"]) == n and float(loss) < 1e-3
    assert int(st["after_special_n"]) == int(batch["after_special"].sum()) > 0
    assert int(st["after_special_n"]) + int(st["other_n"]) == n
    assert int(st["after_special_correct"]) + int(st["other_correct"]) == n


def _cfg(d, name, data, **kw):
    base = dict(steps=4, warmup_steps=2, batch_size=4, num_workers=0, log_interval=2, eval_interval=0,
                ckpt_interval=0, keep_interval=0, wandb="off", device="cpu")
    return TrainConfig(run_name=name, out_dir=d, data_config=data, **dict(base, **kw))


def test_training_and_init_from():
    with tempfile.TemporaryDirectory() as d:
        # Phase 1: the induction stream, with its metrics
        p1 = _cfg(d, "p1", CONFIG)
        assert isinstance(load_task(p1), InductionConfig)
        assert train(p1, ModelConfig(**TINY))["step"] == 4
        rows = [json.loads(l) for l in open(os.path.join(d, "p1", "metrics.jsonl"))]
        logs = [r for r in rows if "train/final_accuracy" in r]
        assert len(logs) == 2
        for k in ("after_special_accuracy", "other_accuracy", "after_special_share", "supervised_fraction"):
            assert all(f"train/{k}" in r for r in logs), k
        assert all(0.2 < r["train/supervised_fraction"] < 0.8 for r in logs)
        ckpt = os.path.join(d, "p1", "ckpt_latest.pt")
        w1 = torch.load(ckpt, weights_only=False)["model"]

        # Phase 2: the graph task from the phase 1 weights, fresh optimizer and schedule
        p2 = _cfg(d, "p2", LOOKUP, init_from=ckpt)
        train(dataclasses.replace(p2, steps=1), ModelConfig(**TINY))
        saved = torch.load(os.path.join(d, "p2", "ckpt_latest.pt"), weights_only=False)
        assert saved["step"] == 1 and saved["train_config"]["init_from"] == ckpt
        assert len(saved["optimizer"]["state"]) > 0 and all(int(s["step"]) == 1 for s in saved["optimizer"]["state"].values())
        rows = [json.loads(l) for l in open(os.path.join(d, "p2", "metrics.jsonl"))]
        assert rows[0]["step"] == 0 and "num_params" in rows[0]
        assert json.load(open(os.path.join(d, "p2", "config.json")))["train"]["init_from"] == ckpt
        assert not any("after_special" in k for r in rows for k in r)

        # The weights loaded are phase 1's: with lr 0 one step changes nothing
        p0 = _cfg(d, "p0", LOOKUP, init_from=ckpt, lr=0.0, steps=1)
        train(p0, ModelConfig(**TINY))
        w0 = torch.load(os.path.join(d, "p0", "ckpt_latest.pt"), weights_only=False)["model"]
        assert all(torch.equal(w0[k], w1[k]) for k in w1)

        # A different model config is refused
        try:
            train(_cfg(d, "bad", LOOKUP, init_from=ckpt), ModelConfig(**dict(TINY, core_layers=2)))
        except ValueError as e:
            assert "core_layers" in str(e)
        else:
            raise AssertionError("mismatched init_from must be refused")

        # Resuming wins over init_from: p2 continues from its own step-1 checkpoint
        out = train(dataclasses.replace(p2, steps=3), ModelConfig(**TINY))
        assert out["step"] == 3
        assert int(torch.load(os.path.join(d, "p2", "ckpt_latest.pt"), weights_only=False)["step"]) == 3


def test_gate():
    """--gate_step: a failing gate stops the run at that step without a checkpoint, writes
    gate.json, exits with status 3 and is not restarted; a passing gate lets the run finish."""
    from looped_lm.train import GATE_FAILED, main
    with tempfile.TemporaryDirectory() as d:
        ok = _cfg(d, "ok", LOOKUP, steps=6, gate_step=4, gate_first_query=-1.0)
        out = train(ok, ModelConfig(**TINY))
        assert out["step"] == 6 and not out["gate_failed"]
        gate = json.load(open(os.path.join(d, "ok", "gate.json")))
        assert gate["passed"] and gate["step"] == 4

        bad = _cfg(d, "bad", LOOKUP, steps=6, gate_step=4, gate_first_query=1.0)  # accuracy is never above 1
        out = train(bad, ModelConfig(**TINY))
        assert out["step"] == 4 and out["gate_failed"]
        assert not os.path.exists(os.path.join(d, "bad", "ckpt_latest.pt"))
        gate = json.load(open(os.path.join(d, "bad", "gate.json")))
        assert not gate["passed"] and gate["step"] == 4 and 0.0 <= gate["first_query_accuracy"] <= 1.0
        rows = [json.loads(l) for l in open(os.path.join(d, "bad", "metrics.jsonl"))]
        assert [r["gate/passed"] for r in rows if "gate/passed" in r] == [0]
        assert max(r["step"] for r in rows) == 4
        # Submitting it again does not restart it, and the command line exits with status 3
        argv = ["--run_name", "bad", "--out_dir", d, "--data_config", LOOKUP, "--steps", "6", "--gate_step", "4",
                "--gate_first_query", "1.0", "--log_interval", "2", "--eval_interval", "0", "--num_workers", "0",
                "--wandb", "off", "--device", "cpu", "--width", "32", "--num_heads", "2", "--head_dim", "16",
                "--mlp_hidden", "64", "--core_layers", "1", "--batch_size", "4"]
        assert main(argv) == GATE_FAILED
        assert max(json.loads(l)["step"] for l in open(os.path.join(d, "bad", "metrics.jsonl"))) == 4

        # Record-only: the gate is written and logged, the run continues, and it can be resubmitted
        rec = _cfg(d, "rec", LOOKUP, steps=6, gate_step=4, gate_first_query=1.0, gate_enforce=False)
        out = train(rec, ModelConfig(**TINY))
        assert out["step"] == 6 and not out["gate_failed"]
        gate = json.load(open(os.path.join(d, "rec", "gate.json")))
        assert not gate["passed"] and not gate["enforced"]
        rows = [json.loads(l) for l in open(os.path.join(d, "rec", "metrics.jsonl"))]
        assert [r["gate/passed"] for r in rows if "gate/passed" in r] == [0] and max(r["step"] for r in rows) == 6
        assert train(dataclasses.replace(rec, steps=8), ModelConfig(**TINY))["step"] == 8

        for kw in (dict(gate_step=3), dict(gate_step=8)):  # not a multiple of log_interval 2; beyond steps
            try:
                train(_cfg(d, "x", LOOKUP, steps=6, **kw), ModelConfig(**TINY))
            except ValueError:
                pass
            else:
                raise AssertionError(kw)
        try:
            train(_cfg(d, "y", CONFIG, gate_step=2), ModelConfig(**TINY))
        except ValueError as e:
            assert "first queries" in str(e)
        else:
            raise AssertionError("the gate must be refused on the induction task")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
