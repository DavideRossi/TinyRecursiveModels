"""Tests of the MQAR diagnostic stream (looped_lm/diagnostics/mqar.py) and its use in training.

    python tests/looped_lm/test_mqar.py
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
from looped_lm.dataset_generation.vocab import ARROW, ASK, BOS, NODE_POOL, PAD, QUERY, TARGETS, VOCAB  # noqa: E402
from looped_lm.diagnostics.mqar import MqarConfig, MqarStream, make_batch, make_sample  # noqa: E402
from looped_lm.model import ModelConfig  # noqa: E402
from looped_lm.objective import loss_and_stats  # noqa: E402
from looped_lm.train import TrainConfig, load_task, train  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "looped_lm", "diagnostics", "configs")
LADDER = ["mqar_base", "mqar_a", "mqar_ab", "mqar_abc", "mqar_abcd"]
HALF = VOCAB.first_node + NODE_POOL // 2


def test_ladder_configs_differ_by_one_flag():
    flags = ("shared_vocab", "arrow", "query_markers", "target_list")
    cfgs = [MqarConfig.load(os.path.join(CONFIGS, n + ".json")) for n in LADDER]
    for prev, cur in zip(cfgs, cfgs[1:]):
        diff = [f for f in flags if getattr(prev, f) != getattr(cur, f)]
        assert len(diff) == 1 and getattr(cur, diff[0]), (prev.name, cur.name, diff)
        assert (prev.num_pairs, prev.seq_len) == (cur.num_pairs, cur.seq_len) == (16, 160)
    assert not any(getattr(cfgs[0], f) for f in flags) and all(getattr(cfgs[-1], f) for f in flags)


def test_sample_layout():
    rng = np.random.default_rng(0)
    for name in LADDER:
        cfg = MqarConfig.load(os.path.join(CONFIGS, name + ".json"))
        for _ in range(50):
            s = make_sample(cfg, rng)
            t, N = s["tokens"], cfg.num_pairs
            keys, values = list(s["keys"]), list(s["values"])
            assert len(t) == cfg.seq_len and t[0] == BOS
            assert len(set(keys + values)) == 2 * N
            if not cfg.shared_vocab:
                assert all(VOCAB.first_node <= k < HALF for k in keys) and all(v >= HALF for v in values)
            # The pairs, then the optional target list
            step = cfg.pair_len
            for i, (k, v) in enumerate(zip(keys, values)):
                pair = list(t[1 + i * step:1 + (i + 1) * step])
                assert pair == ([k, ARROW, v] if cfg.arrow else [k, v])
            if cfg.target_list:
                start = 1 + N * step
                assert t[start] == TARGETS and sorted(t[start + 1:start + 1 + N]) == sorted(values)
            # Queries: every key once, its value right after it (or after "?"), filler elsewhere
            answers = np.nonzero(s["loss_mask"])[0]
            assert len(answers) == N and (s["role"][answers] == 1).all() and (s["role"] != 0).sum() == N
            assert (answers >= cfg.context_len).all()
            assert sorted(s["query_index"][answers]) == list(range(N))
            asked = []
            for a in answers:
                assert s["pred_pos"][a] == a - 1
                if cfg.query_markers:
                    assert list(t[a - 3:a]) == [QUERY, t[a - 2], ASK]
                    k = t[a - 2]
                else:
                    k = t[a - 1]
                asked.append(k)
                assert t[a] == values[keys.index(k)]
            assert sorted(asked) == sorted(keys)
            used = np.zeros(cfg.seq_len, bool)
            for a in answers:
                used[a - cfg.query_len + 1:a + 1] = True
            region = np.arange(cfg.seq_len) >= cfg.context_len
            assert (t[region & ~used] == PAD).all() and (s["pred_pos"][~s["loss_mask"]] == -1).all()


def test_stream_deterministic_and_worker_independent():
    cfg = MqarConfig.load(os.path.join(CONFIGS, "mqar_abcd.json"))
    a, b = make_batch(cfg, 3, 5, 8), make_batch(cfg, 3, 5, 8)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(make_batch(cfg, 3, 6, 8)["tokens"], a["tokens"])
    assert not torch.equal(make_batch(cfg, 4, 5, 8)["tokens"], a["tokens"])
    one = [x["tokens"] for _, x in zip(range(6), DataLoader(MqarStream(cfg, 3, 8, start=2), batch_size=None))]
    two = [x["tokens"] for _, x in zip(range(6), DataLoader(MqarStream(cfg, 3, 8, start=2), batch_size=None,
                                                             num_workers=2))]
    assert all(torch.equal(x, y) for x, y in zip(one, two))
    assert torch.equal(one[0], make_batch(cfg, 3, 2, 8)["tokens"])


class ValueOracle(torch.nn.Module):
    """Stub: the hidden state at position p is the one-hot of the token at p + 1, the head the identity."""

    def forward(self, tokens):
        return 30.0 * F.one_hot(torch.roll(tokens, -1, dims=1), VOCAB.size).float()

    def head(self, h):
        return h


def test_objective_alignment():
    """Read one position ahead, every answer and every first query is right: the loss positions and
    the prediction positions of the MQAR batches agree with the objective."""
    for name in LADDER:
        cfg = MqarConfig.load(os.path.join(CONFIGS, name + ".json"))
        batch = make_batch(cfg, 0, 0, 6)
        loss, st = loss_and_stats(ValueOracle(), batch)
        assert int(st["final_n"]) == int(st["final_correct"]) == 6 * cfg.num_pairs
        assert int(st["first_n"]) == int(st["first_correct"]) == 6 and int(st["end_n"]) == 0
        assert float(loss) < 1e-3


def test_training_on_mqar():
    with tempfile.TemporaryDirectory() as d:
        cfg = TrainConfig(run_name="m", out_dir=d, data_config=os.path.join(CONFIGS, "mqar_base.json"), steps=6,
                          warmup_steps=2, batch_size=4, num_workers=0, log_interval=3, eval_interval=3,
                          ckpt_interval=0, keep_interval=0, wandb="off", device="cpu")
        assert isinstance(load_task(cfg), MqarConfig)
        try:
            load_task(dataclasses.replace(cfg, answer_format="path"))
        except ValueError:
            pass
        else:
            raise AssertionError("--answer_format must be refused for MQAR")
        out = train(cfg, ModelConfig(width=32, num_heads=2, head_dim=16, mlp_hidden=64, core_layers=1))
        assert out["step"] == 6
        rows = [json.loads(l) for l in open(os.path.join(d, "m", "metrics.jsonl"))]
        assert sum("train/first_query_accuracy" in r for r in rows) == 2
        assert sum("model/embed_norm_mean" in r for r in rows) == 3  # step 0, 3, 6: norms even without eval sets
        assert not any(k.startswith("eval/") for r in rows for k in r)
        saved = json.load(open(os.path.join(d, "m", "config.json")))
        assert saved["task"]["kind"] == "mqar" and saved["task"]["name"] == "mqar_base"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
