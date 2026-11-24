"""Tests of the looped LM model, objective, evaluation and training loop (CPU, tiny models).

    python tests/looped_lm/test_model.py
"""
import dataclasses
import math
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from looped_lm.dataset_generation.config import TaskConfig  # noqa: E402
from looped_lm.dataset_generation.dataset import collate, to_item  # noqa: E402
from looped_lm.dataset_generation.generate import make_sample  # noqa: E402
from looped_lm.dataset_generation.vocab import PAD, VOCAB  # noqa: E402
from looped_lm.evaluation import evaluate_set, wilson  # noqa: E402
from looped_lm.model import LoopedLM, ModelConfig  # noqa: E402
from looped_lm.objective import loss_and_stats  # noqa: E402
from looped_lm.train import TrainConfig, lr_at, train  # noqa: E402

TINY = dict(width=32, num_heads=2, head_dim=16, mlp_hidden=64, core_layers=2, max_positions=512)
# A small task: one query, 2-3 targets, short chains; worst case 1 x 4 + 2 chains x 2 = 8 edges
TASK = TaskConfig(targets_mode="fixed", depth={"range": [1, 2]}, branching={"range": [1, 2]}, num_queries=1,
                  num_targets={"range": [2, 3]}, total_edges={"range": [8, 12]})
MULTI = TaskConfig(targets_mode="fixed", depth={"range": [1, 3]}, branching={"range": [1, 2]}, num_queries=2,
                   num_targets=4, total_edges=30)


def tiny_model(seed=0, **kw):
    torch.manual_seed(seed)
    return LoopedLM(ModelConfig(**dict(TINY, **kw)))


def batch_of(task, n, seed=0):
    return collate([to_item(make_sample(task, seed, i), i) for i in range(n)])


def test_parameter_counts():
    for kw in (dict(core_layers=2), dict(core_layers=16), dict(core_layers=8, width=512, num_heads=8, mlp_hidden=1408),
               dict(prelude_layers=1, core_layers=2, coda_layers=3), dict(TINY), dict(TINY, num_heads=3)):
        cfg = ModelConfig(**kw)
        d, a, m, L = cfg.width, cfg.num_heads * cfg.head_dim, cfg.mlp_hidden, cfg.num_layers
        formula = VOCAB.size * d + d + L * (4 * d * a + 3 * d * m + 2 * d)
        assert LoopedLM(cfg).num_params() == cfg.expected_num_params() == formula, kw
    assert ModelConfig(core_layers=2).expected_num_params() == 2_134_016
    assert ModelConfig(core_layers=8, width=512, num_heads=8, mlp_hidden=1408).expected_num_params() == 26_753_024


def test_structure_and_initialization():
    try:
        LoopedLM(ModelConfig(**dict(TINY, n_loops=2)))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("n_loops = 2 must raise NotImplementedError")
    m = LoopedLM(ModelConfig(width=256, core_layers=8, prelude_layers=1, coda_layers=1))
    assert len(m.prelude) == 1 and len(m.core) == 8 and len(m.coda) == 1
    out_std = 0.02 / math.sqrt(2 * 10)
    o = torch.cat([b.attn.o_proj.weight.flatten() for b in m.core] + [b.mlp.down_proj.weight.flatten() for b in m.core])
    q = torch.cat([b.attn.qkv_proj.weight.flatten() for b in m.core])
    assert abs(o.std().item() - out_std) / out_std < 0.02 and abs(q.std().item() - 0.02) / 0.02 < 0.02
    assert abs(m.embed.weight.std().item() - 0.02) < 0.001
    assert all(torch.equal(p, torch.ones_like(p)) for n, p in m.named_parameters() if n.endswith("norm.weight"))
    assert not any(isinstance(x, torch.nn.Linear) and x.bias is not None for x in m.modules())
    # Tied head: logits are hidden states times the embedding
    tokens = torch.randint(0, VOCAB.size, (2, 7))
    assert torch.allclose(m.logits(tokens), m(tokens) @ m.embed.weight.t())


def test_causality():
    m = tiny_model().eval()
    tokens = torch.randint(0, VOCAB.size, (3, 40))
    with torch.no_grad():
        ref = m.logits(tokens)
        for t in (0, 10, 25, 38):
            changed = tokens.clone()
            changed[:, t + 1:] = torch.randint(0, VOCAB.size, (3, 40 - t - 1))
            out = m.logits(changed)
            assert torch.allclose(out[:, :t + 1], ref[:, :t + 1], atol=1e-5), t
            assert not torch.allclose(out[:, t + 1:], ref[:, t + 1:])


def test_loss_only_on_masked_positions():
    m = tiny_model()
    batch = batch_of(TASK, 4)
    loss, stats = loss_and_stats(m, batch)
    # The same as the cross-entropy of the full logits at the masked positions (targets shifted by one)
    logits = m.logits(batch["tokens"])
    mask = batch["loss_mask"][:, 1:]
    manual = F.cross_entropy(logits[:, :-1][mask], batch["tokens"][:, 1:][mask])
    assert torch.allclose(loss, manual, atol=1e-5)
    assert int(stats["loss_tokens"]) == int(batch["loss_mask"].sum())
    assert int(stats["final_n"]) == int((batch["role"] == 1).sum()) and int(stats["end_n"]) == int((batch["role"] == 5).sum())
    # Unmasking one answer removes exactly its term
    b, t = (batch["role"] == 1).nonzero()[0].tolist()
    fewer = dict(batch, loss_mask=batch["loss_mask"].clone())
    fewer["loss_mask"][b, t] = False
    mask2 = fewer["loss_mask"][:, 1:]
    manual2 = F.cross_entropy(logits[:, :-1][mask2], batch["tokens"][:, 1:][mask2])
    assert torch.allclose(loss_and_stats(m, fewer)[0], manual2, atol=1e-5)
    # Padding never contributes: arbitrary tokens in place of <pad> leave the loss unchanged
    pad = batch["tokens"] == PAD
    assert pad.any()
    noisy = dict(batch, tokens=torch.where(pad, torch.randint(0, VOCAB.size, pad.shape), batch["tokens"]))
    assert torch.allclose(loss_and_stats(m, noisy)[0], loss, atol=1e-6)
    # Nor does the batch it is padded in: each sample alone gives the same per-token terms
    alone = sum(float(loss_and_stats(m, batch_of_one(batch, i))[1]["loss_sum"]) for i in range(4))
    assert abs(alone - float(stats["loss_sum"])) < 1e-4


def batch_of_one(batch, i):
    L = int(batch["length"][i])
    return {k: (v[i:i + 1, :L] if torch.is_tensor(v) and v.dim() == 2 else v) for k, v in batch.items()}


class NextTokenOracle(torch.nn.Module):
    """Stub model: the hidden state at position p is the one-hot of the token at p + offset, and the
    head is the identity. offset 1: correct predictions at pred_pos; offset 0: correct only if read
    at the answer's own position."""

    def __init__(self, offset=1, wrong_rows=()):
        super().__init__()
        self.offset, self.wrong_rows = offset, set(wrong_rows)

    def forward(self, tokens):
        shifted = torch.roll(tokens, -self.offset, dims=1) if self.offset else tokens.clone()
        for r in self.wrong_rows:
            shifted[r] = (shifted[r] + 1) % VOCAB.size
        return 30.0 * F.one_hot(shifted, VOCAB.size).float()

    def head(self, h):
        return h


def loaded(task, n):
    batch = batch_of(task, n)
    batch["num_targets"] = torch.tensor([make_sample(task, 0, i).num_targets for i in range(n)])
    return {"batches": [batch], "config": task, "num_samples": n, "tokens_mean": 0.0}


def test_evaluation_uses_pred_pos():
    data = loaded(MULTI, 6)
    r = evaluate_set(NextTokenOracle(offset=1), data, "cpu")
    assert r["n"] == 12 and r["correct"] == 12 and r["accuracy"] == 1.0  # 2 queries x 6 samples
    assert abs(r["chance"] - 0.25) < 1e-12 and r["loss"] < 1e-3
    # Read at the answer's own position, the same oracle would be right; the evaluation must not be
    r0 = evaluate_set(NextTokenOracle(offset=0), data, "cpu")
    assert r0["correct"] == 0
    # Wrong answers in two samples: 4 of 12 wrong; the interval is Wilson's; per (d, b) counts add up
    r2 = evaluate_set(NextTokenOracle(offset=1, wrong_rows=(1, 4)), data, "cpu")
    assert r2["correct"] == 8 and (r2["ci_low"], r2["ci_high"]) == wilson(8, 12)
    assert sum(s["n"] for s in r2["by_db"].values()) == 12 and sum(s["correct"] for s in r2["by_db"].values()) == 8
    lo, hi = wilson(8, 12)
    assert 0.39 < lo < 0.40 and 0.86 < hi < 0.87  # 8/12: [0.391, 0.862]


def test_learning_rate_schedule():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, steps=110, min_lr_ratio=0.1)
    assert abs(lr_at(0, cfg) - 1e-4) < 1e-12 and abs(lr_at(9, cfg) - 1e-3) < 1e-12
    assert abs(lr_at(10, cfg) - 1e-3) < 1e-12 and abs(lr_at(60, cfg) - 0.55e-3) < 1e-12
    assert abs(lr_at(110, cfg) - 1e-4) < 1e-12


def test_tiny_model_overfits_one_batch():
    m = tiny_model(width=64, num_heads=2, head_dim=32, mlp_hidden=128)
    batch = batch_of(TASK, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(300):
        loss, _ = loss_and_stats(m, batch)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert float(loss) < 0.05, float(loss)


def test_resume_reproduces_uninterrupted_run():
    common = dict(steps=12, warmup_steps=3, batch_size=4, num_workers=0, eval_interval=0, ckpt_interval=0,
                  keep_interval=0, log_interval=4, wandb="off", device="cpu", lr=3e-3)
    model_cfg = ModelConfig(**TINY)
    with tempfile.TemporaryDirectory() as d:
        full = train(TrainConfig(run_name="full", out_dir=d, **common), model_cfg, TASK, record_losses=True)
        first = train(TrainConfig(run_name="split", out_dir=d, **common), model_cfg, TASK, stop_after=6,
                      record_losses=True)
        assert first["step"] == 6 and os.path.exists(os.path.join(d, "split", "ckpt_latest.pt"))
        second = train(TrainConfig(run_name="split", out_dir=d, **common), model_cfg, TASK, record_losses=True)
        assert second["step"] == 12 and len(second["losses"]) == 6
        assert np.allclose(first["losses"], full["losses"][:6], rtol=0, atol=1e-7)
        assert np.allclose(second["losses"], full["losses"][6:], rtol=0, atol=1e-6), (second["losses"], full["losses"][6:])
        # A different model or data stream refuses to resume
        try:
            train(TrainConfig(run_name="split", out_dir=d, **dict(common, steps=20, seed=1)), model_cfg, TASK)
        except ValueError as e:
            assert "different seed" in str(e)
        else:
            raise AssertionError("resuming with another seed must fail")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
