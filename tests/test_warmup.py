"""Supervised window at random depth (arch.halt_warmup_prob): CPU tests.

Run with `python -m pytest tests/` or `python tests/test_warmup.py`.

- p = 0 reproduces the code before the option bit for bit (tests/warmup_reference.json),
  including the random numbers consumed;
- warm-up steps contribute exactly zero to the loss and the gradients;
- no halting during warm-up, at most s + halt_max_steps steps, exploration counted from the
  window's start and never beyond it;
- evaluation does not depend on the setting.
"""
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Carry  # noqa: E402
from models.losses import ACTLossHead, IGNORE_LABEL_ID, METRIC_DENOMINATORS, METRIC_MAX  # noqa: E402
from test_aux_halt_heads import AUX, BATCH, SEQ_LEN, VOCAB, make_batch, make_config, make_model, make_optimizers  # noqa: E402
from test_eval_loop import evaluate_batch, make_eval_batch, make_eval_model  # noqa: E402
import warmup_reference  # noqa: E402

HMS = 4
EVAL_STEPS = 8  # make_eval_batch needs at least 8 steps
WARMUP = dict(halt_warmup_prob=0.5, halt_warmup_min=1, halt_warmup_max=6)


# p = 0 ---------------------------------------------------------------------------------------

def test_off_reproduces_previous_code():
    with open(warmup_reference.PATH) as f:
        ref = json.load(f)
    if ref["torch"] != torch.__version__:
        print(f"SKIP test_off_reproduces_previous_code: reference written with torch {ref['torch']}, running {torch.__version__}")
        return
    for overrides in ({}, {"halt_warmup_prob": 0.0, "halt_warmup_min": 3, "halt_warmup_max": 9}):
        cur = {name: warmup_reference.fingerprints(**o, **overrides) for name, o in warmup_reference.CASES.items()}
        for case, steps in ref["cases"].items():
            assert cur[case]["rng_state"] == steps["rng_state"], f"{case}: different random numbers consumed"
            for i, rec in steps.items():
                if i == "rng_state":
                    continue
                now = cur[case][i]
                assert set(now) - set(rec) == {"metric/steps_total"}, f"{case}: unexpected new keys"
                diff = [k for k, v in rec.items() if now.get(k) != v]
                assert not diff, f"{case}, step {i}: differs for {diff}"


def test_invalid_range():
    for bad in (dict(halt_warmup_prob=1.5), dict(halt_warmup_prob=0.2, halt_warmup_min=0),
                dict(halt_warmup_prob=0.2, halt_warmup_min=5, halt_warmup_max=4)):
        try:
            TinyRecursiveReasoningModel_ACTV1(make_config(**bad))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    # The range is not checked when the option is off
    TinyRecursiveReasoningModel_ACTV1(make_config(halt_warmup_prob=0.0, halt_warmup_min=0))
    # halt_warmup_max defaults to halt_max_steps
    assert TinyRecursiveReasoningModel_ACTV1(make_config(halt_warmup_prob=0.2)).warmup_max == make_config()["halt_max_steps"]


# Loss masking (loss head with a stub model) ----------------------------------------------------

class StubModel(torch.nn.Module):
    """Returns a fixed carry and fixed outputs; the logits are leaves, to read their gradients."""

    def __init__(self, carry, outputs, prob):
        super().__init__()
        self.config = SimpleNamespace(halt_warmup_prob=prob)
        self.carry, self.outputs = carry, outputs

    def forward(self, carry=None, batch=None):
        return self.carry, dict(self.outputs, aux_halt_logits=dict(self.outputs["aux_halt_logits"]))


def stub_inputs(rows=None, warmup_zero=False):
    g = torch.Generator().manual_seed(5)
    B = 8
    labels = torch.randint(1, VOCAB, (B, SEQ_LEN), generator=g, dtype=torch.int32)
    logits = torch.randn(B, SEQ_LEN, VOCAB, generator=g)
    logits[1] = 0
    logits[1].scatter_(-1, labels[1].long().unsqueeze(-1), 5.0)  # row 1 solved
    labels[2, :5] = IGNORE_LABEL_ID
    steps = torch.tensor([1, 3, 5, 2, 7, 4, 6, 9], dtype=torch.int32)
    warmup = torch.tensor([0, 3, 2, 4, 0, 5, 6, 3], dtype=torch.int32)  # rows 1, 3, 5, 6 in warm-up
    halted = torch.tensor([False, False, False, False, True, False, False, True])
    tensors = dict(logits=logits, q_halt_logits=torch.randn(B, generator=g), q_continue_logits=torch.randn(B, generator=g),
                   target_q_continue=torch.rand(B, generator=g), aux=torch.randn(B, generator=g))
    if rows is not None:
        labels, steps, warmup, halted = labels[rows], steps[rows], warmup[rows], halted[rows]
        tensors = {k: v[rows] for k, v in tensors.items()}
    if warmup_zero:
        warmup = torch.zeros_like(warmup)
    leaves = {k: v.clone().requires_grad_(k != "target_q_continue") for k, v in tensors.items()}
    carry = TinyRecursiveReasoningModel_ACTV1Carry(inner_carry=None, steps=steps, halted=halted,
                                                   current_data={"labels": labels}, warmup=warmup)
    outputs = {k: leaves[k] for k in ("logits", "q_halt_logits", "q_continue_logits", "target_q_continue")}
    outputs["aux_halt_logits"] = {"x": leaves["aux"]}
    return carry, outputs, leaves


def run_head(carry, outputs, prob):
    head = ACTLossHead(StubModel(carry, outputs, prob), loss_type="stablemax_cross_entropy")
    head.train()
    _, loss, metrics, _, _ = head(return_keys=[])
    loss.backward()
    return loss, metrics


def test_warmup_steps_contribute_exactly_zero():
    carry, outputs, leaves = stub_inputs()
    loss, metrics = run_head(carry, outputs, prob=0.5)
    supervised = carry.steps > carry.warmup
    assert supervised.tolist() == [True, False, True, False, True, False, False, True]

    # Gradients of warm-up rows: exactly zero, for every loss term
    for k, leaf in leaves.items():
        if leaf.requires_grad:
            assert torch.count_nonzero(leaf.grad[~supervised]) == 0, f"warm-up rows get gradient through {k}"
            assert torch.count_nonzero(leaf.grad[supervised]) > 0, f"no gradient through {k}"

    # Same loss, gradients and loss metrics as a batch made of the supervised sequences only
    rows = supervised.nonzero().squeeze(-1)
    sub_carry, sub_outputs, sub_leaves = stub_inputs(rows=rows, warmup_zero=True)
    sub_loss, sub_metrics = run_head(sub_carry, sub_outputs, prob=0.0)
    torch.testing.assert_close(loss, sub_loss, rtol=1e-12, atol=0)
    for k, leaf in leaves.items():
        if leaf.requires_grad:
            assert torch.equal(leaf.grad[rows], sub_leaves[k].grad), f"gradient through {k} differs"
    for k in ("lm_loss", "q_halt_loss", "q_continue_loss", "aux_x_q_halt_loss"):
        torch.testing.assert_close(metrics[k], sub_metrics[k], rtol=1e-6, atol=0)


def test_warmup_metrics():
    carry, outputs, _ = stub_inputs()
    _, m = run_head(carry, outputs, prob=0.5)
    supervised = carry.steps > carry.warmup
    assert m["batch_count"] == 8 and m["supervised_share"] == 4 and m["supervised_count"] == 4
    assert m["supervised_depth_mean"] == carry.steps[supervised].sum()  # 1 + 5 + 7 + 9
    assert m["supervised_depth_max"] == 9
    # Last warm-up step (steps == warmup > 0): rows 1 (solved) and 6
    assert m["warmup_end_count"] == 2 and m["warmup_solved_share"] == 1
    # Halted rows 4 (no warm-up, 7 steps) and 7 (3 warm-up + 6 window steps)
    assert m["count"] == 2 and m["steps"] == 7 + 6 and m["steps_total"] == 7 + 9
    assert set(METRIC_DENOMINATORS) | set(METRIC_DENOMINATORS.values()) | METRIC_MAX <= set(m)

    # Off: none of the window metrics
    carry, outputs, _ = stub_inputs(warmup_zero=True)
    _, m = run_head(carry, outputs, prob=0.0)
    assert not (set(METRIC_DENOMINATORS) | set(METRIC_DENOMINATORS.values()) | METRIC_MAX) & set(m)


def test_postprocess_train_metrics():
    try:
        import pretrain
    except ImportError as e:
        print(f"SKIP test_postprocess_train_metrics: {e}")
        return
    reduced = {"count": 4, "steps": 10, "lm_loss": 8.0, "batch_count": 16, "supervised_share": 12,
               "supervised_count": 12, "supervised_depth_mean": 60, "warmup_end_count": 0, "warmup_solved_share": 0}
    out = pretrain.postprocess_train_metrics(reduced, global_batch_size=16)
    assert out == {"train/count": 1, "train/steps": 2.5, "train/lm_loss": 0.5,
                   "train/supervised_share": 0.75, "train/supervised_depth_mean": 5.0}


def test_whole_batch_in_warmup_has_zero_gradient():
    model = make_model(AUX, halt_max_steps=HMS, halt_warmup_prob=1.0, halt_warmup_min=2, halt_warmup_max=5)
    model.train()
    batch = make_batch()
    carry = model.initial_carry(batch)
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    assert (carry.warmup >= 2).all() and (carry.steps == 1).all()
    assert loss.item() == 0.0 and metrics["supervised_share"] == 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None] + [model.model.puzzle_emb.local_weights.grad]
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)


# Halting -------------------------------------------------------------------------------------

def force_halt_logit(model, value):
    q_head = model.model.inner.q_head
    with torch.no_grad():
        q_head.weight.zero_()
        q_head.bias.fill_(value)


def trace(model, num_steps=120, seed=3):
    """Per step: (warmup, steps, halted) of the carry after the step."""
    model.train()
    batch = make_batch()
    torch.manual_seed(seed)
    carry = model.initial_carry(batch)
    out = []
    with torch.no_grad():
        for _ in range(num_steps):
            carry, _, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
            out.append((carry.warmup.clone(), carry.steps.clone(), carry.halted.clone()))
    return out


def halts(records):
    """(warmup, total steps) of every halting."""
    return [(int(w[i]), int(s[i])) for w, s, h in records for i in h.nonzero().squeeze(-1).tolist()]


def test_halting_respects_the_window():
    for halt_logit, exploration in [(10.0, 0.0), (-10.0, 0.0), (10.0, 1.0), (-10.0, 1.0), (0.0, 0.5)]:
        check_window(halt_logit, exploration)


def check_window(halt_logit, exploration):
    model = make_model(AUX, halt_max_steps=HMS, halt_exploration_prob=exploration, **WARMUP)
    force_halt_logit(model, halt_logit)
    records = trace(model)

    prev_halted = torch.ones(BATCH, dtype=torch.bool)
    prev_warmup = None
    for warmup, steps, halted in records:
        window = steps - warmup
        # Warm-up lengths: 0 or in the range, fixed for the life of a sequence
        assert ((warmup == 0) | ((warmup >= WARMUP["halt_warmup_min"]) & (warmup <= WARMUP["halt_warmup_max"]))).all()
        if prev_warmup is not None:
            assert torch.equal(warmup[~prev_halted], prev_warmup[~prev_halted])
        assert (window[halted] >= 1).all(), "halted during warm-up"
        assert (window <= HMS).all(), "ran beyond s + halt_max_steps"
        prev_halted, prev_warmup = halted, warmup

    window_at_halt = [s - w for w, s in halts(records)]
    assert len(window_at_halt) > 20 and any(w > 0 for w, _ in halts(records)) and any(w == 0 for w, _ in halts(records))
    if halt_logit < 0:
        # Only the step limit halts: exactly s + halt_max_steps steps
        assert set(window_at_halt) == {HMS}
    elif exploration == 0.0:
        # The halt head halts at the window's first step
        assert set(window_at_halt) == {1}
    elif exploration == 1.0:
        # Exploration minimum counted from the window's start, in [2, halt_max_steps]
        assert set(window_at_halt) <= set(range(2, HMS + 1)) and len(set(window_at_halt)) > 1


def test_exploration_counted_from_window_start():
    # Exploration with the largest minimum and a halt head that always halts: every sequence
    # runs exactly s + halt_max_steps steps, never more
    model = make_model(AUX, halt_max_steps=HMS, halt_exploration_prob=1.0, **WARMUP)
    force_halt_logit(model, 10.0)
    torch.manual_seed(0)
    original = torch.randint_like

    def largest(t, low=0, high=None, **kw):
        return torch.full_like(t, high - 1)

    torch.randint_like = largest
    try:
        records = trace(model, num_steps=60)
    finally:
        torch.randint_like = original
    assert {s - w for w, s in halts(records)} == {HMS}
    assert any(w > 0 for w, _ in halts(records))


# Evaluation ----------------------------------------------------------------------------------

def same(a, b, what):
    """Bit-equal (NaN equal to NaN)."""
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True, msg=f"{what} differs")


def test_evaluation_unchanged():
    base = make_eval_model(eval_halt_max_steps=EVAL_STEPS)
    batch = make_eval_batch(base)
    ref = evaluate_batch(base, batch, capture_steps=True)
    for overrides in (dict(halt_warmup_prob=1.0, halt_warmup_min=2, halt_warmup_max=6), WARMUP):
        model = make_eval_model(eval_halt_max_steps=EVAL_STEPS, **overrides)
        out = evaluate_batch(model, batch, capture_steps=True)
        assert out["inference_steps"] == ref["inference_steps"] == EVAL_STEPS
        assert ref["metrics"].keys() == out["metrics"].keys()
        for k in ref["metrics"]:
            same(ref["metrics"][k], out["metrics"][k], k)
        for k in ref["step_records"]:
            for a, b in zip(ref["step_records"][k], out["step_records"][k]):
                same(a, b, k)
        for k in ref["preds"]:
            same(ref["preds"][k], out["preds"][k], k)

    # A carry left by training with warm-up does not leak into an evaluation step
    model = make_eval_model(eval_halt_max_steps=EVAL_STEPS, **WARMUP)
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry.warmup = torch.full_like(carry.warmup, 3)
        carry, _, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
    assert (carry.warmup == 0).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
