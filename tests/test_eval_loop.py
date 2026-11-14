"""Tests for evaluation-time step limit, frozen-mask halting and the extended per-step capture.

Run with `python -m pytest tests/` or `python tests/test_eval_loop.py`.
"""
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from analysis import eval_steps as es  # noqa: E402
from utils.eval_loop import FrozenOutputs, run_eval_batch, halt_logit_key, grid_hash, grid_hash_coefficients, cell_count_dtype  # noqa: E402
from test_aux_halt_heads import make_model, make_batch, AUX, BATCH  # noqa: E402

COEF_0 = -539115568831144627  # grid_hash_coefficients(1)[0]

TRAIN_STEPS, EVAL_STEPS = 4, 8


# ----------------------------------------------------------------------------- frozen mask


def _run_frozen(logits, threshold):
    """logits: [P, S]. Outputs at step s are the tensor filled with s (so frozen values = halting step)."""
    P, S = logits.shape
    frozen = FrozenOutputs(threshold)
    for s in range(S):
        outputs = {"value": torch.full((P, ), float(s + 1)), "grid": torch.full((P, 3), s + 1)}
        frozen.update(s + 1, logits[:, s], outputs, is_last=(s == S - 1))
    return frozen


def test_frozen_mask_synthetic():
    logits = torch.tensor([
        [-1.0, -1.0, -1.0, -1.0],   # never crosses           -> last step (4)
        [ 2.0, -1.0, -1.0, -1.0],   # crosses at step 1       -> 1
        [-1.0,  0.5, -3.0,  9.0],   # crosses, drops, crosses -> 2 (stays frozen)
        [-1.0, -1.0,  0.0, -1.0],   # exactly at the threshold -> 3 (>=)
        [-1.0, -1.0, -1.0,  3.0],   # crosses at the last step -> 4
    ])
    frozen = _run_frozen(logits, 0.0)
    expected = torch.tensor([4, 1, 2, 3, 4], dtype=torch.int32)
    assert torch.equal(frozen.halt_step, expected)
    assert frozen.halt_step.dtype == torch.int32
    assert frozen.halted.all()
    # Outputs are the ones of the halting step, for 1-D and 2-D outputs
    assert torch.equal(frozen.outputs["value"], expected.float())
    assert torch.equal(frozen.outputs["grid"], expected.long().view(-1, 1).expand(-1, 3))

    # Extreme thresholds
    assert torch.equal(_run_frozen(logits, float("-inf")).halt_step, torch.ones(5, dtype=torch.int32))
    assert torch.equal(_run_frozen(logits, float("inf")).halt_step, torch.full((5, ), 4, dtype=torch.int32))


def test_frozen_mask_matches_offline_rule():
    g = torch.Generator().manual_seed(0)
    logits = torch.randint(-4, 5, (200, 7), generator=g).float() / 2  # many ties
    for t in [-1.0, 0.0, 0.5, 1.25]:
        frozen = _run_frozen(logits, t)
        assert np.array_equal(frozen.halt_step.numpy(), es.halting_steps(logits.numpy(), t))


# ----------------------------------------------------------------------------- small evaluation run


def make_eval_model(seed=0, **overrides):
    """Tiny model whose halting heads and predictions vary (the default init gives constant logits)."""
    model = make_model(AUX, seed=seed, **overrides)
    g = torch.Generator().manual_seed(seed + 100)
    inner = model.model.inner
    with torch.no_grad():
        inner.q_head.weight.copy_(torch.randn(inner.q_head.weight.shape, generator=g) * 0.3)
        inner.q_head.bias.zero_()
        head = inner.aux_halt_heads["detached_linear"].linear
        head.weight.copy_(torch.randn(head.weight.shape, generator=g) * 0.3)
        head.bias.zero_()
    model.eval()
    return model


def per_step_preds(model, batch):
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        out = []
        while True:
            carry, _, _, preds, done = model(carry=carry, batch=batch, return_keys={"preds"})
            out.append(preds["preds"])
            if done:
                return torch.stack(out, dim=1)  # [B, S, L]


def make_eval_batch(model):
    """Labels are the model's own prediction at a puzzle-specific step, so correctness varies by step."""
    batch = make_batch()
    preds = per_step_preds(model, batch)
    target_step = torch.tensor([0, 1, 2, 3, 5, 7])[:BATCH]
    batch["labels"] = preds[torch.arange(BATCH), target_step].to(batch["labels"].dtype)
    return batch


def evaluate_batch(model, batch, **kwargs):
    with torch.inference_mode():
        return run_eval_batch(model, model.initial_carry(batch), batch, set(), **kwargs)


def save_capture(path, result, halt=None):
    """Same layout as pretrain.evaluate writes."""
    capture = {k: torch.stack(v, dim=1).cpu() for k, v in result["step_records"].items()}
    P, S = capture["seq_is_correct"].shape
    capture.update(puzzle_identifiers=torch.zeros(P, dtype=torch.int32), example_index=torch.arange(P),
                   set_id=torch.zeros(P, dtype=torch.int32), num_steps=torch.full((P, ), S, dtype=torch.int32),
                   set_names=["all"])
    if halt is not None:
        capture.update(halt_step=result["halt_step"].cpu(), eval_halt_head=halt[0], eval_halt_threshold=float(halt[1]))
    torch.save(capture, path)


def test_eval_step_limit():
    batch = make_batch()
    # Default: evaluation runs halt_max_steps
    assert evaluate_batch(make_eval_model(), batch)["inference_steps"] == TRAIN_STEPS
    # More (and fewer) steps than in training
    for limit in [EVAL_STEPS, 2]:
        model = make_eval_model(eval_halt_max_steps=limit)
        assert evaluate_batch(model, batch, expected_steps=limit)["inference_steps"] == limit
    # A model that ignores the option is reported
    try:
        evaluate_batch(make_eval_model(), batch, expected_steps=EVAL_STEPS)
        raise AssertionError("expected an error")
    except RuntimeError:
        pass

    # Training keeps using halt_max_steps
    model = make_model(AUX, eval_halt_max_steps=EVAL_STEPS, halt_exploration_prob=0.0)
    model.train()
    carry = model.initial_carry(batch)
    for step in range(1, TRAIN_STEPS + 1):
        carry, _, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
        assert bool(carry.halted.all()) == (step == TRAIN_STEPS)  # q_head starts at -5: no early halt

    # The first steps of a longer evaluation are the shorter evaluation
    short = evaluate_batch(make_eval_model(), batch, capture_steps=True)["step_records"]
    long = evaluate_batch(make_eval_model(eval_halt_max_steps=EVAL_STEPS), batch, capture_steps=True)["step_records"]
    for k in short:
        a, b = torch.stack(short[k], 1), torch.stack(long[k], 1)[:, :TRAIN_STEPS]
        assert torch.equal(a, b) or (a.is_floating_point() and torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(), b.nan_to_num())), k


def test_halting_off_is_unchanged_and_inf_threshold_matches():
    model = make_eval_model(eval_halt_max_steps=EVAL_STEPS)
    batch = make_eval_batch(model)
    plain = evaluate_batch(model, batch)
    with_capture = evaluate_batch(model, batch, capture_steps=True)
    never = evaluate_batch(model, batch, halt_head="q_head", halt_threshold=float("inf"))

    assert "halted_early" not in plain["metrics"]
    for k, v in plain["metrics"].items():
        assert torch.equal(v, with_capture["metrics"][k]), k  # capture does not change metrics
        assert torch.allclose(v.double(), never["metrics"][k].double(), rtol=1e-6, atol=0), k
    assert set(never["metrics"]) == set(plain["metrics"]) | {"halted_early"}
    assert int(never["metrics"]["halted_early"]) == 0
    assert int(never["metrics"]["steps"]) == EVAL_STEPS * BATCH
    assert torch.equal(never["preds"]["preds"], evaluate_batch(model, batch, capture_steps=True)["preds"]["preds"])


def test_in_loop_halting_matches_offline_replay():
    model = make_eval_model(eval_halt_max_steps=EVAL_STEPS)
    batch = make_eval_batch(model)
    seen_early, seen_correct = False, False
    for head in ["q_head", "detached_linear"]:
        raw = evaluate_batch(model, batch, capture_steps=True)["step_records"]
        logits = torch.stack(raw[halt_logit_key(head)], 1)
        for t in [float("-inf"), float(logits.quantile(0.3)), float(logits.median()), float(logits.quantile(0.8)), 0.0]:
            result = evaluate_batch(model, batch, capture_steps=True, halt_head=head, halt_threshold=t)
            m = result["metrics"]
            count = int(m["count"])
            assert count == BATCH

            with tempfile.TemporaryDirectory() as d:
                save_capture(os.path.join(d, "step_1_eval_steps.0"), result, halt=(head, t))
                data, checks = es.load(d, 1)
            correct = data["seq_is_correct"]
            assert correct.shape == (BATCH, EVAL_STEPS)  # all steps captured, whatever the halting

            # Offline replay of the capture at the same threshold
            replay = es.HaltCurve(data["logits"][head], correct)(float(np.float32(t)))
            assert np.isclose(replay["accuracy"][0], int(m["exact_accuracy"]) / count)
            assert np.isclose(replay["avg_steps"][0], int(m["steps"]) / count)
            assert np.isclose(replay["frac_halted_early"][0], int(m["halted_early"]) / count)
            assert np.array_equal(data["halt_step"], es.halting_steps(data["logits"][head], float(np.float32(t))))
            assert checks["in_loop_halting"]["matches_replay"]

            # Frozen predictions are the ones of the halting step
            all_preds = per_step_preds(model, batch)
            halt_step = result["halt_step"].long()
            assert torch.equal(result["preds"]["preds"], all_preds[torch.arange(BATCH), halt_step - 1])
            assert torch.equal(result["preds"][halt_logit_key(head)], logits[torch.arange(BATCH), halt_step - 1])

            seen_early |= 0 < int(m["halted_early"]) < count
            seen_correct |= 0 < int(m["exact_accuracy"]) < count
    assert seen_early and seen_correct  # the test exercised mixed outcomes


def test_capture_change_records():
    model = make_eval_model(eval_halt_max_steps=EVAL_STEPS)
    batch = make_eval_batch(model)
    rec = {k: torch.stack(v, 1) for k, v in evaluate_batch(model, batch, capture_steps=True)["step_records"].items()}
    all_preds = per_step_preds(model, batch)

    assert rec["preds_changed"].dtype == torch.bool and not rec["preds_changed"][:, 0].any()
    expected = (all_preds[:, 1:] != all_preds[:, :-1]).any(-1)
    assert torch.equal(rec["preds_changed"][:, 1:], expected)
    for k in ["z_H_rel_change", "z_L_rel_change"]:
        assert rec[k].shape == (BATCH, EVAL_STEPS) and rec[k].dtype == torch.float16
        assert rec[k][:, 0].isnan().all() and (rec[k][:, 1:] > 0).all() and rec[k][:, 1:].isfinite().all()

    # Reference value for z_H at step 2, from the carries
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry, *_ = model(carry=carry, batch=batch, return_keys=set())
        z1 = carry.inner_carry.z_H.float()
        carry, *_ = model(carry=carry, batch=batch, return_keys=set())
        z2 = carry.inner_carry.z_H.float()
    ref = (z2 - z1).flatten(1).norm(dim=1) / z1.flatten(1).norm(dim=1)
    assert torch.allclose(rec["z_H_rel_change"][:, 1].float(), ref, rtol=2e-3)  # stored as float16

    # Grid hash: equal at consecutive steps exactly when the grid did not change, and a pure
    # function of the grid
    assert rec["pred_hash"].dtype == torch.int64 and rec["pred_hash"].shape == (BATCH, EVAL_STEPS)
    assert torch.equal(rec["pred_hash"][:, 1:] != rec["pred_hash"][:, :-1], expected)
    assert torch.equal(rec["pred_hash"], torch.stack([grid_hash(all_preds[:, s]) for s in range(EVAL_STEPS)], 1))


def test_capture_cell_counts():
    model = make_eval_model(eval_halt_max_steps=EVAL_STEPS)
    batch = make_eval_batch(model)
    rec = {k: torch.stack(v, 1) for k, v in evaluate_batch(model, batch, capture_steps=True)["step_records"].items()}
    all_preds = per_step_preds(model, batch)  # [B, S, L]

    for k in ("wrong_cells", "cells_changed"):
        assert rec[k].dtype == torch.uint8 and rec[k].shape == (BATCH, EVAL_STEPS)
    # Wrong cells: direct per-cell count over the labelled cells
    labels = batch["labels"][:, None, :]
    direct = ((all_preds != labels) & (labels != -100)).sum(-1)
    assert torch.equal(rec["wrong_cells"].long(), direct)
    # 0 wrong cells exactly when the puzzle is correct, and the capture has both outcomes
    assert torch.equal(rec["wrong_cells"] == 0, rec["seq_is_correct"])
    assert rec["seq_is_correct"].any() and (~rec["seq_is_correct"]).any()
    # Changed cells: direct count, 0 at step 1, > 0 exactly when the grid changed
    assert torch.equal(rec["cells_changed"][:, 1:].long(), (all_preds[:, 1:] != all_preds[:, :-1]).sum(-1))
    assert not rec["cells_changed"][:, 0].any()
    assert torch.equal(rec["cells_changed"] > 0, rec["preds_changed"])

    # Ignored cells do not count as wrong
    masked = dict(batch, labels=batch["labels"].clone())
    masked["labels"][:, :5] = -100
    rec2 = {k: torch.stack(v, 1) for k, v in evaluate_batch(model, masked, capture_steps=True)["step_records"].items()}
    direct2 = ((all_preds[..., 5:] != masked["labels"][:, None, 5:])).sum(-1)
    assert torch.equal(rec2["wrong_cells"].long(), direct2)
    assert torch.equal(rec2["wrong_cells"] == 0, rec2["seq_is_correct"])

    assert [cell_count_dtype(n) for n in (81, 255, 256, 900, 32767, 40000)] == \
        [torch.uint8, torch.uint8, torch.int16, torch.int16, torch.int16, torch.int32]


def test_grid_hash():
    coef = grid_hash_coefficients(81)
    assert coef == grid_hash_coefficients(81) and coef[:5] == grid_hash_coefficients(5)
    assert all(c % 2 == 1 and -2 ** 63 <= c < 2 ** 63 for c in coef) and len(set(coef)) == 81
    assert coef[0] == COEF_0  # fixed constant: hashes are comparable across runs and versions

    state = torch.get_rng_state()
    g = torch.Generator().manual_seed(0)
    grids = torch.randint(0, 11, (500, 81), generator=g)
    h = grid_hash(grids)
    assert torch.equal(torch.get_rng_state(), state)  # the global generator is untouched
    assert h.dtype == torch.int64 and h.unique().numel() == 500
    assert torch.equal(h, grid_hash(grids.clone().to(torch.int32)))
    # Wraparound arithmetic, checked against exact Python integers
    exact = sum((int(t) + 1) * c for t, c in zip(grids[0], coef)) % 2 ** 64
    assert int(h[0]) % 2 ** 64 == exact
    # One changed cell, a swap of two cells, and an all-zero grid all change the hash
    one = grids.clone(); one[:, 40] = (one[:, 40] + 1) % 11
    swap = grids.clone(); swap[:, [3, 4]] = swap[:, [4, 3]]
    assert (grid_hash(one) != h).all()
    assert torch.equal(grid_hash(swap) != h, grids[:, 3] != grids[:, 4])
    assert int(grid_hash(torch.zeros(1, 81, dtype=torch.long))) != 0


def test_unknown_halt_head():
    model = make_eval_model()
    try:
        evaluate_batch(model, make_batch(), halt_head="nope")
        raise AssertionError("expected an error")
    except ValueError as e:
        assert "detached_linear" in str(e)


def batch_independence(**overrides):
    """Each puzzle's per-step results do not depend on which other puzzles share its batch.

    Returns whether the results were also bit-identical."""
    model = make_eval_model(eval_halt_max_steps=EVAL_STEPS, **overrides)
    batch = make_eval_batch(model)
    full = {k: torch.stack(v, 1) for k, v in evaluate_batch(model, batch, capture_steps=True)["step_records"].items()}

    perm = torch.tensor([3, 0, 5])  # a different batch: subset, other order, other size
    sub = {k: v[perm] for k, v in batch.items()}
    part = {k: torch.stack(v, 1) for k, v in evaluate_batch(model, sub, capture_steps=True)["step_records"].items()}

    bit_identical = True
    for k in full:
        a, b = full[k][perm], part[k]
        if a.is_floating_point():
            a, b = a.nan_to_num(), b.nan_to_num()
            assert torch.allclose(a, b, rtol=1e-2, atol=1e-2), k
        bit_identical &= torch.equal(a, b)
    assert torch.equal(full["seq_is_correct"][perm], part["seq_is_correct"])
    return bit_identical


MLP_T = dict(mlp_t=True, pos_encodings="none")  # the Sudoku configuration


def test_puzzles_in_a_batch_are_independent():
    batch_independence()
    batch_independence(**MLP_T)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"Bit-identical across batch compositions: attention {batch_independence()}, mlp_t {batch_independence(**MLP_T)}")
    print(f"{len(tests)} tests passed")
