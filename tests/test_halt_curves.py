"""Tests for analysis/halt_curves.py on synthetic captures with known answers.

Run with `python -m pytest tests/` or `python tests/test_halt_curves.py`.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from analysis import halt_curves as hc  # noqa: E402


def brute_force(logits, correct, t):
    """Direct application of the rule: halt at the first step with logit >= t, else the last."""
    P, S = correct.shape
    steps, acc, early = [], [], []
    for i in range(P):
        crossing = np.flatnonzero(logits[i] >= t)
        h = crossing[0] + 1 if len(crossing) else S
        steps.append(h)
        acc.append(correct[i, h - 1])
        early.append(h < S)
    return np.mean(steps), np.mean(acc), np.mean(early)


# Hand-made correctness patterns (S = 4) with known oracle and flip statistics
PATTERNS = np.array([
    [0, 0, 0, 0],  # never correct                     oracle: wrong, 4 steps
    [1, 1, 1, 1],  # always correct                    oracle: 1 step
    [0, 1, 0, 0],  # correct, then wrong at the end    oracle: 2 steps   (lost at last)
    [1, 0, 1, 1],  # correct, wrong, correct           oracle: 1 step    (c-w-c)
    [0, 1, 0, 1],  # correct, wrong, correct           oracle: 2 steps   (c-w-c)
    [1, 0, 0, 1],  # correct, wrong, correct           oracle: 1 step    (c-w-c)
    [0, 0, 1, 0],  # correct only at step 3            oracle: 3 steps   (lost at last)
], dtype=bool)


def test_oracle_and_flips():
    orc = hc.oracle(PATTERNS)
    assert orc["accuracy"] == 6 / 7
    assert orc["avg_steps"] == (4 + 1 + 2 + 1 + 2 + 1 + 3) / 7
    f = hc.flips(PATTERNS)
    assert f["correct_at_some_step"] == 6
    assert f["correct_at_some_step_but_wrong_at_last"] == 2
    assert f["correct_then_wrong_then_correct"] == 3
    assert hc.fixed_step_accuracy(PATTERNS) == [3 / 7, 3 / 7, 3 / 7, 4 / 7]


def test_curve_known_by_construction():
    # Logits cross 0 exactly at the oracle step, and stay below -10 before it:
    # t = 0 reproduces the oracle; t = -inf is step 1; t = +inf is the last step.
    S = PATTERNS.shape[1]
    logits = np.full(PATTERNS.shape, -20.0, dtype=np.float32)
    for i, row in enumerate(PATTERNS):
        if row.any():
            logits[i, row.argmax():] = 5.0
    curve = hc.HaltCurve(logits, PATTERNS)
    pts = curve([-np.inf, 0.0, np.inf])
    orc = hc.oracle(PATTERNS)
    fixed = hc.fixed_step_accuracy(PATTERNS)
    np.testing.assert_allclose(pts["accuracy"], [fixed[0], orc["accuracy"], fixed[-1]])
    np.testing.assert_allclose(pts["avg_steps"], [1, orc["avg_steps"], S])
    np.testing.assert_allclose(pts["frac_halted_early"], [1, 6 / 7, 0])  # all but the never-correct row


def test_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    P, S = 300, 6
    # Coarse values to create many ties (like bf16 logits), including non-monotone sequences
    logits = rng.integers(-6, 7, size=(P, S)).astype(np.float32) / 2
    correct = rng.random((P, S)) < np.linspace(0.2, 0.7, S)
    curve = hc.HaltCurve(logits, correct)
    thresholds = np.concatenate([[-np.inf, np.inf], np.unique(logits), np.unique(logits) + 0.25])
    pts = curve(thresholds)
    for j, t in enumerate(thresholds):
        steps, acc, early = brute_force(logits, correct, t)
        assert np.isclose(pts["avg_steps"][j], steps), t
        assert np.isclose(pts["accuracy"][j], acc), t
        assert np.isclose(pts["frac_halted_early"][j], early), t
    # halting_steps agrees with the curve
    for t in [-1.0, 0.0, 1.5]:
        assert np.isclose(hc.halting_steps(logits, t).mean(), curve(t)["avg_steps"][0])


def test_selection_respects_tolerance():
    rng = np.random.default_rng(1)
    P, S = 2000, 8
    logits = rng.normal(size=(P, S)).cumsum(axis=1).astype(np.float32)
    correct = rng.random((P, S)) < 1 / (1 + np.exp(-logits))
    curve = hc.HaltCurve(logits, correct)
    full = curve(np.inf)["accuracy"][0]
    levels = curve.level_thresholds()
    all_pts = curve(levels)
    prev_steps = np.inf
    for tol in [0.0, 0.001, 0.01, 0.05]:
        chosen = hc.select_threshold(curve, tol)
        assert chosen["accuracy"] >= full - tol - 1e-12
        assert np.isclose(curve(chosen["threshold"])["avg_steps"][0], chosen["avg_steps"])
        # No feasible threshold uses fewer steps
        feasible = all_pts["accuracy"] >= full - tol - 1e-12
        assert np.isclose(all_pts["avg_steps"][feasible].min(), chosen["avg_steps"])
        assert chosen["avg_steps"] <= prev_steps  # larger tolerance never needs more steps
        prev_steps = chosen["avg_steps"]


def test_bootstrap_interval():
    rng = np.random.default_rng(2)
    P, S = 5000, 5
    logits = rng.normal(size=(P, S)).astype(np.float32)
    correct = rng.random((P, S)) < 0.6
    res = hc.evaluate_with_ci(logits, correct, 0.5, 2000, np.random.default_rng(0))
    steps, acc, early = brute_force(logits, correct, 0.5)
    assert np.isclose(res["accuracy"], acc) and np.isclose(res["avg_steps"], steps)
    assert np.isclose(res["accuracy_loss"], correct[:, -1].mean() - acc)
    for k in ["accuracy", "avg_steps", "accuracy_loss"]:
        lo, hi = res[f"{k}_ci"]
        assert lo <= res[k] <= hi and hi > lo
    # Binomial standard error check for accuracy: 95% interval ~ +-1.96 sqrt(p(1-p)/P)
    half = (res["accuracy_ci"][1] - res["accuracy_ci"][0]) / 2
    assert abs(half - 1.96 * np.sqrt(acc * (1 - acc) / P)) < 0.003


def _write_rank(path, logits_q, logits_aux, correct, example_index, set_id, set_names):
    P, S = correct.shape
    torch.save({
        "q_halt_logits": torch.from_numpy(logits_q),
        "aux_halt_logits/probe": torch.from_numpy(logits_aux),
        "seq_is_correct": torch.from_numpy(correct),
        "puzzle_identifiers": torch.zeros(P, dtype=torch.int32),
        "example_index": torch.from_numpy(example_index.astype(np.int64)),
        "set_id": torch.from_numpy(set_id.astype(np.int32)),
        "num_steps": torch.full((P,), S, dtype=torch.int32),
        "set_names": set_names,
    }, path)


def test_load_deduplicates_and_reports_missing():
    rng = np.random.default_rng(3)
    S = 4
    with tempfile.TemporaryDirectory() as d:
        # Set "a": examples 0..9 split over two ranks, with 2..3 duplicated (identical values);
        # set "b": examples 0..4 without example 3 (missing).
        idx_a = np.arange(10)
        idx_b = np.array([0, 1, 2, 4])
        logits = rng.normal(size=(14, S)).astype(np.float32)
        correct = rng.random((14, S)) < 0.5
        ex = np.concatenate([idx_a, idx_b])
        sid = np.concatenate([np.zeros(10), np.ones(4)])
        r0 = np.concatenate([np.arange(0, 4), np.arange(10, 12)])
        r1 = np.concatenate([np.arange(2, 10), np.arange(12, 14)])  # rows 2, 3 repeated
        for rank, rows in enumerate([r0, r1]):
            _write_rank(os.path.join(d, f"step_7_eval_steps.{rank}"), logits[rows], logits[rows] * 2,
                        correct[rows], ex[rows], sid[rows], ["a", "b"])
        data, checks = hc.load(d, 7)

        assert checks["rows_loaded"] == 16 and checks["unique_puzzles"] == 14
        assert checks["duplicate_rows_dropped"] == 2 and checks["duplicates_with_differing_values"] == 0
        assert checks["per_set"]["a"]["missing"] == 0
        assert checks["per_set"]["b"]["missing"] == 1  # inferred from max index 4
        assert list(data["logits"]) == ["q_head", "probe"]
        np.testing.assert_array_equal(data["logits"]["q_head"], logits)  # sorted back by (set, example)
        np.testing.assert_array_equal(data["seq_is_correct"], correct)

        # Differing duplicate is reported
        _write_rank(os.path.join(d, "step_8_eval_steps.0"), logits[:3], logits[:3], correct[:3],
                    np.array([0, 1, 1]), np.zeros(3), ["a"])
        _, checks = hc.load(d, 8)
        assert checks["duplicates_with_differing_values"] == 1


def test_end_to_end_outputs():
    rng = np.random.default_rng(4)
    P, S = 1000, 6
    logits = rng.normal(size=(P, S)).cumsum(axis=1).astype(np.float32)
    correct = rng.random((P, S)) < 1 / (1 + np.exp(-logits))
    with tempfile.TemporaryDirectory() as d:
        sid = (np.arange(P) % 2)
        ex = np.arange(P) // 2
        half = P // 2
        for rank, rows in enumerate([np.arange(half), np.arange(half, P)]):
            _write_rank(os.path.join(d, f"step_3_eval_steps.{rank}"), logits[rows], logits[rows] + 1,
                        correct[rows], ex[rows], sid[rows], ["x", "y"])
        out = os.path.join(d, "out")
        summary = hc.run(d, 3, out, num_resamples=200, num_thresholds=50, verbose=False)

        assert set(summary["results"]) == {"x", "y", "overall"}
        r = summary["results"]["overall"]
        assert r["num_selection"] + r["num_report"] == P
        assert summary["results"]["x"]["num_puzzles"] + summary["results"]["y"]["num_puzzles"] == P
        assert np.isclose(r["full_step_accuracy_all_puzzles"], correct[:, -1].mean())
        assert [op["tolerance"] for op in r["heads"]["probe"]["operating_points"]] == [0.0, 0.001, 0.005]
        for name in ["x", "y", "overall"]:
            for head in ["q_head", "probe"]:
                path = os.path.join(out, f"curve_{name}_{head}.csv")
                rows = np.genfromtxt(path, delimiter=",", names=True)
                assert rows["threshold"][0] == -np.inf and rows["threshold"][-1] == np.inf
                assert rows["avg_steps"][0] == 1 and rows["avg_steps"][-1] == S
                assert np.all(np.diff(rows["avg_steps"]) >= 0)
        with open(os.path.join(out, "summary.json")) as f:
            loaded = json.load(f)
        assert loaded["results"]["overall"]["heads"]["q_head"]["operating_points"][0]["report"]["accuracy_ci"]
        try:
            import matplotlib  # noqa: F401
            assert os.path.exists(os.path.join(out, "curves_overall.png"))
        except ImportError:
            pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
