"""Tests for analysis/eval_steps.py on synthetic captures with known answers.

Run with `python -m pytest tests/` or `python tests/test_eval_steps.py`.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from analysis import eval_steps as es  # noqa: E402


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
    orc = es.oracle(PATTERNS)
    assert orc["accuracy"] == 6 / 7
    assert orc["avg_steps"] == (4 + 1 + 2 + 1 + 2 + 1 + 3) / 7
    f = es.flips(PATTERNS)
    assert f["correct_at_some_step"] == 6
    assert f["correct_at_some_step_but_wrong_at_last"] == 2
    assert f["correct_then_wrong_then_correct"] == 3
    assert es.fixed_step_accuracy(PATTERNS) == [3 / 7, 3 / 7, 3 / 7, 4 / 7]


def test_curve_known_by_construction():
    # Logits cross 0 exactly at the oracle step, and stay below -10 before it:
    # t = 0 reproduces the oracle; t = -inf is step 1; t = +inf is the last step.
    S = PATTERNS.shape[1]
    logits = np.full(PATTERNS.shape, -20.0, dtype=np.float32)
    for i, row in enumerate(PATTERNS):
        if row.any():
            logits[i, row.argmax():] = 5.0
    curve = es.HaltCurve(logits, PATTERNS)
    pts = curve([-np.inf, 0.0, np.inf])
    orc = es.oracle(PATTERNS)
    fixed = es.fixed_step_accuracy(PATTERNS)
    np.testing.assert_allclose(pts["accuracy"], [fixed[0], orc["accuracy"], fixed[-1]])
    np.testing.assert_allclose(pts["avg_steps"], [1, orc["avg_steps"], S])
    np.testing.assert_allclose(pts["frac_halted_early"], [1, 6 / 7, 0])  # all but the never-correct row


def test_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    P, S = 300, 6
    # Coarse values to create many ties (like bf16 logits), including non-monotone sequences
    logits = rng.integers(-6, 7, size=(P, S)).astype(np.float32) / 2
    correct = rng.random((P, S)) < np.linspace(0.2, 0.7, S)
    curve = es.HaltCurve(logits, correct)
    thresholds = np.concatenate([[-np.inf, np.inf], np.unique(logits), np.unique(logits) + 0.25])
    pts = curve(thresholds)
    for j, t in enumerate(thresholds):
        steps, acc, early = brute_force(logits, correct, t)
        assert np.isclose(pts["avg_steps"][j], steps), t
        assert np.isclose(pts["accuracy"][j], acc), t
        assert np.isclose(pts["frac_halted_early"][j], early), t
    # halting_steps agrees with the curve
    for t in [-1.0, 0.0, 1.5]:
        assert np.isclose(es.halting_steps(logits, t).mean(), curve(t)["avg_steps"][0])


def test_selection_respects_tolerance():
    rng = np.random.default_rng(1)
    P, S = 2000, 8
    logits = rng.normal(size=(P, S)).cumsum(axis=1).astype(np.float32)
    correct = rng.random((P, S)) < 1 / (1 + np.exp(-logits))
    curve = es.HaltCurve(logits, correct)
    full = curve(np.inf)["accuracy"][0]
    levels = curve.level_thresholds()
    all_pts = curve(levels)
    prev_steps = np.inf
    for tol in [0.0, 0.001, 0.01, 0.05]:
        chosen = es.select_threshold(curve, tol)
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
    res = es.evaluate_with_ci(logits, correct, 0.5, 2000, np.random.default_rng(0))
    steps, acc, early = brute_force(logits, correct, 0.5)
    assert np.isclose(res["accuracy"], acc) and np.isclose(res["avg_steps"], steps)
    assert np.isclose(res["accuracy_loss"], correct[:, -1].mean() - acc)
    for k in ["accuracy", "avg_steps", "accuracy_loss"]:
        lo, hi = res[f"{k}_ci"]
        assert lo <= res[k] <= hi and hi > lo
    # Binomial standard error check for accuracy: 95% interval ~ +-1.96 sqrt(p(1-p)/P)
    half = (res["accuracy_ci"][1] - res["accuracy_ci"][0]) / 2
    assert abs(half - 1.96 * np.sqrt(acc * (1 - acc) / P)) < 0.003


def test_unchanged_among_unsolved():
    correct = np.array([[0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 1]], dtype=bool)
    changed = np.array([[0, 1, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0]], dtype=bool)
    out = es.unchanged_among_unsolved(correct, changed)
    assert out[0] is None
    # step 2: wrong at step 1 = rows 0, 1, 3; unchanged at step 2 = row 1
    assert (out[1]["step"], out[1]["unsolved_at_previous_step"], out[1]["unchanged"]) == (2, 3, 1)
    # step 3: wrong at step 2 = rows 0, 1; both unchanged
    assert (out[2]["unsolved_at_previous_step"], out[2]["unchanged"], out[2]["share"]) == (2, 2, 1.0)
    # step 4: wrong at step 3 = rows 0, 1, 2; unchanged = rows 1, 2
    assert (out[3]["unsolved_at_previous_step"], out[3]["unchanged"]) == (3, 2)


def test_truncation_equals_shorter_capture():
    """Analyzing the first K steps of a 32-step capture = analyzing a K-step capture."""
    rng = np.random.default_rng(5)
    P, S, K = 800, 32, 16
    logits = rng.normal(size=(P, S)).cumsum(axis=1).astype(np.float32)
    correct = rng.random((P, S)) < 1 / (1 + np.exp(-logits / 3))
    changed = rng.random((P, S)) < 0.5
    changed[:, 0] = False
    with tempfile.TemporaryDirectory() as d:
        for step, n in [(1, S), (2, K)]:
            _write_rank(os.path.join(d, f"step_{step}_eval_steps.0"), logits[:, :n], logits[:, :n] * 0.5 + 1,
                        correct[:, :n], np.arange(P), np.zeros(P), ["all"], preds_changed=changed[:, :n])
        full = es.run(d, 1, os.path.join(d, "o1"), num_resamples=50, num_thresholds=40, verbose=False, truncate_steps=[K])
        short = es.run(d, 2, os.path.join(d, "o2"), num_resamples=50, num_thresholds=40, verbose=False)
        assert os.path.exists(os.path.join(d, "o1", f"curve_all_q_head_first{K}.csv"))

        r_full, r_short = full["results"]["all"], short["results"]["all"]
        assert r_full["num_steps"] == S and len(r_full["fixed_step_accuracy"]) == S
        r_trunc = r_full["truncated"][str(K)]
        assert r_trunc["num_steps"] == K
        for key in ["full_step_accuracy", "fixed_step_accuracy", "oracle", "flips", "unchanged_among_unsolved"]:
            assert r_trunc[key] == r_short[key], key
        assert r_trunc["fixed_step_accuracy"] == r_full["fixed_step_accuracy"][:K]
        for head in ["q_head", "probe"]:
            assert r_trunc["heads"][head]["t0"] == r_short["heads"][head]["t0"]
            for a, b in zip(r_trunc["heads"][head]["operating_points"], r_short["heads"][head]["operating_points"]):
                assert a["selection"] == b["selection"]
                assert a["report"]["accuracy"] == b["report"]["accuracy"] and a["report"]["avg_steps"] == b["report"]["avg_steps"]
        try:
            es.run(d, 2, os.path.join(d, "o3"), verbose=False, truncate_steps=[K])
            raise AssertionError("expected an error")
        except ValueError:
            pass


def test_old_capture_without_change_records():
    rng = np.random.default_rng(6)
    P, S = 100, 4
    logits = rng.normal(size=(P, S)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        _write_rank(os.path.join(d, "step_1_eval_steps.0"), logits, logits, rng.random((P, S)) < 0.5,
                    np.arange(P), np.zeros(P), ["all"])
        r = es.run(d, 1, os.path.join(d, "o"), num_resamples=0, verbose=False)["results"]["all"]
        assert isinstance(r["unchanged_among_unsolved"], str)


def _write_rank(path, logits_q, logits_aux, correct, example_index, set_id, set_names, preds_changed=None,
                rel_change=None, pred_hash=None, cells=None):
    P, S = correct.shape
    extra = {} if preds_changed is None else {"preds_changed": torch.from_numpy(preds_changed)}
    if rel_change is not None:
        extra.update({f"{z}_rel_change": torch.from_numpy(v) for z, v in rel_change.items()})
    if pred_hash is not None:
        extra["pred_hash"] = torch.from_numpy(pred_hash)
    if cells is not None:  # {"wrong_cells": ..., "cells_changed": ...}
        extra.update({k: torch.from_numpy(v) for k, v in cells.items()})
    torch.save({
        **extra,
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
        data, checks = es.load(d, 7)

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
        _, checks = es.load(d, 8)
        assert checks["duplicates_with_differing_values"] == 1


def test_state_change_among_unsolved():
    correct = np.array([[0, 0, 1],
                        [0, 0, 0],
                        [0, 1, 1],
                        [1, 1, 1],
                        [0, 0, 0]], dtype=bool)
    nan = np.nan
    rel = np.array([[nan, 0.1, 9.0],
                    [nan, 0.2, 0.4],
                    [nan, 0.3, 7.0],
                    [nan, 5.0, 8.0],
                    [nan, 0.4, nan]], dtype=np.float32)
    out = es.state_change_among_unsolved(correct, rel)
    assert out[0] is None
    # step 2: wrong at step 1 = rows 0, 1, 2, 4 -> values 0.1, 0.2, 0.3, 0.4
    assert (out[1]["step"], out[1]["unsolved_at_previous_step"]) == (2, 4)
    np.testing.assert_allclose([out[1]["q25"], out[1]["median"], out[1]["q75"]], [0.175, 0.25, 0.325], rtol=1e-6)
    # step 3: wrong at step 2 = rows 0, 1, 4 -> values 9.0, 0.4, NaN (ignored)
    assert out[2]["unsolved_at_previous_step"] == 3
    np.testing.assert_allclose(out[2]["median"], 4.7, rtol=1e-6)
    # nobody unsolved -> no quartiles
    assert es.state_change_among_unsolved(np.ones((2, 2), dtype=bool), rel[:2, :2])[1]["median"] is None


def test_halt_step_counts():
    logits = np.array([[1.0, -1.0, -1.0],    # halts at step 1
                       [-1.0, 0.0, -1.0],    # at step 2 (logit == threshold halts)
                       [-1.0, -1.0, 5.0],    # crosses at the last step
                       [-1.0, -1.0, -1.0],   # never crosses: last step
                       [-1.0, 2.0, 2.0]], dtype=np.float32)
    assert es.halt_step_counts(logits, 0.0) == [1, 2, 2]
    assert es.halt_step_counts(logits, np.inf) == [0, 0, 5]
    assert es.halt_step_counts(logits, -np.inf) == [5, 0, 0]
    correct = np.ones(logits.shape, dtype=bool)
    rep = es.evaluate_with_ci(logits, correct, 0.0, 0, np.random.default_rng(0))
    assert rep["halt_step_counts"] == [1, 2, 2]
    assert np.isclose(rep["avg_steps"], np.dot(rep["halt_step_counts"], [1, 2, 3]) / len(logits))
    assert np.isclose(rep["frac_halted_early"], 1 - rep["halt_step_counts"][-1] / len(logits))


# Wrong cells (S = 4); 0 wrong cells <=> correct
_WRONG = np.array([[5, 3, 1, 0],     # solved at step 4
                   [9, 9, 2, 2],     # unsolved, minimum 2
                   [0, 0, 0, 0],     # always correct
                   [1, 0, 4, 3],     # solved at step 2, lost: minimum 0
                   [12, 12, 10, 9],  # far away
                   [2, 1, 1, 1]], dtype=np.uint8)


def test_wrong_cells_statistics():
    correct = _WRONG == 0
    by_step = es.wrong_cells_among_unsolved(correct, _WRONG)
    # step 1, unsolved rows 0, 1, 3, 4, 5 -> 5, 9, 1, 12, 2
    s1 = by_step[0]
    assert (s1["step"], s1["unsolved"], s1["median"], s1["q25"], s1["q75"]) == (1, 5, 5.0, 2.0, 9.0)
    assert s1["share_at_most"] == {"1": 0.2, "2": 0.4, "4": 0.4, "8": 0.6}
    # step 4, unsolved rows 1, 3, 4, 5 -> 2, 3, 9, 1
    s4 = by_step[3]
    assert (s4["unsolved"], s4["median"]) == (4, 2.5)
    assert s4["share_at_most"] == {"1": 0.25, "2": 0.5, "4": 0.75, "8": 0.75}
    assert es.wrong_cells_among_unsolved(np.ones((2, 2), dtype=bool), np.zeros((2, 2), dtype=np.uint8))[0]["median"] is None

    end = es.wrong_cells_unsolved_at_last(correct, _WRONG)
    # unsolved at the last step: rows 1, 3, 4, 5 -> last 2, 3, 9, 1; minimum 2, 0, 9, 1
    assert end["num_unsolved_at_last"] == 4
    assert end["last_step_counts"] == [0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    assert end["minimum_counts"] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    assert (end["last_step_median"], end["minimum_median"]) == (2.5, 1.5)


def test_solving_rate_by_distance():
    correct = _WRONG == 0
    out = {b["bucket"]: b for b in es.solving_rate_by_distance(correct, _WRONG)}
    assert list(out) == ["1", "2", "3-4", "5-8", "9+"]
    # Pairs (puzzle, s - 1 -> s) with the puzzle wrong at s - 1, by its wrong cells at s - 1:
    #   row 0: 5 -> 3 (5-8), 3 -> 1 (3-4), 1 -> 0 solved (1)
    #   row 1: 9 -> 9, 9 -> 2 (9+, 9+), 2 -> 2 (2)
    #   row 3: 1 -> 0 solved (1); 4 -> 3 (3-4)        (step 2 -> 3 starts from a correct grid)
    #   row 4: 12 -> 12, 12 -> 10, 10 -> 9 (9+ x3)
    #   row 5: 2 -> 1 (2), 1 -> 1, 1 -> 1 (1, 1)
    expect = {"1": (4, 3, 2), "2": (2, 2, 0), "3-4": (2, 2, 0), "5-8": (1, 1, 0), "9+": (5, 2, 0)}
    for k, (n, puzzles, solved) in expect.items():
        assert (out[k]["observations"], out[k]["puzzles"], out[k]["solved"]) == (n, puzzles, solved), k
    assert out["1"]["rate"] == 0.5 and out["5-8"]["rate"] == 0.0
    lo, hi = out["1"]["rate_ci_approx"]
    assert lo < 0.5 < hi
    assert sum(b["observations"] for b in out.values()) == int((~correct[:, :-1]).sum())
    # Custom buckets; an empty bucket has no rate
    custom = es.solving_rate_by_distance(correct, _WRONG, buckets=((1, 2), (50, None)))
    assert (custom[0]["bucket"], custom[0]["observations"]) == ("1-2", 6)
    assert (custom[1]["bucket"], custom[1]["observations"], custom[1]["rate"]) == ("50+", 0, None)


def test_cells_changed_and_solved_state_change():
    correct = _WRONG == 0
    changed = np.array([[0, 2, 2, 1],
                        [0, 0, 7, 0],
                        [0, 0, 0, 0],
                        [0, 1, 4, 1],
                        [0, 3, 2, 1],
                        [0, 1, 0, 0]], dtype=np.uint8)
    out = es.cells_changed_among_unsolved(correct, changed)
    assert out[0] is None
    # step 2: wrong at step 1 = rows 0, 1, 3, 4, 5 -> 2, 0, 1, 3, 1
    assert (out[1]["unsolved_at_previous_step"], out[1]["median"], out[1]["q25"], out[1]["q75"]) == (5, 1.0, 1.0, 2.0)
    assert np.isclose(out[1]["mean"], 7 / 5)
    # step 3: wrong at step 2 = rows 0, 1, 4, 5 -> 2, 7, 2, 0
    assert (out[2]["unsolved_at_previous_step"], out[2]["median"]) == (4, 2.0)

    rel = np.arange(24, dtype=np.float32).reshape(6, 4)
    rel[:, 0] = np.nan
    solved = es.state_change_among_unsolved(correct, rel, solved=True)
    # correct at step 1: row 2 -> value at step 2 = 9; correct at step 2: rows 2, 3 -> 10, 14
    assert (solved[1]["solved_at_previous_step"], solved[1]["median"]) == (1, 9.0)
    assert (solved[2]["solved_at_previous_step"], solved[2]["median"]) == (2, 12.0)
    unsolved = es.state_change_among_unsolved(correct, rel)
    assert unsolved[1]["unsolved_at_previous_step"] + solved[1]["solved_at_previous_step"] == 6


def test_load_checks_wrong_cells_against_correctness():
    rng = np.random.default_rng(9)
    P, S = 50, 5
    logits = rng.normal(size=(P, S)).astype(np.float32)
    wrong = rng.integers(0, 4, size=(P, S)).astype(np.uint8)
    changed = rng.integers(0, 3, size=(P, S)).astype(np.uint8)
    args = (logits, logits, wrong == 0, np.arange(P), np.zeros(P, dtype=np.int64), ["all"])
    with tempfile.TemporaryDirectory() as d:
        _write_rank(os.path.join(d, "step_1_eval_steps.0"), *args, cells={"wrong_cells": wrong, "cells_changed": changed})
        data, checks = es.load(d, 1)
        assert checks["wrong_cells_match_correctness"] and data["wrong_cells"].dtype == np.uint8
        r = es.run(d, 1, os.path.join(d, "out"), num_resamples=0, num_thresholds=10, verbose=False, truncate_steps=[3])
        res = r["results"]["all"]
        assert len(res["wrong_cells"]["among_unsolved"]) == S and len(res["cells_changed_among_unsolved"]) == S
        assert len(res["truncated"]["3"]["wrong_cells"]["among_unsolved"]) == 3
        assert "not available" in res["state_change_among_solved"]["z_H"]
        json.dumps(es._json_safe(r))

        bad = wrong.copy()
        bad[3, 2] = 0 if bad[3, 2] else 1  # disagrees with the correctness flag
        _write_rank(os.path.join(d, "step_2_eval_steps.0"), *args, cells={"wrong_cells": bad, "cells_changed": changed})
        try:
            es.load(d, 2)
        except ValueError as e:
            assert "1 (puzzle, step)" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_solving_rate():
    correct = np.array([[0, 0, 1, 1],
                        [0, 1, 1, 1],
                        [0, 0, 0, 0],
                        [1, 1, 0, 1],   # solved, lost, solved again: counts as solved at step 4
                        [0, 1, 1, 1]], dtype=bool)
    out = es.solving_rate(correct)
    assert out[0] is None
    got = [(e["step"], e["unsolved_at_previous_step"], e["solved"], e["unsolved"]) for e in out[1:]]
    assert got == [(2, 4, 2, 2), (3, 2, 1, 2), (4, 2, 1, 1)]
    assert [e["rate"] for e in out[1:]] == [0.5, 0.5, 0.5]
    lo, hi = out[1]["rate_ci"]
    assert lo < 0.5 < hi and np.isclose(lo + hi, 1.0)  # symmetric around 1/2
    # Wilson interval against known values: 1 of 10 -> [0.0179, 0.4042]; 0 of n has lower bound 0
    np.testing.assert_allclose(es.wilson_interval(1, 10), [0.017876, 0.404150], atol=1e-5)
    assert es.wilson_interval(0, 50)[0] == 0.0 and 0 < es.wilson_interval(0, 50)[1] < 0.08
    # Nobody unsolved at the previous step: no rate
    assert es.solving_rate(np.ones((3, 2), dtype=bool))[1]["rate"] is None
    # Large sample: the interval contains the true rate
    rng = np.random.default_rng(0)
    big = np.stack([np.zeros(200000, dtype=bool), rng.random(200000) < 0.03], axis=1)
    e = es.solving_rate(big)[1]
    assert e["rate_ci"][0] < 0.03 < e["rate_ci"][1] and e["rate_ci"][1] - e["rate_ci"][0] < 0.002


# Hash sequences (S = 10) with known behavior. Values are arbitrary, including negative ones.
_G = np.array([-(2 ** 62) - 12345, 7, -3, 2 ** 61 + 5, 11, 13, 17, 19, 23, 29], dtype=np.int64)
_SEQUENCES = {
    "frozen":        [0, 1, 2, 3, 3, 3, 3, 3, 3, 3],  # first repeat at step 5, period 1; 4 grids
    "cycle3":        [0, 1, 2, 3, 4, 2, 3, 4, 2, 3],  # first repeat at step 6, period 3; 5 grids
    "revisiting":    [0, 1, 0, 2, 3, 4, 5, 6, 7, 8],  # repeat at step 3 (period 2), then new grids
    "wandering":     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "late_freeze":   [0, 1, 0, 2, 3, 4, 5, 5, 5, 5],  # transient repeat at step 3, frozen at the end
    "broken_cycle":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 2],  # period 2 until the last step breaks it
    "solved_frozen": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # solved at the end: not analysed
}


def test_cycle_records_and_classes():
    names = [n for n in _SEQUENCES if n != "solved_frozen"]
    hashes = _G[np.array([_SEQUENCES[n] for n in names])]
    rec = es.cycle_records(hashes)
    cls = [es.CYCLE_CLASSES[i] for i in es.classify_cycles(rec)]
    expect = {  # first_repeat_step, period, confirmed, distinct, ends_periodic, class
        "frozen":       (5, 1, True, 4, True, "frozen"),
        "cycle3":       (6, 3, True, 5, True, "cycling"),
        "revisiting":   (3, 2, False, 9, False, "revisiting"),
        "wandering":    (0, 0, False, 10, False, "wandering"),
        "late_freeze":  (3, 2, False, 6, True, "revisiting"),
        "broken_cycle": (3, 2, False, 3, False, "revisiting"),
    }
    for i, n in enumerate(names):
        got = (int(rec["first_repeat_step"][i]), int(rec["period"][i]), bool(rec["confirmed"][i]),
               int(rec["distinct"][i]), bool(rec["ends_periodic"][i]), cls[i])
        assert got == expect[n], (n, got)

    # Chunking does not change anything; the previous occurrence matches a direct search
    small = es.cycle_records(hashes, chunk=2)
    assert all(np.array_equal(rec[k], small[k]) for k in rec)
    rng = np.random.default_rng(0)
    h = rng.integers(0, 6, size=(200, 12)).astype(np.int64) * 2 ** 40 - 2 ** 41
    prev = es.previous_occurrence(h)
    for i in range(len(h)):
        for s in range(h.shape[1]):
            earlier = np.flatnonzero(h[i, :s] == h[i, s])
            assert prev[i, s] == (earlier[-1] if len(earlier) else -1)
    # ... and so do the records, against a direct per-row computation
    rec = es.cycle_records(h)
    for i in range(len(h)):
        repeats = np.flatnonzero(prev[i] >= 0)
        f = repeats[0]  # 12 steps over 6 values: every row repeats
        p = f - prev[i, f]
        assert (rec["first_repeat_step"][i], rec["period"][i]) == (f + 1, p)
        assert rec["confirmed"][i] == all(h[i, s] == h[i, s - p] for s in range(f, 12))
        assert rec["distinct"][i] == len(set(h[i].tolist()))


def test_cycle_analysis():
    names = list(_SEQUENCES)
    hashes = _G[np.array([_SEQUENCES[n] for n in names])]
    correct = np.zeros(hashes.shape, dtype=bool)
    correct[names.index("solved_frozen"), 1:] = True
    out = es.cycle_analysis(correct, hashes)
    assert out["num_unsolved_at_last"] == 6
    assert {k: v["count"] for k, v in out["classes"].items()} == {"frozen": 1, "cycling": 1, "revisiting": 3, "wandering": 1}
    assert np.isclose(sum(v["share"] for v in out["classes"].values()), 1.0)
    assert out["repeats"]["with_any_repeat"] == 5
    assert out["repeats"]["first_repeat_step_counts"] == [0, 0, 3, 0, 1, 1, 0, 0, 0, 0]
    assert out["repeats"]["first_repeat_period_counts"] == {"1": 1, "2": 3, "3": 1}
    assert out["repeats"]["distinct_grids_counts"] == [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    assert out["confirmed"]["period_counts"] == {"1": 1, "3": 1}
    assert out["confirmed"]["start_step_counts"] == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert out["revisiting_ending_periodic"] == 1  # late_freeze
    by = out["by_step"]
    assert [e["unsolved"] for e in by] == [7] + [6] * 9   # the solved puzzle is unsolved at step 1 only
    assert [e["in_frozen"] for e in by] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert [e["in_cycling"] for e in by] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert by[-1]["share_in_confirmed_cycle"] == 2 / 6 and by[0]["unsolved_at_last"] == 6

    # Nobody unsolved at the end
    empty = es.cycle_analysis(np.ones((3, 4), dtype=bool), np.zeros((3, 4), dtype=np.int64))
    assert empty["num_unsolved_at_last"] == 0 and empty["classes"]["frozen"] == {"count": 0, "share": None}
    assert empty["confirmed"]["period_counts"] == {}


def test_end_to_end_with_hashes_and_compact_dtypes():
    """A capture with hashes and float16 state changes loads; one without hashes still does."""
    rng = np.random.default_rng(8)
    P, S = 400, 10
    logits = rng.normal(size=(P, S)).astype(np.float32)
    correct = np.logical_or.accumulate(rng.random((P, S)) < 0.1, axis=1)
    hashes = rng.integers(-2 ** 62, 2 ** 62, size=(P, S), dtype=np.int64)
    hashes[:100, 4:] = hashes[:100, 4:5]            # frozen from step 5 (first repeat at step 6)
    changed = np.ones((P, S), dtype=bool)
    changed[:, 0] = False
    changed[:, 1:] = hashes[:, 1:] != hashes[:, :-1]
    rel = {z: rng.random((P, S)).astype(np.float16) for z in ("z_H", "z_L")}
    with tempfile.TemporaryDirectory() as d:
        args = (logits, logits, correct, np.arange(P), np.zeros(P, dtype=np.int64), ["all"])
        _write_rank(os.path.join(d, "step_1_eval_steps.0"), *args, preds_changed=changed, rel_change=rel, pred_hash=hashes)
        _write_rank(os.path.join(d, "step_2_eval_steps.0"), *args, preds_changed=changed)
        summary = es.run(d, 1, os.path.join(d, "out1"), num_resamples=0, num_thresholds=20, verbose=False, truncate_steps=[5])
        r = summary["results"]["all"]
        data, _ = es.load(d, 1)
        report = ~es.split_puzzles(P, 0.2, 0)
        unsolved = report & ~correct[:, -1]
        cyc = r["cycles"]
        assert cyc["num_unsolved_at_last"] == unsolved.sum()
        assert cyc["classes"]["frozen"]["count"] == (unsolved[:100]).sum() == cyc["confirmed"]["start_step_counts"][5]
        assert cyc["classes"]["wandering"]["count"] == (unsolved[100:]).sum()
        assert len(r["solving_rate"]) == S and r["solving_rate"][1]["unsolved_at_previous_step"] == (report & ~correct[:, 0]).sum()
        assert isinstance(r["state_change_among_unsolved"]["z_H"][1]["median"], float)
        # The first 5 steps end before anything repeats
        assert r["truncated"]["5"]["cycles"]["classes"]["frozen"]["count"] == 0
        json.dumps(es._json_safe(summary))

        old = es.run(d, 2, os.path.join(d, "out2"), num_resamples=0, num_thresholds=20, verbose=False)
        assert "not available" in old["results"]["all"]["cycles"]
        assert "not available" in old["results"]["all"]["wrong_cells"]
        assert "not available" in old["results"]["all"]["cells_changed_among_unsolved"]
        assert len(old["results"]["all"]["solving_rate"]) == S


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
        summary = es.run(d, 3, out, num_resamples=200, num_thresholds=50, verbose=False)

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
        q = loaded["results"]["overall"]["heads"]["q_head"]
        assert q["operating_points"][0]["report"]["accuracy_ci"]
        for point in [q["t0"]] + [op["report"] for op in q["operating_points"]]:
            assert len(point["halt_step_counts"]) == S
            assert sum(point["halt_step_counts"]) == loaded["results"]["overall"]["num_report"]
        assert "not available" in loaded["results"]["overall"]["state_change_among_unsolved"]["z_H"]
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
