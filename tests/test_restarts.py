"""Tests for experiments/restarts.py on a tiny CPU model.

Run with `python -m pytest tests/` or `python tests/test_restarts.py`.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from analysis import eval_steps as es  # noqa: E402
from experiments import restarts as rs  # noqa: E402
from utils.eval_loop import run_eval_batch  # noqa: E402
from test_aux_halt_heads import make_batch  # noqa: E402
from test_eval_loop import make_eval_model, per_step_preds, save_capture  # noqa: E402

N1, N2 = 4, 4
SEEDS = [0, 1]


def make_puzzles(model, n=24):
    """n puzzles whose labels are the model's own prediction at a puzzle-specific step, so that
    correctness varies by step (and some puzzles are solved only in phase 2)."""
    parts = [make_batch(seed=100 + i) for i in range((n + 5) // 6)]
    batch = {k: torch.cat([p[k] for p in parts])[:n] for k in parts[0]}
    preds = per_step_preds(model, batch)
    target = torch.arange(n) % preds.shape[1]
    batch["labels"] = preds[torch.arange(n), target].to(batch["labels"].dtype)
    return batch


def setup(n=24):
    model = make_eval_model(eval_halt_max_steps=N1 + N2)
    batch = make_puzzles(model, n)
    with torch.inference_mode():
        plain = run_eval_batch(model, model.initial_carry(batch), batch, set(), capture_steps=True)
    rec = {k: torch.stack(v, 1) for k, v in plain["step_records"].items()}
    # Threshold: about half of the puzzles accepted by step N1
    threshold = float(rec["q_halt_logits"][:, :N1].max(dim=1).values.median())
    settings = rs.Settings(n1=N1, n2=N2, threshold=threshold, perturb_scales=[0.001, 0.3], restart_scales=[0.5], seeds=SEEDS)
    return model, batch, plain, rec, settings


def run(model, batch, settings, idx=None):
    idx = np.arange(batch["inputs"].shape[0]) if idx is None else np.asarray(idx)
    sub = {k: v[torch.as_tensor(idx)] for k, v in batch.items()}
    with torch.inference_mode():
        return rs.run_batch(model, sub, np.zeros(len(idx), dtype=np.int64), idx.astype(np.int64), settings)


def test_control_reproduces_plain_run():
    model, batch, plain, rec, settings = setup()
    out = run(model, batch, settings)
    logits = rec["q_halt_logits"].numpy()
    thr = float(np.float32(settings.threshold))

    # Phase 1: acceptance and halting steps are those of the plain run's first N1 steps
    accepted = logits[:, :N1].max(axis=1) >= thr
    assert np.array_equal(out["phase1"]["accepted"], accepted)
    assert 0 < accepted.sum() < len(accepted)
    h = es.halting_steps(logits[:, :N1], thr)
    assert np.array_equal(out["phase1"]["halt_step"][accepted], h[accepted])
    correct = rec["seq_is_correct"].numpy()
    assert np.array_equal(out["phase1"]["correct_at_halt"][accepted], correct[accepted, h[accepted] - 1])
    assert np.array_equal(out["phase1"]["correct_at_n1"], correct[:, N1 - 1])
    retried = out["retried"]
    assert np.array_equal(retried, np.flatnonzero(~accepted))

    # Control: steps N1 + 1 ... N1 + N2 of the plain run, every capture field
    ctrl = out["branches"]["continue"]
    for key, value in ctrl["records"].items():
        expected = rec[key][retried][:, N1:]
        if value.is_floating_point():
            assert torch.allclose(value.float(), expected.float(), atol=1e-6, equal_nan=True), key
        else:
            assert torch.equal(value, expected), key
    h2 = N1 + es.halting_steps(logits[retried, N1:], thr)
    assert np.array_equal(ctrl["halt_step"], h2)
    assert np.array_equal(ctrl["correct_at_halt"], correct[retried, h2 - 1])
    assert np.array_equal(ctrl["correct_at_last"], correct[retried, -1])
    assert np.array_equal(ctrl["accepted"], logits[retried, N1:].max(axis=1) >= thr)
    assert not ctrl["realized_scale_z_H"].any()

    # Noisy branches exist for every scale and seed, and differ from the control
    names = [b.name for b in settings.branches()]
    assert list(out["branches"]) == names and len(names) == 1 + 2 * 2 + 1 * 2
    big = out["branches"]["perturb_s0.3_seed0"]["records"]["pred_hash"]
    assert not torch.equal(big, ctrl["records"]["pred_hash"])


def test_chunking_and_padding_change_nothing():
    model, batch, _, _, settings = setup()
    whole = run(model, batch, settings)
    settings.chunk_size = 5  # pads the last chunk
    chunked = run(model, batch, settings)
    for name, b in whole["branches"].items():
        for k in ("accepted", "halt_step", "correct_at_halt", "correct_at_last"):
            assert np.array_equal(b[k], chunked["branches"][name][k]), (name, k)
        for k, v in b["records"].items():
            assert torch.equal(v, chunked["branches"][name]["records"][k]), (name, k)


def test_noise_scaling():
    state = torch.randn(5, 7, 16) * torch.arange(1, 6).view(5, 1, 1)  # rows with different norms
    unit_h, unit_l = rs.unit_noise((7, 16), 0, [0] * 5, range(5))
    assert torch.allclose(torch.linalg.vector_norm(unit_h.flatten(1), dim=1), torch.ones(5))
    assert torch.allclose(torch.linalg.vector_norm(unit_l.flatten(1), dim=1), torch.ones(5))
    assert not torch.allclose(unit_h, unit_l)  # separate directions for z_H and z_L
    for scale in (0.001, 0.1, 1.0):
        new = rs.add_noise(state, unit_h, scale)
        assert torch.allclose(rs.realized_scale(new, state), torch.full((5, ), scale), rtol=1e-4)
    # z_L has its own norm: the noise follows it, not z_H's
    state_l = state * 10
    assert torch.allclose(rs.realized_scale(rs.add_noise(state_l, unit_l, 0.1), state_l), torch.full((5, ), 0.1), rtol=1e-4)
    # bfloat16 states: the noise is rounded with the state; realized_scale reports what was applied
    s16 = state.to(torch.bfloat16)
    r = rs.realized_scale(rs.add_noise(s16, unit_h, 0.1), s16)
    assert torch.allclose(r, torch.full((5, ), 0.1), rtol=0.05)


def test_noise_depends_only_on_seed_and_puzzle():
    state = torch.get_rng_state()
    a_h, a_l = rs.unit_noise((3, 4), 7, [0, 0, 1, 0], [2, 5, 5, 9])
    b_h, b_l = rs.unit_noise((3, 4), 7, [0, 0], [9, 2])
    assert torch.equal(torch.get_rng_state(), state)  # no global generator touched
    assert torch.equal(a_h[3], b_h[0]) and torch.equal(a_h[0], b_h[1]) and torch.equal(a_l[3], b_l[0])
    assert not torch.equal(a_h[1], a_h[2])  # same example index, other set
    c_h, _ = rs.unit_noise((3, 4), 8, [0], [2])
    assert not torch.equal(c_h[0], a_h[0])  # other seed
    assert rs.puzzle_noise_seed(7, 0, 2) == rs.puzzle_noise_seed(7, 0, 2) != rs.puzzle_noise_seed(7, 0, 3)


def test_results_do_not_depend_on_batch_size_or_order():
    model, batch, _, _, settings = setup()
    whole = run(model, batch, settings)
    order = np.random.default_rng(0).permutation(24)
    parts = [run(model, batch, settings, order[:10]), run(model, batch, settings, order[10:])]
    for name in whole["branches"]:
        per_puzzle = {}
        for idx, part in zip((order[:10], order[10:]), parts):
            for row, j in enumerate(part["retried"]):
                per_puzzle[int(idx[j])] = {k: part["branches"][name]["records"][k][row] for k in ("pred_hash", "q_halt_logits")}
        assert sorted(per_puzzle) == whole["retried"].tolist()
        for row, j in enumerate(whole["retried"]):
            for k in ("pred_hash", "q_halt_logits"):
                assert torch.equal(per_puzzle[int(j)][k], whole["branches"][name]["records"][k][row]), (name, k)


def test_noisy_restart_starts_from_the_initial_state():
    model, batch, _, _, settings = setup()
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        for _ in range(N1):
            carry, *_ = model(carry=carry, batch=batch, return_keys=set())
        inner = rs.inner_model(model)
        unit = rs.unit_noise(carry.inner_carry.z_H.shape[1:], 0, [0] * 24, range(24))
        c0, rh, _ = rs.branch_carry(model, carry, rs.Branch("noisy_restart", 0.0, 0), unit)
        assert torch.equal(c0.inner_carry.z_H, inner.H_init.expand_as(c0.inner_carry.z_H))
        assert torch.equal(c0.inner_carry.z_L, inner.L_init.expand_as(c0.inner_carry.z_L))
        assert not rh.any() and torch.equal(c0.steps, carry.steps)
        c1, rh, rl = rs.branch_carry(model, carry, rs.Branch("noisy_restart", 0.5, 0), unit)
        assert torch.allclose(rh, torch.full_like(rh, 0.5), rtol=0.05) and torch.allclose(rl, torch.full_like(rl, 0.5), rtol=0.05)
        cp, rh, _ = rs.branch_carry(model, carry, rs.Branch("perturb", 0.5, 0), unit)
        assert torch.allclose(rs.realized_scale(cp.inner_carry.z_H, carry.inner_carry.z_H), rh)
        # Same direction for both noisy variants of a (seed, puzzle) pair
        d_p = cp.inner_carry.z_H.float() - carry.inner_carry.z_H.float()
        d_r = c1.inner_carry.z_H.float() - c0.inner_carry.z_H.float()
        cos = torch.nn.functional.cosine_similarity(d_p.flatten(1), d_r.flatten(1))
        assert (cos > 0.95).all()  # not exactly 1: both are rounded to bfloat16


def _write_run(d, phase1, phase2, n1=10, n2=5, set_sizes=(8, )):
    np.savez(os.path.join(d, "phase1.npz"), **phase1)
    np.savez(os.path.join(d, "phase2.npz"), **phase2)
    with open(os.path.join(d, "settings.json"), "w") as f:
        json.dump({"n1": n1, "n2": n2, "head": "q_head", "threshold": 0.0, "set_sizes": list(set_sizes),
                   "ckpt_step": 1, "selection_frac": 0.0, "split_seed": 0}, f)


def write_known_run(d):
    """A finished run with hand-made outcomes. 8 puzzles; phase 1 accepts 0-2 (1 wrongly), retries 3-7.
    N1 = 10, N2 = 5; control and two perturb seeds at scale 0.1."""
    phase1 = {"set_id": np.zeros(8, int), "example_index": np.arange(8),
              "accepted": np.array([1, 1, 1, 0, 0, 0, 0, 0], bool), "halt_step": np.array([2, 3, 4, 0, 0, 0, 0, 0]),
              "correct_at_halt": np.array([1, 1, 0, 0, 0, 0, 0, 0], bool),
              "correct_at_n1": np.array([1, 1, 0, 0, 0, 0, 1, 0], bool)}
    names = ["continue", "perturb_s0.1_seed0", "perturb_s0.1_seed1"]
    #                        retried: 3   4   5   6   7
    accepted = np.array([[1, 0, 0, 1, 0],      # control
                         [0, 1, 0, 1, 1],      # seed 0
                         [1, 1, 1, 0, 0]], bool)  # seed 1
    halt = np.array([[12, 15, 15, 11, 15],
                     [15, 13, 15, 12, 14],
                     [11, 12, 14, 15, 15]])
    correct = np.array([[1, 0, 0, 1, 0],
                        [0, 1, 0, 1, 0],       # seed 0: puzzle 7 is a false acceptance
                        [1, 1, 0, 0, 0]], bool)  # seed 1: puzzle 5 is a false acceptance
    phase2 = {"set_id": np.zeros(5, int), "example_index": np.arange(3, 8), "branches": np.array(names),
              "accepted": accepted, "halt_step": halt, "correct_at_halt": correct, "correct_at_last": correct,
              "realized_scale_z_H": np.zeros((3, 5)), "realized_scale_z_L": np.zeros((3, 5))}
    _write_run(d, phase1, phase2)


def test_summary_on_known_outcomes():
    with tempfile.TemporaryDirectory() as d:
        write_known_run(d)
        s = rs.summarize(d, num_resamples=200, verbose=False)
        assert s["split"]["report_puzzles"] == 8 and s["split"]["retried_report_puzzles"] == 5
        assert (s["phase1"]["accepted"], s["phase1"]["false_acceptances"], s["phase1"]["retried_correct_at_n1"]) == (3, 1, 1)
        # Stopping at step 10: correct 0, 1, 6 -> 3 / 8; steps 2 + 3 + 4 + 5 * 10
        assert s["phase1"]["overall_stop_at_n1"]["accuracy"] == 3 / 8
        assert s["phase1"]["overall_stop_at_n1"]["avg_steps"] == (2 + 3 + 4 + 50) / 8

        b = {x["branch"]: x for x in s["branches"]}
        c, s0 = b["continue"], b["perturb_s0.1_seed0"]
        assert (c["accepted"], c["accepted_correct"], c["false_acceptances"], c["solved"]) == (2, 2, 0, 2)
        assert (s0["accepted"], s0["false_acceptances"], s0["solved"]) == (3, 1, 2)
        assert s0["mean_steps"] == (15 + 13 + 15 + 12 + 14) / 5
        p = s0["paired_with_control"]  # control solves 3, 6; seed 0 solves 4, 6
        assert (p["both"], p["variant_only"], p["control_only"], p["neither"]) == (1, 1, 1, 2) and p["difference"] == 0
        assert p["difference_ci"][0] <= 0 <= p["difference_ci"][1]
        # Overall: phase-1 correct 2 + control's 2 = 4 / 8; steps 2 + 3 + 4 + (12 + 15 + 15 + 11 + 15)
        assert c["overall"]["accuracy"] == 4 / 8 and c["overall"]["avg_steps"] == (9 + 68) / 8

        g = s["groups"][0]
        assert (g["variant"], g["scale"], g["seeds"]) == ("perturb", 0.1, [0, 1])
        assert g["across_seeds"]["solved_share"] == {"mean": 0.4, "min": 0.4, "max": 0.4}
        assert g["any_seed_oracle"]["solved"] == 3 and "ground truth" in g["any_seed_oracle"]["note"]  # puzzles 3, 4, 6
        # Sequential: seed 0 first, then seed 1
        #   3: seed 0 not accepted, seed 1 accepted at 11 (correct): 10 + 5 + 1 = 16
        #   4: seed 0 accepted at 13 (correct): 13
        #   5: seed 0 no, seed 1 accepted at 14 (wrong): 10 + 5 + 4 = 19
        #   6: seed 0 accepted at 12 (correct): 12
        #   7: seed 0 accepted at 14 (wrong): 14
        q = g["sequential"]
        assert (q["accepted"], q["solved"], q["false_acceptances"]) == (5, 3, 2)
        assert q["mean_steps"] == (16 + 13 + 19 + 12 + 14) / 5
        assert q["attempts_used"] == [3, 2]
        assert q["overall"]["accuracy"] == (2 + 3) / 8 and q["overall"]["avg_steps"] == (9 + 74) / 8
        sp = q["paired_with_control"]  # control 3, 6; sequential 3, 4, 6
        assert (sp["both"], sp["variant_only"], sp["control_only"]) == (2, 1, 0)

        # None accepted: the last attempt's answer, all attempts counted
        acc, cor, steps, first = rs.sequential(np.zeros((3, 1), bool), np.full((3, 1), 15), np.array([[1], [1], [0]], bool), 10, 5)
        assert (bool(acc[0]), bool(cor[0]), int(steps[0]), int(first[0])) == (False, False, 25, 2)

        with open(os.path.join(d, "summary.csv")) as f:
            rows = f.read().splitlines()
        assert rows[0].startswith("kind,variant") and len(rows) == 1 + 3 + 1


def test_run_experiment_with_reference_capture():
    model, batch, plain, _, settings = setup(n=24)
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "plain")
        os.makedirs(ref)
        save_capture(os.path.join(ref, "step_1_eval_steps.0"), plain)
        # Two loader batches of 16 rows (the second has 8 real puzzles and 8 padding rows)
        pad = {k: torch.cat([v[16:], v[:8]]) for k, v in batch.items()}
        batches = [("all", {k: v[:16] for k, v in batch.items()}, 16), ("all", pad, 8)]
        out = os.path.join(d, "out")
        settings.chunk_size = 16
        rs.run_experiment(model, batches, settings, out, ["all"], [24], ckpt_step=1,
                          extra_settings={"selection_frac": 0.2, "split_seed": 0})
        s = rs.summarize(out, reference=(ref, 1), num_resamples=100, verbose=False)

        ref_checks = s["reference"]
        assert ref_checks["phase1_check"]["same_acceptance"] == 24 and ref_checks["phase1_check"]["same_halt_step"] == 24
        cc = ref_checks["control_check"]
        assert cc["same_pred_hash"] == cc["same_seq_is_correct"] == cc["retried_puzzles"] > 0
        assert cc["max_abs_halt_logit_difference"] < 1e-6
        # Plain continuation to N1 + N2 is the control
        ctrl = next(b for b in s["branches"] if b["branch"] == "continue")
        plain_cont = ref_checks["plain_continuation"][0]
        assert plain_cont["budget"] == N1 + N2 and np.isclose(plain_cont["overall"]["accuracy"], ctrl["overall"]["accuracy"])
        assert np.isclose(plain_cont["overall"]["avg_steps"], ctrl["overall"]["avg_steps"])
        assert len(ref_checks["plain_continuation"]) == 1  # the capture has only N1 + N2 steps

        # The captures are readable by the analysis script, with steps numbered from N1 + 1
        cap = os.path.join(out, "captures", "perturb_s0.3_seed1")
        a = es.run(cap, 1, os.path.join(cap, "analysis"), num_resamples=0, num_thresholds=10, verbose=False)
        assert a["checks"]["step_offset"] == N1 and a["checks"]["in_loop_halting"]["matches_replay"]
        assert a["results"]["all"]["solving_rate"][1]["step"] == N1 + 2
        assert a["results"]["all"]["num_report"] == s["split"]["retried_report_puzzles"]

        # Same results with one batch of 24
        out2 = os.path.join(d, "out2")
        settings.chunk_size = None
        rs.run_experiment(model, [("all", batch, 24)], settings, out2, ["all"], [24], ckpt_step=1,
                          extra_settings={"selection_frac": 0.2, "split_seed": 0})
        for f in ("phase1.npz", "phase2.npz"):
            x, y = np.load(os.path.join(out, f)), np.load(os.path.join(out2, f))
            assert all(np.array_equal(x[k], y[k]) for k in x.files), f

        # --max-puzzles
        out3 = os.path.join(d, "out3")
        rs.run_experiment(model, batches, settings, out3, ["all"], [24], ckpt_step=1, max_puzzles=20)
        assert len(np.load(os.path.join(out3, "phase1.npz"))["set_id"]) == 20


def test_no_puzzle_retried():
    model, batch, _, _, settings = setup(n=12)
    settings.threshold = -1e9  # everything accepted at step 1
    with tempfile.TemporaryDirectory() as d:
        rs.run_experiment(model, [("all", batch, 12)], settings, d, ["all"], [12], ckpt_step=1,
                          extra_settings={"selection_frac": 0.0, "split_seed": 0})
        s = rs.summarize(d, num_resamples=10, verbose=False)
        assert s["split"]["retried_report_puzzles"] == 0 and s["phase1"]["accepted"] == 12
        assert s["branches"][0]["retried"] == 0 and s["branches"][0]["solved_share"] is None


def short_settings(settings, scales=(0.0, 0.5), lengths=(2, 4)):
    """Noisy restarts only, attempts of the given lengths (N2 = 4: 2 x 2 and 1 x 4 steps)."""
    settings.perturb_scales, settings.restart_scales = [], list(scales)
    settings.seeds, settings.attempt_steps = [], list(lengths)
    return settings


def test_attempt_branches():
    names = [b.name for b in rs.make_branches([0.1], [0.3], [], n2=8, attempt_steps=[4, 8])]
    assert names == ["continue", "perturb_s0.1_a4_seed0", "perturb_s0.1_a4_seed1", "noisy_restart_s0.3_a4_seed0",
                     "noisy_restart_s0.3_a4_seed1", "perturb_s0.1_a8_seed0", "noisy_restart_s0.3_a8_seed0"]
    for name in names[1:]:
        assert rs._parse(name)[:2] in {("perturb", 0.1), ("noisy_restart", 0.3)}
    assert rs._parse("noisy_restart_s0.3_a4_seed1") == ("noisy_restart", 0.3, 1)
    assert rs._parse("perturb_s0.001_seed2") == ("perturb", 0.001, 2)  # default mode: names unchanged
    assert [b.length(8) for b in rs.make_branches([], [0.3], [0], n2=8)] == [8, 8]
    try:
        rs.make_branches([], [0.3], [], n2=8, attempt_steps=[3])
        raise AssertionError("expected an error")
    except ValueError as e:
        assert "divide" in str(e)


def test_short_attempts():
    model, batch, plain, rec, settings = setup()
    out = run(model, batch, short_settings(settings))
    retried = out["retried"]
    assert np.array_equal(out["phase1_pred_hash"], rec["pred_hash"][retried, :N1].numpy())
    assert list(out["branches"]) == ["continue", "noisy_restart_s0_a2_seed0", "noisy_restart_s0_a2_seed1",
                                     "noisy_restart_s0.5_a2_seed0", "noisy_restart_s0.5_a2_seed1",
                                     "noisy_restart_s0_a4_seed0", "noisy_restart_s0.5_a4_seed0"]
    logits = rec["q_halt_logits"].numpy()
    thr = float(np.float32(settings.threshold))
    for name, b in out["branches"].items():
        length = 4 if name == "continue" or "_a4_" in name else 2
        assert all(v.shape == (len(retried), length) for v in b["records"].values()), name
        assert np.array_equal(b["final_pred_hash"], b["records"]["pred_hash"][:, -1].numpy())
        assert ((b["halt_step"] > N1) & (b["halt_step"] <= N1 + length)).all()
        # Not accepted within the attempt: the answer of its last step
        assert np.array_equal(b["correct_at_halt"][~b["accepted"]], b["correct_at_last"][~b["accepted"]])
        if "_s0_" in name:
            # Without noise, a restart repeats the plain run's first steps
            for key in ("pred_hash", "q_halt_logits", "seq_is_correct"):
                assert torch.equal(b["records"][key], rec[key][retried, :length]), (name, key)
            h = N1 + es.halting_steps(logits[retried, :length], thr)
            assert np.array_equal(b["halt_step"], h)
    # One attempt of N2 steps is the default mode's attempt with the same seed
    default = run(model, batch, rs.Settings(n1=N1, n2=N2, threshold=settings.threshold, perturb_scales=[],
                                            restart_scales=[0.5], seeds=[0]))
    for k in ("accepted", "halt_step", "correct_at_halt", "final_pred_hash"):
        assert np.array_equal(out["branches"]["noisy_restart_s0.5_a4_seed0"][k], default["branches"]["noisy_restart_s0.5_seed0"][k])
    # Chunking and padding change nothing
    settings.chunk_size = 5
    chunked = run(model, batch, settings)
    for name, b in out["branches"].items():
        for k in ("accepted", "halt_step", "correct_at_halt", "final_pred_hash"):
            assert np.array_equal(b[k], chunked["branches"][name][k]), (name, k)


def test_sampled_puzzles():
    a = rs.sample_positions([10, 14], 9, seed=3)
    assert np.array_equal(a, rs.sample_positions([10, 14], 9, seed=3)) and len(np.unique(a)) == 9 and (np.diff(a) > 0).all()
    assert not np.array_equal(a, rs.sample_positions([10, 14], 9, seed=4))
    assert rs.sample_positions([10, 14], 24, seed=3) is None and rs.sample_positions([10, 14], None, seed=3) is None

    # Two sets, loader batches of 4 (the last of each set padded); the sample repacked into batches of 4
    data = {"inputs": torch.arange(24).view(24, 1), "labels": torch.arange(24).view(24, 1)}
    loader = []
    for name, lo, hi in (("a", 0, 10), ("b", 10, 24)):
        for s in range(lo, hi, 4):
            rows = torch.arange(s, s + 4).clamp(max=hi - 1)
            loader.append((name, {k: v[rows] for k, v in data.items()}, min(4, hi - s)))
    out = list(rs.keyed_batches(loader, {"a": 0, "b": 1}, [10, 14], a))
    assert all(b["inputs"].shape[0] == 4 for b, _, _ in out) and len(out) == 3
    sid = np.concatenate([s for _, s, _ in out])
    ex = np.concatenate([e for _, _, e in out])
    assert np.array_equal(np.where(sid == 0, 0, 10) + ex, a)
    rows = torch.cat([b["inputs"][:len(s), 0] for b, s, _ in out]).numpy()
    assert np.array_equal(rows, a)  # each kept row is the sampled puzzle
    # Without a sample: the loader batches as they are
    plain = list(rs.keyed_batches(loader, {"a": 0, "b": 1}, [10, 14]))
    assert len(plain) == len(loader) and np.array_equal(plain[2][2], [8, 9]) and np.array_equal(plain[3][2], [0, 1, 2, 3])

    # Same sample and same results whatever the loader's batches
    model, batch, _, _, settings = setup(n=24)
    settings.chunk_size = 8
    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full")
        rs.run_experiment(model, [("all", batch, 24)], settings, full, ["all"], [24], ckpt_step=1)
        p_full = np.load(os.path.join(full, "phase1.npz"))
        sample = rs.sample_positions([24], 10, seed=0)
        for i, batches in enumerate(([("all", batch, 24)],
                                     [("all", {k: v[:16] for k, v in batch.items()}, 16),
                                      ("all", {k: torch.cat([v[16:], v[:8]]) for k, v in batch.items()}, 8)])):
            out = os.path.join(d, f"s{i}")
            rs.run_experiment(model, batches, settings, out, ["all"], [24], ckpt_step=1, max_puzzles=10, sample_seed=0)
            p = np.load(os.path.join(out, "phase1.npz"))
            assert np.array_equal(p["example_index"], sample)
            for k in p.files:
                assert np.array_equal(p[k], p_full[k][sample]), k
            with open(os.path.join(out, "settings.json")) as f:
                assert json.load(f)["sample_seed"] == 0


def write_short_run(d):
    """A finished run of the tiny model with noisy restarts of 2 x 2 and 1 x 4 steps (N2 = 4), summarized."""
    model, batch, _, _, settings = setup(n=24)
    short_settings(settings)
    rs.run_experiment(model, [("all", batch, 24)], settings, d, ["all"], [24], ckpt_step=1,
                      extra_settings={"selection_frac": 0.0, "split_seed": 0})
    rs.summarize(d, num_resamples=50, verbose=False)
    return d


def test_summary_of_short_attempts():
    model, batch, plain, rec, settings = setup(n=24)
    short_settings(settings)
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "plain")
        os.makedirs(ref)
        save_capture(os.path.join(ref, "step_1_eval_steps.0"), plain)
        out = os.path.join(d, "out")
        rs.run_experiment(model, [("all", batch, 24)], settings, out, ["all"], [24], ckpt_step=1,
                          extra_settings={"selection_frac": 0.0, "split_seed": 0})
        s = rs.summarize(out, reference=(ref, 1), num_resamples=100, verbose=False)
        p1 = np.load(os.path.join(out, "phase1.npz"))
        retried = np.flatnonzero(~p1["accepted"])
        R = len(retried)
        assert s["split"]["retried_report_puzzles"] == R > 0
        correct = rec["seq_is_correct"].numpy()[retried]
        before = correct[:, N1 - 1]

        # Control: solving rate at steps N1 + 1 ... N1 + N2, as in the plain run
        rate = s["control_solving_rate"]
        assert [r["step"] for r in rate] == list(range(N1 + 1, N1 + N2 + 1))
        for k, r in enumerate(rate):
            prev = before if k == 0 else correct[:, N1 + k - 1]
            assert (r["unsolved_at_previous_step"], r["solved"]) == ((~prev).sum(), (~prev & correct[:, N1 + k]).sum())
            assert "rate_ci" in r or r["rate"] is None

        g = {(x["scale"], x["attempt_steps"]): x for x in s["groups"]}
        assert sorted(g) == [(0.0, 2), (0.0, 4), (0.5, 2), (0.5, 4)]
        assert [(x["attempts"], x["budget"]) for x in (g[0.0, 2], g[0.0, 4])] == [(2, N1 + N2), (1, N1 + N2)]
        assert "same total budget" in g[0.0, 2]["sequential"]["note"]

        # Noise 0: both attempts repeat the plain run's first 2 steps
        zero = g[0.0, 2]
        for k, r in enumerate(zero["solving_rate"]):
            prev = before if k == 0 else correct[:, k - 1]
            n, solved = (~prev).sum(), (~prev & correct[:, k]).sum()
            assert (r["step"], r["unsolved_at_previous_step"], r["solved"]) == (k + 1, 2 * n, 2 * solved)
            assert "rate_ci_approx" in r or r["rate"] is None
        div = zero["diversity"]
        assert div["distinct_final_grids"] == {"mean": 1.0, "counts": [R, 0]}
        assert div["same_as_unperturbed"]["share_of_attempts"] == 1.0 and div["same_as_unperturbed"]["counts"] == [0, 0, R]
        assert "step 2" in div["same_as_unperturbed"]["unperturbed"]
        # Sequential with identical attempts: the first attempt decides when it is accepted
        h = es.halting_steps(rec["q_halt_logits"].numpy()[retried, :2], float(np.float32(settings.threshold)))
        acc = rec["q_halt_logits"].numpy()[retried, :2].max(axis=1) >= float(np.float32(settings.threshold))
        seq = zero["sequential"]
        assert seq["accepted"] == acc.sum() and seq["solved"] == correct[np.arange(R), h - 1].sum()
        assert np.isclose(seq["mean_steps"], np.mean(np.where(acc, N1 + h, N1 + 2 + 2)))
        assert seq["attempts_used"] == [int(acc.sum()), int(R - acc.sum())]

        # Noise 0.5: diversity counts add up; the reference covers the budget N1 + 2 x 2
        div = g[0.5, 2]["diversity"]
        assert sum(div["distinct_final_grids"]["counts"]) == R and sum(div["same_as_unperturbed"]["counts"]) == R
        assert g[0.5, 2]["sequential"]["paired_with_plain_same_budget"]["budget"] == N1 + N2
        assert s["reference"]["control_check"]["same_pred_hash"] == R

        # A reference with only N1 steps still checks phase 1
        short_ref = os.path.join(d, "short")
        os.makedirs(short_ref)
        save_capture(os.path.join(short_ref, "step_1_eval_steps.0"), {"step_records": {k: v[:N1] for k, v in plain["step_records"].items()}})
        s2 = rs.summarize(out, reference=(short_ref, 1), num_resamples=10, verbose=False)
        assert s2["reference"]["phase1_check"]["same_acceptance"] == 24
        assert "fewer than" in s2["reference"]["control_check"] and s2["reference"]["plain_continuation"] == []

        with open(os.path.join(out, "summary.csv")) as f:
            rows = f.read().splitlines()
        assert "attempt_steps" in rows[0] and len(rows) == 1 + 7 + 4  # no plain rows: the last summary had the short reference


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
