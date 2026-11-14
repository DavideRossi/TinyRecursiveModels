"""Early-halting analysis of the per-step evaluation capture (eval_capture_steps=True).

Standalone: reads the capture files only (no model, no GPU, no training code).

    python analysis/eval_steps.py CKPT_DIR STEP OUT_DIR [options]

CKPT_DIR holds step_<STEP>_eval_steps.<rank> files (one per rank), written by
pretrain.evaluate. Each is a torch.save dict with

    q_halt_logits, aux_halt_logits/<name>   float32 [P, S]   halt logit per step
    seq_is_correct                          bool    [P, S]   all cells correct at the step
    puzzle_identifiers, example_index, set_id, num_steps     [P]
    set_names                               list of str (indexed by set_id)
  and, in newer captures,
    preds_changed                           bool    [P, S]   predicted grid differs from the previous step
    z_H_rel_change, z_L_rel_change          float   [P, S]   relative change of the states from the previous step
    pred_hash                               int64   [P, S]   hash of the predicted grid (equal grids, equal hashes)
    halt_step, eval_halt_head, eval_halt_threshold           in-loop halting, if it was enabled

The capture always holds every step up to the evaluation step limit S (any value), whatever
halting was applied in the loop. --truncate-steps K repeats the analysis on the first K steps:
a capture with 32 steps contains the 16-step evaluation as a prefix.

Halting rule for a threshold t: a puzzle halts at the first step whose halt logit
is >= t, or at the last step if none is; its outcome is seq_is_correct at that
step. Steps are counted from 1. (In training the model halts when logit > 0;
the two rules differ only for logits exactly equal to 0.)

The curves are computed exactly, not on a grid: with M the running maximum of
the logits along the steps, the halting step is h(t) = 1 + #{s < S : M[s] < t}.
As t rises past M[i, s], puzzle i moves from halting at step s to step s + 1,
which changes its correctness by c[i, s+1] - c[i, s]. Sorting these events once
gives accuracy, average steps and fraction halted early for any set of
thresholds via cumulative sums and a binary search.

Puzzles (identified by test-set id + example_index) are split at random into a
selection split, used only to choose thresholds, and a report split, on which
everything is reported. Bootstrap intervals resample report puzzles; since each
puzzle's outcome at a fixed threshold is one of S x 2 x 2 categories (halting
step, correct there, correct at the last step), a resample is a multinomial
draw over those category counts.

Outputs in OUT_DIR:
    curve_<set>_<head>.csv   threshold, avg_steps, accuracy, frac_halted_early
                             (curve_<set>_<head>_first<K>.csv for --truncate-steps)
    summary.json             checks, split, per set: full-step accuracy, fixed-step
                             table, oracle, flips, t = 0 point, operating points (each with
                             the number of puzzles halting at every step), share of unsolved
                             puzzles whose grid did not change, quartiles of their relative
                             state change, solving rate per step, repeated grids and cycles
                             among the puzzles unsolved at the last step; under "truncated"
                             the same for the first K steps
    curves_<set>.png         accuracy vs average steps (needs matplotlib; skipped otherwise)
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

AUX_PREFIX = "aux_halt_logits/"
Q_HEAD = "q_head"
PER_PUZZLE_KEYS = ("puzzle_identifiers", "example_index", "set_id", "num_steps")
OPTIONAL_KEYS = ("preds_changed", "z_H_rel_change", "z_L_rel_change", "pred_hash", "halt_step")  # newer captures
HALT_META_KEYS = ("eval_halt_head", "eval_halt_threshold")


# --------------------------------------------------------------------------- loading


def capture_files(ckpt_dir: str, step: int) -> List[str]:
    pattern = re.compile(rf"step_{step}_eval_steps\.(\d+)$")
    files = []
    for path in glob.glob(os.path.join(ckpt_dir, f"step_{step}_eval_steps.*")):
        m = pattern.search(os.path.basename(path))
        if m:
            files.append((int(m.group(1)), path))
    if not files:
        sys.exit(f"No step_{step}_eval_steps.<rank> files in {ckpt_dir}")
    return [p for _, p in sorted(files)]


def read_capture_file(path: str) -> dict:
    import torch  # only needed to read the torch.save format

    raw = torch.load(path, map_location="cpu", weights_only=True)
    out = {"set_names": list(raw["set_names"]), "logits": {}}
    for k, v in raw.items():
        if k == "q_halt_logits":
            out["logits"][Q_HEAD] = v.numpy()
        elif k.startswith(AUX_PREFIX):
            out["logits"][k[len(AUX_PREFIX):]] = v.numpy()
        elif k in PER_PUZZLE_KEYS or k in OPTIONAL_KEYS or k == "seq_is_correct":
            out[k] = v.numpy()
        elif k in HALT_META_KEYS:
            out[k] = v
    return out


def merge_captures(parts: List[dict], files: List[str]) -> dict:
    """Concatenate per-rank captures, checking they describe the same evaluation."""
    first = parts[0]
    for part, path in zip(parts[1:], files[1:]):
        if part["set_names"] != first["set_names"]:
            raise ValueError(f"{path}: set_names {part['set_names']} differ from {first['set_names']}")
        if list(part["logits"]) != list(first["logits"]):
            raise ValueError(f"{path}: heads {list(part['logits'])} differ from {list(first['logits'])}")
        if part["seq_is_correct"].shape[1] != first["seq_is_correct"].shape[1]:
            raise ValueError(f"{path}: number of steps differs between files")
        for k in OPTIONAL_KEYS + HALT_META_KEYS:
            if (k in part) != (k in first) or (k in HALT_META_KEYS and part.get(k) != first.get(k)):
                raise ValueError(f"{path}: '{k}' differs between files")

    data = {
        "set_names": first["set_names"],
        "logits": {h: np.concatenate([p["logits"][h] for p in parts]) for h in first["logits"]},
        "seq_is_correct": np.concatenate([p["seq_is_correct"] for p in parts]),
    }
    for k in PER_PUZZLE_KEYS + OPTIONAL_KEYS:
        if k in first:
            data[k] = np.concatenate([p[k] for p in parts])
    for k in HALT_META_KEYS:
        if k in first:
            data[k] = first[k]
    return data


def _rows_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eq = a == b
    if np.issubdtype(a.dtype, np.floating):
        eq |= np.isnan(a) & np.isnan(b)
    return eq.reshape(len(a), -1).all(axis=1)


def deduplicate(data: dict) -> (dict, dict):
    """Keep one row per (set_id, example_index), sorted by that key."""
    set_id = data["set_id"].astype(np.int64)
    example = data["example_index"].astype(np.int64)
    key = set_id * (int(example.max()) + 1) + example
    _, first, inverse, counts = np.unique(key, return_index=True, return_inverse=True, return_counts=True)

    duplicated_rows = int(len(key) - len(first))
    inconsistent = 0
    if duplicated_rows:
        ref = first[inverse]  # row kept for each row's key
        same = np.ones(len(key), dtype=bool)
        for arr in [data["seq_is_correct"], data["puzzle_identifiers"], data["num_steps"], *data["logits"].values(),
                    *[data[k] for k in OPTIONAL_KEYS if k in data]]:
            same &= _rows_equal(arr, arr[ref])
        inconsistent = int(np.unique(key[~same]).size)

    out = {"set_names": data["set_names"], "logits": {h: v[first] for h, v in data["logits"].items()},
           "seq_is_correct": data["seq_is_correct"][first]}
    for k in PER_PUZZLE_KEYS + OPTIONAL_KEYS:
        if k in data:
            out[k] = data[k][first]
    for k in HALT_META_KEYS:
        if k in data:
            out[k] = data[k]
    report = {
        "rows_loaded": int(len(key)),
        "unique_puzzles": int(len(first)),
        "duplicate_rows_dropped": duplicated_rows,
        "puzzles_with_duplicates": int((counts > 1).sum()),
        "duplicates_with_differing_values": inconsistent,
    }
    return out, report


def check_missing(data: dict, data_dir: Optional[str]) -> dict:
    """Per set: expected count (from the dataset if given, else max index + 1) vs present."""
    report = {}
    for sid, name in enumerate(data["set_names"]):
        idx = data["example_index"][data["set_id"] == sid]
        expected, source = (int(idx.max()) + 1 if len(idx) else 0), "max example_index + 1"
        if data_dir is not None:
            path = os.path.join(data_dir, "test", f"{name}__inputs.npy")
            expected, source = int(np.load(path, mmap_mode="r").shape[0]), path
        in_range = idx[(idx >= 0) & (idx < expected)]
        report[name] = {
            "expected": expected,
            "expected_source": source,
            "present": int(len(idx)),
            "missing": int(expected - np.unique(in_range).size),
            "out_of_range": int(len(idx) - len(in_range)),
        }
    return report


def load(ckpt_dir: str, step: int, data_dir: Optional[str] = None) -> (dict, dict):
    files = capture_files(ckpt_dir, step)
    data = merge_captures([read_capture_file(f) for f in files], files)
    data, dup_report = deduplicate(data)

    num_steps = data["seq_is_correct"].shape[1]
    if not np.all(data["num_steps"] == num_steps):
        raise ValueError(f"num_steps differs between puzzles ({np.unique(data['num_steps'])}); "
                         "evaluation is expected to run halt_max_steps for every puzzle")
    for head, logits in data["logits"].items():
        if np.isnan(logits).any():
            raise ValueError(f"NaN halt logits for head {head}")

    checks = {"files": files, "num_steps": num_steps, "heads": list(data["logits"]),
              "set_names": data["set_names"], **dup_report, "per_set": check_missing(data, data_dir)}
    if "halt_step" in data:
        checks["in_loop_halting"] = in_loop_halting(data)
    return data, checks


def in_loop_halting(data: dict) -> dict:
    """Result of the halting applied during evaluation, and its replay from the captured logits."""
    head, threshold = data["eval_halt_head"], float(np.float32(data["eval_halt_threshold"]))
    correct, halt_step = data["seq_is_correct"], data["halt_step"].astype(np.int64)
    S = correct.shape[1]
    replay_step = halting_steps(data["logits"][head], threshold)
    replay_correct = correct[np.arange(len(correct)), replay_step - 1]
    in_loop_correct = correct[np.arange(len(correct)), halt_step - 1]
    return {
        "head": head, "threshold": threshold,
        "accuracy": float(in_loop_correct.mean()), "avg_steps": float(halt_step.mean()),
        "frac_halted_early": float((halt_step < S).mean()),
        "replay_accuracy": float(replay_correct.mean()), "replay_avg_steps": float(replay_step.mean()),
        "puzzles_with_different_halt_step": int((replay_step != halt_step).sum()),
        "matches_replay": bool(np.array_equal(replay_step, halt_step)),
    }


# --------------------------------------------------------------------------- core computations


class HaltCurve:
    """Exact accuracy / average steps / fraction halted early as functions of the threshold."""

    def __init__(self, logits: np.ndarray, correct: np.ndarray):
        P, S = correct.shape
        self.P, self.S = P, S
        c = correct.astype(np.int64)
        M = np.maximum.accumulate(logits.astype(np.float64), axis=1)

        events = M[:, :S - 1].ravel()
        delta = (c[:, 1:] - c[:, :-1]).ravel()
        order = np.argsort(events, kind="stable")
        self.event_values = events[order]
        self.cum_delta = np.concatenate([[0], np.cumsum(delta[order])])
        self.correct_at_first = int(c[:, 0].sum())
        self.last_event = np.sort(M[:, S - 2]) if S > 1 else np.empty(0)

    def __call__(self, thresholds) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
        below = np.searchsorted(self.event_values, t, side="left")  # events with M < t
        reached_last = np.searchsorted(self.last_event, t, side="left") if self.S > 1 else np.full(t.shape, self.P)
        return {
            "avg_steps": (self.P + below) / self.P,
            "accuracy": (self.correct_at_first + self.cum_delta[below]) / self.P,
            "frac_halted_early": 1.0 - reached_last / self.P,
        }

    def level_thresholds(self) -> np.ndarray:
        """One threshold per distinct curve point: -inf, midpoints between event values, +inf."""
        u = np.unique(self.event_values)
        return np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])


def halting_steps(logits: np.ndarray, threshold: float) -> np.ndarray:
    """Per-puzzle halting step (1-based) under the threshold rule."""
    M = np.maximum.accumulate(logits.astype(np.float64), axis=1)
    return 1 + (M[:, :-1] < threshold).sum(axis=1)


def fixed_step_accuracy(correct: np.ndarray) -> List[float]:
    return correct.mean(axis=0).tolist()


def oracle(correct: np.ndarray) -> dict:
    ever = correct.any(axis=1)
    steps = np.where(ever, correct.argmax(axis=1) + 1, correct.shape[1])
    return {"accuracy": float(ever.mean()), "avg_steps": float(steps.mean())}


def flips(correct: np.ndarray) -> dict:
    c = correct.astype(bool)
    correct_before = np.zeros_like(c)
    correct_before[:, 1:] = np.logical_or.accumulate(c, axis=1)[:, :-1]
    wrong_after_correct = ~c & correct_before
    wac_before = np.zeros_like(c)
    wac_before[:, 1:] = np.logical_or.accumulate(wrong_after_correct, axis=1)[:, :-1]
    return {
        "num_puzzles": int(len(c)),
        "correct_at_some_step": int(c.any(axis=1).sum()),
        "correct_at_some_step_but_wrong_at_last": int((c.any(axis=1) & ~c[:, -1]).sum()),
        "correct_then_wrong_then_correct": int((c & wac_before).any(axis=1).sum()),
    }


def unchanged_among_unsolved(correct: np.ndarray, preds_changed: np.ndarray) -> List[Optional[dict]]:
    """Per step s >= 2: of the puzzles wrong at step s - 1, the share whose grid is the same at step s.

    Entry s - 1 of the list describes step s; step 1 has no previous step (None).
    """
    out: List[Optional[dict]] = [None]
    for s in range(1, correct.shape[1]):
        unsolved = ~correct[:, s - 1]
        unchanged = unsolved & ~preds_changed[:, s]
        n = int(unsolved.sum())
        out.append({"step": s + 1, "unsolved_at_previous_step": n, "unchanged": int(unchanged.sum()),
                    "share": float(unchanged.sum() / n) if n else None})
    return out


def state_change_among_unsolved(correct: np.ndarray, rel_change: np.ndarray) -> List[Optional[dict]]:
    """Per step s >= 2: quartiles of the relative state change at step s, over puzzles wrong at step s - 1.

    Same layout and same puzzles as unchanged_among_unsolved. Non-finite values are ignored.
    """
    out: List[Optional[dict]] = [None]
    for s in range(1, correct.shape[1]):
        values = rel_change[~correct[:, s - 1], s].astype(np.float64)
        values = values[np.isfinite(values)]
        q = np.percentile(values, [25, 50, 75]).tolist() if len(values) else [None, None, None]
        out.append({"step": s + 1, "unsolved_at_previous_step": int((~correct[:, s - 1]).sum()),
                    "q25": q[0], "median": q[1], "q75": q[2]})
    return out


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> List[float]:
    """Wilson score interval (95% by default) for a binomial proportion."""
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [0.0 if successes == 0 else max(0.0, center - half), 1.0 if successes == n else min(1.0, center + half)]


def solving_rate(correct: np.ndarray) -> List[Optional[dict]]:
    """Per step s >= 2: of the puzzles wrong at step s - 1, the fraction correct at step s.

    Entry s - 1 describes step s (None for step 1). "unsolved" is the number wrong at step s.
    """
    out: List[Optional[dict]] = [None]
    for s in range(1, correct.shape[1]):
        n = int((~correct[:, s - 1]).sum())
        solved = int((~correct[:, s - 1] & correct[:, s]).sum())
        out.append({"step": s + 1, "unsolved_at_previous_step": n, "solved": solved,
                    "rate": solved / n if n else None, "rate_ci": wilson_interval(solved, n) if n else None,
                    "unsolved": int((~correct[:, s]).sum())})
    return out


CYCLE_CLASSES = ("frozen", "cycling", "revisiting", "wandering")


def previous_occurrence(hashes: np.ndarray) -> np.ndarray:
    """[U, S] -> for each step, the index of the most recent earlier step of the same row with
    the same hash, or -1. One stable sort per row: equal hashes become neighbours, in step order."""
    order = np.argsort(hashes, axis=1, kind="stable")
    ordered = np.take_along_axis(hashes, order, axis=1)
    prev_sorted = np.full(order.shape, -1, dtype=np.int64)
    prev_sorted[:, 1:] = np.where(ordered[:, 1:] == ordered[:, :-1], order[:, :-1], -1)
    prev = np.empty_like(prev_sorted)
    np.put_along_axis(prev, order, prev_sorted, axis=1)
    return prev


def _periodic_from(hashes: np.ndarray, start: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Per row: hashes[s] == hashes[s - period] for every index s >= start (rows with period 0: False)."""
    out = np.zeros(len(hashes), dtype=bool)
    S = hashes.shape[1]
    for p in np.unique(period[period > 0]):
        rows = np.flatnonzero(period == p)
        equal = hashes[rows, p:] == hashes[rows, :-p]
        out[rows] = (equal | (np.arange(p, S)[None, :] < start[rows, None])).all(axis=1)
    return out


def cycle_records(hashes: np.ndarray, chunk: int = 50000) -> Dict[str, np.ndarray]:
    """Per row of grid hashes [U, S] (steps counted from 1 in the results):

    first_repeat_step  first step whose grid equals a grid of an earlier step (0: no grid repeats)
    period             at that step, the distance to the most recent identical grid (0: none)
    confirmed          from that step to the last, every grid equals the one `period` steps before
    distinct           number of distinct grids
    ends_periodic      with p the distance from the last grid to its most recent copy, the last p
                       grids all equal the ones p steps before (one full period at the end)
    """
    parts: Dict[str, list] = {k: [] for k in ("first_repeat_step", "period", "confirmed", "distinct", "ends_periodic")}
    S = hashes.shape[1]
    for a in range(0, len(hashes), chunk):
        h = hashes[a:a + chunk]
        rows = np.arange(len(h))
        prev = previous_occurrence(h)
        repeat = prev >= 0
        any_repeat = repeat.any(axis=1)
        first = repeat.argmax(axis=1)
        period = np.where(any_repeat, first - prev[rows, first], 0)
        last_period = np.where(prev[:, -1] >= 0, S - 1 - prev[:, -1], 0)
        last_period[2 * last_period > S] = 0  # a full period at the end cannot be checked
        parts["first_repeat_step"].append(np.where(any_repeat, first + 1, 0))
        parts["period"].append(period)
        parts["confirmed"].append(_periodic_from(h, first, period))
        parts["distinct"].append((~repeat).sum(axis=1))
        parts["ends_periodic"].append(_periodic_from(h, S - last_period, last_period))
    if not parts["period"]:
        return {k: np.zeros(0, dtype=bool if k in ("confirmed", "ends_periodic") else np.int64) for k in parts}
    return {k: np.concatenate(v) for k, v in parts.items()}


def classify_cycles(records: Dict[str, np.ndarray]) -> np.ndarray:
    """Index into CYCLE_CLASSES per row: frozen (period 1, confirmed to the end), cycling
    (period > 1, confirmed), revisiting (a grid repeats, not periodic), wandering (no repeat)."""
    repeats = records["first_repeat_step"] > 0
    return np.where(~repeats, 3, np.where(~records["confirmed"], 2, np.where(records["period"] == 1, 0, 1)))


def _value_counts(values: np.ndarray) -> Dict[str, int]:
    u, n = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u, n)}


def cycle_analysis(correct: np.ndarray, hashes: np.ndarray) -> dict:
    """Repeated grids and cycles among the puzzles unsolved at the last step.

    Equal hashes mean equal predicted grids; they do not prove the latent state repeats, which
    is why a cycle only counts when it is confirmed up to the last step.
    """
    P, S = correct.shape
    unsolved_last = ~correct[:, -1]
    rec = cycle_records(hashes[unsolved_last])
    cls = classify_cycles(rec)
    U = int(unsolved_last.sum())
    confirmed = cls <= 1
    steps = np.arange(1, S + 1)

    # Per step: puzzles unsolved at that step that are already in their confirmed cycle
    wrong = ~correct[unsolved_last]
    in_cycle = confirmed[:, None] & (rec["first_repeat_step"][:, None] <= steps[None, :]) & wrong
    by_step = []
    for s in range(S):
        n = int((~correct[:, s]).sum())
        frozen, cycling = int(in_cycle[cls == 0, s].sum()), int(in_cycle[cls == 1, s].sum())
        by_step.append({"step": s + 1, "unsolved": n, "unsolved_at_last": int(wrong[:, s].sum()),
                        "in_frozen": frozen, "in_cycling": cycling,
                        "share_in_confirmed_cycle": (frozen + cycling) / n if n else None})

    step_counts = lambda v: np.bincount(v, minlength=S + 1)[1:].tolist()  # entry s - 1 is step s
    repeats = rec["first_repeat_step"] > 0
    return {
        "num_unsolved_at_last": U,
        "classes": {name: {"count": int((cls == i).sum()), "share": float((cls == i).mean()) if U else None}
                    for i, name in enumerate(CYCLE_CLASSES)},
        "repeats": {
            "with_any_repeat": int(repeats.sum()),
            "first_repeat_step_counts": step_counts(rec["first_repeat_step"][repeats]),
            "first_repeat_period_counts": _value_counts(rec["period"][repeats]),
            "distinct_grids_counts": step_counts(rec["distinct"]),  # entry d - 1: puzzles with d distinct grids
        },
        "confirmed": {
            "period_counts": _value_counts(rec["period"][confirmed]),
            "start_step_counts": step_counts(rec["first_repeat_step"][confirmed]),
        },
        # Not confirmed from the first repeat, but the last full period repeats: an early
        # transient repeat followed by a cycle. Counted as revisiting above.
        "revisiting_ending_periodic": int(((cls == 2) & rec["ends_periodic"]).sum()),
        "by_step": by_step,
    }


def halt_step_counts(logits: np.ndarray, threshold: float) -> List[int]:
    """Number of puzzles halting at each step (entry s - 1 is step s; the last entry includes
    the puzzles that never reach the threshold)."""
    return np.bincount(halting_steps(logits, threshold), minlength=logits.shape[1] + 1)[1:].tolist()


def select_threshold(curve: HaltCurve, tolerance: float) -> dict:
    """Fewest average steps with accuracy >= full-step accuracy - tolerance (on curve's puzzles)."""
    thresholds = curve.level_thresholds()
    pts = curve(thresholds)
    full = float(curve(np.inf)["accuracy"][0])
    feasible = pts["accuracy"] >= full - tolerance - 1e-12
    # Fewest steps; among equal steps the highest accuracy
    candidates = np.flatnonzero(feasible)
    best = candidates[np.lexsort((-pts["accuracy"][candidates], pts["avg_steps"][candidates]))[0]]
    return {"threshold": float(thresholds[best]), "accuracy": float(pts["accuracy"][best]),
            "avg_steps": float(pts["avg_steps"][best]), "full_step_accuracy": full}


def evaluate_with_ci(logits: np.ndarray, correct: np.ndarray, threshold: float,
                     num_resamples: int, rng: np.random.Generator, ci: float = 0.95) -> dict:
    P, S = correct.shape
    h = halting_steps(logits, threshold)
    c_h = correct[np.arange(P), h - 1].astype(np.int64)
    c_last = correct[:, -1].astype(np.int64)
    counts = np.bincount((h - 1) * 4 + c_h * 2 + c_last, minlength=S * 4)

    cat_steps = np.repeat(np.arange(1, S + 1), 4)
    cat_c_h = np.tile([0, 0, 1, 1], S)
    cat_c_last = np.tile([0, 1, 0, 1], S)

    def stats(n):  # n: [..., 4S] category counts
        return {
            "accuracy": (n * cat_c_h).sum(-1) / P,
            "avg_steps": (n * cat_steps).sum(-1) / P,
            "accuracy_loss": (n * (cat_c_last - cat_c_h)).sum(-1) / P,
        }

    point = stats(counts)
    boot = stats(rng.multinomial(P, counts / P, size=num_resamples)) if num_resamples > 0 else None
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out = {"threshold": threshold, "frac_halted_early": float((h < S).mean()),
           "halt_step_counts": np.bincount(h, minlength=S + 1)[1:].tolist()}
    for k, v in point.items():
        out[k] = float(v)
        if boot is not None:
            out[f"{k}_ci"] = [float(np.percentile(boot[k], lo)), float(np.percentile(boot[k], hi))]
    return out


def curve_grid(curve: HaltCurve, num_thresholds: int) -> np.ndarray:
    """Quantiles of the event values (where the curve changes), plus -inf, 0, +inf.

    -inf and +inf are the two ends of the curve (always step 1, always the last step).
    """
    q = np.quantile(curve.event_values, np.linspace(0, 1, num_thresholds)) if len(curve.event_values) else []
    return np.unique(np.concatenate([[-np.inf, 0.0, np.inf], q]))


# --------------------------------------------------------------------------- analysis driver


def split_puzzles(num_puzzles: int, selection_frac: float, seed: int) -> np.ndarray:
    """Boolean mask: True = selection split. Rows are sorted by (set_id, example_index)."""
    return np.random.default_rng(seed).random(num_puzzles) < selection_frac


def analyze_group(data: dict, rows: np.ndarray, selection: np.ndarray, tolerances: List[float],
                  num_resamples: int, num_thresholds: int, rng: np.random.Generator,
                  truncate: Optional[int] = None) -> (dict, dict):
    """truncate=K: use only the first K steps, as if the evaluation step limit had been K."""
    steps = slice(None, truncate)
    correct = data["seq_is_correct"][rows][:, steps]
    sel, rep = selection[rows], ~selection[rows]
    c_rep = correct[rep]

    result = {
        "num_steps": int(correct.shape[1]),
        "num_puzzles": int(rows.sum()),
        "num_selection": int(sel.sum()),
        "num_report": int(rep.sum()),
        "full_step_accuracy_all_puzzles": float(correct[:, -1].mean()),
        "full_step_accuracy": float(c_rep[:, -1].mean()),
        "fixed_step_accuracy": fixed_step_accuracy(c_rep),
        "oracle": oracle(c_rep),
        "flips": flips(c_rep),
        "heads": {},
    }
    if "preds_changed" in data:
        result["unchanged_among_unsolved"] = unchanged_among_unsolved(c_rep, data["preds_changed"][rows][:, steps][rep])
    else:
        result["unchanged_among_unsolved"] = "not available: the capture has no preds_changed"
    result["state_change_among_unsolved"] = {
        z: state_change_among_unsolved(c_rep, data[f"{z}_rel_change"][rows][:, steps][rep])
        if f"{z}_rel_change" in data else f"not available: the capture has no {z}_rel_change"
        for z in ("z_H", "z_L")}
    result["solving_rate"] = solving_rate(c_rep)
    if "pred_hash" in data:
        result["cycles"] = cycle_analysis(c_rep, data["pred_hash"][rows][:, steps][rep])
    else:
        result["cycles"] = "not available: the capture has no pred_hash"
    curves = {}
    for head, all_logits in data["logits"].items():
        logits = all_logits[rows][:, steps]
        report_curve = HaltCurve(logits[rep], c_rep)
        grid = curve_grid(report_curve, num_thresholds)
        curves[head] = {"threshold": grid, **report_curve(grid)}

        t0 = report_curve(0.0)
        head_result = {"t0": {**{k: float(v[0]) for k, v in t0.items()},
                              "halt_step_counts": halt_step_counts(logits[rep], 0.0)},
                       "operating_points": []}
        selection_curve = HaltCurve(logits[sel], correct[sel])
        for tol in tolerances:
            chosen = select_threshold(selection_curve, tol)
            head_result["operating_points"].append({
                "tolerance": tol,
                "selection": chosen,
                "report": evaluate_with_ci(logits[rep], c_rep, chosen["threshold"], num_resamples, rng),
            })
        result["heads"][head] = head_result
    return result, curves


def write_curve_csv(path: str, curve: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "avg_steps", "accuracy", "frac_halted_early"])
        for row in zip(curve["threshold"], curve["avg_steps"], curve["accuracy"], curve["frac_halted_early"]):
            w.writerow([repr(float(x)) for x in row])


def plot_group(path: str, name: str, result: dict, curves: dict, truncated: Optional[dict] = None) -> bool:
    """truncated: {K: (result, curves)} for the first-K-steps analyses, drawn on the same axes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fixed = result["fixed_step_accuracy"]
    ax.plot(range(1, len(fixed) + 1), fixed, "o--", color="0.5", markersize=4, label="fixed number of steps")

    # One color per head; distinct line styles and decreasing widths keep overlapping curves
    # (e.g. a twin head) visible. Truncated analyses reuse the head's color, thinner.
    styles = ["-", "--", ":", "-."]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    variants = [(None, result, curves)] + [(k, r, c) for k, (r, c) in sorted((truncated or {}).items())]
    for v, (k, res, crv) in enumerate(variants):
        suffix = "" if k is None else f", first {k} steps"
        for i, (head, curve) in enumerate(crv.items()):
            order = np.argsort(curve["avg_steps"], kind="stable")
            color = colors[i % len(colors)]
            ax.plot(curve["avg_steps"][order], curve["accuracy"][order], styles[i % len(styles)], color=color,
                    linewidth=max(2.5 - 0.5 * i, 1.0) if k is None else 1.0, alpha=1.0 if k is None else 0.7,
                    label=head + suffix)
            t0 = res["heads"][head]["t0"]
            ax.plot(t0["avg_steps"], t0["accuracy"], "D" if k is None else "s", color=color, markeredgecolor="black",
                    markersize=max(8 - 2 * i, 4), label=f"{head}, t = 0{suffix}")
        orc = res["oracle"]
        ax.plot(orc["avg_steps"], orc["accuracy"], "*", color="black" if k is None else "0.45",
                markersize=14 if k is None else 10, label="oracle" + suffix)
    ax.set_xlabel("average steps")
    ax.set_ylabel("accuracy (all cells correct)")
    ax.set_title(f"{name}: accuracy vs average steps (report split, {result['num_report']} puzzles)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)) and math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    if isinstance(x, np.generic):
        return x.item()
    return x


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def run(ckpt_dir: str, step: int, out_dir: str, selection_frac: float = 0.2,
        tolerances=(0.0, 0.001, 0.005), seed: int = 0, num_resamples: int = 10000,
        num_thresholds: int = 2000, data_dir: Optional[str] = None, verbose: bool = True,
        truncate_steps=()) -> dict:
    data, checks = load(ckpt_dir, step, data_dir)
    truncate_steps = sorted(set(truncate_steps))
    for k in truncate_steps:
        if not 1 <= k < checks["num_steps"]:
            raise ValueError(f"--truncate-steps {k}: must be between 1 and {checks['num_steps'] - 1} (the capture has {checks['num_steps']} steps)")
    selection = split_puzzles(len(data["set_id"]), selection_frac, seed)
    rng = np.random.default_rng(seed + 1)

    groups = {name: data["set_id"] == sid for sid, name in enumerate(data["set_names"])
              if (data["set_id"] == sid).any()}
    if len(groups) > 1:
        groups["overall"] = np.ones(len(data["set_id"]), dtype=bool)

    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "inputs": {"ckpt_dir": ckpt_dir, "step": step, "selection_frac": selection_frac,
                   "tolerances": list(tolerances), "seed": seed, "bootstrap_resamples": num_resamples,
                   "num_thresholds": num_thresholds, "data_dir": data_dir,
                   "truncate_steps": truncate_steps},
        "checks": checks,
        "split": {"num_selection": int(selection.sum()), "num_report": int((~selection).sum())},
        "results": {},
    }
    plots = []
    for name, rows in groups.items():
        result, curves = analyze_group(data, rows, selection, list(tolerances), num_resamples, num_thresholds, rng)
        summary["results"][name] = result
        for head, curve in curves.items():
            write_curve_csv(os.path.join(out_dir, f"curve_{_safe_name(name)}_{_safe_name(head)}.csv"), curve)

        truncated = {}
        for k in truncate_steps:
            truncated[k] = analyze_group(data, rows, selection, list(tolerances), num_resamples, num_thresholds, rng, truncate=k)
            for head, curve in truncated[k][1].items():
                write_curve_csv(os.path.join(out_dir, f"curve_{_safe_name(name)}_{_safe_name(head)}_first{k}.csv"), curve)
        if truncated:
            result["truncated"] = {str(k): r for k, (r, _) in truncated.items()}

        plot_path = os.path.join(out_dir, f"curves_{_safe_name(name)}.png")
        if plot_group(plot_path, name, result, curves, truncated):
            plots.append(plot_path)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    if verbose:
        print_summary(summary, plots, out_dir)
    return summary


def print_summary(summary: dict, plots: List[str], out_dir: str):
    c = summary["checks"]
    print(f"Loaded {len(c['files'])} file(s): {c['rows_loaded']} rows, {c['unique_puzzles']} unique puzzles, "
          f"{c['num_steps']} steps, heads {c['heads']}")
    if c["duplicate_rows_dropped"]:
        print(f"  WARNING: dropped {c['duplicate_rows_dropped']} duplicate rows "
              f"({c['duplicates_with_differing_values']} puzzles with differing values)")
    for name, s in c["per_set"].items():
        if s["missing"] or s["out_of_range"]:
            print(f"  WARNING: set {name}: {s['missing']} missing, {s['out_of_range']} out of range "
                  f"(expected {s['expected']} from {s['expected_source']})")
    il = c.get("in_loop_halting")
    if il:
        print(f"  In-loop halting ({il['head']}, t={il['threshold']:g}): accuracy {il['accuracy']:.4f}, avg steps "
              f"{il['avg_steps']:.2f}; replay from the capture " + ("matches" if il["matches_replay"] else
              f"DIFFERS for {il['puzzles_with_different_halt_step']} puzzles"))
    groups = []
    for name, r0 in summary["results"].items():
        groups.append((name, r0))
        groups.extend((f"{name}, first {k} steps", rk) for k, rk in r0.get("truncated", {}).items())
    for label, r in groups:
        print(f"\n[{label}] report split: {r['num_report']} puzzles; full-step accuracy {r['full_step_accuracy']:.4f} "
              f"(all puzzles: {r['full_step_accuracy_all_puzzles']:.4f})")
        print(f"  oracle: accuracy {r['oracle']['accuracy']:.4f}, avg steps {r['oracle']['avg_steps']:.2f}")
        cyc = r.get("cycles")
        if isinstance(cyc, dict) and cyc["num_unsolved_at_last"]:
            print(f"  unsolved at the last step: {cyc['num_unsolved_at_last']} -> " + ", ".join(
                f"{name} {c['share']:.1%}" for name, c in cyc["classes"].items()))
        for head, h in r["heads"].items():
            t0 = h["t0"]
            print(f"  {head}: t=0 -> accuracy {t0['accuracy']:.4f}, avg steps {t0['avg_steps']:.2f}")
            for op in h["operating_points"]:
                rep = op["report"]
                ci = lambda k: f" [{rep[k + '_ci'][0]:.4f}, {rep[k + '_ci'][1]:.4f}]" if k + "_ci" in rep else ""
                print(f"    tol {op['tolerance']:g}: t={op['selection']['threshold']:.4g} -> accuracy "
                      f"{rep['accuracy']:.4f}{ci('accuracy')}, avg steps {rep['avg_steps']:.2f}{ci('avg_steps')}")
    print(f"\nWrote {out_dir}/summary.json, curve CSVs"
          + (f" and {len(plots)} plot(s)" if plots else " (no plot: matplotlib not installed)"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ckpt_dir", help="directory with step_<STEP>_eval_steps.<rank> files")
    p.add_argument("step", type=int, help="training step of the evaluation")
    p.add_argument("out_dir", help="output directory")
    p.add_argument("--selection-frac", type=float, default=0.2, help="fraction of puzzles used to choose thresholds")
    p.add_argument("--tolerances", type=float, nargs="+", default=[0.0, 0.001, 0.005],
                   help="allowed absolute accuracy loss vs running all steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples (0 disables intervals)")
    p.add_argument("--num-thresholds", type=int, default=2000, help="quantile thresholds in the curve CSVs")
    p.add_argument("--truncate-steps", type=int, nargs="+", default=[], metavar="K",
                   help="also analyze only the first K steps (as if the step limit had been K)")
    p.add_argument("--data-dir", default=None,
                   help="dataset root (e.g. data/sudoku-extreme-1k-aug-1000) to check for missing puzzles")
    a = p.parse_args(argv)
    if not 0 < a.selection_frac < 1:
        p.error("--selection-frac must be in (0, 1)")
    run(a.ckpt_dir, a.step, a.out_dir, a.selection_frac, a.tolerances, a.seed, a.bootstrap,
        a.num_thresholds, a.data_dir, truncate_steps=a.truncate_steps)


if __name__ == "__main__":
    main()
