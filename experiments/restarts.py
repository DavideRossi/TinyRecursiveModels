"""Perturbed restarts for the puzzles the halt head has not accepted.

Phase 1 runs every test puzzle for N1 steps with halting by a head and threshold (the frozen-mask
rule of utils/eval_loop.py). The puzzles the head has not accepted by step N1 are retried: ground
truth is never used to choose them, only to score the results.

Phase 2 gives each variant the same retried puzzles and the same extra budget of N2 steps, with
the same halting rule:
    continue        the state of step N1, unchanged (the control: a plain run to N1 + N2 steps)
    perturb         the state of step N1 plus noise
    noisy_restart   the model's initial state (H_init, L_init) plus noise
Noise is i.i.d. Gaussian over the whole state (all positions and dimensions), rescaled per puzzle
and separately for z_H and z_L to norm = scale * norm of the state it is added to. Its direction
depends only on (seed, set, example index): it does not change with batch size or order, and it is
the same for every scale and both noisy variants of a (seed, puzzle) pair.

How the branching works: the model's evaluation step limit is set to N1 + N2, phase 1 stops after
N1 calls, and every variant continues from that in-memory state with the step counter at N1, so
the wrapper never resets it. For each batch, all variants of its retried puzzles are stacked into
batches of the evaluation batch size (padded, so the compiled model sees one shape).

    # On a GPU node (from the repository root):
    python experiments/restarts.py run CKPT OUT_DIR [--max-puzzles 5000] [--reference DIR STEP]
    # Recompute the summary, e.g. on a laptop with a plain capture as reference:
    python experiments/restarts.py summarize OUT_DIR --reference DIR STEP

OUT_DIR gets:
    settings.json                    the settings of the run
    phase1.npz                       per puzzle: key, accepted by step N1, halting step, correctness
    phase2.npz                       per branch and retried puzzle: accepted, halting step, correctness,
                                     the relative size of the noise actually applied (after rounding
                                     to the state's dtype)
    captures/<branch>/step_<N>_eval_steps.0
                                     phase-2 steps of the retried puzzles in the per-step capture format
                                     (step_offset = N1); readable by analysis/eval_steps.py
    summary.json, summary.csv        see summarize()

--reference DIR STEP: a plain capture of the same checkpoint with at least N1 + N2 steps (e.g. the
128-step one). The summary then checks phase 1 and the control against it, and adds plain
continuation to the same budgets as reference points.
"""
import argparse
import csv
import dataclasses
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from analysis import eval_steps as es  # noqa: E402
from models.losses import compute_correctness  # noqa: E402
from utils.eval_loop import CAPTURE_RETURN_KEYS, FrozenOutputs, StepRecorder, halt_logit  # noqa: E402

VARIANTS = ("continue", "perturb", "noisy_restart")
DEFAULTS = dict(n1=32, n2=32, head="q_head", threshold=5.27, perturb_scales=[0.001, 0.1, 0.3, 1.0],
                restart_scales=[0.1, 0.3, 1.0], seeds=[0, 1, 2], selection_frac=0.2, split_seed=0)
NOISE_SEED = 0x7E57_0F_4E015E  # constant: noise is comparable across runs
_MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------- branches


@dataclasses.dataclass(frozen=True)
class Branch:
    variant: str
    scale: float = 0.0
    seed: Optional[int] = None

    @property
    def name(self) -> str:
        return self.variant if self.variant == "continue" else f"{self.variant}_s{self.scale:g}_seed{self.seed}"


def make_branches(perturb_scales: Sequence[float], restart_scales: Sequence[float], seeds: Sequence[int]) -> List[Branch]:
    """The control first, then every (scale, seed) of each noisy variant."""
    out = [Branch("continue")]
    out += [Branch("perturb", float(s), int(k)) for s in perturb_scales for k in seeds]
    out += [Branch("noisy_restart", float(s), int(k)) for s in restart_scales for k in seeds]
    return out


# --------------------------------------------------------------------------- noise


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def puzzle_noise_seed(seed: int, set_id: int, example_index: int) -> int:
    """Generator seed for one (seed, puzzle) pair; a pure function of its arguments."""
    z = NOISE_SEED
    for v in (seed, set_id, example_index):
        z = _splitmix64(z ^ (int(v) & _MASK64))
    return z & ((1 << 63) - 1)


def unit_noise(shape: Sequence[int], seed: int, set_ids: Sequence[int], example_indices: Sequence[int]):
    """Per puzzle, two unit-norm Gaussian directions of `shape` (for z_H and z_L), float32 on the CPU.

    Each puzzle has its own generator, seeded from (seed, set, example index) only; no global
    generator is read or changed."""
    out_h, out_l = [], []
    gen = torch.Generator()
    for sid, ex in zip(set_ids, example_indices):
        gen.manual_seed(puzzle_noise_seed(seed, sid, ex))
        for out in (out_h, out_l):
            n = torch.randn(tuple(shape), generator=gen, dtype=torch.float32)
            out.append(n / torch.linalg.vector_norm(n))
    return torch.stack(out_h), torch.stack(out_l)


def add_noise(state: torch.Tensor, unit: torch.Tensor, scale: float) -> torch.Tensor:
    """state + scale * ||state|| * unit, per puzzle (row), computed in float32 and returned in the state's dtype."""
    s = state.to(torch.float32)
    norm = torch.linalg.vector_norm(s.flatten(1), dim=1).view((-1, ) + (1, ) * (s.ndim - 1))
    return (s + scale * norm * unit.to(s.device)).to(state.dtype)


def realized_scale(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """||new - old|| / ||old|| per row, after rounding to the state's dtype."""
    new, old = new.to(torch.float32).flatten(1), old.to(torch.float32).flatten(1)
    return torch.linalg.vector_norm(new - old, dim=1) / torch.linalg.vector_norm(old, dim=1)


# --------------------------------------------------------------------------- carries


def map_tensors(obj: Any, fn):
    """Apply fn to every tensor of a (nested) carry: dataclasses, dicts, lists."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **{f.name: map_tensors(getattr(obj, f.name), fn) for f in dataclasses.fields(obj)})
    if isinstance(obj, dict):
        return {k: map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(v, fn) for v in obj)
    return obj


def cat_tensors(objs: List[Any]):
    """Concatenate along the batch dimension a list of carries (or batches) of the same structure."""
    first = objs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(objs, dim=0)
    if dataclasses.is_dataclass(first):
        return dataclasses.replace(first, **{f.name: cat_tensors([getattr(o, f.name) for o in objs]) for f in dataclasses.fields(first)})
    if isinstance(first, dict):
        return {k: cat_tensors([o[k] for o in objs]) for k in first}
    return first


def inner_model(model: Any) -> Any:
    """The recursive model inside the loss head (through torch.compile's wrapper)."""
    model = getattr(model, "_orig_mod", model)
    return model.model.inner


def branch_carry(model: Any, base: Any, branch: Branch, unit: Optional[tuple]) -> (Any, torch.Tensor, torch.Tensor):
    """Starting carry of a branch from the carry of step N1 (rows = retried puzzles).

    Returns the carry and the realized relative noise for z_H and z_L (0 for the control)."""
    inner = base.inner_carry
    zero = torch.zeros(inner.z_H.shape[0], dtype=torch.float32, device=inner.z_H.device)
    if branch.variant == "continue":
        return base, zero, zero
    if branch.variant == "perturb":
        start = inner
    elif branch.variant == "noisy_restart":
        reset = torch.ones(inner.z_H.shape[0], dtype=torch.bool, device=inner.z_H.device)
        start = inner_model(model).reset_carry(reset, inner)  # H_init / L_init, as the wrapper resets
    else:
        raise ValueError(f"unknown variant {branch.variant}")
    z_H = add_noise(start.z_H, unit[0], branch.scale)
    z_L = add_noise(start.z_L, unit[1], branch.scale)
    new_inner = dataclasses.replace(inner, z_H=z_H, z_L=z_L)
    return (dataclasses.replace(base, inner_carry=new_inner), realized_scale(z_H, start.z_H),
            realized_scale(z_L, start.z_L))


# --------------------------------------------------------------------------- one batch


@dataclasses.dataclass
class Settings:
    n1: int = DEFAULTS["n1"]
    n2: int = DEFAULTS["n2"]
    head: str = DEFAULTS["head"]
    threshold: float = DEFAULTS["threshold"]
    perturb_scales: List[float] = dataclasses.field(default_factory=lambda: list(DEFAULTS["perturb_scales"]))
    restart_scales: List[float] = dataclasses.field(default_factory=lambda: list(DEFAULTS["restart_scales"]))
    seeds: List[int] = dataclasses.field(default_factory=lambda: list(DEFAULTS["seeds"]))
    chunk_size: Optional[int] = None  # phase-2 batch size (padded to it); None: one batch, unpadded

    def branches(self) -> List[Branch]:
        return make_branches(self.perturb_scales, self.restart_scales, self.seeds)


def run_batch(model: Any, batch: Dict[str, torch.Tensor], set_id: np.ndarray, example_index: np.ndarray,
              settings: Settings) -> dict:
    """Both phases for one batch. Rows beyond len(set_id) are padding and are ignored.

    The model's evaluation step limit must be N1 + N2. Returns phase-1 arrays for the valid rows,
    the retried rows, and per branch: phase-2 arrays and capture records of the retried puzzles."""
    n1, n2 = settings.n1, settings.n2
    num_valid = len(set_id)
    return_keys = set(CAPTURE_RETURN_KEYS)
    threshold = settings.threshold

    # On the model's device, as pretrain.evaluate does (the test loader yields CPU tensors, and the
    # initial carry must be created on the same device as the model's buffers)
    device = next(model.parameters()).device
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    # Phase 1
    with torch.device(device):
        carry = model.initial_carry(batch)
    frozen = FrozenOutputs(threshold)
    for step in range(1, n1 + 1):
        carry, _, _, preds, all_finish = model(carry=carry, batch=batch, return_keys=return_keys)
        if all_finish:
            raise RuntimeError(f"the model stopped at step {step}: its evaluation step limit must be N1 + N2 = {n1 + n2}")
        frozen.update(step, halt_logit(preds, settings.head), {"preds": preds["preds"]}, is_last=False)
    labels = batch["labels"]
    correct_at_halt = compute_correctness(frozen.outputs["preds"], labels)[3]
    correct_at_n1 = compute_correctness(preds["preds"], labels)[3]
    accepted = frozen.halted[:num_valid].cpu().numpy()
    phase1 = {"accepted": accepted, "halt_step": np.where(accepted, frozen.halt_step[:num_valid].cpu().numpy(), 0),
              "correct_at_halt": (correct_at_halt[:num_valid].cpu().numpy() & accepted),
              "correct_at_n1": correct_at_n1[:num_valid].cpu().numpy()}
    retried = np.flatnonzero(~accepted)
    out = {"phase1": phase1, "retried": retried, "branches": {}}
    if len(retried) == 0:
        return out

    # Phase 2: every branch on the retried rows, stacked
    idx = torch.as_tensor(retried, device=labels.device)
    base = map_tensors(carry, lambda t: t[idx])
    sub_batch = {k: v[idx] for k, v in batch.items()}
    last_preds = preds["preds"][idx]
    units = {seed: unit_noise(base.inner_carry.z_H.shape[1:], seed, set_id[retried], example_index[retried])
             for seed in settings.seeds}
    branches = settings.branches()
    starts, scale_h, scale_l = [], [], []
    for b in branches:
        c, rh, rl = branch_carry(model, base, b, units[b.seed] if b.seed is not None else None)
        starts.append(c)
        scale_h.append(rh)
        scale_l.append(rl)
    all_carry = cat_tensors(starts)
    all_batch = cat_tensors([sub_batch] * len(branches))
    all_prev = torch.cat([last_preds] * len(branches))
    total = len(branches) * len(retried)

    chunk = settings.chunk_size or total
    pieces = {"accepted": [], "halt_step": [], "correct_at_halt": [], "correct_at_last": [], "records": []}
    for a in range(0, total, chunk):
        rows = torch.arange(a, min(a + chunk, total), device=labels.device)
        n = len(rows)
        if settings.chunk_size and n < chunk:  # pad to the fixed batch size with copies of the first row
            rows = torch.cat([rows, rows[:1].repeat(chunk - n)])
        c = map_tensors(all_carry, lambda t: t[rows])
        bt = {k: v[rows] for k, v in all_batch.items()}
        recorder = StepRecorder(prev_preds=all_prev[rows], prev_carry=c)
        frozen2 = FrozenOutputs(threshold)
        max_logit = None
        for k in range(1, n2 + 1):
            c, _, _, p, all_finish = model(carry=c, batch=bt, return_keys=return_keys)
            if bool(all_finish) != (k == n2):
                raise RuntimeError(f"phase 2 finished at step {n1 + k}, expected {n1 + n2}")
            recorder.record(p, c, bt["labels"])
            logit = halt_logit(p, settings.head)
            frozen2.update(n1 + k, logit, {"preds": p["preds"]}, is_last=(k == n2))
            max_logit = logit if max_logit is None else torch.maximum(max_logit, logit)
        thr = torch.tensor(threshold, dtype=max_logit.dtype, device=max_logit.device)
        pieces["accepted"].append((max_logit >= thr)[:n].cpu())
        pieces["halt_step"].append(frozen2.halt_step[:n].cpu())
        pieces["correct_at_halt"].append(compute_correctness(frozen2.outputs["preds"], bt["labels"])[3][:n].cpu())
        pieces["correct_at_last"].append(compute_correctness(p["preds"], bt["labels"])[3][:n].cpu())
        pieces["records"].append({key: torch.stack(v, 1)[:n].cpu() for key, v in recorder.step_records.items()})

    flat = {k: torch.cat(v) for k, v in pieces.items() if k != "records"}
    records = {key: torch.cat([r[key] for r in pieces["records"]]) for key in pieces["records"][0]}
    R = len(retried)
    for i, b in enumerate(branches):
        sl = slice(i * R, (i + 1) * R)
        out["branches"][b.name] = {
            "accepted": flat["accepted"][sl].numpy(), "halt_step": flat["halt_step"][sl].numpy(),
            "correct_at_halt": flat["correct_at_halt"][sl].numpy(), "correct_at_last": flat["correct_at_last"][sl].numpy(),
            "realized_scale_z_H": scale_h[i].cpu().numpy(), "realized_scale_z_L": scale_l[i].cpu().numpy(),
            "records": {key: v[sl] for key, v in records.items()},
            "puzzle_identifiers": sub_batch["puzzle_identifiers"].cpu(),
        }
    return out


# --------------------------------------------------------------------------- whole test set


def run_experiment(model: Any, batches, settings: Settings, out_dir: str, set_names: List[str], set_sizes: List[int],
                   ckpt_step: int, max_puzzles: Optional[int] = None, extra_settings: Optional[dict] = None):
    """batches: iterable of (set_name, batch, global_batch_size) as the test loader yields them
    (one process). Writes settings.json, phase1.npz, phase2.npz and the captures to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    branches = settings.branches()
    set_ids = {name: i for i, name in enumerate(set_names)}
    next_start: Dict[str, int] = {}
    p1: Dict[str, list] = {k: [] for k in ("set_id", "example_index", "accepted", "halt_step", "correct_at_halt", "correct_at_n1")}
    p2: Dict[str, Dict[str, list]] = {b.name: {} for b in branches}
    keys: Dict[str, list] = {"set_id": [], "example_index": []}
    done = 0
    for batch_i, (set_name, batch, global_batch_size) in enumerate(batches):
        start = next_start.get(set_name, 0)
        next_start[set_name] = start + batch["inputs"].shape[0]
        num_valid = min(global_batch_size, batch["inputs"].shape[0])
        if max_puzzles is not None:
            num_valid = min(num_valid, max_puzzles - done)
        if num_valid <= 0:
            break
        sid = np.full(num_valid, set_ids[set_name], dtype=np.int64)
        ex = np.arange(start, start + num_valid, dtype=np.int64)
        with torch.inference_mode():
            res = run_batch(model, batch, sid, ex, settings)
        p1["set_id"].append(sid)
        p1["example_index"].append(ex)
        for k, v in res["phase1"].items():
            p1[k].append(v)
        keys["set_id"].append(sid[res["retried"]])
        keys["example_index"].append(ex[res["retried"]])
        for name, r in res["branches"].items():
            for k, v in r.items():
                p2[name].setdefault(k, []).append(v)
        done += num_valid
        print(f"batch {batch_i + 1}: {done} puzzles, {len(res['retried'])} retried in this batch", flush=True)

    phase1 = {k: np.concatenate(v) for k, v in p1.items()}
    order1 = np.lexsort((phase1["example_index"], phase1["set_id"]))
    phase1 = {k: v[order1] for k, v in phase1.items()}
    np.savez(os.path.join(out_dir, "phase1.npz"), **phase1)

    r_sid, r_ex = np.concatenate(keys["set_id"]), np.concatenate(keys["example_index"])
    order = np.lexsort((r_ex, r_sid))
    phase2 = {"set_id": r_sid[order], "example_index": r_ex[order], "branches": np.array([b.name for b in branches])}
    dtypes = {"accepted": bool, "halt_step": np.int64, "correct_at_halt": bool, "correct_at_last": bool,
              "realized_scale_z_H": np.float32, "realized_scale_z_L": np.float32}
    for field, dtype in dtypes.items():
        phase2[field] = np.stack([np.concatenate(p2[b.name][field])[order] for b in branches]) if len(order) else \
            np.zeros((len(branches), 0), dtype=dtype)
    np.savez(os.path.join(out_dir, "phase2.npz"), **phase2)

    # Captures in the per-step format (steps N1 + 1 ... N1 + N2 of the retried puzzles)
    subset = (f"puzzles not accepted by {settings.head} (threshold {settings.threshold:g}) by step {settings.n1}"
              + (f", among the first {max_puzzles}" if max_puzzles is not None else ""))
    torder = torch.as_tensor(order)
    for b in branches:
        if not len(order):
            break
        rec = {key: torch.cat([r[key] for r in p2[b.name]["records"]])[torder] for key in p2[b.name]["records"][0]}
        rec.update(
            puzzle_identifiers=torch.cat(p2[b.name]["puzzle_identifiers"])[torder],
            example_index=torch.as_tensor(phase2["example_index"]), set_id=torch.as_tensor(phase2["set_id"], dtype=torch.int32),
            num_steps=torch.full((len(order), ), settings.n2, dtype=torch.int32),
            halt_step=torch.as_tensor(phase2["halt_step"][branches.index(b)] - settings.n1, dtype=torch.int32),
            set_names=list(set_names), eval_halt_head=settings.head, eval_halt_threshold=float(settings.threshold),
            step_offset=settings.n1, set_sizes=list(set_sizes), puzzle_subset=f"{subset}; branch {b.name}")
        d = os.path.join(out_dir, "captures", b.name)
        os.makedirs(d, exist_ok=True)
        torch.save(rec, os.path.join(d, f"step_{ckpt_step}_eval_steps.0"))

    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump({**dataclasses.asdict(settings), "set_names": list(set_names), "set_sizes": list(set_sizes),
                   "ckpt_step": ckpt_step, "max_puzzles": max_puzzles, "num_puzzles": int(len(phase1["set_id"])),
                   "branches": [b.name for b in branches], **(extra_settings or {})}, f, indent=2)


# --------------------------------------------------------------------------- summary


def _key(set_id, example_index) -> np.ndarray:
    return np.asarray(set_id, dtype=np.int64) * (1 << 40) + np.asarray(example_index, dtype=np.int64)


def paired(variant: np.ndarray, control: np.ndarray, num_resamples: int, rng: np.random.Generator) -> dict:
    """2 x 2 table of solved / not solved for the same puzzles, and the difference in solved share
    with a bootstrap interval (resampling puzzles)."""
    n = len(variant)
    cells = np.array([(variant & control).sum(), (variant & ~control).sum(), (~variant & control).sum(),
                      (~variant & ~control).sum()])
    out = {"both": int(cells[0]), "variant_only": int(cells[1]), "control_only": int(cells[2]), "neither": int(cells[3]),
           "difference": float((cells[1] - cells[2]) / n) if n else None}
    if n and num_resamples:
        boot = rng.multinomial(n, cells / n, size=num_resamples)
        diff = (boot[:, 1] - boot[:, 2]) / n
        out["difference_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    return out


def _outcome_stats(accepted, correct, steps) -> dict:
    n = len(accepted)
    return {"retried": int(n), "accepted": int(accepted.sum()), "accepted_correct": int((accepted & correct).sum()),
            "false_acceptances": int((accepted & ~correct).sum()), "solved": int(correct.sum()),
            "solved_share": float(correct.mean()) if n else None, "mean_steps": float(steps.mean()) if n else None}


def _overall(p1: dict, report: np.ndarray, retried_rows: np.ndarray, correct, steps) -> dict:
    """Whole procedure on the report split: phase-1 answers for accepted puzzles, phase-2 ones for retried."""
    accepted = p1["accepted"] & report
    n = int(report.sum())
    total_correct = int((p1["correct_at_halt"] & accepted).sum()) + int(correct[retried_rows].sum())
    total_steps = int(p1["halt_step"][accepted].sum()) + int(steps[retried_rows].sum())
    return {"puzzles": n, "accuracy": total_correct / n if n else None, "avg_steps": total_steps / n if n else None}


def sequential(accepted: np.ndarray, halt_step: np.ndarray, correct: np.ndarray, n1: int, n2: int):
    """Deployable restart strategy over seeds in order ([seeds, R] arrays): keep the first answer the
    head accepts; if none is accepted, the last attempt's answer. Steps count every attempt."""
    S, R = accepted.shape
    any_acc = accepted.any(axis=0)
    first = np.where(any_acc, accepted.argmax(axis=0), S - 1)
    cols = np.arange(R)
    steps = n1 + n2 * first + (halt_step[first, cols] - n1)
    return any_acc, correct[first, cols], steps, first


def summarize(out_dir: str, reference: Optional[tuple] = None, num_resamples: int = 10000,
              selection_frac: Optional[float] = None, split_seed: Optional[int] = None, verbose: bool = True) -> dict:
    """Summary over the retried puzzles of the report split. Writes summary.json and summary.csv."""
    with open(os.path.join(out_dir, "settings.json")) as f:
        settings = json.load(f)
    frac = settings.get("selection_frac", DEFAULTS["selection_frac"]) if selection_frac is None else selection_frac
    sseed = settings.get("split_seed", DEFAULTS["split_seed"]) if split_seed is None else split_seed
    n1, n2 = settings["n1"], settings["n2"]
    p1 = dict(np.load(os.path.join(out_dir, "phase1.npz")))
    p2 = dict(np.load(os.path.join(out_dir, "phase2.npz")))
    names = [str(x) for x in p2["branches"]]
    rng = np.random.default_rng(sseed + 1)

    report1 = ~es.split_subset(p1["set_id"], p1["example_index"], settings["set_sizes"], frac, sseed)
    k1, k2 = _key(p1["set_id"], p1["example_index"]), _key(p2["set_id"], p2["example_index"])
    pos = np.searchsorted(k1, k2)  # retried puzzles' rows in phase 1
    report2 = report1[pos]
    R = int(report2.sum())

    def overall(correct, steps):
        c, s = np.zeros(len(k1), bool), np.zeros(len(k1), np.int64)
        c[pos], s[pos] = correct, steps
        return _overall(p1, report1, np.flatnonzero(report1 & ~p1["accepted"]), c, s)

    ctrl = names.index("continue")
    get = lambda field, i: p2[field][i][report2]
    control_correct = get("correct_at_halt", ctrl)

    summary = {"settings": settings, "split": {"selection_frac": frac, "seed": sseed, "report_puzzles": int(report1.sum()),
                                                "retried_report_puzzles": R}}
    a1 = p1["accepted"] & report1
    summary["phase1"] = {
        "accepted": int(a1.sum()), "accepted_correct": int((a1 & p1["correct_at_halt"]).sum()),
        "false_acceptances": int((a1 & ~p1["correct_at_halt"]).sum()), "retried": R,
        "retried_correct_at_n1": int((report1 & ~p1["accepted"] & p1["correct_at_n1"]).sum()),
        # Stopping at N1 (answers of step N1 for the puzzles not accepted)
        "overall_stop_at_n1": overall(p1["correct_at_n1"][pos], np.full(len(pos), n1)),
    }

    branches = []
    for i, name in enumerate(names):
        acc, cor, steps = get("accepted", i), get("correct_at_halt", i), get("halt_step", i).astype(np.int64)
        variant, scale, seed = _parse(name)
        entry = {"branch": name, "variant": variant, "scale": scale, "seed": seed,
                 **_outcome_stats(acc, cor, steps),
                 "realized_scale_z_H": float(np.mean(get("realized_scale_z_H", i))) if R else None,
                 "realized_scale_z_L": float(np.mean(get("realized_scale_z_L", i))) if R else None,
                 "paired_with_control": paired(cor, control_correct, num_resamples, rng),
                 "overall": overall(p2["correct_at_halt"][i], p2["halt_step"][i].astype(np.int64))}
        branches.append(entry)
    summary["branches"] = branches

    groups = []
    for variant, scale in dict.fromkeys((b["variant"], b["scale"]) for b in branches if b["variant"] != "continue"):
        members = [i for i, b in enumerate(branches) if (b["variant"], b["scale"]) == (variant, scale)]
        per_seed = [branches[i] for i in members]
        stat = lambda f: {"mean": float(np.mean([f(b) for b in per_seed])), "min": float(np.min([f(b) for b in per_seed])),
                          "max": float(np.max([f(b) for b in per_seed]))}
        g = {"variant": variant, "scale": scale, "seeds": [b["seed"] for b in per_seed],
             "across_seeds": {k: stat(lambda b, k=k: b[k] or 0) for k in
                              ("solved_share", "accepted", "false_acceptances", "mean_steps")},
             "difference_from_control": stat(lambda b: b["paired_with_control"]["difference"] or 0),
             "overall_accuracy": stat(lambda b: b["overall"]["accuracy"] or 0),
             "overall_avg_steps": stat(lambda b: b["overall"]["avg_steps"] or 0)}
        cor_all = np.stack([get("correct_at_halt", i) for i in members])
        g["any_seed_oracle"] = {
            "note": f"uses ground truth to pick among {len(members)} attempts, each with {n2} extra steps: an upper bound, "
                    "not a deployable strategy",
            "solved": int(cor_all.any(axis=0).sum()), "solved_share": float(cor_all.any(axis=0).mean()) if R else None}
        acc_s, cor_s, steps_s, first = sequential(np.stack([get("accepted", i) for i in members]),
                                                  np.stack([get("halt_step", i) for i in members]).astype(np.int64),
                                                  cor_all, n1, n2)
        full = sequential(np.stack([p2["accepted"][i] for i in members]), np.stack([p2["halt_step"][i] for i in members]).astype(np.int64),
                          np.stack([p2["correct_at_halt"][i] for i in members]), n1, n2)
        g["sequential"] = {
            "note": f"seeds tried in order {g['seeds']}; the first answer the head accepts is kept, otherwise the last "
                    f"attempt's; steps count every attempt (up to {n1 + len(members) * n2} per puzzle)",
            **_outcome_stats(acc_s, cor_s, steps_s),
            "attempts_used": np.bincount(first + 1, minlength=len(members) + 1)[1:].tolist(),
            "paired_with_control": paired(cor_s, control_correct, num_resamples, rng),
            "overall": overall(full[1], full[2])}
        groups.append(g)
    summary["groups"] = groups

    if reference is not None:
        summary["reference"] = reference_results(out_dir, settings, reference, p1, p2, report1, pos, report2, groups,
                                                 names, num_resamples, rng)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(es._json_safe(summary), f, indent=2)
    write_csv(os.path.join(out_dir, "summary.csv"), summary)
    if verbose:
        print_summary(summary)
    return summary


def _parse(name: str):
    if name == "continue":
        return "continue", 0.0, None
    m = re.fullmatch(r"(.+)_s([^_]+)_seed(-?\d+)", name)
    if not m:
        raise ValueError(f"unexpected branch name {name}")
    return m.group(1), float(m.group(2)), int(m.group(3))


def reference_results(out_dir, settings, reference, p1, p2, report1, pos, report2, groups, names, num_resamples, rng) -> dict:
    """Checks against a plain capture, and plain continuation to the same budgets."""
    n1, n2 = settings["n1"], settings["n2"]
    ref_dir, ref_step = reference
    data, _ = es.load(ref_dir, int(ref_step))
    S = data["seq_is_correct"].shape[1]
    if S < n1 + n2:
        raise ValueError(f"the reference capture has {S} steps; at least N1 + N2 = {n1 + n2} are needed")
    kr = _key(data["set_id"], data["example_index"])
    k1 = _key(p1["set_id"], p1["example_index"])
    rows = np.searchsorted(kr, k1)
    if np.any(rows >= len(kr)) or np.any(kr[np.minimum(rows, len(kr) - 1)] != k1):
        raise ValueError("the reference capture lacks some of the puzzles of this run")
    logits = data["logits"][settings["head"]][rows]
    correct = data["seq_is_correct"][rows]
    thr = float(np.float32(settings["threshold"]))
    out = {"capture": f"{ref_dir} (step {ref_step}, {S} steps)"}

    # Phase 1 replayed from the reference
    h1 = es.halting_steps(logits[:, :n1], thr)
    acc1 = (np.maximum.accumulate(logits[:, :n1], axis=1)[:, -1] >= thr)
    out["phase1_check"] = {"puzzles": int(len(k1)), "same_acceptance": int((acc1 == p1["accepted"]).sum()),
                           "same_halt_step": int(((h1 == p1["halt_step"]) | ~p1["accepted"]).sum())}

    # Control against steps N1 + 1 ... N1 + N2
    control_dir = os.path.join(out_dir, "captures", "continue")
    if not os.path.isdir(control_dir):
        out["control_check"] = f"not available: {control_dir} not found"
        cap = None
    else:
        cap, _ = es.load(control_dir, settings["ckpt_step"])
    rr = rows[pos]
    window = slice(n1, n1 + n2)
    check = {"retried_puzzles": int(len(rr))}
    for key in ("pred_hash", "seq_is_correct", "wrong_cells") if cap is not None else ():
        if key in cap and key in data:
            same = (cap[key] == data[key][rr][:, window]).all(axis=1)
            check[f"same_{key}"] = int(same.sum())
    if cap is not None:
        check["max_abs_halt_logit_difference"] = float(np.abs(cap["logits"][settings["head"]] - data["logits"][settings["head"]][rr][:, window]).max()) \
            if len(rr) else 0.0
        out["control_check"] = check

    # Plain continuation (halting by the same head) to each budget N1 + k * N2 the capture covers
    k_max = max([len(g["seeds"]) for g in groups] + [1])
    plain = []
    for k in range(1, k_max + 1):
        budget = n1 + k * n2
        if budget > S:
            break
        h = es.halting_steps(logits[:, :budget], thr)
        c = correct[np.arange(len(h)), h - 1]
        retried = report1 & ~p1["accepted"]
        acc = (np.maximum.accumulate(logits[:, :budget], axis=1)[:, -1] >= thr)
        plain.append({"budget": budget, "name": f"plain continuation, halting within {budget} steps",
                      "overall": {"puzzles": int(report1.sum()), "accuracy": float(c[report1].mean()),
                                  "avg_steps": float(h[report1].mean())},
                      **_outcome_stats(acc[retried], c[retried], h[retried].astype(np.int64))})
    out["plain_continuation"] = plain
    # Fixed numbers of steps, no halting
    out["fixed_steps"] = [{"steps": s, "name": f"plain, {s} steps (no halting)", "accuracy": float(correct[report1, s - 1].mean())}
                          for s in sorted({n1, n1 + n2, S} | {n1 + k * n2 for k in range(1, k_max + 1) if n1 + k * n2 <= S})]

    # Sequential strategy against plain continuation with the same budget, on the retried puzzles
    retried_rows = np.flatnonzero(report1 & ~p1["accepted"])
    for g in groups:
        budget = n1 + len(g["seeds"]) * n2
        if budget > S:
            continue
        h = es.halting_steps(logits[retried_rows, :budget], thr)
        c_plain = correct[retried_rows][np.arange(len(h)), h - 1]
        members = [i for i, name in enumerate(names) if _parse(name)[:2] == (g["variant"], g["scale"])]
        _, cor_s, _, _ = sequential(np.stack([p2["accepted"][i][report2] for i in members]),
                                    np.stack([p2["halt_step"][i][report2] for i in members]).astype(np.int64),
                                    np.stack([p2["correct_at_halt"][i][report2] for i in members]), n1, n2)
        g["sequential"]["paired_with_plain_same_budget"] = {"budget": budget, **paired(cor_s, c_plain, num_resamples, rng)}
    return out


CSV_COLUMNS = ["kind", "variant", "scale", "seed", "retried", "accepted", "accepted_correct", "false_acceptances", "solved",
               "solved_share", "mean_steps", "both", "variant_only", "control_only", "neither", "difference",
               "difference_ci_low", "difference_ci_high", "overall_accuracy", "overall_avg_steps"]


def write_csv(path: str, summary: dict):
    rows = []

    def row(kind, variant, scale, seed, stats, pair, overall):
        r = {"kind": kind, "variant": variant, "scale": scale, "seed": seed,
             **{k: stats.get(k) for k in ("retried", "accepted", "accepted_correct", "false_acceptances", "solved",
                                          "solved_share", "mean_steps")}}
        if pair:
            r.update({k: pair[k] for k in ("both", "variant_only", "control_only", "neither", "difference")})
            ci = pair.get("difference_ci") or [None, None]
            r.update(difference_ci_low=ci[0], difference_ci_high=ci[1])
        if overall:
            r.update(overall_accuracy=overall["accuracy"], overall_avg_steps=overall["avg_steps"])
        rows.append(r)

    for b in summary["branches"]:
        row("seed", b["variant"], b["scale"], b["seed"], b, b["paired_with_control"], b["overall"])
    for g in summary["groups"]:
        s = g["sequential"]
        row("sequential", g["variant"], g["scale"], "+".join(map(str, g["seeds"])), s, s["paired_with_control"], s["overall"])
    for p in summary.get("reference", {}).get("plain_continuation", []):
        row("plain", "plain", None, None, p, None, p["overall"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def print_summary(summary: dict):
    sp, p1 = summary["split"], summary["phase1"]
    print(f"Report split: {sp['report_puzzles']} puzzles; phase 1 accepted {p1['accepted']} "
          f"({p1['false_acceptances']} wrongly); retried {p1['retried']}")
    ctrl = next(b for b in summary["branches"] if b["variant"] == "continue")
    print(f"  control: solved {ctrl['solved']}/{ctrl['retried']} ({ctrl['solved_share']:.4f}), "
          f"overall accuracy {ctrl['overall']['accuracy']:.4f}, avg steps {ctrl['overall']['avg_steps']:.2f}")
    for g in summary["groups"]:
        a, d, s = g["across_seeds"]["solved_share"], g["difference_from_control"], g["sequential"]
        print(f"  {g['variant']} {g['scale']:g}: solved {a['mean']:.4f} [{a['min']:.4f}, {a['max']:.4f}] "
              f"(vs control {d['mean']:+.4f}); sequential {s['solved_share']:.4f}, mean steps {s['mean_steps']:.1f}, "
              f"overall {s['overall']['accuracy']:.4f} @ {s['overall']['avg_steps']:.2f}; "
              f"any seed (oracle) {g['any_seed_oracle']['solved_share']:.4f}")
    ref = summary.get("reference")
    if ref:
        c = ref["control_check"]
        p1c = ref["phase1_check"]
        print(f"  reference {ref['capture']}: phase 1 same acceptance for {p1c['same_acceptance']}/{p1c['puzzles']} puzzles")
        if isinstance(c, dict):
            print(f"    control: same grid hashes for {c.get('same_pred_hash')}/{c['retried_puzzles']} retried puzzles, "
                  f"same correctness for {c.get('same_seq_is_correct')}, max halt-logit difference "
                  f"{c['max_abs_halt_logit_difference']:.3g}")
        else:
            print(f"    control check {c}")
        for p in ref["plain_continuation"]:
            print(f"    {p['name']}: overall {p['overall']['accuracy']:.4f} @ {p['overall']['avg_steps']:.2f}")


# --------------------------------------------------------------------------- command line


def load_model_and_data(ckpt: str, settings: Settings, batch_size: Optional[int]):
    """Model and test batches as the evaluation-only run builds them (same weights)."""
    import yaml
    from pretrain import PretrainConfig, create_dataloader, create_model
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
    with open(os.path.join(ckpt_dir, "all_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(load_checkpoint=ckpt, eval_halt_max_steps=settings.n1 + settings.n2, eval_only=True,
               checkpoint_path=None, eval_capture_steps=False, eval_halt_enable=False)
    if batch_size is not None:
        cfg["global_batch_size"] = batch_size
    config = PretrainConfig(**cfg)

    train_metadata = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed, dataset_paths=config.data_paths, rank=0, num_replicas=1, test_set_mode=False,
        epochs_per_iter=1, global_batch_size=config.global_batch_size), split="train").metadata
    loader, metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1,
                                         global_batch_size=config.global_batch_size, rank=0, world_size=1)
    model, _, _ = create_model(config, train_metadata, rank=0, world_size=1)
    model.eval()

    paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    if len(paths) != 1:
        raise ValueError("one test dataset path is supported")
    set_sizes = [int(np.load(os.path.join(paths[0], "test", f"{name}__inputs.npy"), mmap_mode="r").shape[0])
                 for name in metadata.sets]
    return model, loader, list(metadata.sets), set_sizes, config.global_batch_size


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run both phases (GPU), then summarize")
    r.add_argument("ckpt", help="checkpoint file step_<N>, with all_config.yaml next to it")
    r.add_argument("out_dir")
    r.add_argument("--n1", type=int, default=DEFAULTS["n1"])
    r.add_argument("--n2", type=int, default=DEFAULTS["n2"])
    r.add_argument("--head", default=DEFAULTS["head"])
    r.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    r.add_argument("--perturb-scales", type=float, nargs="*", default=DEFAULTS["perturb_scales"])
    r.add_argument("--restart-scales", type=float, nargs="*", default=DEFAULTS["restart_scales"])
    r.add_argument("--seeds", type=int, nargs="+", default=DEFAULTS["seeds"])
    r.add_argument("--max-puzzles", type=int, default=None, help="only the first N test puzzles (quick trials)")
    r.add_argument("--batch-size", type=int, default=None, help="default: the checkpoint run's global_batch_size")
    s = sub.add_parser("summarize", help="recompute summary.json / summary.csv from a finished run")
    s.add_argument("out_dir")
    for q in (r, s):
        q.add_argument("--reference", nargs=2, metavar=("DIR", "STEP"), default=None,
                       help="plain capture with at least N1 + N2 steps, for checks and reference points")
        q.add_argument("--bootstrap", type=int, default=10000)
        q.add_argument("--selection-frac", type=float, default=DEFAULTS["selection_frac"])
        q.add_argument("--split-seed", type=int, default=DEFAULTS["split_seed"])
    a = p.parse_args(argv)

    if a.command == "run":
        if os.path.abspath(a.out_dir) == os.path.dirname(os.path.abspath(a.ckpt)):
            p.error("out_dir must not be the checkpoint's directory")
        settings = Settings(n1=a.n1, n2=a.n2, head=a.head, threshold=a.threshold, perturb_scales=a.perturb_scales,
                            restart_scales=a.restart_scales, seeds=a.seeds)
        model, loader, set_names, set_sizes, batch_size = load_model_and_data(a.ckpt, settings, a.batch_size)
        settings.chunk_size = batch_size
        m = re.fullmatch(r"step_(\d+)", os.path.basename(a.ckpt))
        run_experiment(model, loader, settings, a.out_dir, set_names, set_sizes, int(m.group(1)) if m else 0,
                       max_puzzles=a.max_puzzles,
                       extra_settings={"ckpt": a.ckpt, "selection_frac": a.selection_frac, "split_seed": a.split_seed})
    summarize(a.out_dir, a.reference, a.bootstrap, a.selection_frac, a.split_seed)


if __name__ == "__main__":
    main()
