"""Loading and plotting functions for analysis/report.ipynb.

Presentation only: everything here reads the output files of analysis/eval_steps.py
(summary.json and the curve CSVs). Nothing is computed from the raw capture files.

A run is one analysis output directory, or the "first K steps" block of one (written by
eval_steps.py --truncate-steps K). Runs may differ in number of steps and in the heads and
quantities available; a view that cannot be drawn for a run skips it with a printed note.

Conventions used by every view: one color per run, one line style per head. The views of the
classes of unsolved puzzles (frozen, cycling, ...) use a separate fixed color per class.
"""
import csv
import html
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#008300", "#e34948", "#eda100", "#e87ba4", "#6250d6"]
CLASS_COLORS = {"frozen": "#6250d6", "cycling": "#e87ba4", "revisiting": "#eda100", "wandering": "#8f8e87"}
CLASS_LABELS = {"frozen": "frozen (period 1, to the end)", "cycling": "cycling (period > 1, to the end)",
                "revisiting": "revisiting (repeats, not periodic)", "wandering": "wandering (no grid repeats)"}
HEAD_STYLES = ["-", "--", "-.", ":"]
TOLERANCE_MARKERS = ["o", "s", "^", "v", "P", "X"]
INK, MUTED, GRID = "#2b2b2a", "#6f6e68", "#d9d8d2"

RunSpec = Union[str, Tuple[str, int]]


# --------------------------------------------------------------------------- loading


@dataclass
class Run:
    label: str
    directory: str
    truncate: Optional[int]  # K: the "first K steps" block of the analysis
    set_name: str
    result: dict             # the summary.json block of this set (and truncation)
    color: str
    inputs: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return self.result["num_steps"]

    @property
    def step_offset(self) -> int:
        """Captures of later steps only (e.g. a restart branch) start at step_offset + 1."""
        return self.result.get("step_offset", 0)

    def steps(self, n: int) -> np.ndarray:
        """Step numbers of n per-step values starting at the first captured step."""
        return np.arange(1, n + 1) + self.step_offset

    def heads(self, wanted: Sequence[str]) -> List[str]:
        """The wanted heads this run has; prints a note for the others."""
        for h in wanted:
            if h not in self.result["heads"]:
                note(self, f"no head '{h}' (available: {', '.join(self.result['heads'])})")
        return [h for h in wanted if h in self.result["heads"]]

    def curve(self, head: str) -> Optional[Dict[str, np.ndarray]]:
        """threshold / avg_steps / accuracy / frac_halted_early arrays from the curve CSV."""
        suffix = "" if self.truncate is None else f"_first{self.truncate}"
        path = os.path.join(self.directory, f"curve_{_safe_name(self.set_name)}_{_safe_name(head)}{suffix}.csv")
        if not os.path.exists(path):
            return None
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None


def _safe_name(s: str) -> str:  # same rule as eval_steps.py uses for file names
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def note(run: Run, why: str):
    print(f"[{run.label}] skipped: {why}")


def load_runs(runs: Dict[str, RunSpec], set_name: str = "all", root: str = ".") -> List[Run]:
    """runs: {label: analysis output directory} or {label: (directory, K)} for the first-K-steps block.

    Relative directories are resolved against root. Colors are assigned in the order given.
    """
    if len(runs) > len(RUN_COLORS):
        raise ValueError(f"at most {len(RUN_COLORS)} runs can be shown together (got {len(runs)})")
    out = []
    for i, (label, spec) in enumerate(runs.items()):
        directory, truncate = (spec, None) if isinstance(spec, (str, os.PathLike)) else spec
        directory = os.path.join(root, directory)
        path = os.path.join(directory, "summary.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"run '{label}': {path} not found (run analysis/eval_steps.py first)")
        with open(path) as f:
            summary = json.load(f)
        if set_name not in summary["results"]:
            raise KeyError(f"run '{label}': no set '{set_name}' (available: {', '.join(summary['results'])})")
        result = summary["results"][set_name]
        if truncate is not None:
            blocks = result.get("truncated", {})
            if str(truncate) not in blocks:
                raise KeyError(f"run '{label}': no first-{truncate}-steps block in {path} (available: "
                               f"{', '.join(blocks) or 'none'}); rerun eval_steps.py with --truncate-steps {truncate}")
            result = blocks[str(truncate)]
        out.append(Run(label, directory, truncate, set_name, result, RUN_COLORS[i], summary.get("inputs", {})))
    return out


def _num(x) -> float:
    """JSON value to float ('+inf' / '-inf' are stored as strings)."""
    return float(x) if x is not None else float("nan")


def _per_step(entries) -> Optional[List[dict]]:
    """Per-step list from the summary (entry for step 1 is null), or None if not available."""
    return [e for e in entries if e] if isinstance(entries, list) else None


# --------------------------------------------------------------------------- tables


class Table:
    """A small table that renders as HTML in a notebook and as text elsewhere."""

    def __init__(self, columns: List[str], rows: List[List[str]], caption: str = ""):
        self.columns, self.rows, self.caption = columns, rows, caption

    def _repr_html_(self) -> str:
        e = html.escape
        head = "".join(f"<th style='text-align:right'>{e(c)}</th>" for c in self.columns)
        body = "".join("<tr>" + "".join(f"<td style='text-align:right;white-space:nowrap'>{e(v)}</td>" for v in r)
                       + "</tr>" for r in self.rows)
        cap = f"<caption style='caption-side:bottom;text-align:left'>{e(self.caption)}</caption>" if self.caption else ""
        return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def __str__(self) -> str:
        widths = [max(len(str(x)) for x in col) for col in zip(self.columns, *self.rows)]
        lines = ["  ".join(str(v).rjust(w) for v, w in zip(r, widths)) for r in [self.columns] + self.rows]
        return "\n".join(lines + ([self.caption] if self.caption else []))

    __repr__ = __str__


def _with_ci(point: dict, key: str, digits: int) -> str:
    text = f"{point[key]:.{digits}f}"
    if key + "_ci" in point:
        lo, hi = point[key + "_ci"]
        text += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return text


def summary_table(runs: List[Run]) -> Table:
    """One row per run: puzzles, full-step accuracy, oracle."""
    rows = []
    for run in runs:
        r = run.result
        rows.append([run.label, str(r["num_steps"]), f"{r['num_puzzles']:,}", f"{r['num_report']:,}",
                     f"{r['full_step_accuracy_all_puzzles']:.4f}", f"{r['full_step_accuracy']:.4f}",
                     f"{r['oracle']['accuracy']:.4f}", f"{r['oracle']['avg_steps']:.2f}"])
    return Table(["run", "steps", "puzzles", "report split", "full-step accuracy (all puzzles)",
                  "full-step accuracy (report)", "oracle accuracy", "oracle avg steps"], rows,
                 "Everything except the 'all puzzles' column is measured on the report split.")


def operating_points_table(runs: List[Run], heads: Sequence[str] = ("q_head",)) -> Table:
    """One row per run and head; for each tolerance the selected threshold, accuracy and average steps.

    Thresholds are chosen on the selection split; accuracy and steps (with 95% bootstrap
    intervals, when the analysis computed them) are measured on the report split.
    """
    tolerances = sorted({op["tolerance"] for run in runs for h in run.result["heads"].values()
                         for op in h["operating_points"]})
    columns = ["run", "head", "t = 0: accuracy", "t = 0: avg steps"]
    for tol in tolerances:
        columns += [f"tol {tol:g}: threshold", f"tol {tol:g}: accuracy", f"tol {tol:g}: avg steps"]
    rows = []
    for run in runs:
        for head in run.heads(heads):
            h = run.result["heads"][head]
            row = [run.label, head, f"{h['t0']['accuracy']:.4f}", f"{h['t0']['avg_steps']:.2f}"]
            points = {op["tolerance"]: op for op in h["operating_points"]}
            for tol in tolerances:
                if tol in points:
                    rep = points[tol]["report"]
                    row += [f"{_num(rep['threshold']):.3f}", _with_ci(rep, "accuracy", 4), _with_ci(rep, "avg_steps", 2)]
                else:
                    row += ["-", "-", "-"]
            rows.append(row)
    return Table(columns, rows, "tol: allowed accuracy loss against running all steps. "
                                "Thresholds chosen on the selection split, values measured on the report split.")


def flip_table(runs: List[Run]) -> Table:
    rows = []
    for run in runs:
        f = run.result["flips"]
        n = f["num_puzzles"]
        share = lambda k: f"{f[k]:,} ({f[k] / n:.3%})"
        rows.append([run.label, f"{n:,}", share("correct_at_some_step"),
                     share("correct_at_some_step_but_wrong_at_last"), share("correct_then_wrong_then_correct")])
    return Table(["run", "puzzles (report split)", "correct at some step",
                  "correct at some step, wrong at the last", "correct, then wrong, then correct"], rows)


def _cycles(run: Run) -> Optional[dict]:
    """The cycle analysis of a run, or None (with a note) if there is nothing to show."""
    cycles = run.result.get("cycles")
    if not isinstance(cycles, dict):
        note(run, "the analysis has no grid hashes (older capture, or rerun eval_steps.py)")
        return None
    if not cycles["num_unsolved_at_last"]:
        note(run, "no puzzle is unsolved at the last step")
        return None
    return cycles


def cycle_table(runs: List[Run]) -> Optional[Table]:
    """One row per run: the classes of the puzzles unsolved at the last step."""
    rows = []
    for run in runs:
        cycles = _cycles(run)
        if cycles:
            n = cycles["num_unsolved_at_last"]
            share = lambda k: f"{k:,} ({k / n:.1%})"
            rows.append([run.label, f"{n:,}"] + [share(c["count"]) for c in cycles["classes"].values()]
                        + [share(cycles["repeats"]["with_any_repeat"]), share(cycles["revisiting_ending_periodic"])])
    if not rows:
        return None
    return Table(["run", "unsolved at the last step", "frozen", "cycling", "revisiting", "wandering",
                  "some grid repeats", "revisiting, last period repeats"], rows,
                 "Report split. Frozen and cycling: periodic from the first repeated grid to the last step. "
                 "The last column counts revisiting puzzles that still end in a repeating pattern.")


# --------------------------------------------------------------------------- plots


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _style(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _back_to_front(runs: List[Run]) -> List[Run]:
    """Drawing order: longest run first, so a shorter run that coincides with the start of a
    longer one (e.g. its first K steps) is drawn on top and both stay visible."""
    return sorted(runs, key=lambda run: -run.num_steps)


def _plain_log(ax):
    """Logarithmic y axis with plain tick labels (0.1, 0.3) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter
    ax.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.grid(which="minor", color=GRID, linewidth=0.3)


def _run_handles(runs: List[Run]):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=run.color, linewidth=2, label=run.label) for run in runs]


def _legend(ax, handles, **kw):
    if handles:
        ax.legend(handles=handles, fontsize=8, frameon=False, labelcolor=INK, **kw)


def plot_accuracy_by_step(runs: List[Run]):
    """Fixed-step accuracy by step; share of unsolved puzzles and solving rate on logarithmic axes."""
    plt = _plt()
    fig, (ax_acc, ax_unsolved, ax_rate) = plt.subplots(1, 3, figsize=(16, 4.2))
    with_rate = []
    for run in _back_to_front(runs):
        entries = _per_step(run.result.get("solving_rate"))
        if entries is None:
            note(run, "the analysis has no solving rate (rerun eval_steps.py)")
            continue
        entries = [e for e in entries if e.get("rate")]  # a rate of zero cannot be drawn on a log axis
        if not entries:
            note(run, "no puzzle is solved after the first step")
            continue
        steps = [e["step"] for e in entries]
        ax_rate.fill_between(steps, [max(e["rate_ci"][0], 1e-12) for e in entries], [e["rate_ci"][1] for e in entries],
                             color=run.color, alpha=0.2, linewidth=0)
        ax_rate.plot(steps, [e["rate"] for e in entries], "-o", color=run.color, linewidth=2, markersize=3.5)
        with_rate.append(run)
    _style(ax_rate, "step", "share of the unsolved puzzles solved at this step (log scale)",
           "Solving rate (95% interval)")
    if with_rate:
        _plain_log(ax_rate)
        _legend(ax_rate, _run_handles([run for run in runs if run in with_rate]), loc="upper right")
    for run in _back_to_front(runs):
        acc = np.array(run.result["fixed_step_accuracy"])
        steps = run.steps(len(acc))
        ax_acc.plot(steps, acc, "-o", color=run.color, linewidth=2, markersize=3.5)
        unsolved = 1 - acc
        ax_unsolved.plot(steps[unsolved > 0], unsolved[unsolved > 0], "-o", color=run.color, linewidth=2, markersize=3.5)
    _style(ax_acc, "step", "accuracy (all cells correct)", "Accuracy when every puzzle stops at the same step")
    _style(ax_unsolved, "step", "share of puzzles unsolved (log scale)", "Share of puzzles still unsolved")
    _plain_log(ax_unsolved)
    _legend(ax_acc, _run_handles(runs), loc="lower right")
    fig.tight_layout()
    return fig


def _draw_operational(ax, runs: List[Run], heads: Sequence[str], notes: bool):
    for run in _back_to_front(runs):
        fixed = np.array(run.result["fixed_step_accuracy"])
        ax.plot(np.arange(1, len(fixed) + 1), fixed, ":", color=run.color, linewidth=1.2, alpha=0.6,
                marker=".", markersize=4)
        orc = run.result["oracle"]
        ax.plot(orc["avg_steps"], orc["accuracy"], "*", color=run.color, markeredgecolor=INK,
                markeredgewidth=0.6, markersize=15, zorder=5)
        available = [h for h in heads if h in run.result["heads"]] if not notes else run.heads(heads)
        for head in available:
            style = HEAD_STYLES[list(heads).index(head) % len(HEAD_STYLES)]
            curve = run.curve(head)
            if curve is None:
                if notes:
                    note(run, f"no curve CSV for head '{head}'")
            else:
                order = np.argsort(curve["avg_steps"], kind="stable")
                ax.plot(curve["avg_steps"][order], curve["accuracy"][order], style, color=run.color, linewidth=2)
            h = run.result["heads"][head]
            ax.plot(h["t0"]["avg_steps"], h["t0"]["accuracy"], "D", color=run.color, markeredgecolor="white",
                    markeredgewidth=1.2, markersize=8, zorder=4)
            for i, op in enumerate(h["operating_points"]):
                ax.plot(op["report"]["avg_steps"], op["report"]["accuracy"], TOLERANCE_MARKERS[i % len(TOLERANCE_MARKERS)],
                        color="white", markeredgecolor=run.color, markeredgewidth=1.8, markersize=8, zorder=6)


def plot_operational_curves(runs: List[Run], heads: Sequence[str] = ("q_head",)):
    """Accuracy against average steps as the halting threshold varies; right panel zooms on the corner."""
    from matplotlib.lines import Line2D
    plt = _plt()
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))
    _draw_operational(ax, runs, heads, notes=True)
    _draw_operational(ax_zoom, runs, heads, notes=False)
    xlabel = "average steps"
    if any(run.step_offset for run in runs):
        xlabel += " (counted from the first captured step: " + ", ".join(
            f"{run.label} from step {run.step_offset + 1}" for run in runs) + ")"
    _style(ax, xlabel, "accuracy (all cells correct)", "Accuracy against average steps (report split)")
    _style(ax_zoom, "average steps", "accuracy (all cells correct)", "Zoom on the corner")

    # Corner: from the oracle / t = 0 points to a little beyond the selected operating points
    xs, ys = [], []
    for run in runs:
        xs.append(run.result["oracle"]["avg_steps"])
        ys.append(run.result["full_step_accuracy"])
        for head in heads:
            h = run.result["heads"].get(head)
            if h:
                points = [h["t0"]] + [op["report"] for op in h["operating_points"]]
                xs += [p["avg_steps"] for p in points]
                ys += [p["accuracy"] for p in points]
    if xs:
        dx, dy = max(max(xs) - min(xs), 0.5), max(max(ys) - min(ys), 0.005)
        ax_zoom.set_xlim(min(xs) - 0.6 * dx, max(xs) + 1.2 * dx)
        ax_zoom.set_ylim(min(ys) - 0.35 * dy, max(ys) + 0.2 * dy)

    gray = dict(color=MUTED, linestyle="none")
    tolerances = sorted({op["tolerance"] for run in runs for h in run.result["heads"].values()
                         for op in h["operating_points"]})
    handles = _run_handles(runs)
    handles += [Line2D([], [], color=MUTED, linestyle=HEAD_STYLES[i % len(HEAD_STYLES)], linewidth=2, label=f"head {h}")
                for i, h in enumerate(heads)]
    handles += [Line2D([], [], color=MUTED, linestyle=":", marker=".", linewidth=1.2, label="fixed number of steps"),
                Line2D([], [], marker="*", markersize=12, label="oracle", **gray),
                Line2D([], [], marker="D", markersize=7, label="threshold t = 0", **gray)]
    handles += [Line2D([], [], marker=TOLERANCE_MARKERS[i % len(TOLERANCE_MARKERS)], markersize=7, markerfacecolor="white",
                       markeredgewidth=1.6, label=f"selected, tolerance {tol:g}", **gray)
                for i, tol in enumerate(tolerances)]
    _legend(ax, handles, loc="lower right")
    fig.tight_layout()
    return fig


def _quartile_band(ax, entries: List[dict], color: str, solid: bool = True, positive: bool = False):
    """Median line with a quartile band; positive=True drops non-positive values (log axes)."""
    keep = [e for e in entries if e.get("median") is not None and (not positive or e["median"] > 0)]
    if not keep:
        return False
    steps = [e["step"] for e in keep]
    lo = [e["q25"] if not positive or e["q25"] > 0 else e["median"] for e in keep]
    ax.fill_between(steps, lo, [e["q75"] for e in keep], color=color, alpha=0.18 if solid else 0.08, linewidth=0)
    ax.plot(steps, [e["median"] for e in keep], "-o" if solid else "--", color=color, linewidth=2 if solid else 1.6,
            markersize=3.5)
    return True


def plot_stuck_vs_moving(runs: List[Run]):
    """Among puzzles unsolved at the previous step: share with an unchanged grid, cells changed, and
    the relative state change, the latter next to the same quantity for puzzles already solved."""
    from matplotlib.lines import Line2D
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    (ax_same, ax_cells), (ax_zh, ax_zl) = axes
    drawn = {ax: [] for ax in axes.ravel()}
    for run in _back_to_front(runs):
        entries = _per_step(run.result.get("unchanged_among_unsolved"))
        entries = [e for e in entries or [] if e.get("share") is not None]
        if entries:
            ax_same.plot([e["step"] for e in entries], [e["share"] for e in entries], "-o", color=run.color,
                         linewidth=2, markersize=3.5)
            drawn[ax_same].append(run)
        else:
            note(run, "the analysis has no record of grid changes (older capture)")

        entries = _per_step(run.result.get("cells_changed_among_unsolved"))
        if entries is None:
            note(run, "the analysis has no count of changed cells (older capture, or rerun eval_steps.py)")
        elif _quartile_band(ax_cells, entries, run.color):
            drawn[ax_cells].append(run)

        for ax, z in ((ax_zh, "z_H"), (ax_zl, "z_L")):
            unsolved = _per_step(run.result.get("state_change_among_unsolved", {}).get(z))
            if unsolved is None:
                note(run, f"the analysis has no relative change of {z} (older capture, or rerun eval_steps.py)")
                continue
            if _quartile_band(ax, unsolved, run.color, positive=True):
                drawn[ax].append(run)
            solved = _per_step(run.result.get("state_change_among_solved", {}).get(z))
            if solved is None:
                note(run, f"the analysis has no relative change of {z} for solved puzzles (rerun eval_steps.py)")
            else:
                _quartile_band(ax, solved, run.color, solid=False, positive=True)

    _style(ax_same, "step", "share with the same grid as the previous step", "Grid did not change")
    _style(ax_cells, "step", "cells changed from the previous step", "Cells changed: median and quartiles")
    for ax, z in ((ax_zh, "z_H"), (ax_zl, "z_L")):
        _style(ax, "step", f"relative change of {z} (log scale)", f"Relative change of {z}: median and quartiles")
        if ax.lines:
            _plain_log(ax)
    groups = [Line2D([], [], color=MUTED, linewidth=2, label="unsolved at the previous step"),
              Line2D([], [], color=MUTED, linewidth=1.6, linestyle="--", label="solved at the previous step")]
    for ax, lst in drawn.items():
        handles = _run_handles([run for run in runs if run in lst])
        _legend(ax, handles + (groups if ax in (ax_zh, ax_zl) and handles else []), loc="best")
    fig.suptitle("Puzzles still unsolved at the previous step (report split); relative change also for solved puzzles",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_halting_steps(runs: List[Run], heads: Sequence[str] = ("q_head",), log_y: bool = False):
    """Number of puzzles halting at each step, one row per run and head, one column per tolerance."""
    plt = _plt()
    rows = []
    for run in runs:
        for head in run.heads(heads):
            points = run.result["heads"][head]["operating_points"]
            if all("halt_step_counts" in op["report"] for op in points) and points:
                rows.append((run, head, points))
            else:
                note(run, f"the analysis has no halting-step counts for '{head}' (rerun eval_steps.py)")
    if not rows:
        return None
    ncols = max(len(points) for _, _, points in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(4.6 * ncols, 2.9 * len(rows)), squeeze=False, sharey="row")
    for axrow, (run, head, points) in zip(axes, rows):
        for ax, op in zip(axrow, points):
            counts = np.array(op["report"]["halt_step_counts"])
            steps = run.steps(len(counts))
            ax.bar(steps[:-1], counts[:-1], width=0.8, color=run.color)
            ax.bar(steps[-1:], counts[-1:], width=0.8, color=run.color, alpha=0.45, hatch="///", edgecolor=run.color)
            _style(ax, "halting step", "puzzles" + (" (log scale)" if log_y else ""),
                   f"{run.label}, {head}: tolerance {op['tolerance']:g} (t = {_num(op['report']['threshold']):.3g})")
            if log_y:
                ax.set_yscale("log")
            ax.grid(axis="x", visible=False)
            ax.annotate(f"reach the last step:\n{counts[-1]:,} ({counts[-1] / counts.sum():.1%})",
                        xy=(0.6, 0.95), xycoords="axes fraction", ha="center", va="top", fontsize=8, color=INK)
        for ax in axrow[len(points):]:
            ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_unsolved_classes(runs: List[Run]):
    """Classes of the puzzles unsolved at the last step: shares per run, and by step for each run.

    By step, the whole is the puzzles still unsolved at that step: those already in their
    confirmed cycle (frozen or cycling), the others that stay unsolved to the end, and those
    that are solved later.
    """
    from matplotlib.patches import Patch
    plt = _plt()
    shown = [(run, c) for run, c in ((run, _cycles(run)) for run in runs) if c]
    if not shown:
        return None
    fig, axes = plt.subplots(1, 1 + len(shown), figsize=(5.2 + 4.6 * len(shown), 4.4), squeeze=False)
    ax, step_axes = axes[0][0], axes[0][1:]

    for y, (run, cycles) in enumerate(shown):
        left = 0.0
        for name, c in cycles["classes"].items():
            ax.barh(y, c["share"], left=left, height=0.6, color=CLASS_COLORS[name], edgecolor="white", linewidth=1.5)
            if c["share"] >= 0.06:
                ax.text(left + c["share"] / 2, y, f"{c['share']:.0%}", ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
            left += c["share"]
    _style(ax, "share of the puzzles unsolved at the last step", "", "Unsolved at the last step, by class")
    ax.set_yticks(range(len(shown)), [f"{run.label}\n({c['num_unsolved_at_last']:,})" for run, c in shown])
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(shown) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=CLASS_COLORS[k], label=CLASS_LABELS[k]) for k in CLASS_COLORS], fontsize=8,
              frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    other, later = "#c9c8c1", "#ecebe6"
    for sax, (run, cycles) in zip(step_axes, shown):
        entries = [e for e in cycles["by_step"] if e["unsolved"]]
        steps = [e["step"] for e in entries]
        n = np.array([e["unsolved"] for e in entries], dtype=float)
        frozen = np.array([e["in_frozen"] for e in entries]) / n
        cycling = np.array([e["in_cycling"] for e in entries]) / n
        at_last = np.array([e["unsolved_at_last"] for e in entries]) / n
        sax.stackplot(steps, frozen, cycling, at_last - frozen - cycling, 1 - at_last,
                      colors=[CLASS_COLORS["frozen"], CLASS_COLORS["cycling"], other, later],
                      edgecolor="white", linewidth=0.8)
        _style(sax, "step", "share of the puzzles unsolved at the step", f"{run.label}: unsolved puzzles by step")
        sax.set_ylim(0, 1)
        sax.set_xlim(steps[0], steps[-1])
    step_axes[0].legend(handles=[Patch(color=CLASS_COLORS["frozen"], label="already frozen"),
                                 Patch(color=CLASS_COLORS["cycling"], label="already cycling"),
                                 Patch(color=other, label="unsolved to the end, not in a cycle (yet)"),
                                 Patch(color=later, label="solved later")],
                        fontsize=8, frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    return fig


def plot_cycle_histograms(runs: List[Run], log_y: bool = False):
    """Confirmed cycles: number of puzzles by period, and by the step of the first repeated grid."""
    plt = _plt()
    shown = []
    for run in runs:
        cycles = _cycles(run)
        if cycles and not cycles["confirmed"]["period_counts"]:
            note(run, "no puzzle is in a confirmed cycle")
        elif cycles:
            shown.append((run, cycles))
    if not shown:
        return None
    fig, axes = plt.subplots(len(shown), 2, figsize=(12, 3.0 * len(shown)), squeeze=False)
    for (ax_period, ax_start), (run, cycles) in zip(axes, shown):
        periods = {int(k): v for k, v in cycles["confirmed"]["period_counts"].items()}
        ax_period.bar(list(periods), list(periods.values()), width=0.8, color=run.color)
        starts = np.array(cycles["confirmed"]["start_step_counts"])
        ax_start.bar(run.steps(len(starts)), starts, width=0.8, color=run.color)
        total = sum(periods.values())
        ylabel = "puzzles" + (" (log scale)" if log_y else "")
        _style(ax_period, "period (steps)", ylabel,
               f"{run.label}: period of the {total:,} confirmed cycles (1 = frozen)")
        _style(ax_start, "step of the first repeated grid", ylabel, f"{run.label}: step at which the cycle starts")
        ax_period.set_xlim(0.2, max(max(periods), 8) + 0.8)
        for ax in (ax_period, ax_start):
            ax.grid(axis="x", visible=False)
            if log_y:
                ax.set_yscale("log")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- distance to the solution


def _wrong_cells(run: Run) -> Optional[dict]:
    w = run.result.get("wrong_cells")
    if not isinstance(w, dict):
        note(run, "the analysis has no wrong-cell counts (older capture, or rerun eval_steps.py)")
        return None
    return w


def plot_wrong_cells_by_step(runs: List[Run]):
    """Puzzles unsolved at each step: wrong cells (median and quartiles), and the share within 1, 2, 4, 8 cells."""
    plt = _plt()
    shown = [(run, w) for run, w in ((run, _wrong_cells(run)) for run in _back_to_front(runs)) if w]
    if not shown:
        return None
    thresholds = list(shown[0][1]["among_unsolved"][0]["share_at_most"])
    fig = plt.figure(figsize=(15, 7.6))
    grid = fig.add_gridspec(2, len(thresholds))
    ax = fig.add_subplot(grid[0, :])
    small = [fig.add_subplot(grid[1, 0])]
    small += [fig.add_subplot(grid[1, k], sharey=small[0]) for k in range(1, len(thresholds))]
    for run, w in shown:
        entries = w["among_unsolved"]
        _quartile_band(ax, entries, run.color)
        steps = [e["step"] for e in entries if e["unsolved"]]
        for sax, k in zip(small, thresholds):
            sax.plot(steps, [e["share_at_most"][k] for e in entries if e["unsolved"]], "-o", color=run.color,
                     linewidth=2, markersize=3)
    _style(ax, "step", "wrong cells", "Wrong cells of the puzzles unsolved at each step: median and quartiles")
    ax.set_ylim(bottom=0)
    for sax, k in zip(small, thresholds):
        _style(sax, "step", "share of unsolved puzzles" if sax is small[0] else "", f"At most {k} wrong cell{'s' if k != '1' else ''}")
        sax.set_ylim(0, None)
    _legend(ax, _run_handles([run for run in runs if run in [r for r, _ in shown]]), loc="upper right")
    fig.tight_layout()
    return fig


def plot_wrong_cells_at_end(runs: List[Run], log_y: bool = False):
    """Puzzles unsolved at the last step: wrong cells at the last step, and the minimum ever reached."""
    plt = _plt()
    shown = []
    for run in runs:
        w = _wrong_cells(run)
        if w and not w["unsolved_at_last"]["num_unsolved_at_last"]:
            note(run, "no puzzle is unsolved at the last step")
        elif w:
            shown.append((run, w["unsolved_at_last"]))
    if not shown:
        return None
    xmax = max(max(len(e["last_step_counts"]), len(e["minimum_counts"])) for _, e in shown)
    fig, axes = plt.subplots(len(shown), 2, figsize=(13, 3.0 * len(shown)), squeeze=False, sharex=True)
    for (ax_last, ax_min), (run, e) in zip(axes, shown):
        for ax, key, med, title in ((ax_last, "last_step_counts", "last_step_median", "wrong cells at the last step"),
                                    (ax_min, "minimum_counts", "minimum_median", "fewest wrong cells at any step")):
            counts = e[key]
            ax.bar(np.arange(len(counts)), counts, width=0.8, color=run.color)
            _style(ax, "wrong cells", "puzzles" + (" (log scale)" if log_y else ""),
                   f"{run.label}: {title} ({e['num_unsolved_at_last']:,} unsolved at the end, median {e[med]:g})")
            ax.grid(axis="x", visible=False)
            ax.set_xlim(-0.8, xmax)
            if log_y:
                ax.set_yscale("log")
        if e["minimum_counts"] and e["minimum_counts"][0]:
            ax_min.annotate(f"{e['minimum_counts'][0]:,} were solved at some step", xy=(0.98, 0.92),
                            xycoords="axes fraction", ha="right", fontsize=8, color=INK)
    fig.tight_layout()
    return fig


def solving_rate_by_distance_table(runs: List[Run]) -> Optional[Table]:
    rows = []
    for run in runs:
        w = _wrong_cells(run)
        for b in (w or {}).get("solving_rate_by_distance", []):
            rate = "-" if b["rate"] is None else (f"{b['rate']:.4f} [{b['rate_ci_approx'][0]:.4f}, "
                                                   f"{b['rate_ci_approx'][1]:.4f}]")
            rows.append([run.label, b["bucket"], f"{b['observations']:,}", f"{b['puzzles']:,}", f"{b['solved']:,}", rate])
    if not rows:
        return None
    return Table(["run", "wrong cells at the previous step", "observations (puzzle, step)", "distinct puzzles",
                  "solved at the step", "solving rate [approx. 95% interval]"], rows,
                 "Pooled over steps. An observation is a puzzle unsolved at step s - 1; it is solved if correct at "
                 "step s. A puzzle contributes one observation per step while unsolved, so the intervals, which treat "
                 "observations as independent, are approximate.")


def plot_solving_rate_by_distance(runs: List[Run], log_y: bool = True):
    """Solving rate by number of wrong cells at the previous step, one bar per run in each bucket."""
    plt = _plt()
    shown = [(run, w["solving_rate_by_distance"]) for run, w in ((run, _wrong_cells(run)) for run in runs) if w]
    if not shown:
        return None
    buckets = [b["bucket"] for b in shown[0][1]]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.8 / len(shown)
    for i, (run, entries) in enumerate(shown):
        by_bucket = {b["bucket"]: b for b in entries}
        for x, name in enumerate(buckets):
            b = by_bucket.get(name)
            if not b or not b["rate"]:
                continue
            pos = x - 0.4 + width * (i + 0.5)
            lo, hi = b["rate_ci_approx"]
            ax.bar(pos, b["rate"], width=width * 0.92, color=run.color)
            ax.errorbar(pos, b["rate"], yerr=[[b["rate"] - lo], [hi - b["rate"]]], color=INK, linewidth=1, capsize=2)
    ax.set_xticks(range(len(buckets)), buckets)
    _style(ax, "wrong cells at the previous step", "share solved at the step" + (" (log scale)" if log_y else ""),
           "Solving rate by distance to the solution (pooled over steps; approximate 95% intervals)")
    ax.grid(axis="x", visible=False)
    if log_y:
        _plain_log(ax)
    _legend(ax, _run_handles([run for run, _ in shown]), loc="upper right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- restarts (experiments/restarts.py)


VARIANT_COLORS = {"continue": MUTED, "perturb": "#2a78d6", "noisy_restart": "#eb6834"}
PLAIN_COLOR = INK


def load_restarts(directory: Optional[str], root: str = ".") -> Optional[dict]:
    """summary.json of an experiments/restarts.py run, or None (with a note)."""
    if directory is None:
        print("[restarts] skipped: RESTARTS_DIR is not set")
        return None
    path = os.path.join(root, directory, "summary.json")
    if not os.path.exists(path):
        print(f"[restarts] skipped: {path} not found (run experiments/restarts.py, or its summarize command)")
        return None
    with open(path) as f:
        summary = json.load(f)
    if not summary["split"]["retried_report_puzzles"]:
        print("[restarts] skipped: no puzzle was retried")
        return None
    return summary


def _label(variant: str, scale) -> str:
    return "control (continue)" if variant == "continue" else f"{variant.replace('_', ' ')} {scale:g}"


def _rng_text(stat: dict, digits: int = 4, signed: bool = False) -> str:
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(stat['mean'])} [{f.format(stat['min'])}, {f.format(stat['max'])}]"


def restarts_table(summary: Optional[dict]) -> Optional[Table]:
    """Per variant and scale, over the retried puzzles: results across seeds, the sequential
    (deployable) strategy, and the any-seed oracle."""
    if summary is None:
        return None
    ctrl = next(b for b in summary["branches"] if b["variant"] == "continue")
    rows = [[_label("continue", 0), "-", f"{ctrl['solved_share']:.4f}", "-", f"{ctrl['false_acceptances']:,}",
             f"{ctrl['mean_steps']:.2f}", "-", "-", "-", "-", f"{ctrl['overall']['accuracy']:.4f} @ {ctrl['overall']['avg_steps']:.2f}"]]
    for g in summary["groups"]:
        a, s = g["across_seeds"], g["sequential"]
        realized = [b["realized_scale_z_H"] for b in summary["branches"] if (b["variant"], b["scale"]) == (g["variant"], g["scale"])]
        plain = s.get("paired_with_plain_same_budget")
        rows.append([_label(g["variant"], g["scale"]), f"{np.mean(realized):.4g}", _rng_text(a["solved_share"]),
                     _rng_text(g["difference_from_control"], signed=True), _rng_text(a["false_acceptances"], 0),
                     _rng_text(a["mean_steps"], 2), f"{s['solved_share']:.4f} ({s['false_acceptances']:,} false)",
                     f"{s['mean_steps']:.2f}",
                     "-" if plain is None else f"{plain['difference']:+.4f} [{plain['difference_ci'][0]:+.4f}, {plain['difference_ci'][1]:+.4f}]"
                     if "difference_ci" in plain else f"{plain['difference']:+.4f}",
                     f"{g['any_seed_oracle']['solved_share']:.4f}",
                     f"{s['overall']['accuracy']:.4f} @ {s['overall']['avg_steps']:.2f}"])
    n, sp = summary["split"]["retried_report_puzzles"], summary["settings"]
    return Table(["variant", "realized noise (z_H)", "solved share: mean [min, max] over seeds", "vs control",
                  "false acceptances", "mean steps", "sequential: solved", "sequential: mean steps",
                  "sequential vs plain, same budget", "any seed (oracle)", "overall: accuracy @ avg steps"], rows,
                 f"{n:,} retried puzzles of the report split (not accepted by {sp['head']} at threshold {sp['threshold']:g} "
                 f"by step {sp['n1']}); each attempt has {sp['n2']} more steps. Sequential: seeds in order, first accepted "
                 "answer kept, every attempt's steps counted (deployable). Any seed (oracle): picks the seed with ground "
                 "truth, several times the budget; an upper bound only. Overall: phase 1 plus phase 2, all report puzzles "
                 "(sequential overall uses the sequential strategy).")


def restarts_paired_table(summary: Optional[dict]) -> Optional[Table]:
    """Paired comparison with the control on the same retried puzzles, per seed and for the sequential strategy."""
    if summary is None:
        return None
    rows = []

    def add(name, p):
        ci = p.get("difference_ci")
        rows.append([name, f"{p['both']:,}", f"{p['variant_only']:,}", f"{p['control_only']:,}", f"{p['neither']:,}",
                     f"{p['difference']:+.4f}" + (f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "")])

    for b in summary["branches"]:
        if b["variant"] != "continue":
            add(f"{_label(b['variant'], b['scale'])}, seed {b['seed']}", b["paired_with_control"])
    for g in summary["groups"]:
        add(f"{_label(g['variant'], g['scale'])}, sequential", g["sequential"]["paired_with_control"])
    return Table(["variant", "solved by both", "variant only", "control only", "neither", "difference [95% interval]"],
                 rows, "Retried puzzles of the report split. Interval: bootstrap over puzzles. Many puzzles changing "
                       "sides with a difference near 0 means the noise re-rolls which puzzles are solved, not how many.")


def plot_restarts_solved(summary: Optional[dict]):
    """Share of retried puzzles solved, and false acceptances, per variant and scale."""
    if summary is None:
        return None
    from matplotlib.lines import Line2D
    plt = _plt()
    n = summary["split"]["retried_report_puzzles"]
    ctrl = next(b for b in summary["branches"] if b["variant"] == "continue")
    groups = summary["groups"]
    fig, (ax, ax_false) = plt.subplots(1, 2, figsize=(14, 4.8))
    labels = [_label("continue", 0)] + [_label(g["variant"], g["scale"]) for g in groups]
    for axis, key, scale in ((ax, "solved", 1 / n), (ax_false, "false_acceptances", 1 / n)):
        axis.scatter([0], [ctrl[key] * scale], s=60, color=VARIANT_COLORS["continue"], zorder=3)
        axis.axhline(ctrl[key] * scale, color=VARIANT_COLORS["continue"], linestyle="--", linewidth=1)
        for x, g in enumerate(groups, start=1):
            color = VARIANT_COLORS.get(g["variant"], MUTED)
            seeds = [b for b in summary["branches"] if (b["variant"], b["scale"]) == (g["variant"], g["scale"])]
            ys = [b[key] * scale for b in seeds]
            axis.scatter([x - 0.12] * len(ys), ys, s=22, color=color, alpha=0.8, zorder=3)
            axis.plot([x - 0.22, x - 0.02], [np.mean(ys)] * 2, color=color, linewidth=2.5)
            axis.scatter([x + 0.15], [g["sequential"][key] * scale], s=70, facecolor="white", edgecolor=color,
                         linewidth=1.8, marker="D", zorder=4)
            if key == "solved":
                axis.scatter([x + 0.15], [g["any_seed_oracle"]["solved_share"]], s=45, color=MUTED, marker="x", zorder=4)
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.set_xlim(-0.6, len(labels) - 0.4)
        axis.grid(axis="x", visible=False)
    _style(ax, "", "share of the retried puzzles solved", f"Retried puzzles solved ({n:,} puzzles, report split)")
    _style(ax_false, "", "share of the retried puzzles", "False acceptances (accepted but wrong)")
    handles = [Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=5, label="one seed"),
               Line2D([], [], color=MUTED, linewidth=2.5, label="mean over seeds"),
               Line2D([], [], color=MUTED, marker="D", markerfacecolor="white", linestyle="none", markersize=7,
                      label="sequential (deployable, several attempts)"),
               Line2D([], [], color=MUTED, marker="x", linestyle="none", markersize=7, label="any seed (oracle, upper bound)"),
               Line2D([], [], color=MUTED, linestyle="--", label="control")]
    _legend(ax, handles, loc="best")
    fig.tight_layout()
    return fig


def plot_restarts_overall(summary: Optional[dict]):
    """Whole procedure (phase 1 + phase 2) on the report split: accuracy against average steps, with
    plain continuation to the same budgets and fixed step counts as references."""
    if summary is None:
        return None
    from matplotlib.lines import Line2D
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    p1 = summary["phase1"]["overall_stop_at_n1"]
    ax.scatter([p1["avg_steps"]], [p1["accuracy"]], marker="s", s=60, color=MUTED, zorder=4)
    ax.annotate(f"stop at step {summary['settings']['n1']}", (p1["avg_steps"], p1["accuracy"]), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color=INK)
    for b in summary["branches"]:
        color = VARIANT_COLORS.get(b["variant"], MUTED)
        ax.scatter([b["overall"]["avg_steps"]], [b["overall"]["accuracy"]], s=60 if b["variant"] == "continue" else 18,
                   color=color, alpha=1 if b["variant"] == "continue" else 0.7, zorder=3)
    for g in summary["groups"]:
        s = g["sequential"]["overall"]
        color = VARIANT_COLORS.get(g["variant"], MUTED)
        ax.scatter([s["avg_steps"]], [s["accuracy"]], marker="D", s=70, facecolor="white", edgecolor=color, linewidth=1.8, zorder=5)
        ax.annotate(f"{g['scale']:g}", (s["avg_steps"], s["accuracy"]), textcoords="offset points", xytext=(6, 2),
                    fontsize=8, color=INK)
    ref = summary.get("reference")
    if ref:
        plain = ref["plain_continuation"]
        xs, ys = [p["overall"]["avg_steps"] for p in plain], [p["overall"]["accuracy"] for p in plain]
        ax.plot(xs, ys, "-o", color=PLAIN_COLOR, linewidth=1.5, markersize=6, zorder=4)
        for p in plain:
            ax.annotate(f"plain, halting within {p['budget']}", (p["overall"]["avg_steps"], p["overall"]["accuracy"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK)
        for f in ref["fixed_steps"]:
            ax.axhline(f["accuracy"], color=PLAIN_COLOR, linestyle=":", linewidth=1)
            ax.annotate(f"{f['steps']} steps, no halting: {f['accuracy']:.4f}", (0.0, f["accuracy"]), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(4, 3), ha="left", fontsize=8, color=MUTED)
    else:
        print("[restarts] no plain reference points: summarize with --reference DIR STEP (a plain capture)")
    _style(ax, "average steps per puzzle", "accuracy (all cells correct)",
           f"Whole procedure on the report split ({summary['split']['report_puzzles']:,} puzzles)")
    handles = [Line2D([], [], color=VARIANT_COLORS[v], marker="o", linestyle="none", markersize=6, label=_label(v, 0).split(" 0")[0])
               for v in ("continue", "perturb", "noisy_restart")]
    handles += [Line2D([], [], color=MUTED, marker="D", markerfacecolor="white", linestyle="none", markersize=7,
                       label="sequential (label: noise scale)"),
                Line2D([], [], color=MUTED, marker="s", linestyle="none", markersize=6, label="phase 1 only")]
    if ref:
        handles.append(Line2D([], [], color=PLAIN_COLOR, marker="o", label="plain continuation, same budgets"))
    _legend(ax, handles, loc="lower right")
    fig.tight_layout()
    return fig
