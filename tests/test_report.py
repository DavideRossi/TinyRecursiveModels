"""Tests for analysis/report.py and a smoke test of analysis/report.ipynb.

Run with `python -m pytest tests/` or `python tests/test_report.py`.
The notebook tests need nbclient and ipykernel and are skipped without them.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from analysis import eval_steps as es  # noqa: E402
from analysis import report  # noqa: E402
from test_eval_steps import _write_rank  # noqa: E402

NOTEBOOK = os.path.join(ROOT, "analysis", "report.ipynb")
REAL_OUTPUT = os.path.join(ROOT, "checkpoints", "sudoku_aux_linear_eval32", "analysis")


def make_analysis(d, name, S, seed, new_capture=True, aux_name="probe"):
    """Synthetic capture -> eval_steps.run -> analysis output directory."""
    rng = np.random.default_rng(seed)
    P = 600
    logits = rng.normal(size=(P, S)).cumsum(axis=1).astype(np.float32)
    correct = np.logical_or.accumulate(rng.random((P, S)) < 0.15, axis=1)
    extra = {}
    if new_capture:
        changed = rng.random((P, S)) < 0.7
        changed[:, 0] = False
        rel = {z: rng.random((P, S)).astype(np.float32) for z in ("z_H", "z_L")}
        for v in rel.values():
            v[:, 0] = np.nan
        hashes = rng.integers(-2 ** 62, 2 ** 62, size=(P, S), dtype=np.int64)
        hashes[:150, 3:] = hashes[:150, 3:4]                    # frozen from step 4
        hashes[150:250, 4:] = hashes[150:250, 2 + np.arange(S - 4) % 2]  # period 2 from step 5
        wrong = np.where(correct, 0, rng.integers(1, 30, size=(P, S))).astype(np.uint8)
        cells_changed = np.where(changed, rng.integers(1, 10, size=(P, S)), 0).astype(np.uint8)
        extra = {"preds_changed": changed, "rel_change": rel, "pred_hash": hashes,
                 "cells": {"wrong_cells": wrong, "cells_changed": cells_changed}}
    ckpt = os.path.join(d, name)
    os.makedirs(ckpt)
    _write_rank(os.path.join(ckpt, "step_1_eval_steps.0"), logits, logits + 0.5, correct, np.arange(P),
                np.zeros(P, dtype=np.int64), ["all"], **extra)
    # The helper names the auxiliary head "probe"
    out = os.path.join(ckpt, "analysis")
    es.run(ckpt, 1, out, num_resamples=100, num_thresholds=50, verbose=False,
           truncate_steps=[S // 2] if new_capture else [])
    return out


def captured(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kw)
    return result, buf.getvalue()


def test_load_runs_and_truncated_block():
    with tempfile.TemporaryDirectory() as d:
        out = make_analysis(d, "a", 8, 0)
        runs = report.load_runs({"half": (out, 4), "full": out})
        assert [r.num_steps for r in runs] == [4, 8]
        assert runs[0].color != runs[1].color
        with open(os.path.join(out, "summary.json")) as f:
            summary = json.load(f)
        assert runs[0].result == summary["results"]["all"]["truncated"]["4"]
        # The curve of the truncated run comes from its own CSV, and ends at its own step limit
        assert runs[0].curve("q_head")["avg_steps"].max() == 4
        assert runs[1].curve("q_head")["avg_steps"].max() == 8
        assert runs[1].curve("no_such_head") is None
        for bad, error in [({"x": (out, 3)}, KeyError), ({"x": os.path.join(d, "missing")}, FileNotFoundError)]:
            try:
                report.load_runs(bad)
            except error:
                pass
            else:
                raise AssertionError(f"expected {error.__name__}")
        try:
            report.load_runs({"x": out}, set_name="other")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError")


def test_tables_show_the_summary_values():
    with tempfile.TemporaryDirectory() as d:
        out = make_analysis(d, "a", 8, 1)
        runs = report.load_runs({"full": out})
        r = runs[0].result
        row = report.summary_table(runs).rows[0]
        assert row[0] == "full" and row[1] == "8"
        assert row[5] == f"{r['full_step_accuracy']:.4f}" and row[7] == f"{r['oracle']['avg_steps']:.2f}"

        table = report.operating_points_table(runs, ["q_head", "probe"])
        assert [x[1] for x in table.rows] == ["q_head", "probe"]
        op = r["heads"]["q_head"]["operating_points"][0]["report"]
        cell = table.rows[0][table.columns.index("tol 0: accuracy")]
        assert cell == f"{op['accuracy']:.4f} [{op['accuracy_ci'][0]:.4f}, {op['accuracy_ci'][1]:.4f}]"
        assert "<table>" in table._repr_html_() and "tol 0.005: avg steps" in str(table)

        flips = report.flip_table(runs).rows[0]
        assert flips[1] == f"{r['flips']['num_puzzles']:,}"
        assert flips[2].startswith(f"{r['flips']['correct_at_some_step']:,} (")


def test_views_skip_what_a_run_lacks():
    with tempfile.TemporaryDirectory() as d:
        new = make_analysis(d, "new", 8, 2)
        old = make_analysis(d, "old", 6, 3, new_capture=False)
        # An analysis written before halt_step_counts existed
        path = os.path.join(old, "summary.json")
        with open(path) as f:
            summary = json.load(f)
        for head in summary["results"]["all"]["heads"].values():
            for op in head["operating_points"]:
                del op["report"]["halt_step_counts"]
        del summary["results"]["all"]["state_change_among_unsolved"]
        with open(path, "w") as f:
            json.dump(summary, f)
        runs = report.load_runs({"new": new, "old": old})

        fig, out = captured(report.plot_accuracy_by_step, runs)
        assert out == "" and len(fig.axes) == 3 and fig.axes[1].get_yscale() == "log" == fig.axes[2].get_yscale()
        assert all(len(ax.lines) == 2 for ax in fig.axes)

        fig, out = captured(report.plot_operational_curves, runs, ["q_head", "missing_head"])
        assert out.count("skipped") == 2 and "no head 'missing_head'" in out
        assert len(fig.axes) == 2

        fig, out = captured(report.plot_stuck_vs_moving, runs)
        assert out.count("[old] skipped") == 4 and "[new]" not in out
        # grid unchanged, cells changed, z_H and z_L (unsolved solid, solved dashed)
        assert [len(ax.lines) for ax in fig.axes] == [1, 1, 2, 2]
        assert [l.get_linestyle() for l in fig.axes[2].lines] == ["-", "--"]
        z = runs[0].result["state_change_among_solved"]["z_H"]
        assert np.allclose(fig.axes[2].lines[1].get_ydata(), [e["median"] for e in z[1:] if e["median"]])

        fig, out = captured(report.plot_halting_steps, runs, ["q_head"], log_y=True)
        assert out.count("[old] skipped") == 1 and "[new]" not in out
        ops = runs[0].result["heads"]["q_head"]["operating_points"]
        assert len(fig.axes) == len(ops) and fig.axes[0].get_yscale() == "log"
        heights = [p.get_height() for p in fig.axes[0].patches]
        assert heights == ops[0]["report"]["halt_step_counts"]

        fig, out = captured(report.plot_halting_steps, runs[1:], ["q_head"])
        assert fig is None and "skipped" in out

        # Cycles: only the new capture has grid hashes
        cyc = runs[0].result["cycles"]
        table, out = captured(report.cycle_table, runs)
        assert out.count("[old] skipped") == 1 and len(table.rows) == 1
        assert table.rows[0][1] == f"{cyc['num_unsolved_at_last']:,}"
        assert table.rows[0][2].startswith(f"{cyc['classes']['frozen']['count']:,} (")
        assert cyc["classes"]["frozen"]["count"] > 0 and cyc["classes"]["cycling"]["count"] > 0

        fig, out = captured(report.plot_unsolved_classes, runs)
        assert out.count("[old] skipped") == 1 and len(fig.axes) == 2
        widths = [p.get_width() for p in fig.axes[0].patches]
        assert np.allclose(widths, [c["share"] for c in cyc["classes"].values()]) and np.isclose(sum(widths), 1)

        fig, out = captured(report.plot_cycle_histograms, runs, log_y=True)
        assert out.count("[old] skipped") == 1 and len(fig.axes) == 2
        assert sorted(p.get_height() for p in fig.axes[0].patches) == sorted(cyc["confirmed"]["period_counts"].values())
        assert [p.get_height() for p in fig.axes[1].patches] == cyc["confirmed"]["start_step_counts"]

        for view in (report.cycle_table, report.plot_unsolved_classes, report.plot_cycle_histograms):
            result, out = captured(view, runs[1:])
            assert result is None and "skipped" in out

        # Distance to the solution: only the new capture has wrong-cell counts
        w = runs[0].result["wrong_cells"]
        fig, out = captured(report.plot_wrong_cells_by_step, runs)
        assert out.count("[old] skipped") == 1 and len(fig.axes) == 5
        assert np.allclose(fig.axes[0].lines[0].get_ydata(), [e["median"] for e in w["among_unsolved"]])
        assert np.allclose(fig.axes[1].lines[0].get_ydata(), [e["share_at_most"]["1"] for e in w["among_unsolved"]])
        fig, out = captured(report.plot_wrong_cells_at_end, runs, log_y=True)
        assert out.count("[old] skipped") == 1 and len(fig.axes) == 2
        assert [p.get_height() for p in fig.axes[0].patches] == w["unsolved_at_last"]["last_step_counts"]
        assert [p.get_height() for p in fig.axes[1].patches] == w["unsolved_at_last"]["minimum_counts"]
        table, out = captured(report.solving_rate_by_distance_table, runs)
        assert out.count("[old] skipped") == 1 and len(table.rows) == 5
        b = w["solving_rate_by_distance"][0]
        assert table.rows[0][1:5] == [b["bucket"], f"{b['observations']:,}", f"{b['puzzles']:,}", f"{b['solved']:,}"]
        fig, out = captured(report.plot_solving_rate_by_distance, runs)
        assert out.count("[old] skipped") == 1
        assert sorted(p.get_height() for p in fig.axes[0].patches) == sorted(
            x["rate"] for x in w["solving_rate_by_distance"] if x["rate"])
        for view in (report.plot_wrong_cells_by_step, report.plot_wrong_cells_at_end,
                     report.solving_rate_by_distance_table, report.plot_solving_rate_by_distance):
            result, out = captured(view, runs[1:])
            assert result is None and "skipped" in out

        # An analysis written before the solving rate existed
        del runs[1].result["solving_rate"]
        fig, out = captured(report.plot_accuracy_by_step, runs)
        assert out.count("[old] skipped") == 1 and len(fig.axes[2].lines) == 1


def _execute_notebook(runs_source):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(NOTEBOOK, as_version=4)
    if runs_source is not None:
        cell = next(c for c in nb.cells if "parameters" in c.metadata.get("tags", []))
        cell.source = runs_source
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": os.path.dirname(NOTEBOOK)}}).execute()
    return nb


def _have_notebook_tools():
    try:
        import nbclient, nbformat, ipykernel  # noqa: F401, E401
        return True
    except ImportError:
        return False


def _check_executed(nb, figures, tables):
    """figures, tables: (min, max) numbers of figures and HTML tables in the outputs."""
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs]
    assert not [o for o in outputs if o.output_type == "error"], [o for o in outputs if o.output_type == "error"]
    assert figures[0] <= sum("image/png" in o.get("data", {}) for o in outputs) <= figures[1]
    assert tables[0] <= sum("text/html" in o.get("data", {}) for o in outputs) <= tables[1]
    return "".join(o.get("text", "") for o in outputs if o.output_type == "stream")


def test_notebook_has_no_stored_outputs():
    with open(NOTEBOOK) as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_notebook_runs_on_synthetic_output():
    if not _have_notebook_tools():
        print("  (skipped: nbclient / ipykernel not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        new = make_analysis(d, "new", 8, 4)
        old = make_analysis(d, "old", 6, 5, new_capture=False)
        source = (f"RUNS = {{'half': ({new!r}, 4), 'full': {new!r}, 'old': {old!r}}}\n"
                  "HEADS = ['q_head', 'probe']\nSET_NAME = 'all'\nLOG_HISTOGRAMS = False")
        notes = _check_executed(_execute_notebook(source), (9, 9), (5, 5))
        assert "[old] skipped" in notes and "[full]" not in notes


def test_notebook_runs_on_the_32_step_analysis():
    """Smoke test: the notebook as committed, on the existing 32-step analysis output."""
    if not _have_notebook_tools():
        print("  (skipped: nbclient / ipykernel not installed)")
        return
    if not os.path.exists(os.path.join(REAL_OUTPUT, "summary.json")):
        print(f"  (skipped: {os.path.relpath(REAL_OUTPUT, ROOT)} not present)")
        return
    # 4 figures and 3 tables from a capture without grid hashes and cell counts; 9 and 5 with them
    _check_executed(_execute_notebook(None), (4, 9), (3, 5))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
