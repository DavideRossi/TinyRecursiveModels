"""Tests of the curriculum (looped_lm/curriculum.py and its use in train.py), CPU, tiny models.

    python tests/looped_lm/test_curriculum.py
"""
import dataclasses
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from looped_lm.curriculum import Curriculum, after_check, initial_state, stream_of  # noqa: E402
from looped_lm.dataset_generation import suite  # noqa: E402
from looped_lm.dataset_generation.config import TaskConfig  # noqa: E402
from looped_lm.dataset_generation.dataset import TrainStream  # noqa: E402
from looped_lm.dataset_generation.storage import stream_id, write_set  # noqa: E402
from looped_lm.model import ModelConfig  # noqa: E402
from looped_lm.train import CURRICULUM_FAILED, TrainConfig, main, train  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "looped_lm")
DEPTH_FILE = os.path.join(ROOT, "configs", "curriculum_depth.json")
TRAIN_JSON = os.path.join(ROOT, "dataset_generation", "configs", "train.json")
SUITE = os.path.join(ROOT, "dataset_generation", "configs", "eval_suite.json")

TINY = dict(width=32, num_heads=2, head_dim=16, mlp_hidden=64, core_layers=1, max_positions=512)
# A small base task; the test curriculum overrides its depth (worst case 4 components x 3 x 2 = 24 edges)
BASE = TaskConfig(name="base", targets_mode="fixed", depth=3, branching={"range": [1, 2]},
                  num_queries={"range": [1, 2]}, num_targets={"range": [3, 4]}, total_edges={"range": [24, 28]})
STAGES = [{"name": "a", "overrides": {"depth": 1}, "check_set": "cfg/c1"},
          {"name": "b", "overrides": {"depth": {"values": [1, 2], "probs": [0.5, 0.5]}}, "check_set": "cfg/c2"},
          {"name": "c", "overrides": {"depth": {"range": [1, 3]}}, "check_set": "cfg/c3"}]


def test_depth_curriculum_file():
    """configs/curriculum_depth.json: the stages are train.json with only the depth changed; half the
    probability on the depths new in the stage, half on the earlier ones, uniform within each;
    checks at each stage's maximum depth, on cfg/train_d<max> sets of the suite."""
    cur = Curriculum.load(DEPTH_FILE)
    base = TaskConfig.load(TRAIN_JSON)
    tasks = cur.stage_tasks(base)
    assert [t.depth.support for t in tasks] == [[1], [1, 2], [1, 2, 3, 4], list(range(1, 9))]
    assert (cur.threshold, cur.check_metric, cur.check_interval, cur.check_samples, cur.max_stage_steps) == \
        (0.9, "first_query_exact_match", 250, 1000, 10000)
    earlier = []
    for t in tasks:
        p = dict(zip(t.depth.values.tolist(), t.depth.probs.tolist() if t.depth.probs is not None
                     else [1.0 / len(t.depth.values)] * len(t.depth.values)))
        new = [d for d in p if d not in earlier]
        if earlier:
            assert abs(sum(p[d] for d in new) - 0.5) < 1e-9 and abs(sum(p[d] for d in earlier) - 0.5) < 1e-9
        assert len({round(p[d], 9) for d in new}) == 1 and len({round(p[d], 9) for d in earlier}) <= 1
        earlier = sorted(p)
        # everything else as train.json
        assert {k: v for k, v in t.to_dict().items() if k not in ("depth", "name")} == \
            {k: v for k, v in base.to_dict().items() if k not in ("depth", "name")}
    specs = {x.name: x for x in suite.expand(suite.load(SUITE))}
    for st, t in zip(cur.stages, tasks):
        assert st.check_set == f"cfg/train_d{t.depth.max}"
    for d in (1, 2, 3, 4, 6, 8):
        cfg = specs[f"cfg/train_d{d}"].config
        assert cfg.depth.support == [d] and specs[f"cfg/train_d{d}"].num_samples == 1000
        assert {k: v for k, v in cfg.to_dict().items() if k not in ("depth", "name")} == \
            {k: v for k, v in base.to_dict().items() if k not in ("depth", "name")}


def test_stage_streams():
    """Stage k's data: stream k + 1 of the training split, batch j a pure function of (seed, stage, j);
    stages draw different data; the depth mix follows the stage."""
    cur = Curriculum.load(DEPTH_FILE)
    tasks = cur.stage_tasks(TaskConfig.load(TRAIN_JSON))
    a = TrainStream(tasks[1], seed=3, batch_size=8, stream=stream_of(1))
    b = TrainStream(tasks[1], seed=3, batch_size=8, stream=stream_of(1), start=2)
    assert torch.equal(a.item(2)["tokens"], next(iter(b))["tokens"])
    assert torch.equal(a.item(5)["tokens"], TrainStream(tasks[1], seed=3, batch_size=8, stream=2).item(5)["tokens"])
    assert not torch.equal(a.item(0)["tokens"], TrainStream(tasks[1], seed=3, batch_size=8, stream=0).item(0)["tokens"])
    assert not torch.equal(a.item(0)["tokens"], TrainStream(tasks[1], seed=4, batch_size=8, stream=2).item(0)["tokens"])
    depths = np.concatenate([[q["depth"] for qs in a.item(j)["queries"] for q in qs] for j in range(40)])
    assert set(depths.tolist()) == {1, 2} and 0.4 < (depths == 2).mean() < 0.6
    d0 = [q["depth"] for qs in TrainStream(tasks[0], seed=3, batch_size=8, stream=1).item(0)["queries"] for q in qs]
    assert set(d0) == {1}


def test_after_check():
    cur = Curriculum(stages=STAGES, threshold=0.5, check_interval=2, max_stage_steps=4)
    s = initial_state(cur)
    assert after_check(cur, s, 2, 0.4) is None and s["stage"] == 0 and s["log"][-1]["check"] == 0.4
    assert after_check(cur, s, 4, 0.5) == "advanced" and (s["stage"], s["stage_start"]) == (1, 4)
    assert s["log"][0] == {"stage": 0, "name": "a", "start_step": 0, "end_step": 4, "reason": "advanced", "check": 0.5}
    assert after_check(cur, s, 6, 0.1) is None
    assert after_check(cur, s, 8, 0.1) == "failed" and s["status"] == "failed" and s["stage"] == 1
    assert s["log"][-1]["reason"] == "max_steps" and s["log"][-1]["end_step"] == 8
    s = initial_state(cur)
    for step in (2, 4, 6):
        after_check(cur, s, step, 1.0)
    assert s["status"] == "completed" and [e["reason"] for e in s["log"]] == ["advanced", "advanced", "completed"]
    for bad in (dict(check_interval=3, max_stage_steps=4), dict(check_metric="loss"), dict(stages=[])):
        try:
            Curriculum(**dict(dict(stages=STAGES), **bad))
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def _setup(d, fmt="direct", **cur_kw):
    """A suite with the three check sets (answers in format fmt) and a curriculum file in d."""
    suite_dir = os.path.join(d, "suite")
    for i, st in enumerate(STAGES, start=1):
        cfg = BASE.replace(name=st["check_set"], depth=i, answer_format=fmt)
        write_set(os.path.join(suite_dir, st["check_set"]), cfg, 6, 1, stream_id(st["check_set"]), readable=0)
    path = os.path.join(d, f"cur_{len(os.listdir(d))}.json")
    with open(path, "w") as f:
        json.dump(dict(dict(stages=STAGES, check_interval=2, check_samples=6, max_stage_steps=4), **cur_kw), f)
    return suite_dir, path


def _cfg(d, name, suite_dir, cur_path, **kw):
    base = dict(steps=12, warmup_steps=2, batch_size=4, num_workers=0, log_interval=2, eval_interval=0,
                ckpt_interval=0, keep_interval=0, wandb="off", device="cpu", lr=3e-3, min_lr_ratio=1.0)
    return TrainConfig(run_name=name, out_dir=d, eval_suite=suite_dir, curriculum=cur_path, **dict(base, **kw))


def _rows(d, name):
    return [json.loads(l) for l in open(os.path.join(d, name, "metrics.jsonl"))]


def test_curriculum_training():
    """A run advances through the stages (a check that always passes), ends when the last one
    passes, logs the stage, writes curriculum.json and a final checkpoint, and is not restarted;
    resuming mid-stage (and across a stage change) gives the same losses; another curriculum
    refuses to resume."""
    with tempfile.TemporaryDirectory() as d:
        suite_dir, path = _setup(d, threshold=-1.0)
        m = ModelConfig(**TINY)
        full = train(_cfg(d, "full", suite_dir, path), m, BASE, record_losses=True)
        assert full["curriculum"] == "completed" and full["step"] == 6 and len(full["losses"]) == 6
        log = json.load(open(os.path.join(d, "full", "curriculum.json")))
        assert log["status"] == "completed"
        assert [(e["name"], e["start_step"], e["end_step"], e["reason"]) for e in log["stages"]] == \
            [("a", 0, 2, "advanced"), ("b", 2, 4, "advanced"), ("c", 4, 6, "completed")]
        rows = _rows(d, "full")
        stages = {r["step"]: r["curriculum/stage"] for r in rows if "curriculum/stage" in r}
        assert stages == {2: 1, 4: 2, 6: 3}  # logged before the check that ends the stage
        assert [r["curriculum/check_stage"] for r in rows if "curriculum/check" in r] == [1, 2, 3]
        assert all("train/first_query_exact_match" in r for r in rows if "train/loss" in r)
        for n in (2, 4, 6):  # at every stage advance and at the end
            assert os.path.exists(os.path.join(d, "full", f"model_step_{n}.pt")), n
        w2 = torch.load(os.path.join(d, "full", "model_step_2.pt"), weights_only=False)
        assert w2["step"] == 2
        saved = torch.load(os.path.join(d, "full", "ckpt_latest.pt"), weights_only=False)
        assert saved["curriculum_state"]["status"] == "completed" and saved["step"] == 6
        # Submitted again: nothing happens
        again = train(_cfg(d, "full", suite_dir, path), m, BASE)
        assert again["curriculum"] == "completed" and again["losses"] == []
        assert len(_rows(d, "full")) == len(rows)

        # Interrupted at step 3 (in stage b) and resumed: the same losses
        first = train(_cfg(d, "split", suite_dir, path), m, BASE, stop_after=3, record_losses=True)
        assert first["step"] == 3 and first["curriculum"] == "running"
        assert torch.load(os.path.join(d, "split", "ckpt_latest.pt"), weights_only=False)["curriculum_state"]["stage"] == 1
        second = train(_cfg(d, "split", suite_dir, path), m, BASE, record_losses=True)
        assert second["curriculum"] == "completed" and second["step"] == 6
        assert np.allclose(first["losses"] + second["losses"], full["losses"], rtol=0, atol=1e-6)

        # Another curriculum file does not resume the run
        train(_cfg(d, "other", suite_dir, path), m, BASE, stop_after=1)
        _, path2 = _setup(d, threshold=-1.0, check_samples=5)
        try:
            train(_cfg(d, "other", suite_dir, path2), m, BASE)
        except ValueError as e:
            assert "curriculum" in str(e)
        else:
            raise AssertionError("a run must not resume under another curriculum")


def test_curriculum_failure():
    """A stage that never passes stops the run at max_stage_steps, with a checkpoint, the stage log
    and exit status 4; submitted again, it does not restart."""
    with tempfile.TemporaryDirectory() as d:
        suite_dir, path = _setup(d, threshold=2.0)
        out = train(_cfg(d, "fail", suite_dir, path), ModelConfig(**TINY), BASE)
        assert out["curriculum"] == "failed" and out["step"] == 4
        log = json.load(open(os.path.join(d, "fail", "curriculum.json")))
        assert log["status"] == "failed" and log["stage"] == 0
        assert log["stages"] == [{"stage": 0, "name": "a", "start_step": 0, "end_step": 4, "reason": "max_steps",
                                  "check": log["stages"][0]["check"]}]
        assert 0.0 <= log["stages"][0]["check"] <= 1.0
        assert os.path.exists(os.path.join(d, "fail", "ckpt_latest.pt"))
        assert os.path.exists(os.path.join(d, "fail", "model_step_4.pt"))
        base_json = os.path.join(d, "base.json")
        with open(base_json, "w") as f:
            f.write(BASE.to_json())
        argv = ["--run_name", "fail2", "--out_dir", d, "--data_config", base_json, "--eval_suite", suite_dir,
                "--curriculum", path, "--steps", "12", "--warmup_steps", "2", "--batch_size", "4",
                "--num_workers", "0", "--log_interval", "2", "--eval_interval", "0", "--wandb", "off",
                "--device", "cpu", "--width", "32", "--num_heads", "2", "--head_dim", "16", "--mlp_hidden", "64",
                "--core_layers", "1", "--max_positions", "512"]
        assert main(argv) == CURRICULUM_FAILED
        n = len(_rows(d, "fail2"))
        assert main(argv) == CURRICULUM_FAILED and len(_rows(d, "fail2")) == n  # not restarted


def test_frontier_curriculum_logs_levels():
    """With frontier answers, the checks and the periodic evaluation log the bracket levels."""
    with tempfile.TemporaryDirectory() as d:
        suite_dir, path = _setup(d, fmt="frontier", threshold=-1.0)
        cfg = _cfg(d, "fr", suite_dir, path, eval_interval=2, eval_sets="cfg/c2,cfg/c3", eval_samples=6)
        out = train(cfg, ModelConfig(**TINY), BASE.replace(answer_format="frontier"))
        assert out["curriculum"] == "completed"
        rows = _rows(d, "fr")
        checks = [r for r in rows if "curriculum/check" in r]
        assert [sorted(k for k in r if k.startswith("curriculum/check_level")) for r in checks] == \
            [["curriculum/check_level1", "curriculum/check_level1_set"],
             ["curriculum/check_level1", "curriculum/check_level1_set", "curriculum/check_level2",
              "curriculum/check_level2_set"],
             [f"curriculum/check_level{k}{s}" for k in (1, 2, 3) for s in ("", "_set")]]
        assert all("curriculum/check_final" in r for r in checks)
        ev = [r for r in rows if "eval/cfg/c3/first_query_exact_match" in r]
        assert ev and all(f"eval/cfg/c3/first_query_level{k}" in r and f"eval/cfg/c2/first_query_level{k}_set" in r
                          for r in ev for k in (1, 2))
        assert all(0.0 <= r[k] <= 1.0 for r in ev for k in r if "level" in k)


def test_missing_check_set():
    with tempfile.TemporaryDirectory() as d:
        suite_dir, path = _setup(d, threshold=-1.0)
        cfg = _cfg(d, "x", os.path.join(d, "nowhere"), path)
        try:
            train(cfg, ModelConfig(**TINY), BASE)
        except FileNotFoundError as e:
            assert "check set" in str(e)
        else:
            raise AssertionError("a missing check set must fail at the start")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
