"""Curriculum of the graph task: stages of data-config overrides, each run until a held-out
check passes (looped_lm/README.md section 2, "Curriculum").

A curriculum file (configs/curriculum_depth.json) holds:

    stages           list of {"name", "overrides" (TaskConfig fields over --data_config),
                     "check_set" (a set of the evaluation suite)}
    threshold        advance when the check reaches this (>=)
    check_metric     the check set's metric: first_query_exact_match (default)
    check_interval   steps between checks, counted within the stage
    check_samples    the first N samples of the check set
    max_stage_steps  a stage still short of the threshold after this many steps fails the run

The model, optimizer and learning-rate schedule run on across stages; only the data change.
Stage k (0-based) draws batch j (its j-th step) from stream k + 1 of the training split, so the
data are a pure function of (seed, stage, step within stage).

State (saved in checkpoints and in curriculum.json in the run directory):

    stage            current stage (0-based)
    stage_start      the global step at which it started
    status           running | completed (the last stage passed) | failed (a stage hit max_stage_steps)
    log              one entry per stage started: stage, name, start_step, end_step, reason
                     (advanced | completed | max_steps | running), check (its last value)
"""
import dataclasses
import json
import os
from typing import List, Optional

from .dataset_generation.config import TaskConfig

CHECK_METRICS = ("first_query_exact_match", "exact_match", "first_query_accuracy", "accuracy")


@dataclasses.dataclass
class Stage:
    name: str
    overrides: dict
    check_set: str


@dataclasses.dataclass
class Curriculum:
    stages: List[Stage]
    threshold: float = 0.9
    check_metric: str = "first_query_exact_match"
    check_interval: int = 250
    check_samples: int = 1000
    max_stage_steps: int = 10000

    def __post_init__(self):
        self.stages = [s if isinstance(s, Stage) else Stage(**s) for s in self.stages]
        if not self.stages:
            raise ValueError("a curriculum needs at least one stage")
        if self.check_metric not in CHECK_METRICS:
            raise ValueError(f"check_metric must be one of {CHECK_METRICS}")
        if self.check_interval < 1 or self.max_stage_steps < self.check_interval:
            raise ValueError("need 1 <= check_interval <= max_stage_steps")
        if self.max_stage_steps % self.check_interval:
            raise ValueError("max_stage_steps must be a multiple of check_interval (the last check is at the limit)")
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError("stage names must be distinct")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path: str) -> "Curriculum":
        with open(path) as f:
            return cls(**json.load(f))

    def stage_task(self, base: TaskConfig, k: int) -> TaskConfig:
        """The data config of stage k: base with the stage's overrides, named base/stage."""
        s = self.stages[k]
        task = TaskConfig.from_dict(dict(base.to_dict(), **s.overrides, name=f"{base.name}/{s.name}"))
        task.check_batch_edges()
        return task

    def stage_tasks(self, base: TaskConfig) -> List[TaskConfig]:
        return [self.stage_task(base, k) for k in range(len(self.stages))]


def stream_of(stage: int) -> int:
    """Training stream of stage k (0-based); stream 0 is the plain, curriculum-free stream."""
    return stage + 1


def initial_state(cur: Curriculum) -> dict:
    return {"stage": 0, "stage_start": 0, "status": "running",
            "log": [{"stage": 0, "name": cur.stages[0].name, "start_step": 0, "end_step": None,
                     "reason": "running", "check": None}]}


def after_check(cur: Curriculum, state: dict, step: int, value: float) -> Optional[str]:
    """Apply a check of the current stage at global step `step` (in place). Returns "advanced",
    "completed", "failed" or None (stay)."""
    k = state["stage"]
    entry = state["log"][-1]
    entry["check"] = value
    if value >= cur.threshold:
        entry.update(end_step=step, reason="advanced" if k + 1 < len(cur.stages) else "completed")
        if k + 1 == len(cur.stages):
            state["status"] = "completed"
            return "completed"
        state.update(stage=k + 1, stage_start=step)
        state["log"].append({"stage": k + 1, "name": cur.stages[k + 1].name, "start_step": step, "end_step": None,
                             "reason": "running", "check": None})
        return "advanced"
    if step - state["stage_start"] >= cur.max_stage_steps:
        entry.update(end_step=step, reason="max_steps")
        state["status"] = "failed"
        return "failed"
    return None


def write_log(run_dir: str, cur: Curriculum, state: dict) -> None:
    path = os.path.join(run_dir, "curriculum.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"status": state["status"], "stage": state["stage"], "stage_start": state["stage_start"],
                   "stages": state["log"], "curriculum": cur.to_dict()}, f, indent=1)
    os.replace(tmp, path)
