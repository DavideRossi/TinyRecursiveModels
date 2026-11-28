"""Training of the looped LM on the graph-reachability stream (looped_lm/README.md).

    python -m looped_lm.train --run_name NAME [--core_layers 8 --width 256 ...] [--steps 50000]

Every field of TrainConfig and ModelConfig is a command-line option (--name value). The run
directory (out_dir/run_name) holds config.json, metrics.jsonl, ckpt_latest.pt (full state, for
resuming) and model_step_N.pt (weights only, every keep_interval steps). Starting a run whose
directory has ckpt_latest.pt resumes it: the model, the optimizer, the schedule step and the data
position (batch j of the stream is a pure function of (data_seed, j), so the stream restarts at
the step); a run started with --init_from takes only the model weights of that checkpoint, with a
fresh optimizer, schedule and data stream (ignored when resuming). With --gate_step N, the run stops
at step N (exit status 3, no checkpoint, gate.json in the run directory) unless the gate metric
over the last log interval is above --gate_threshold: first-query accuracy, or on the induction
task the accuracy on supervised tokens (--gate_enforce false: only record it). With --curriculum
FILE (looped_lm/curriculum.py), the data go through the file's stages, each a set of overrides of
--data_config, advancing when a held-out check passes; the run ends when the last stage passes,
or with exit status 4 when a stage reaches its step limit (curriculum.json in the run directory). SIGUSR1 or SIGTERM (Slurm's time limit, see slurm/train_looped_lm.sbatch) saves a
checkpoint and stops.
"""
import argparse
import dataclasses
import json
import math
import os
import signal
import statistics
import sys
import time
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from .curriculum import Curriculum, after_check, initial_state, stream_of, write_log
from .dataset_generation.config import TaskConfig
from .dataset_generation.dataset import TrainStream
from .diagnostics.induction import InductionConfig, InductionStream, is_induction
from .diagnostics.mqar import MqarConfig, MqarStream, is_mqar
from .evaluation import evaluate_set, load_set
from .model import LoopedLM, ModelConfig
from .objective import loss_and_stats, pad_to_multiple, to_device

GATE_FAILED = 3  # exit status of a run stopped by its gate
CURRICULUM_FAILED = 4  # exit status of a run whose curriculum stage reached its step limit
DEFAULT_EVAL_SETS = ("grid/d1_b1,grid/d2_b2,grid/d4_b2,grid/d8_b1,grid/d8_b4,grid/d16_b4,"
                     "mixed/q4_in_distribution")


@dataclasses.dataclass
class TrainConfig:
    run_name: str = "looped_lm"
    out_dir: str = "checkpoints/looped_lm"           # the run directory is out_dir/run_name
    data_config: str = "looped_lm/dataset_generation/configs/train.json"
    answer_format: str = ""                          # "": the data config's; otherwise overrides it
    allow_small_pool: bool = False                   # diagnostics only: train on a config with pool_size < 2048
    init_from: str = ""                              # checkpoint (ckpt_latest.pt or model_step_N.pt): its model weights only
    gate_step: int = 0                               # 0: off; else stop at this step unless the gate metric passes
    gate_threshold: float = 0.9                      # the gate: gate_metric(task) above this at gate_step
    gate_enforce: bool = True                        # false: record the gate (gate.json, gate/passed) but never stop
    curriculum: str = ""                             # curriculum file (looped_lm/curriculum.py); "": data_config throughout
    batch_size: int = 64
    steps: int = 50000
    lr: float = 1e-3
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1                        # cosine decay to min_lr_ratio * lr at the last step
    weight_decay: float = 0.1                        # on the 2-D weights of the layers (not norms, embedding)
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 0                                    # model initialization
    data_seed: int = -1                              # training stream; -1: same as seed
    num_workers: int = 7
    log_interval: int = 50
    eval_interval: int = 1000                        # 0: no periodic evaluation
    eval_suite: str = "looped_lm/data/eval_suite"
    eval_sets: str = DEFAULT_EVAL_SETS               # comma-separated set names of the suite
    eval_samples: int = 200                          # the first N samples of each set
    eval_max_tokens: int = 131072                    # padded tokens per evaluation batch
    ckpt_interval: int = 2000                        # ckpt_latest.pt (0: only at the end and on signals)
    keep_interval: int = 10000                       # model_step_N.pt (0: never)
    compile: bool = False
    pad_multiple: int = 0                            # pad batches to a multiple of this length; 0: 128 with compile
    device: str = "auto"
    wandb: str = "auto"                              # auto (if installed) | off
    wandb_project: str = "looped_lm"
    benchmark: bool = False                          # time the steps only: no evaluation, checkpoints or logs

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.run_name)

    @property
    def stream_seed(self) -> int:
        return self.seed if self.data_seed < 0 else self.data_seed


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate of update number `step` (0-based): linear warm-up to cfg.lr over warmup_steps,
    then cosine decay to min_lr_ratio * lr at the last step."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
    lo = cfg.lr * cfg.min_lr_ratio
    return lo + (cfg.lr - lo) * 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(model: LoopedLM, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and not name.startswith("embed.") else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def load_task(cfg: TrainConfig):
    """The data config: a TaskConfig, or a diagnostic config (its JSON has "kind": "mqar" or "induction")."""
    with open(cfg.data_config) as f:
        d = json.load(f)
    if is_mqar(d) or is_induction(d):
        if cfg.answer_format:
            raise ValueError(f"--answer_format does not apply to a {d['kind']} config")
        return MqarConfig.from_dict(d) if is_mqar(d) else InductionConfig.from_dict(d)
    task = TaskConfig.from_dict(d)
    return task.replace(answer_format=cfg.answer_format) if cfg.answer_format else task


class _Stop:
    """Set by SIGUSR1 / SIGTERM: save a checkpoint at the end of the current step and stop."""

    def __init__(self):
        self.requested = None

    def __call__(self, signum, frame):
        self.requested = signal.Signals(signum).name


def _save(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


class _Logger:
    def __init__(self, cfg: TrainConfig, configs: dict, resume_step: int):
        self.path = os.path.join(cfg.run_dir, "metrics.jsonl")
        self.wandb = None
        if cfg.wandb != "off":
            try:
                import wandb
            except ImportError:
                if cfg.wandb != "auto":
                    raise
            else:
                os.environ.setdefault("WANDB_MODE", "offline")
                self.wandb = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=configs,
                                        dir=cfg.run_dir, resume="allow", id=_wandb_id(cfg))

    def log(self, metrics: dict, step: int) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(dict(metrics, step=step)) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()


def _wandb_id(cfg: TrainConfig) -> str:
    """A stable wandb run id per run directory, so a resumed run continues the same wandb run."""
    path = os.path.join(cfg.run_dir, "wandb_id.txt")
    if not os.path.exists(path):
        import uuid
        with open(path, "w") as f:
            f.write(uuid.uuid4().hex[:12])
    with open(path) as f:
        return f.read().strip()


def _check_resume(saved: dict, model_cfg: ModelConfig, cfg: TrainConfig, task: TaskConfig,
                  cur: Optional[Curriculum] = None) -> None:
    """A resumed run must have the same model and the same data stream (and curriculum)."""
    problems = []
    if saved.get("curriculum") != (cur.to_dict() if cur is not None else None):
        problems.append("curriculum")
    if saved["model_config"] != model_cfg.to_dict():
        problems.append("model config")
    if saved["task_config"] != task.to_dict():
        problems.append("task config")
    for k in ("batch_size", "seed", "data_seed"):  # the answer format is part of the task config
        if saved["train_config"][k] != getattr(cfg, k):
            problems.append(k)
    if problems:
        raise ValueError(f"{cfg.run_dir} has a checkpoint with a different {', '.join(problems)}; "
                         f"use another run_name or out_dir")


def init_weights(model: LoopedLM, path: str, model_cfg: ModelConfig, device) -> None:
    """Load the model weights of a checkpoint (ckpt_latest.pt or model_step_N.pt) into model; the
    model configs must agree."""
    saved = torch.load(path, map_location=device, weights_only=False)
    theirs, ours = saved["model_config"], model_cfg.to_dict()
    diff = sorted(k for k in set(theirs) | set(ours) if theirs.get(k) != ours.get(k))
    if diff:
        raise ValueError(f"--init_from {path}: model config differs in " +
                         ", ".join(f"{k} ({theirs.get(k)} there, {ours.get(k)} here)" for k in diff))
    model.load_state_dict(saved["model"])


def gate_metric(task) -> str:
    """What the gate measures: train/first_query_accuracy, or on the induction task (no queries)
    train/final_accuracy, the accuracy on supervised tokens."""
    return "final_accuracy" if isinstance(task, InductionConfig) else "first_query_accuracy"


def train(cfg: TrainConfig, model_cfg: ModelConfig, task: Optional[TaskConfig] = None,
          stop_after: Optional[int] = None, record_losses: bool = False) -> dict:
    """Train (or resume) a run. stop_after: stop with a checkpoint once this step is reached, as if
    interrupted (tests). Returns {"step", "losses" (per step, with record_losses), "benchmark",
    "gate_failed", "curriculum" (its status, with a curriculum)}."""
    task = task if task is not None else load_task(cfg)
    cur = Curriculum.load(cfg.curriculum) if cfg.curriculum else None
    stage_tasks = None
    if cur is not None:
        if not isinstance(task, TaskConfig):
            raise ValueError("a curriculum needs a graph-task data config")
        stage_tasks = cur.stage_tasks(task)
        log_path = os.path.join(cfg.run_dir, "curriculum.json")
        if not cfg.benchmark and os.path.exists(log_path):
            with open(log_path) as f:
                done = json.load(f)
            if done["status"] != "running" and done["curriculum"] == cur.to_dict():
                print(f"{cfg.run_dir}: curriculum already {done['status']} at stage {done['stage'] + 1}; "
                      f"not restarting it", flush=True)
                return {"step": done["stages"][-1]["end_step"], "losses": [], "curriculum": done["status"]}
    if cfg.gate_step:
        if cfg.gate_step % cfg.log_interval or cfg.gate_step > cfg.steps:
            raise ValueError(f"gate_step {cfg.gate_step} must be a multiple of log_interval {cfg.log_interval} "
                             f"and at most steps {cfg.steps}")
    gate_path = os.path.join(cfg.run_dir, "gate.json")
    if cfg.gate_step and not cfg.benchmark and os.path.exists(gate_path):
        with open(gate_path) as f:
            gate = json.load(f)
        if not gate["passed"] and gate.get("enforced", True):
            metric = gate.get("metric", "first_query_accuracy")
            print(f"{cfg.run_dir}: failed its gate at step {gate['step']} ({metric} "
                  f"{gate[metric]:.3f}); not restarting it", flush=True)
            return {"step": gate["step"], "losses": [], "gate_failed": True}
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    amp = device.type == "cuda"
    bench = cfg.benchmark
    os.makedirs(cfg.run_dir, exist_ok=True)
    configs = {"model": model_cfg.to_dict(), "train": dataclasses.asdict(cfg), "task": task.to_dict()}
    if cur is not None:
        configs["curriculum"] = cur.to_dict()

    torch.manual_seed(cfg.seed)
    model = LoopedLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  fused=device.type == "cuda")
    step = 0
    cstate = initial_state(cur) if cur is not None else None
    ckpt_path = os.path.join(cfg.run_dir, "ckpt_latest.pt")
    if os.path.exists(ckpt_path) and not bench:
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        _check_resume(saved, model_cfg, cfg, task, cur)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        if cur is not None:
            cstate = saved["curriculum_state"]
        print(f"resumed {cfg.run_dir} at step {step}", flush=True)
    elif cfg.init_from:
        init_weights(model, cfg.init_from, model_cfg, device)
        print(f"{cfg.run_dir}: model weights from {cfg.init_from}", flush=True)
    if step == 0 and not bench:
        with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
            json.dump(configs, f, indent=1)
        if cur is not None:
            write_log(cfg.run_dir, cur, cstate)
    if step >= cfg.steps:
        print(f"{cfg.run_dir}: already at step {step} of {cfg.steps}", flush=True)
        return {"step": step, "losses": []}

    def checkpoint(milestone: bool = False):
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                   "model_config": model_cfg.to_dict(), "train_config": dataclasses.asdict(cfg),
                   "task_config": task.to_dict(), "curriculum": cur.to_dict() if cur is not None else None,
                   "curriculum_state": cstate}
        _save(ckpt_path, payload)
        if milestone:
            _save(os.path.join(cfg.run_dir, f"model_step_{step}.pt"),
                  {k: payload[k] for k in ("model", "step", "model_config", "train_config", "task_config")})

    diagnostic = isinstance(task, (MqarConfig, InductionConfig))

    def make_loader():
        """The data from the current step on; with a curriculum, the current stage's stream from the
        step within the stage."""
        if diagnostic:
            cls = MqarStream if isinstance(task, MqarConfig) else InductionStream
            stream = cls(task, seed=cfg.stream_seed, batch_size=cfg.batch_size, start=step)
        elif cur is not None:
            k = cstate["stage"]
            stream = TrainStream(stage_tasks[k], seed=cfg.stream_seed, batch_size=cfg.batch_size,
                                 start=step - cstate["stage_start"], allow_small_pool=cfg.allow_small_pool,
                                 stream=stream_of(k))
        else:
            stream = TrainStream(task, seed=cfg.stream_seed, batch_size=cfg.batch_size, start=step,
                                 allow_small_pool=cfg.allow_small_pool)
        return DataLoader(stream, batch_size=None, num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                          prefetch_factor=4 if cfg.num_workers > 0 else None, persistent_workers=False)

    loader = make_loader()
    fwd = torch.compile(model) if cfg.compile else model
    pad_multiple = cfg.pad_multiple or (128 if cfg.compile else 1)

    eval_data = {}
    if cfg.eval_interval > 0 and not bench and not diagnostic:  # diagnostics: the stream is held out (no repeats)
        for name in [s for s in cfg.eval_sets.split(",") if s]:
            path = os.path.join(cfg.eval_suite, name)
            if not os.path.exists(os.path.join(path, "meta.json")):
                raise FileNotFoundError(f"evaluation set {path} not found: write the suite first "
                                        f"(python -m looped_lm.dataset_generation.cli write --suite ...)")
            eval_data[name] = load_set(path, cfg.eval_samples, cfg.eval_max_tokens)
    check_data = {}
    if cur is not None and not bench:  # every stage's check set, so a missing one fails at the start
        for st in cur.stages:
            path = os.path.join(cfg.eval_suite, st.check_set)
            if not os.path.exists(os.path.join(path, "meta.json")):
                raise FileNotFoundError(f"curriculum check set {path} not found: write the suite first")
            check_data[st.check_set] = load_set(path, cur.check_samples, cfg.eval_max_tokens)

    stop = _Stop()
    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGUSR1, signal.SIGTERM)}
    logger = None if bench else _Logger(cfg, configs, step)
    if logger is not None and step == 0:
        logger.log({"num_params": model.num_params(), **{f"model/{k}": v for k, v in model.embed_norms().items()}},
                   step=0)

    def run_eval():
        out = {}
        for name, data in eval_data.items():
            r = evaluate_set(model, data, device)
            out.update({f"eval/{name}/accuracy": r["accuracy"], f"eval/{name}/loss": r["loss"],
                        f"eval/{name}/chance": r["chance"], f"eval/{name}/n": r["n"],
                        f"eval/{name}/first_query_accuracy": r["first_query_accuracy"],
                        f"eval/{name}/first_query_chance": r["first_query_chance"]})
            for key in ("first_query_by_depth", "first_query_by_branching"):  # sets mixing d (b) only
                out.update({f"eval/{name}/first_query_accuracy_{k}": s["accuracy"] for k, s in r.get(key, {}).items()})
            out.update({f"eval/{name}/exact_match": r["exact_match"],
                        f"eval/{name}/first_query_exact_match": r["first_query_exact_match"]})
            for key in ("exact_match_by_depth", "first_query_exact_match_by_depth"):  # sets mixing depths only
                metric = key[:-len("_by_depth")]
                out.update({f"eval/{name}/{metric}_{k}": s["accuracy"] for k, s in r.get(key, {}).items()})
        out.update({f"model/{k}": v for k, v in model.embed_norms().items()})
        return out

    losses: List[float] = []
    acc, gate_failed = None, False
    t_last, data_wait, step_times, waits = time.time(), 0.0, [], []
    it = iter(loader)
    model.train()
    try:
        while step < cfg.steps:
            t0 = time.time()
            batch = next(it)
            t1 = time.time()
            batch = to_device(pad_to_multiple(batch, pad_multiple), device)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, cfg)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss, stats = loss_and_stats(fwd, batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            data_wait += t1 - t0
            if record_losses:
                losses.append(float(loss.detach()))
            if bench:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.time() - t0)
                waits.append(t1 - t0)
                continue
            stats = dict(stats, grad_norm=grad_norm.detach().float(), steps=torch.tensor(1, device=device),
                         clipped=(grad_norm.detach() > cfg.grad_clip).float(),
                         real_tokens=batch["length"].sum().to(device) if "length" in batch else torch.tensor(0),
                         padded_tokens=torch.tensor(batch["tokens"].numel(), device=device))
            acc = stats if acc is None else {k: acc[k] + stats[k] for k in acc}
            if step % cfg.log_interval == 0 or step == cfg.steps:
                a = {k: (v.tolist() if v.dim() else float(v)) for k, v in acc.items()}
                by_db = {}  # first-query accuracy by the query's d and b (graph tasks; bin 0: no d, b)
                for key, prefix in (("depth", "d"), ("branching", "b")):
                    n, k = a[f"first_n_by_{key}"], a[f"first_correct_by_{key}"]
                    by_db.update({f"train/first_query_accuracy_{prefix}{i}": k[i] / n[i]
                                  for i in range(1, len(n)) if n[i] > 0})
                # exact match (every answer token right), all queries and first queries, also by depth
                exact = {}  # (none on tasks without queries: induction)
                if a["exact_n"] > 0:
                    exact = {"train/exact_match": a["exact_correct"] / a["exact_n"],
                             "train/first_query_exact_match": a["first_exact_correct"] / max(1, a["first_n"])}
                n, nf = a["exact_n_by_depth"], a["first_n_by_depth"]
                for i in range(1, len(n)):
                    if n[i] > 0:
                        exact[f"train/exact_match_d{i}"] = a["exact_correct_by_depth"][i] / n[i]
                    if nf[i] > 0:
                        exact[f"train/first_query_exact_match_d{i}"] = a["first_exact_correct_by_depth"][i] / nf[i]
                if cur is not None:
                    exact.update({"curriculum/stage": cstate["stage"] + 1,
                                  "curriculum/stage_step": step - cstate["stage_start"]})
                dt = time.time() - t_last
                if cfg.gate_step and step == cfg.gate_step:
                    metric = gate_metric(task)
                    value = (a["final_correct"] / max(1, a["final_n"]) if metric == "final_accuracy"
                             else a["first_correct"] / max(1, a["first_n"]))
                    gate = {"step": step, "metric": metric, metric: value, "threshold": cfg.gate_threshold,
                            "passed": value > cfg.gate_threshold, "enforced": cfg.gate_enforce,
                            "init_from": cfg.init_from}
                    with open(gate_path, "w") as f:
                        json.dump(gate, f, indent=1)
                    logger.log({f"gate/{metric}": value, "gate/passed": int(gate["passed"])}, step=step)
                    gate_failed = cfg.gate_enforce and not gate["passed"]
                logger.log({"train/loss": a["loss_sum"] / max(1, a["loss_tokens"]),
                            "train/final_loss": a["final_loss_sum"] / max(1, a["final_n"]),
                            "train/final_accuracy": a["final_correct"] / max(1, a["final_n"]),
                            "train/first_query_accuracy": a["first_correct"] / max(1, a["first_n"]),
                            "train/end_accuracy": a["end_correct"] / max(1, a["end_n"]),
                            **({"train/after_special_accuracy": a["after_special_correct"] / max(1, a["after_special_n"]),
                                "train/other_accuracy": a["other_correct"] / max(1, a["other_n"]),
                                "train/after_special_share": a["after_special_n"] / max(1, a["final_n"]),
                                "train/supervised_fraction": a["loss_tokens"] / max(1, a["real_tokens"])}
                               if "after_special_n" in a else {}), **by_db, **exact,
                            "train/grad_norm": a["grad_norm"] / a["steps"], "train/clip_fraction": a["clipped"] / a["steps"],
                            "train/lr": lr_at(step - 1, cfg),
                            "train/tokens_per_s": a["real_tokens"] / dt, "train/padded_tokens_per_s": a["padded_tokens"] / dt,
                            "train/step_time": dt / a["steps"], "train/data_wait": data_wait / a["steps"]}, step=step)
                acc, t_last, data_wait = None, time.time(), 0.0
            if gate_failed:
                print(f"gate failed at step {step}: {gate['metric']} {gate[gate['metric']]:.3f}, "
                      f"not above {cfg.gate_threshold}; stopping without a checkpoint", flush=True)
                break
            stage_event = None
            if cur is not None and (step - cstate["stage_start"]) % cur.check_interval == 0 \
                    and step > cstate["stage_start"]:
                k = cstate["stage"]
                r = evaluate_set(model, check_data[cur.stages[k].check_set], device)
                value = r[cur.check_metric]
                stage_event = after_check(cur, cstate, step, value)
                logger.log({"curriculum/check": value, "curriculum/check_stage": k + 1}, step=step)
                t_last = time.time()
                if stage_event is not None:
                    write_log(cfg.run_dir, cur, cstate)
                    print(f"step {step}: stage {k + 1} ({cur.stages[k].name}) {stage_event}, "
                          f"{cur.check_metric} {value:.3f}", flush=True)
                if stage_event == "advanced":  # the next stage's stream, from its step 0
                    del it
                    loader = make_loader()
                    it = iter(loader)
            cur_done = cstate is not None and cstate["status"] != "running"
            if cfg.eval_interval > 0 and (step % cfg.eval_interval == 0 or step == cfg.steps or stage_event):
                logger.log(run_eval(), step=step)  # also at every stage change; norms even without sets
                t_last = time.time()
            done = step == cfg.steps or stop.requested or (stop_after is not None and step >= stop_after) or cur_done
            if done or (cfg.ckpt_interval and step % cfg.ckpt_interval == 0) or \
                    (cfg.keep_interval and step % cfg.keep_interval == 0):
                checkpoint(milestone=bool(cfg.keep_interval and step % cfg.keep_interval == 0) or step == cfg.steps
                           or cur_done)
            if cur_done:
                print(f"curriculum {cstate['status']} at step {step}", flush=True)
                break
            if stop.requested:
                print(f"{stop.requested}: checkpoint saved at step {step}, stopping", flush=True)
                break
            if stop_after is not None and step >= stop_after:
                break
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)
        if logger is not None:
            logger.finish()
    result = {"step": step, "losses": losses, "gate_failed": gate_failed,
              "curriculum": cstate["status"] if cstate is not None else None}
    if bench:
        skip = min(30, len(step_times) // 3)
        st, wt = step_times[skip:], waits[skip:]
        result["benchmark"] = {
            "num_params": model.num_params(), "steps_timed": len(st),
            "step_time_median": statistics.median(st), "step_time_mean": statistics.mean(st),
            "data_wait_median": statistics.median(wt),
            "max_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0}
        print(json.dumps(result["benchmark"]), flush=True)
    return result


# ---------------------------------------------------------------------------- command line
def _add_fields(parser: argparse.ArgumentParser, cls, seen: set):
    for f in dataclasses.fields(cls):
        if f.name in seen:
            raise ValueError(f"option {f.name} defined twice")
        seen.add(f.name)
        if f.type in (bool, "bool"):
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__, str)
            parser.add_argument(f"--{f.name}", type=typ, default=f.default)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m looped_lm.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    seen = set()
    _add_fields(p, TrainConfig, seen)
    _add_fields(p, ModelConfig, seen)
    a = vars(p.parse_args(argv))
    cfg = TrainConfig(**{f.name: a[f.name] for f in dataclasses.fields(TrainConfig)})
    model_cfg = ModelConfig(**{f.name: a[f.name] for f in dataclasses.fields(ModelConfig)})
    return cfg, model_cfg


def main(argv=None):
    cfg, model_cfg = parse_args(argv)
    if cfg.benchmark:
        cfg = dataclasses.replace(cfg, eval_interval=0)
    result = train(cfg, model_cfg)
    if result.get("gate_failed"):
        return GATE_FAILED
    if result.get("curriculum") == "failed":
        return CURRICULUM_FAILED


if __name__ == "__main__":
    sys.exit(main())
