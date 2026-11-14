"""Evaluation helpers: per-puzzle halting as a frozen mask, and the per-batch evaluation loop.

Halting rule: a puzzle halts at the first step whose halt logit (from the chosen head) is
>= the threshold, or at the last step. From then on its reported outputs and its step count
are frozen. The model keeps running the whole batch: nothing is removed from it, and the ACT
wrapper's own halting (which resets a halted sequence) is not used.
"""
from typing import Any, Dict, List, Optional, Set

import torch

from models.losses import AUX_HALT_LOGITS_KEY, act_eval_metrics, compute_correctness

Q_HEAD = "q_head"


def halt_logit_key(head: str) -> str:
    """Name of the output holding the halt logit of `head` ("q_head" or an auxiliary head name)."""
    return "q_halt_logits" if head == Q_HEAD else f"{AUX_HALT_LOGITS_KEY}/{head}"


class FrozenOutputs:
    """Keeps, per puzzle, the outputs of its halting step.

    Call `update` once per step, in order. After the step with `is_last=True`, every puzzle has
    halted; `outputs` holds each puzzle's outputs at its halting step and `halt_step` (int32,
    counted from 1) the step itself.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.halted: Optional[torch.Tensor] = None
        self.halt_step: Optional[torch.Tensor] = None
        self.outputs: Dict[str, torch.Tensor] = {}

    def update(self, step: int, halt_logit: torch.Tensor, outputs: Dict[str, torch.Tensor], is_last: bool) -> torch.Tensor:
        """Returns the mask of puzzles that halt at this step."""
        if self.halted is None:
            self.halted = torch.zeros_like(halt_logit, dtype=torch.bool)
            self.halt_step = torch.zeros_like(halt_logit, dtype=torch.int32)

        # Compare in the logit's own dtype, so the rule is the same wherever it is replayed
        threshold = torch.tensor(self.threshold, dtype=halt_logit.dtype, device=halt_logit.device)
        newly = ~self.halted & (halt_logit >= threshold)
        if is_last:
            newly = ~self.halted

        for k, v in outputs.items():
            if k not in self.outputs:
                self.outputs[k] = v  # placeholder for puzzles not halted yet; replaced when they halt
            else:
                self.outputs[k] = torch.where(newly.view((-1, ) + (1, ) * (v.ndim - 1)), v, self.outputs[k])
        self.halt_step = torch.where(newly, step, self.halt_step).to(torch.int32)
        self.halted = self.halted | newly
        return newly


GRID_HASH_SEED = 0x5EED0F6B1D5  # constant: grid hashes are comparable across runs
_MASK64 = (1 << 64) - 1
_grid_hash_coefficients: Dict[Any, torch.Tensor] = {}


def grid_hash_coefficients(length: int) -> List[int]:
    """One fixed odd 64-bit integer per grid position (as signed int64 values).

    splitmix64 of (GRID_HASH_SEED + position) in plain integer arithmetic: the values depend on
    no library's random generator, and no generator state is read or changed.
    """
    out = []
    for i in range(length):
        z = (GRID_HASH_SEED + (i + 1) * 0x9E3779B97F4A7C15) & _MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z = (z ^ (z >> 31)) | 1
        out.append(z - (1 << 64) if z >= (1 << 63) else z)
    return out


def grid_hash(preds: torch.Tensor) -> torch.Tensor:
    """64-bit hash (int64) of each predicted grid: sum over positions of (token + 1) * coefficient,
    with wraparound. Two grids differing in one position always get different hashes."""
    tokens = preds.flatten(1).to(torch.int64)
    key = (tokens.shape[1], tokens.device)
    if key not in _grid_hash_coefficients:
        _grid_hash_coefficients[key] = torch.tensor(grid_hash_coefficients(tokens.shape[1]), dtype=torch.int64, device=tokens.device)
    return ((tokens + 1) * _grid_hash_coefficients[key]).sum(dim=1)


def cell_count_dtype(num_cells: int) -> torch.dtype:
    """Smallest integer dtype holding counts from 0 to num_cells."""
    if num_cells <= 255:
        return torch.uint8
    return torch.int16 if num_cells <= 32767 else torch.int32


def relative_change(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Per puzzle: ||new - old|| / ||old|| over all positions and dimensions, in float32."""
    new, old = new.to(torch.float32).flatten(1), old.to(torch.float32).flatten(1)
    return torch.linalg.vector_norm(new - old, dim=1) / torch.linalg.vector_norm(old, dim=1)


def halt_logit(preds: Dict[str, torch.Tensor], head: str) -> torch.Tensor:
    """The halt logit of `head` from the model outputs, with a clear error for unknown heads."""
    key = halt_logit_key(head)
    if key not in preds:
        available = [Q_HEAD] + [k[len(AUX_HALT_LOGITS_KEY) + 1:] for k in preds if k.startswith(AUX_HALT_LOGITS_KEY + "/")]
        raise ValueError(f"eval_halt_head='{head}' is not a halting head of this model. Available: {available}")
    return preds[key]


CAPTURE_RETURN_KEYS = {"preds", "q_halt_logits", AUX_HALT_LOGITS_KEY}


class StepRecorder:
    """Per-step capture records of one batch: call `record` after every model step, in order.

    step_records maps each quantity to a list over steps of [B] tensors. Compact dtypes: bool
    flags, float16 relative state changes, int64 grid hashes (pred_hash), float32 halt logits,
    cell counts (wrong_cells, cells_changed) in the smallest integer dtype for the sequence length.

    Changes are relative to the previous recorded step. To continue a run that was not recorded
    (e.g. from a branch point), pass the predictions and the carry of the step before the first
    recorded one as prev_preds / prev_carry; otherwise the first step has no change (False / 0 / NaN).
    """

    def __init__(self, prev_preds: Optional[torch.Tensor] = None, prev_carry: Any = None):
        self.step_records: Dict[str, List[torch.Tensor]] = {}
        self.prev_preds = prev_preds
        inner = getattr(prev_carry, "inner_carry", None)
        self.prev_z = {name: getattr(inner, name) for name in ("z_H", "z_L") if getattr(inner, name, None) is not None}

    def _add(self, key: str, value: torch.Tensor):
        self.step_records.setdefault(key, []).append(value)

    def record(self, preds: Dict[str, torch.Tensor], carry: Any, labels: torch.Tensor):
        self._add("q_halt_logits", preds["q_halt_logits"])
        for k, v in preds.items():
            if k.startswith(AUX_HALT_LOGITS_KEY + "/"):
                self._add(k, v)
        _, loss_counts, is_correct, seq_is_correct = compute_correctness(preds["preds"], labels)
        self._add("seq_is_correct", seq_is_correct)
        # Same mask and definition as seq_is_correct: 0 wrong cells <=> seq_is_correct
        count_dtype = cell_count_dtype(preds["preds"].flatten(1).shape[1])
        self._add("wrong_cells", (loss_counts - is_correct.sum(-1)).to(count_dtype))

        if self.prev_preds is None:
            cells_changed = torch.zeros(preds["preds"].shape[0], dtype=count_dtype, device=preds["preds"].device)
        else:
            cells_changed = (preds["preds"] != self.prev_preds).flatten(1).sum(dim=1).to(count_dtype)
        self._add("cells_changed", cells_changed)
        self._add("preds_changed", cells_changed > 0)
        self.prev_preds = preds["preds"]
        self._add("pred_hash", grid_hash(preds["preds"]))

        inner_carry = getattr(carry, "inner_carry", None)
        for name in ("z_H", "z_L"):
            z = getattr(inner_carry, name, None)
            if z is None:
                continue
            if name in self.prev_z:
                change = relative_change(z, self.prev_z[name])
            else:
                change = torch.full((z.shape[0], ), float("nan"), dtype=torch.float32, device=z.device)
            self._add(f"{name}_rel_change", change.to(torch.float16))
            self.prev_z[name] = z


def run_eval_batch(
    model: Any,
    carry: Any,
    batch: Dict[str, torch.Tensor],
    return_keys: Set[str],
    capture_steps: bool = False,
    halt_head: Optional[str] = None,
    halt_threshold: float = 0.0,
    expected_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one evaluation batch to the model's step limit.

    model: the loss head (e.g. ACTLossHead), in eval mode. `halt_head=None` disables halting:
    outputs and metrics are then those the model reports at the last step. With a head name,
    outputs are frozen per puzzle (see FrozenOutputs) and metrics are computed from them.

    Returns a dict with: metrics, preds (the reported outputs), inference_steps, and
    - step_records (if capture_steps): quantity -> list over steps of [B] tensors with the RAW
      values of every step, whatever the halting settings (see StepRecorder);
    - halt_step (if halting): [B] int32.
    """
    return_keys = set(return_keys)
    if capture_steps:
        return_keys.update(CAPTURE_RETURN_KEYS)
    frozen = None
    if halt_head is not None:
        return_keys.update({"logits", "preds", "q_halt_logits", AUX_HALT_LOGITS_KEY})
        frozen = FrozenOutputs(halt_threshold)

    inference_steps = 0
    recorder = StepRecorder() if capture_steps else None
    while True:
        carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=return_keys)
        inference_steps += 1

        if recorder is not None:
            recorder.record(preds, carry, batch["labels"])

        if frozen is not None:
            frozen.update(inference_steps, halt_logit(preds, halt_head), preds, is_last=bool(all_finish))

        if all_finish:
            break

    if expected_steps is not None and inference_steps != expected_steps:
        raise RuntimeError(f"Evaluation ran {inference_steps} steps, but eval_halt_max_steps={expected_steps} was requested: "
                           "this model does not support eval_halt_max_steps")

    result: Dict[str, Any] = {"metrics": metrics, "preds": preds, "inference_steps": inference_steps}
    if recorder is not None:
        result["step_records"] = recorder.step_records
    if frozen is not None:
        out = frozen.outputs
        result["preds"] = out
        result["halt_step"] = frozen.halt_step
        result["metrics"] = act_eval_metrics(
            loss_fn=model.loss_fn,
            logits=out["logits"], preds=out["preds"], q_halt_logits=out["q_halt_logits"],
            aux_halt_logits={k[len(AUX_HALT_LOGITS_KEY) + 1:]: v for k, v in out.items() if k.startswith(AUX_HALT_LOGITS_KEY + "/")},
            labels=batch["labels"], steps=frozen.halt_step, max_steps=inference_steps,
        )
    return result
