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


def relative_change(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Per puzzle: ||new - old|| / ||old|| over all positions and dimensions, in float32."""
    new, old = new.to(torch.float32).flatten(1), old.to(torch.float32).flatten(1)
    return torch.linalg.vector_norm(new - old, dim=1) / torch.linalg.vector_norm(old, dim=1)


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
      values of every step, whatever the halting settings;
    - halt_step (if halting): [B] int32.
    """
    return_keys = set(return_keys)
    if capture_steps:
        return_keys.update({"preds", "q_halt_logits", AUX_HALT_LOGITS_KEY})
    frozen = None
    if halt_head is not None:
        return_keys.update({"logits", "preds", "q_halt_logits", AUX_HALT_LOGITS_KEY})
        frozen = FrozenOutputs(halt_threshold)

    inference_steps = 0
    step_records: Dict[str, List[torch.Tensor]] = {}
    prev_preds, prev_z = None, {}
    while True:
        carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=return_keys)
        inference_steps += 1

        if capture_steps:
            step_records.setdefault("q_halt_logits", []).append(preds["q_halt_logits"])
            for k, v in preds.items():
                if k.startswith(AUX_HALT_LOGITS_KEY + "/"):
                    step_records.setdefault(k, []).append(v)
            step_records.setdefault("seq_is_correct", []).append(compute_correctness(preds["preds"], batch["labels"])[3])

            # Changes with respect to the previous step (step 1 has none: False / NaN)
            if prev_preds is None:
                changed = torch.zeros(preds["preds"].shape[0], dtype=torch.bool, device=preds["preds"].device)
            else:
                changed = (preds["preds"] != prev_preds).flatten(1).any(dim=1)
            step_records.setdefault("preds_changed", []).append(changed)
            prev_preds = preds["preds"]

            inner_carry = getattr(carry, "inner_carry", None)
            for name in ("z_H", "z_L"):
                z = getattr(inner_carry, name, None)
                if z is None:
                    continue
                if name in prev_z:
                    change = relative_change(z, prev_z[name])
                else:
                    change = torch.full((z.shape[0], ), float("nan"), dtype=torch.float32, device=z.device)
                step_records.setdefault(f"{name}_rel_change", []).append(change)
                prev_z[name] = z

        if frozen is not None:
            key = halt_logit_key(halt_head)  # type: ignore
            if key not in preds:
                available = [Q_HEAD] + [k[len(AUX_HALT_LOGITS_KEY) + 1:] for k in preds if k.startswith(AUX_HALT_LOGITS_KEY + "/")]
                raise ValueError(f"eval_halt_head='{halt_head}' is not a halting head of this model. Available: {available}")
            frozen.update(inference_steps, preds[key], preds, is_last=bool(all_finish))

        if all_finish:
            break

    if expected_steps is not None and inference_steps != expected_steps:
        raise RuntimeError(f"Evaluation ran {inference_steps} steps, but eval_halt_max_steps={expected_steps} was requested: "
                           "this model does not support eval_halt_max_steps")

    result: Dict[str, Any] = {"metrics": metrics, "preds": preds, "inference_steps": inference_steps}
    if capture_steps:
        result["step_records"] = step_records
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
