"""Training until a target number of supervised sequence-steps (pretrain.py, train_supervised_target).

A supervised sequence-step is one sequence receiving a loss in one training step: every slot of
the batch without warm-up, only the sequences in their supervised window with it
(arch.halt_warmup_prob).

Learning-rate schedule: the usual linear warm-up over the first `lr_warmup_steps` optimizer steps,
then the cosine, whose progress is the expected share of training done. The expected total number
of optimizer steps is re-estimated at every step from the supervised share measured since the
end of the learning-rate warm-up:

    T_hat(t) = W + (t - W) * (target - c_W) / (c_t - c_W)

with W the warm-up length and c_t the cumulative count after step t. The cosine's progress
(t - W) / (T_hat(t) - W) is then (c_t - c_W) / (target - c_W): it is 0 at step W and reaches 1
exactly at the step that reaches the target, whatever the supervised share does on the way.
"""
import math
from typing import Optional


def lr_at(base_lr: float, step: int, warmup_steps: int, progress: Optional[float], min_ratio: float,
          num_cycles: float = 0.5) -> float:
    """Linear warm-up, then the cosine at `progress` (clamped to [0, 1]); as
    pretrain.cosine_schedule_with_warmup_lr_lambda."""
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


class SupervisedBudget:
    """Cumulative supervised sequence-steps, the stopping rule and the cosine's progress.

    Call record(step, count) once per optimizer step, before computing that step's learning rate,
    with the step's global count (summed over ranks)."""

    def __init__(self, target: int, warmup_steps: int):
        if target <= 0:
            raise ValueError(f"train_supervised_target must be positive, got {target}")
        self.target = int(target)
        self.warmup_steps = int(warmup_steps)
        self.count = 0
        self.step = 0
        # Count at the end of the learning-rate warm-up (step W): the cosine starts there
        self.count_at_warmup_end: Optional[int] = 0 if self.warmup_steps <= 0 else None

    def record(self, step: int, count: int) -> None:
        if step != self.step + 1:
            raise ValueError(f"steps must be recorded in order: got {step} after {self.step}")
        self.step = step
        self.count += int(count)
        if step == self.warmup_steps:
            self.count_at_warmup_end = self.count

    @property
    def reached(self) -> bool:
        return self.count >= self.target

    @property
    def fraction(self) -> float:
        return self.count / self.target

    def progress(self) -> float:
        """The cosine's progress at the last recorded step (only used from step W on)."""
        if self.count_at_warmup_end is None:
            return 0.0
        remaining = self.target - self.count_at_warmup_end
        if remaining <= 0:
            return 1.0  # target reached during the warm-up
        return min(1.0, (self.count - self.count_at_warmup_end) / remaining)

    def expected_total_steps(self) -> Optional[float]:
        """Current estimate of the optimizer steps until the target (None before any measurement)."""
        if self.reached:
            return float(self.step)
        if self.count_at_warmup_end is None or self.step <= self.warmup_steps or self.count == self.count_at_warmup_end:
            return None
        per_step = (self.count - self.count_at_warmup_end) / (self.step - self.warmup_steps)
        return self.step + (self.target - self.count) / per_step

    def lr(self, base_lr: float, min_ratio: float) -> float:
        return lr_at(base_lr, self.step, self.warmup_steps, self.progress(), min_ratio)
