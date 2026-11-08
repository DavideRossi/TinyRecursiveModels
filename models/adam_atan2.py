"""Pure-PyTorch AdamATan2 optimizer.

Drop-in replacement for the `adam-atan2` package (v0.0.3), which ships a
custom CUDA kernel that must be compiled against the installed PyTorch.
The math is identical to that kernel:

    p      <- p * (1 - lr * weight_decay)
    m      <- lerp(m, g,   1 - beta1)
    v      <- lerp(v, g^2, 1 - beta2)
    p      <- p - lr / (1 - beta1^t) * atan2(m, sqrt(v) / sqrt(1 - beta2^t))

i.e. AdamW where `m / (sqrt(v) + eps)` is replaced by `atan2(m, sqrt(v))`,
which removes the epsilon hyperparameter (https://arxiv.org/abs/2407.05872).
Math runs in float32 for half-precision parameters, as in the kernel.
"""
from typing import Tuple

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class AdamATan2(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd_alpha = 1 - lr * group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    # Step kept on-device (as in the original) so no host sync is needed.
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                step_t: torch.Tensor = state["step"]
                step_t.add_(1)

                # Compute in float32 (or float64) like the kernel's opmath type.
                math_dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                param = p.to(math_dtype)
                grad = p.grad.to(math_dtype)
                exp_avg = state["exp_avg"].to(math_dtype)
                exp_avg_sq = state["exp_avg_sq"].to(math_dtype)

                param.mul_(wd_alpha)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad * grad, 1 - beta2)

                step_f = step_t.to(math_dtype)
                bias_correction1 = 1 - beta1 ** step_f
                bias_correction2_sqrt = (1 - beta2 ** step_f).sqrt()

                denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
                param.sub_(torch.atan2(exp_avg, denom) * (lr / bias_correction1))

                # Write back (no-ops when already in math dtype, since .to() returned the same tensor).
                if param is not p:
                    p.copy_(param)
                if exp_avg is not state["exp_avg"]:
                    state["exp_avg"].copy_(exp_avg)
                if exp_avg_sq is not state["exp_avg_sq"]:
                    state["exp_avg_sq"].copy_(exp_avg_sq)

        return loss
