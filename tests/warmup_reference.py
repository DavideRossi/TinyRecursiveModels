"""Reference for the p = 0 test of the supervised window (tests/test_warmup.py).

Fingerprints (SHA-256 of the raw bytes) of every quantity of a short CPU training run, written
once from the code before the warm-up option existed:
    python tests/warmup_reference.py write
The test recomputes them with the current code and requires them to be identical. Bit equality
holds for the same torch version only, so the test is skipped on another version.
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_aux_halt_heads import AUX, BATCH, make_batch, make_model, make_optimizers  # noqa: E402

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warmup_reference.json")
CASES = {"q_halt_only": dict(no_ACT_continue=True), "with_continue": dict(no_ACT_continue=False)}
NUM_STEPS = 8


def digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() + str(t.dtype).encode()).hexdigest()


def fingerprints(**overrides) -> dict:
    model = make_model(AUX, **overrides)
    model.train()
    optimizers = make_optimizers(model)
    batch = make_batch()
    torch.manual_seed(7)
    carry = model.initial_carry(batch)
    out = {}
    for i in range(NUM_STEPS):
        carry, loss, metrics, outputs, _ = model(carry=carry, batch=batch, return_keys=["q_halt_logits", "aux_halt_logits"])
        ((1 / BATCH) * loss).backward()
        rec = {"loss": digest(loss), "halted": digest(carry.halted), "steps": digest(carry.steps),
               "z_H": digest(carry.inner_carry.z_H)}
        rec.update({f"metric/{k}": digest(v) for k, v in metrics.items()})
        rec.update({f"output/{k}": digest(v) for k, v in outputs.items()})
        rec.update({f"grad/{n}": digest(p.grad) for n, p in model.named_parameters() if p.grad is not None})
        rec["grad/puzzle_emb.local_weights"] = digest(model.model.puzzle_emb.local_weights.grad)
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        model.model.puzzle_emb.local_weights.grad = None
        rec.update({f"param/{n}": digest(p) for n, p in model.named_parameters()})
        out[str(i)] = rec
    out["rng_state"] = digest(torch.get_rng_state())
    return out


def compute() -> dict:
    return {"torch": torch.__version__, "cases": {name: fingerprints(**o) for name, o in CASES.items()}}


if __name__ == "__main__":
    assert sys.argv[1:] == ["write"], "usage: python tests/warmup_reference.py write"
    with open(PATH, "w") as f:
        json.dump(compute(), f, indent=1, sort_keys=True)
    print("wrote", PATH)
