"""References for the "behavior unchanged" tests of the supervised window (tests/test_warmup.py).

Fingerprints (SHA-256 of the raw bytes) of every quantity of a short CPU training run:
- off: warm-up off, written from the code before the warm-up option existed;
- on: warm-up on (p = 0.5, s in 1-6), written from the code before the release option existed.
    python tests/warmup_reference.py write off|on
The tests recompute them with the current code and require them to be identical. Bit equality
holds for the same torch version only, so the tests are skipped on another version.
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_aux_halt_heads import AUX, BATCH, make_batch, make_model, make_optimizers  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {"q_halt_only": dict(no_ACT_continue=True), "with_continue": dict(no_ACT_continue=False)}
# name -> (file, overrides of every case, training steps)
REFERENCES = {
    "off": (os.path.join(HERE, "warmup_reference.json"), {}, 8),
    "on": (os.path.join(HERE, "warmup_on_reference.json"),
           dict(halt_warmup_prob=0.5, halt_warmup_min=1, halt_warmup_max=6), 16),
}
PATH = REFERENCES["off"][0]


def digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().reshape(-1).contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() + str(t.dtype).encode()).hexdigest()


def fingerprints(num_steps=8, **overrides) -> dict:
    model = make_model(AUX, **overrides)
    model.train()
    optimizers = make_optimizers(model)
    batch = make_batch()
    torch.manual_seed(7)
    carry = model.initial_carry(batch)
    out = {}
    for i in range(num_steps):
        carry, loss, metrics, outputs, _ = model(carry=carry, batch=batch, return_keys=["q_halt_logits", "aux_halt_logits"])
        ((1 / BATCH) * loss).backward()
        rec = {"loss": digest(loss), "halted": digest(carry.halted), "steps": digest(carry.steps),
               "z_H": digest(carry.inner_carry.z_H)}
        if overrides.get("halt_warmup_prob", 0) > 0:
            rec["warmup"] = digest(carry.warmup)
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


def compute(reference: str = "off", **extra) -> dict:
    _, overrides, num_steps = REFERENCES[reference]
    return {"torch": torch.__version__,
            "cases": {name: fingerprints(num_steps, **o, **overrides, **extra) for name, o in CASES.items()}}


def compare(reference: str, allowed_new_keys=(), **extra) -> None:
    """Recompute a reference with the current code (plus `extra` overrides) and assert equality."""
    path = REFERENCES[reference][0]
    with open(path) as f:
        ref = json.load(f)
    cur = compute(reference, **extra)
    for case, steps in ref["cases"].items():
        assert cur["cases"][case]["rng_state"] == steps["rng_state"], f"{case}: different random numbers consumed"
        for i, rec in steps.items():
            if i == "rng_state":
                continue
            now = cur["cases"][case][i]
            assert set(now) - set(rec) <= set(allowed_new_keys), f"{case}: unexpected new keys {set(now) - set(rec)}"
            diff = [k for k, v in rec.items() if now.get(k) != v]
            assert not diff, f"{case}, step {i}: differs for {diff}"


def same_torch(reference: str) -> bool:
    with open(REFERENCES[reference][0]) as f:
        return json.load(f)["torch"] == torch.__version__


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "write" and sys.argv[2] in REFERENCES, \
        "usage: python tests/warmup_reference.py write off|on"
    path = REFERENCES[sys.argv[2]][0]
    with open(path, "w") as f:
        json.dump(compute(sys.argv[2]), f, indent=1, sort_keys=True)
    print("wrote", path)
