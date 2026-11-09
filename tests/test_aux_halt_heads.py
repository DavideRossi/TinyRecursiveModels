"""Isolation tests for auxiliary halting heads (CPU, eager mode).

Run with `python -m pytest tests/` or `python tests/test_aux_halt_heads.py`.

The central check: a model with an auxiliary head and a model without one, built and
driven with the same seed, must produce bit-identical backbone / embedding / q_head
gradients, parameters, halting decisions and outputs over several ACT steps with
optimizer updates, while the auxiliary head itself does get trained.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1  # noqa: E402
from models.losses import ACTLossHead, IGNORE_LABEL_ID, compute_correctness  # noqa: E402
from models.adam_atan2 import AdamATan2  # noqa: E402
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed  # noqa: E402

BATCH, SEQ_LEN, VOCAB, NUM_IDS = 6, 81, 11, 3
AUX = [{"type": "detached_linear"}]


def make_config(aux_heads=(), **overrides):
    cfg = dict(
        batch_size=BATCH, seq_len=SEQ_LEN, vocab_size=VOCAB, num_puzzle_identifiers=NUM_IDS,
        puzzle_emb_ndim=64, puzzle_emb_len=4,
        H_cycles=2, L_cycles=2, H_layers=0, L_layers=2,
        hidden_size=64, expansion=4, num_heads=2, pos_encodings="rope",
        halt_max_steps=4, halt_exploration_prob=0.5,  # exploration on: exercises the RNG
        forward_dtype="bfloat16", mlp_t=False, no_ACT_continue=True,
        aux_halt_heads=list(aux_heads),
    )
    cfg.update(overrides)
    return cfg


def make_model(aux_heads=(), seed=0, **overrides):
    torch.manual_seed(seed)
    model = ACTLossHead(TinyRecursiveReasoningModel_ACTV1(make_config(aux_heads, **overrides)), loss_type="stablemax_cross_entropy")
    return model


def make_batch(seed=123):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, VOCAB, (BATCH, SEQ_LEN), generator=g, dtype=torch.int32)
    inputs = torch.where(torch.rand(BATCH, SEQ_LEN, generator=g) < 0.3, labels, torch.ones_like(labels))
    labels[0, :10] = IGNORE_LABEL_ID  # some ignored cells
    return {"inputs": inputs, "labels": labels,
            "puzzle_identifiers": torch.randint(0, NUM_IDS, (BATCH,), generator=g, dtype=torch.int32)}


def is_aux(name: str) -> bool:
    return ".aux_halt_heads." in f".{name}"


def make_optimizers(model):
    return [
        CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=1e-2, weight_decay=0.1, world_size=1),
        AdamATan2(model.parameters(), lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95)),
    ]


def run_training(model, num_steps=6, seed=7):
    """Mimics pretrain.train_batch; returns per-step records for comparison."""
    model.train()
    optimizers = make_optimizers(model)
    batch = make_batch()
    torch.manual_seed(seed)  # same RNG stream for ACT exploration in both runs
    carry = model.initial_carry(batch)
    records = []
    for _ in range(num_steps):
        carry, loss, metrics, outputs, _ = model(carry=carry, batch=batch, return_keys=["q_halt_logits", "aux_halt_logits"])
        ((1 / BATCH) * loss).backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        grads["puzzle_emb.local_weights"] = model.model.puzzle_emb.local_weights.grad.clone()
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        model.model.puzzle_emb.local_weights.grad = None
        records.append(dict(
            grads=grads,
            params={n: p.detach().clone() for n, p in model.named_parameters()},
            buffers={n: b.detach().clone() for n, b in model.named_buffers()},
            halted=carry.halted.clone(), steps=carry.steps.clone(),
            z_H=carry.inner_carry.z_H.clone(), outputs=outputs, metrics=metrics,
        ))
    return records


def assert_equal_except_aux(a: dict, b: dict, what: str):
    a_keys = {k for k in a if not is_aux(k)}
    assert a_keys == set(b), f"{what}: key mismatch {a_keys ^ set(b)}"
    for k in a_keys:
        assert torch.equal(a[k], b[k]), f"{what} differs for {k}"


def test_init_and_rng_unchanged():
    torch.manual_seed(0)
    TinyRecursiveReasoningModel_ACTV1(make_config())
    after_plain = torch.rand(8)
    torch.manual_seed(0)
    TinyRecursiveReasoningModel_ACTV1(make_config(AUX))
    after_aux = torch.rand(8)
    assert torch.equal(after_plain, after_aux), "creating auxiliary heads consumed random numbers"

    plain, aux = make_model(), make_model(AUX)
    assert_equal_except_aux(aux.state_dict(), plain.state_dict(), "initial state_dict")
    assert any(is_aux(k) for k in aux.state_dict())
    assert not any(is_aux(k) for k in plain.state_dict())

    head = aux.model.inner.aux_halt_heads["detached_linear"].linear
    q_head = aux.model.inner.q_head
    assert torch.equal(head.weight, torch.zeros_like(head.weight))
    assert torch.equal(head.bias, torch.full_like(head.bias, -5.0))
    assert torch.equal(head.weight[0], q_head.weight[0]) and torch.equal(head.bias[0], q_head.bias[0])


def _check_training_isolation(**overrides):
    plain = run_training(make_model(**overrides))
    aux = run_training(make_model(AUX, **overrides))

    for step, (p, a) in enumerate(zip(plain, aux)):
        # Backbone, embeddings and q_head: identical gradients, parameters and state
        assert_equal_except_aux(a["grads"], p["grads"], f"step {step} grads")
        assert_equal_except_aux(a["params"], p["params"], f"step {step} params")
        assert_equal_except_aux(a["buffers"], p["buffers"], f"step {step} buffers")
        # Halting and outputs of the original head are unchanged
        assert torch.equal(a["halted"], p["halted"]) and torch.equal(a["steps"], p["steps"]), f"step {step} halting differs"
        assert torch.equal(a["z_H"], p["z_H"])
        assert torch.equal(a["outputs"]["q_halt_logits"], p["outputs"]["q_halt_logits"])
        for k, v in p["metrics"].items():
            assert torch.equal(a["metrics"][k], v), f"step {step} metric {k} differs"
        # Auxiliary head: exposed at every step, has its own metrics, and is trained
        assert a["outputs"]["aux_halt_logits/detached_linear"].shape == (BATCH,)
        assert {"aux_detached_linear_q_halt_loss", "aux_detached_linear_q_halt_accuracy"} <= set(a["metrics"])
        aux_grads = [g for k, g in a["grads"].items() if is_aux(k)]
        assert len(aux_grads) == 2 and any(g.abs().sum() > 0 for g in aux_grads), f"step {step}: aux head got no gradient"

    head_params = {k: v for k, v in aux[-1]["params"].items() if is_aux(k)}
    assert not torch.equal(head_params["model.inner.aux_halt_heads.detached_linear.linear.bias"], torch.tensor([-5.0])), "aux head not updated"


def test_training_isolation_attention():
    _check_training_isolation()


def test_training_isolation_mlp_t():
    _check_training_isolation(mlp_t=True, pos_encodings="none")


def test_training_isolation_q_continue():
    # no_ACT_continue=False: halting compares q_halt vs q_continue and a second (no-grad) inner
    # pass computes the bootstrapped target_q_continue
    _check_training_isolation(no_ACT_continue=False)


def test_aux_loss_removed_gives_same_backbone_grads():
    """Same model, same forward graph: gradients of the total loss vs. the total loss with the
    auxiliary term subtracted. Backbone, embedding and q_head gradients must be bit-identical."""
    import torch.nn.functional as F

    model = make_model(AUX)
    model.train()
    batch = make_batch()
    captured = {}
    head = model.model.inner.aux_halt_heads["detached_linear"]
    # Non-zero head weights: with the zero init, d(logit)/d(input) = 0, so even an attached head
    # would send no gradient to the backbone and the test could not detect a missing detach.
    with torch.no_grad():
        head.linear.weight.copy_(torch.randn(head.linear.weight.shape, generator=torch.Generator().manual_seed(5)))
    hook = head.register_forward_hook(lambda mod, inp, out: captured.__setitem__("logit", out))

    torch.manual_seed(3)
    carry = model.initial_carry(batch)
    new_carry, loss, metrics, outputs, _ = model(carry=carry, batch=batch, return_keys=["preds"])
    hook.remove()

    # The exact auxiliary term that ACTLossHead added, rebuilt on the same graph
    seq_ok = compute_correctness(outputs["preds"], new_carry.current_data["labels"])[3]
    aux_loss = F.binary_cross_entropy_with_logits(captured["logit"], seq_ok.to(captured["logit"].dtype), reduction="sum")
    assert torch.equal(aux_loss.detach(), metrics["aux_detached_linear_q_halt_loss"])

    def grads_of(objective):
        model.zero_grad()
        model.model.puzzle_emb.local_weights.grad = None
        objective.backward(retain_graph=True)
        g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        g["puzzle_emb.local_weights"] = model.model.puzzle_emb.local_weights.grad.clone()
        return g

    with_aux = grads_of(loss)
    without_aux = grads_of(loss - 0.5 * aux_loss)

    backbone_keys = {k for k in with_aux if not is_aux(k)}
    assert "model.inner.q_head.weight" in backbone_keys and "puzzle_emb.local_weights" in backbone_keys
    assert backbone_keys == {k for k in without_aux if not is_aux(k)}
    for k in backbone_keys:
        assert torch.equal(with_aux[k], without_aux[k]), f"grad of {k} depends on the aux loss"
    # Sanity: the subtracted term really was the aux loss (its head gradient cancels)
    assert any(with_aux[k].abs().sum() > 0 for k in with_aux if is_aux(k))
    for k in without_aux:
        if is_aux(k):
            assert without_aux[k].abs().max() < 1e-6


def test_aux_loss_has_no_path_to_backbone():
    """Autograd check: d(aux loss)/d(every non-aux parameter) is None (no graph connection)."""
    model = make_model(AUX)
    model.train()
    batch = make_batch()
    carry = model.initial_carry(batch)
    new_carry, outputs = model.model(carry=carry, batch=batch)
    aux_logit = outputs["aux_halt_logits"]["detached_linear"]
    params = [p for n, p in model.named_parameters() if not is_aux(n)] + [model.model.puzzle_emb.local_weights]
    grads = torch.autograd.grad(aux_logit.sum(), params, allow_unused=True)
    assert all(g is None for g in grads), "auxiliary head output depends on model parameters"
    # and it does depend on its own parameters
    head_params = list(model.model.inner.aux_halt_heads.parameters())
    assert all(g is not None for g in torch.autograd.grad(aux_logit.sum(), head_params))


def test_eval_outputs_and_correctness():
    model = make_model(AUX)
    model.eval()
    batch = make_batch()
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        for _ in range(4):
            carry, _, metrics, outputs, all_finish = model(carry=carry, batch=batch, return_keys=["preds", "q_halt_logits", "aux_halt_logits"])
            assert "aux_halt_logits/detached_linear" in outputs and "q_halt_logits" in outputs
            assert "aux_detached_linear_q_halt_accuracy" in metrics
        assert all_finish  # eval always runs halt_max_steps
        seq_ok = compute_correctness(outputs["preds"], batch["labels"])[3]
        assert int((seq_ok & carry.halted).sum()) == int(metrics["exact_accuracy"])


def test_disabled_outputs_unchanged():
    model = make_model()
    model.train()
    batch = make_batch()
    carry = model.initial_carry(batch)
    _, outputs = model.model(carry=carry, batch=batch)
    assert set(outputs) == {"logits", "q_halt_logits", "q_continue_logits"}
    assert model.model.inner.aux_halt_heads is None


def test_load_state_dict_without_aux_heads():
    plain = make_model(seed=1)
    aux = make_model(AUX, seed=2)
    fresh_head = {k: v.clone() for k, v in aux.state_dict().items() if is_aux(k)}

    # Same logic as pretrain.load_checkpoint
    state_dict = dict(plain.state_dict())
    model_state = aux.state_dict()
    for k in model_state:
        if is_aux(k) and k not in state_dict:
            state_dict[k] = model_state[k]
    aux.load_state_dict(state_dict, assign=True)

    assert_equal_except_aux(aux.state_dict(), plain.state_dict(), "loaded state_dict")
    for k, v in fresh_head.items():
        assert torch.equal(aux.state_dict()[k], v)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
