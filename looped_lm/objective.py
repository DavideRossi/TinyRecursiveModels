"""Training objective: next-token cross-entropy on the positions where loss_mask is true.

The logits at position t - 1 predict the token at t (dataset README). The head is applied only at
the predicting positions of loss tokens (a few per sample), not at every position.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .dataset_generation.vocab import PAD

ROLE_FINAL, ROLE_END = 1, 5
ANSWER_ROLES = (1, 2, 3, 4)  # final, intermediate, bracket, backtrack: the answer, without its end token
MAX_DEPTH, MAX_BRANCHING = 32, 8  # bins of the first-query counts by d and b (larger values go in the last)
DEVICE_KEYS = ("tokens", "loss_mask", "role", "pred_pos", "depth", "branching", "query_index", "after_special")


def to_device(batch: dict, device) -> dict:
    out = dict(batch)
    for k in DEVICE_KEYS:
        if k in batch:
            out[k] = batch[k].to(device, non_blocking=True)
    return out


def pad_to_multiple(batch: dict, multiple: int) -> dict:
    """Right-pad the [B, L] tensors to a length multiple of `multiple` (fewer shapes for torch.compile).
    Padding comes after every real token, so with causal attention it changes nothing."""
    L = batch["tokens"].shape[1]
    extra = (-L) % multiple if multiple > 1 else 0
    if not extra:
        return batch
    fill = {"tokens": PAD, "loss_mask": False, "query_index": -1, "depth": 0, "branching": 0, "role": 0,
            "pred_pos": -1, "after_special": False}
    out = dict(batch)
    for k, v in fill.items():
        if k in batch:
            t = batch[k]
            out[k] = torch.cat([t, torch.full((t.shape[0], extra), v, dtype=t.dtype, device=t.device)], dim=1)
    return out


def loss_positions(loss_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(batch index, target position) of every loss token; its prediction comes from position - 1."""
    m = loss_mask.clone()
    m[:, 0] = False
    return m.nonzero(as_tuple=True)


def query_exact(batch: dict, b: torch.Tensor, t: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """For each scored token (b, t): whether every answer token of its query (roles 1-4) is correct.
    Meaningful at the tokens of a query (query_index >= 0); no host synchronization."""
    qi = batch["query_index"][b, t]
    answer = torch.zeros_like(correct)
    for r in ANSWER_ROLES:
        answer |= batch["role"][b, t] == r
    answer &= qi >= 0
    key = b * batch["tokens"].shape[1] + qi.clamp(min=0)  # one bin per (row, query); query index < length
    wrong = torch.zeros(batch["tokens"].numel(), dtype=torch.long, device=correct.device)
    wrong = wrong.index_add(0, key, (answer & ~correct).long())
    return wrong[key] == 0


def exact_match_counts(batch: dict, b, t, correct, final) -> Dict[str, torch.Tensor]:
    """Exact-match counts (see loss_and_stats), one per query: at its final answer token."""
    q = final & (batch["query_index"][b, t] >= 0)
    first = q & (batch["query_index"][b, t] == 0)
    ok = query_exact(batch, b, t, correct)
    d = batch["depth"][b, t].clamp(0, MAX_DEPTH).long()
    bins = torch.zeros(MAX_DEPTH + 1, dtype=torch.long, device=d.device)
    return {"exact_n": q.sum(), "exact_correct": (q & ok).sum(), "first_exact_correct": (first & ok).sum(),
            "exact_n_by_depth": bins.index_add(0, d, q.long()),
            "exact_correct_by_depth": bins.index_add(0, d, (q & ok).long()),
            "first_exact_correct_by_depth": bins.index_add(0, d, (first & ok).long())}


def loss_and_stats(model, batch: dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mean cross-entropy over the loss tokens of the batch, and device-side counts for logging:
    final answers (role 1), the final answers of each sample's first query (query_index 0: no
    earlier answer in the context, so no elimination), and end tokens (role 5), each with the
    number correct (argmax). The first queries are also counted by the query's d and b
    (first_n_by_depth[d], first_correct_by_depth[d], and _by_branching; bin 0 holds the tasks
    without d and b: MQAR, induction). Exact match: a query is right when every token of its answer
    (roles 1-4, not the end token) is right under teacher forcing, which is exactly when greedy
    decoding reproduces the answer; in the direct format it is the final answer's accuracy.
    Counted over all queries (exact_n, exact_correct) and first queries (first_exact_correct;
    their number is first_n), and both by depth. Batches with an after_special field (the induction diagnostic) also
    count the final answers that follow a special token, and the other final answers."""
    tokens = batch["tokens"]
    hidden = model(tokens)
    b, t = loss_positions(batch["loss_mask"])
    logits = model.head(hidden[b, t - 1]).float()
    target = tokens[b, t]
    ce = F.cross_entropy(logits, target, reduction="none")
    correct = logits.argmax(-1) == target
    role = batch["role"][b, t]
    final, end = role == ROLE_FINAL, role == ROLE_END
    first = final & (batch["query_index"][b, t] == 0)
    stats = {"loss_sum": ce.detach().sum(), "loss_tokens": torch.tensor(len(ce), device=ce.device),
             "final_n": final.sum(), "final_correct": (correct & final).sum(),
             "final_loss_sum": (ce.detach() * final).sum(),
             "first_n": first.sum(), "first_correct": (correct & first).sum(),
             "end_n": end.sum(), "end_correct": (correct & end).sum()}
    for key, top in (("depth", MAX_DEPTH), ("branching", MAX_BRANCHING)):
        v = batch[key][b, t].clamp(0, top).long()
        bins = torch.zeros(top + 1, dtype=torch.long, device=v.device)  # fixed size: no host sync (bincount has one)
        stats[f"first_n_by_{key}"] = bins.index_add(0, v, first.long())
        stats[f"first_correct_by_{key}"] = bins.index_add(0, v, (first & correct).long())
    stats.update(exact_match_counts(batch, b, t, correct, final))
    if "after_special" in batch:
        after = final & batch["after_special"][b, t]
        other = final & ~after
        stats.update({"after_special_n": after.sum(), "after_special_correct": (correct & after).sum(),
                      "other_n": other.sum(), "other_correct": (correct & other).sum()})
    return ce.mean(), stats
