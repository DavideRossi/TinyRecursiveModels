"""Teacher-forced evaluation of final answers on stored sets (used by evaluate.py and by the
periodic evaluation in train.py).

A final answer is a token with role 1; its prediction is the argmax of the logits at its pred_pos
(the position before it). One forward pass per batch scores every query of every sample.
"""
import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .dataset_generation.dataset import collate, to_item
from .dataset_generation.storage import StoredSet
from .dataset_generation.vocab import OPEN
from .objective import ANSWER_ROLES, MAX_DEPTH, ROLE_FINAL, query_exact, to_device

ROLE_INTERMEDIATE = 2


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    """95% Wilson score interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def score_batch(model, batch: dict) -> Dict[str, np.ndarray]:
    """Per final answer: correct (argmax at pred_pos == answer token), its cross-entropy, exact
    (every answer token of its query correct, objective.query_exact), the query's d and b, its index
    among the sample's queries, and the sample's row in the batch. Answer tokens other than the
    final one are predicted at their position - 1."""
    tokens, role = batch["tokens"], batch["role"]
    answer = role == ROLE_FINAL
    for r in ANSWER_ROLES:
        answer |= (role == r) & (batch["query_index"] >= 0)
    hidden = model(tokens)
    b, t = answer.nonzero(as_tuple=True)
    final = role[b, t] == ROLE_FINAL
    p = torch.where(final, batch["pred_pos"][b, t], t - 1)
    logits = model.head(hidden[b, p]).float()
    target = tokens[b, t]
    pred = logits.argmax(-1)
    correct = pred == target
    exact = query_exact(batch, b, t, correct)
    level_ok, level_set = bracket_levels(b.cpu().numpy(), batch["query_index"][b, t].cpu().numpy(),
                                         role[b, t].cpu().numpy(), target.cpu().numpy(), pred.cpu().numpy(),
                                         correct.cpu().numpy())
    fin = final.cpu().numpy()
    level_ok, level_set = level_ok[fin], level_set[fin]
    b, t, logits, target, correct, exact = b[final], t[final], logits[final], target[final], correct[final], exact[final]
    return {"correct": correct.cpu().numpy(), "exact": exact.cpu().numpy(), "level_ok": level_ok, "level_set": level_set,
            "loss": F.cross_entropy(logits, target, reduction="none").cpu().numpy(),
            "depth": batch["depth"][b, t].cpu().numpy(), "branching": batch["branching"][b, t].cpu().numpy(),
            "query_index": batch["query_index"][b, t].cpu().numpy(), "row": b.cpu().numpy()}


def bracket_levels(rows, queries, roles, targets, preds, correct):
    """Per answer token (in row-major order, as nonzero returns them; the tokens of a query are
    contiguous), for answers with brackets (frontier): at the query's final token, row k - 1 of
    level_ok says whether every token of bracket level k ([, its nodes, ]) is right under teacher
    forcing, and of level_set whether its nodes' predictions are the right nodes as a multiset
    (right set, any order). -1 elsewhere and for answers without brackets. Level k is the k-th
    bracket of the answer (the nodes at distance k from the source)."""
    n = len(rows)
    level_ok = np.full((n, MAX_DEPTH), -1, dtype=np.int8)
    level_set = np.full((n, MAX_DEPTH), -1, dtype=np.int8)
    if n == 0:
        return level_ok, level_set
    starts = np.flatnonzero((np.diff(rows) != 0) | (np.diff(queries) != 0)) + 1
    for g in np.split(np.arange(n), starts):
        if queries[g[0]] < 0:
            continue
        lvl = np.cumsum(targets[g] == OPEN)
        fin = g[roles[g] == ROLE_FINAL]
        if lvl[-1] == 0 or len(fin) != 1:
            continue
        inner = roles[g] != ROLE_FINAL
        nodes = inner & (roles[g] == ROLE_INTERMEDIATE)
        for k in range(1, min(int(lvl[-1]), MAX_DEPTH) + 1):
            m = inner & (lvl == k)
            level_ok[fin[0], k - 1] = bool(correct[g][m].all())
            mk = nodes & (lvl == k)
            level_set[fin[0], k - 1] = bool(np.array_equal(np.sort(preds[g][mk]), np.sort(targets[g][mk])))
    return level_ok, level_set


def length_batches(lengths: List[int], max_tokens: int, max_batch: int = 256) -> List[List[int]]:
    """Indices grouped by length (sorted), each group with at most max_tokens padded tokens."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    groups, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * lengths[i] > max_tokens or len(cur) >= max_batch):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def load_set(path: str, num: Optional[int] = None, max_tokens: int = 131072) -> dict:
    """The first `num` samples (all if None) of a stored set, collated into length batches on the CPU,
    with each sample's number of targets."""
    st = StoredSet(path)
    n = len(st) if num is None else min(num, len(st))
    recs = [st.get(i) for i in range(n)]
    batches = []
    for g in length_batches([len(r["tokens"]) for r in recs], max_tokens):
        batch = collate([to_item(recs[i], i) for i in g])
        batch["num_targets"] = torch.tensor([recs[i]["num_targets"] for i in g])
        batches.append(batch)
    return {"name": path, "config": st.config, "batches": batches, "num_samples": n,
            "tokens_mean": float(np.mean([len(r["tokens"]) for r in recs]))}


@torch.no_grad()
def evaluate_set(model, loaded: dict, device, amp: bool = True, by_db: Optional[bool] = None) -> dict:
    """Accuracy of final answers with a 95% interval, number of answers, chance (mean 1 / #targets),
    mean loss on final answers; the same for each sample's first query only (first_query_*: no
    earlier answer in the context, so the elimination heuristic can't help); for sets with several
    queries per sample, also by the query's (d, b); for sets whose first queries have several
    depths (branchings), first-query results by depth (first_query_by_depth["d1"], ...) and by
    branching (first_query_by_branching["b1"], ...). Exact match (every answer token right, see
    objective.query_exact; the final answer's accuracy in the direct format): exact_match and
    first_query_exact_match, and for sets mixing depths, exact_match_by_depth["d1"] and
    first_query_exact_match_by_depth["d1"], ... (summaries like the others). For answers with
    brackets (frontier), first_query_levels["d<d>"] per depth of the first queries: n, level[k - 1]
    (bracket level k entirely right under teacher forcing), level_set[k - 1] (its nodes right as a
    set, in any order) and final (the final selection, given the correct brackets)."""
    was_training = model.training
    model.eval()
    cols = {k: [] for k in ("correct", "exact", "level_ok", "level_set", "loss", "depth", "branching", "query_index",
                            "chance")}
    for batch in loaded["batches"]:
        dev = to_device(batch, device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=amp and torch.device(device).type == "cuda"):
            s = score_batch(model, dev)
        for k in ("correct", "exact", "level_ok", "level_set", "loss", "depth", "branching", "query_index"):
            cols[k].append(s[k])
        cols["chance"].append(1.0 / batch["num_targets"].numpy()[s["row"]])
    model.train(was_training)
    c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}

    def summary(m, col="correct"):
        n, k = int(m.sum()), int(c[col][m].sum())
        lo, hi = wilson(k, n)
        return {"n": n, "correct": k, "accuracy": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi,
                "chance": float(c["chance"][m].mean()) if n else float("nan"),
                "loss": float(c["loss"][m].mean()) if n else float("nan")}
    out = summary(np.ones(len(c["correct"]), dtype=bool))
    out.update({f"first_query_{k}": v for k, v in summary(c["query_index"] == 0).items()})
    first = c["query_index"] == 0
    for key, col, prefix in (("first_query_by_depth", "depth", "d"), ("first_query_by_branching", "branching", "b")):
        values = sorted(set(c[col][first].tolist()))
        if len(values) > 1:
            out[key] = {f"{prefix}{int(v)}": summary(first & (c[col] == v)) for v in values}
    everything = np.ones(len(c["correct"]), dtype=bool)
    out["exact_match"] = summary(everything, "exact")["accuracy"]
    out["first_query_exact_match"] = summary(first, "exact")["accuracy"]
    depths = sorted(set(c["depth"].tolist()))
    if len(depths) > 1:
        for key, m in (("exact_match_by_depth", everything), ("first_query_exact_match_by_depth", first)):
            out[key] = {f"d{int(v)}": summary(m & (c["depth"] == v), "exact") for v in depths}
    lv, ls = c["level_ok"], c["level_set"]
    if lv.ndim == 2 and (lv >= 0).any():
        out["first_query_levels"] = {}
        for d in sorted(set(c["depth"][first].tolist())):
            m = first & (c["depth"] == d)
            levels = [k for k in range(MAX_DEPTH) if (lv[m, k] >= 0).any()]
            out["first_query_levels"][f"d{int(d)}"] = {
                "n": int(m.sum()), "final": float(c["correct"][m].mean()),
                "level": [float(lv[m, k][lv[m, k] >= 0].mean()) for k in levels],
                "level_set": [float(ls[m, k][ls[m, k] >= 0].mean()) for k in levels]}
    out["num_samples"] = loaded["num_samples"]
    out["tokens_mean"] = loaded["tokens_mean"]
    if by_db is None:
        by_db = loaded["config"].num_queries.max > 1
    if by_db:
        out["by_db"] = {f"d{d}_b{b}": dict(summary((c["depth"] == d) & (c["branching"] == b)), depth=int(d),
                                           branching=int(b))
                        for d, b in sorted(set(zip(c["depth"].tolist(), c["branching"].tolist())))}
    return out


def find_sets(suite_dir: str, only: Optional[Iterable[str]] = None, skip: Optional[Iterable[str]] = None) -> List[str]:
    """Names (paths relative to suite_dir) of the stored sets, selected by name prefixes."""
    names = sorted(os.path.relpath(dp, suite_dir) for dp, _, files in os.walk(suite_dir) if "meta.json" in files)
    if only:
        names = [n for n in names if any(n.startswith(p) for p in only)]
    if skip:
        names = [n for n in names if not any(n.startswith(p) for p in skip)]
    return names
