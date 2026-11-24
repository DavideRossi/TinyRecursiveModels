"""Tests of the generator, the datasets, storage and the audit (looped_lm/dataset_generation).

    python tests/looped_lm/test_generator.py
"""
import itertools
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from looped_lm.dataset_generation import audit as audit_mod  # noqa: E402
from looped_lm.dataset_generation import formats, suite  # noqa: E402
from looped_lm.dataset_generation.checker import CheckError, check_sample, parse  # noqa: E402
from looped_lm.dataset_generation.config import ANSWER_FORMATS, EDGE_ORDERS, TARGET_POSITIONS, TaskConfig  # noqa: E402
from looped_lm.dataset_generation.dataset import GeneratedDataset, StoredDataset, TrainStream, collate  # noqa: E402
from looped_lm.dataset_generation.generate import batch_edges, make_batch, make_sample  # noqa: E402
from looped_lm.dataset_generation.storage import StoredSet, write_set  # noqa: E402
from looped_lm.dataset_generation.vocab import ARROW, BACKTRACK, CLOSE, OPEN, VOCAB  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_checker import N, seq  # noqa: E402

MIXED = dict(depth={"range": [1, 6]}, branching={"range": [1, 4]}, num_queries={"range": [1, 3]},
             num_distractors={"range": [1, 3]}, total_edges={"range": [8, 150]})
# For the training stream: one distractor, and every edge count fits the worst case 3 * 24 + 24 = 96
STREAM = dict(MIXED, num_distractors=1, total_edges={"range": [96, 150]})
# Fixed mode: 4-7 targets; worst case 3 queries x 24 + 4 distractors as chains of depth 6 = 96
FIXED = dict(targets_mode="fixed", depth={"range": [1, 6]}, branching={"range": [1, 4]},
             num_queries={"range": [1, 3]}, num_targets={"range": [4, 7]}, total_edges={"range": [96, 150]})
# Fixed mode, trimming often: worst case 2 x 36 + 9 chains x 6 = 126 edges, untrimmed up to 10 x 36 = 360
TIGHT = dict(targets_mode="fixed", depth={"range": [2, 6]}, branching={"range": [2, 6]},
             num_queries={"range": [1, 2]}, num_targets={"range": [6, 10]}, total_edges={"range": [126, 200]})
SUITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "looped_lm", "dataset_generation",
                     "configs", "eval_suite.json")


def same_sample(a, b):
    for k in ("tokens", "loss_mask", "query_index", "depth", "branching", "role", "pred_pos"):
        x = a[k] if isinstance(a, dict) else getattr(a, k)
        y = b[k] if isinstance(b, dict) else getattr(b, k)
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    qa = a["queries"] if isinstance(a, dict) else a.queries
    qb = b["queries"] if isinstance(b, dict) else b.queries
    return qa == qb


def test_checker_accepts_every_configuration():
    n = 0
    for fmt, order, pos in itertools.product(ANSWER_FORMATS, EDGE_ORDERS, TARGET_POSITIONS):
        cfg = TaskConfig(**MIXED, answer_format=fmt, edge_order=order, targets_position=pos)
        for i in range(25):
            s = make_sample(cfg, 5, i)
            check_sample(s, fmt, order, pos, num_edges=s.num_edges)
            n += 1
    # Separate distractor distributions and no padding
    cfg = TaskConfig(depth=3, branching=2, num_queries=2, num_distractors=4, distractor_depth={"range": [1, 9]},
                     distractor_branching=[1, 5], answer_format="dfs")
    for i in range(50):
        check_sample(make_sample(cfg, 0, i), "dfs")
    assert n == 600


def test_edge_count_and_targets():
    cfg = TaskConfig(**dict(MIXED, total_edges=97))
    for i in range(100):
        s = make_sample(cfg, 0, i)
        assert s.num_edges == 97
        assert s.num_targets >= 2
    # total_edges smaller than what the sample needs: the sample keeps its components
    cfg = TaskConfig(depth=5, branching=4, num_queries=2, num_distractors=1, total_edges=10)
    s = make_sample(cfg, 0, 0)
    assert s.num_edges == 60 and s.num_components == 3


def test_queried_and_distractor_components_have_one_target():
    cfg = TaskConfig(depth=3, branching=3, num_queries=2, num_distractors=2, total_edges=50)
    for i in range(50):
        s = make_sample(cfg, 1, i)
        # 4 components with a target (9 edges each), padding of 9-edge components with a target,
        # then a filler without a target: 50 = 5 * 9 + 5
        assert s.num_targets == 5 and s.num_components == 6


def test_determinism():
    cfg = TaskConfig(**MIXED, answer_format="frontier")
    a, b = make_sample(cfg, 3, 17), make_sample(cfg, 3, 17)
    assert same_sample(a, b)
    others = [make_sample(cfg, 3, 18), make_sample(cfg, 4, 17), make_sample(cfg, 3, 17, split="eval"),
              make_sample(cfg, 3, 17, stream=1)]
    assert not any(same_sample(a, o) for o in others)


def stream_items(num_workers, batch_size, n):
    cfg = TaskConfig(**STREAM, pool_size=2048)
    loader = DataLoader(TrainStream(cfg, seed=2, batch_size=batch_size), batch_size=None, num_workers=num_workers,
                        collate_fn=None if batch_size is None else (lambda x: x))
    return list(itertools.islice(iter(loader), n))


def test_stream_independent_of_worker_count():
    ref = stream_items(0, None, 12)
    assert [int(it["index"]) for it in ref] == list(range(12))
    for w in (1, 2, 3):
        got = stream_items(w, None, 12)
        for x, y in zip(ref, got):
            assert int(x["index"]) == int(y["index"]) and torch.equal(x["tokens"], y["tokens"])
            assert torch.equal(x["loss_mask"], y["loss_mask"]) and x["queries"] == y["queries"]
    ref = stream_items(0, 4, 5)
    for w in (2, 3):
        got = stream_items(w, 4, 5)
        for x, y in zip(ref, got):
            assert torch.equal(x["index"], y["index"]) and torch.equal(x["tokens"], y["tokens"])
    assert ref[1]["index"].tolist() == [4, 5, 6, 7]


def test_ranks_partition_the_stream():
    cfg = TaskConfig(**STREAM)
    single = [int(it["index"]) for it in itertools.islice(iter(TrainStream(cfg, seed=0)), 12)]
    ranks = [[int(it["index"]) for it in itertools.islice(iter(TrainStream(cfg, seed=0, rank=r, world_size=3)), 4)]
             for r in range(3)]
    assert sorted(sum(ranks, [])) == single
    # Batches: rank r of R takes batches r, r + R, ... of the same global sequence
    batches = list(itertools.islice(iter(TrainStream(cfg, seed=0, batch_size=4)), 6))
    for r in range(3):
        mine = list(itertools.islice(iter(TrainStream(cfg, seed=0, batch_size=4, rank=r, world_size=3)), 2))
        for k, b in enumerate(mine):
            ref = batches[r + 3 * k]
            assert torch.equal(b["index"], ref["index"]) and torch.equal(b["tokens"], ref["tokens"])


def test_map_style_independent_of_access_order():
    cfg = TaskConfig(**MIXED)
    ds = GeneratedDataset(cfg, 20, seed=9, split="eval", stream=4)
    fwd = [ds[i] for i in range(20)]
    for i in np.random.default_rng(0).permutation(20):
        assert torch.equal(ds[int(i)]["tokens"], fwd[int(i)]["tokens"])
    assert torch.equal(fwd[7]["tokens"], torch.as_tensor(make_sample(cfg, 9, 7, "eval", 4).tokens, dtype=torch.long))


def test_canonical_order_in_frontier_and_dfs():
    for order in EDGE_ORDERS:
        for fmt in ("frontier", "dfs"):
            cfg = TaskConfig(depth={"range": [2, 5]}, branching={"range": [2, 5]}, num_queries=3, num_distractors=1,
                             edge_order=order, answer_format=fmt)
            for i in range(30):
                s = make_sample(cfg, 0, i)
                toks = s.tokens.tolist()
                edges = [(toks[k], toks[k + 2]) for k in range(len(toks) - 2) if toks[k + 1] == ARROW]
                for q in s.queries:
                    pos = [edges.index((q["source"], f)) for f in q["branch_order"]]
                    assert pos == sorted(pos), "branch order is not the order of the first edges"
                    ans = toks[q["answer_start"]:q["answer_end"]]
                    if fmt == "frontier":
                        b = q["branching"]
                        for level in range(q["depth"]):
                            block = ans[level * (b + 2):(level + 1) * (b + 2)]
                            assert block[0] == OPEN and block[-1] == CLOSE
                            if level == 0:
                                assert block[1:-1] == q["branch_order"]
                            # Every node of the level continues the branch at the same rank
                            prev = [q["source"]] * b if level == 0 else \
                                ans[(level - 1) * (b + 2) + 1:level * (b + 2) - 1]
                            assert all((p, c) in edges for p, c in zip(prev, block[1:-1]))
                        assert ans[-1] == q["target"]
                    else:
                        segments, cur = [], []
                        for t in ans:
                            if t == BACKTRACK:
                                segments.append(cur)
                                cur = []
                            else:
                                cur.append(t)
                        segments.append(cur)
                        assert [seg[0] for seg in segments] == q["branch_order"][:len(segments)]
                        assert segments[-1] == q["path"]
                        assert all(len(seg) == q["depth"] for seg in segments)


def test_loss_mask():
    for fmt in ANSWER_FORMATS:
        cfg = TaskConfig(**MIXED, answer_format=fmt, targets_position="before")
        for i in range(30):
            s = make_sample(cfg, 0, i)
            expected = np.zeros(len(s), bool)
            for q in s.queries:
                expected[q["answer_start"]:q["answer_end"] + 1] = True  # answer + its ; or EOS
                assert not s.loss_mask[q["prompt_pos"]:q["answer_start"]].any()
            assert np.array_equal(s.loss_mask, expected)
            assert not s.loss_mask[:s.queries[0]["prompt_pos"]].any()  # never the edges or the target list
            assert (s.role[s.loss_mask] != formats.ROLE_OTHER).all()
            assert (s.role[~s.loss_mask] == formats.ROLE_OTHER).all()
            finals = np.nonzero(s.role == formats.ROLE_FINAL)[0]
            assert len(finals) == len(s.queries)
            assert np.array_equal(s.pred_pos[finals], finals - 1)
            assert (s.pred_pos[s.role != formats.ROLE_FINAL] == -1).all()
            assert [int(s.tokens[f]) for f in finals] == [q["target"] for q in s.queries]


def test_query_order_independent_of_graph_order():
    cfg = TaskConfig(depth=2, branching=2, num_queries=2, num_distractors=2)
    in_edge_order = 0
    n = 400
    for i in range(n):
        s = make_sample(cfg, 0, i)
        p = parse(s.tokens)
        first = [min(k for k, (u, _) in enumerate(p.edges) if u == q["source"]) for q in s.queries]
        in_edge_order += first[0] < first[1]
    assert 0.4 < in_edge_order / n < 0.6, in_edge_order / n


def test_pool_size_checks():
    try:
        TaskConfig(depth=32, branching=8, num_queries=4, num_distractors=4)
    except ValueError as e:
        assert "pool" in str(e)
    else:
        raise AssertionError("expected the pool check to fail")
    small = TaskConfig(**STREAM, pool_size=1000)
    try:
        TrainStream(small)
    except ValueError as e:
        assert "full pool" in str(e)
    else:
        raise AssertionError("training must refuse a small pool")
    next(iter(TrainStream(small, allow_small_pool=True)))
    s = make_sample(small, 0, 0)
    assert all(t < VOCAB.node(1000) for t in s.tokens if VOCAB.is_node(t))


def test_storage_round_trip_and_collate():
    cfg = TaskConfig(**MIXED, answer_format="dfs", edge_order="reverse")
    with tempfile.TemporaryDirectory() as d:
        write_set(d, cfg, 30, seed=1, stream=7, split="eval", readable=3)
        st = StoredSet(d)
        assert len(st) == 30 and st.config.to_dict() == cfg.to_dict()
        for i in range(30):
            rec = st.get(i)
            assert same_sample(rec, make_sample(cfg, 1, i, "eval", 7))
            check_sample(rec, "dfs", "reverse", num_edges=rec["num_edges"])
        with open(os.path.join(d, "samples.jsonl")) as f:
            assert len(f.readlines()) == 3
        ds = StoredDataset(d)
        batch = collate([ds[i] for i in range(5)])
        L = int(batch["length"].max())
        assert batch["tokens"].shape == (5, L)
        for i in range(5):
            n = int(batch["length"][i])
            assert torch.equal(batch["tokens"][i, :n], ds[i]["tokens"])
            assert (batch["tokens"][i, n:] == 0).all() and not batch["loss_mask"][i, n:].any()
            assert (batch["pred_pos"][i, n:] == -1).all()


def test_suite_expansion():
    s = {"seed": 3, "base": {"num_distractors": 1, "answer_format": "path"},
         "grid": {"depths": [2, 4], "branchings": [1, 3], "num_samples": 10},
         "mixed": [{"name": "m", "num_samples": 5, "overrides": {"depth": {"range": [1, 4]}, "num_queries": 3}}]}
    specs = suite.expand(s)
    assert [x.name for x in specs] == ["grid/d2_b1", "grid/d2_b3", "grid/d4_b1", "grid/d4_b3", "mixed/m"]
    grid = [x for x in specs if x.name.startswith("grid")]
    assert {x.config.total_edges.max for x in grid} == {24}  # 2 components of the largest cell (4 * 3 each)
    assert len({x.stream for x in specs}) == len(specs)
    for x in grid:
        for i in range(5):
            sm = make_sample(x.config, x.seed, i, "eval", x.stream)
            assert sm.num_edges == 24 and len(sm.queries) == 1
            check_sample(sm, "path")


def test_audit_at_chance_and_detects_a_planted_shortcut():
    cfg = TaskConfig(depth=3, branching=3, num_queries=1, num_distractors=5, total_edges=54)
    r = audit_mod.audit(cfg, 600)
    for name in audit_mod.HEURISTICS:
        assert r[name]["flag"] == "", (name, r[name])
    # Plant a shortcut: the answer moved to the front of the target list
    orig = audit_mod.make_sample

    def leaky(*a, **k):
        s = orig(*a, **k)
        toks = s.tokens.tolist()
        p = parse(toks)
        t = s.queries[0]["target"]
        lst = [t] + [x for x in p.targets if x != t]
        toks[p.targets_pos + 1:p.targets_pos + 1 + len(lst)] = lst
        s.tokens = np.asarray(toks, dtype=np.int32)
        return s

    audit_mod.make_sample = leaky
    try:
        r = audit_mod.audit(cfg, 300)
    finally:
        audit_mod.make_sample = orig
    assert r["first_listed"]["flag"] == "ABOVE CHANCE" and r["first_listed"]["accuracy"] == 1.0


def edges_per_row(tokens: torch.Tensor):
    return (tokens == ARROW).sum(1).tolist()


def test_batch_shares_edge_count_for_any_worker_count():
    cfg = TaskConfig(**STREAM)
    ref = stream_items(0, 8, 6)
    for j, b in enumerate(ref):
        assert edges_per_row(b["tokens"]) == [batch_edges(cfg, 2, j)] * 8
    assert len({edges_per_row(b["tokens"])[0] for b in ref}) > 1  # different batches, different counts
    for w in (1, 2, 3):
        for x, y in zip(ref, stream_items(w, 8, 6)):
            assert torch.equal(x["index"], y["index"]) and torch.equal(x["tokens"], y["tokens"])
            assert torch.equal(x["loss_mask"], y["loss_mask"]) and x["queries"] == y["queries"]
    # Without batch_size, every sample is a batch of one
    for j, it in enumerate(stream_items(0, None, 5)):
        assert edges_per_row(it["tokens"][None]) == [batch_edges(cfg, 2, j)]
    # Every other draw is the sample's own: the same components as the map-style sample, only padded
    # to another edge count
    s, t = make_batch(cfg, 2, 3, 8)[5], make_sample(cfg, 2, 3 * 8 + 5)
    assert [(q["depth"], q["branching"]) for q in s.queries] == [(q["depth"], q["branching"]) for q in t.queries]
    for x in make_batch(cfg, 2, 3, 8):
        check_sample(x, cfg.answer_format, cfg.edge_order, cfg.targets_position, num_edges=x.num_edges)


def test_batch_edge_count_uncorrelated_with_depth_and_branching():
    # Per sample: its edge count against what it drew. Under independence each correlation has sd
    # about 1/sqrt(#samples) (the sample's draws are independent across samples, even within a batch)
    cfg = TaskConfig(**STREAM)
    n, bs = 3000, 4
    edges, cols = [], {"depth": [], "branching": [], "query edges (sum d*b)": [], "queries": [], "tokens": []}
    for j in range(n):
        for x in make_batch(cfg, 11, j, bs):
            edges.append(x.num_edges)
            cols["depth"].append(np.mean([q["depth"] for q in x.queries]))
            cols["branching"].append(np.mean([q["branching"] for q in x.queries]))
            cols["query edges (sum d*b)"].append(sum(q["depth"] * q["branching"] for q in x.queries))
            cols["queries"].append(len(x.queries))
            cols["tokens"].append(len(x))
    assert set(edges) == set(cfg.total_edges.support)  # the whole range, drawn per batch
    bound = 4 / np.sqrt(len(edges))
    for k, v in cols.items():
        r = np.corrcoef(edges, v)[0, 1]
        if k == "tokens":
            assert r > 0.5, r  # the test can see a dependence: length follows the edge count
        else:
            assert abs(r) < bound, (k, r, bound)


def test_batch_edges_config_check():
    TrainStream(TaskConfig(**STREAM))  # minimum 96 = 3 queries x 24 + 1 distractor x 24: fits exactly
    for bad, msg in ((dict(STREAM, total_edges={"range": [95, 150]}), "starts at 95, but a sample can need 96"),
                     (dict(STREAM, num_distractors=2), "starts at 96, but a sample can need 120"),
                     (dict(STREAM, total_edges=None), "needs total_edges")):
        try:
            TrainStream(TaskConfig(**bad))
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"expected the check to refuse {bad}")
    train = TaskConfig.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "looped_lm",
                                         "dataset_generation", "configs", "train.json"))
    TrainStream(train)


def test_exclusion_heuristics_on_a_hand_built_sample():
    # Query s (s > a1 > a2, target a2); distractors u > c1 > c2 and x > y > z (targets c2, z): T = 3
    rng = np.random.default_rng(0)
    cases = [
        # s's edge at 0; incoming edges of a2, c2, z at 1, 3, 5: the nearest is a2, the answer.
        # Target list after the edges, a2 last: the most recent is a2 too. Both exclude the answer.
        ("s > a1 a1 > a2 u > c1 c1 > c2 x > y y > z", "T: c2 z a2", N["a2"], N["a2"], 0.0),
        # s's edge at 1; incoming edges of c2, a2, z at 0, 4, 5: the nearest is c2. Last listed: c2.
        ("c1 > c2 s > a1 u > c1 x > y a1 > a2 y > z", "T: a2 z c2", N["c2"], N["c2"], 0.5),
    ]
    for edges, targets, nearest, recent, expected in cases:
        toks = seq(f"<bos> {edges} {targets} Q: s ? a2 <eos>")
        g = audit_mod.heuristic_guesses(toks, [2], rng)
        assert g["proximity"] == [nearest] and g["recent"] == [recent]
        for name, base in audit_mod.EXCLUSION.items():
            assert audit_mod.exclusion_hits(g[base], [N["a2"]], 3) == [expected], name
    # Chance and ceiling: an exclusion heuristic whose excluded target is the answer 1/T of the time
    # scores 1/T exactly; one that never excludes the answer scores the ceiling 1/(T-1)
    T = 4
    hits = audit_mod.exclusion_hits([0, 1, 2, 3], [0, 0, 0, 0], T)
    assert np.isclose(np.mean(hits), 1 / T) and max(hits) == 1 / (T - 1)


def test_old_mode_reproduces_previous_samples():
    import per_component_reference as ref
    with open(ref.PATH) as f:
        expected = json.load(f)
    assert json.loads(json.dumps(ref.compute())) == expected


def test_fixed_mode_targets_trimming_and_queries():
    tight = TaskConfig(**dict(TIGHT, total_edges=126))
    loose = TaskConfig(**dict(TIGHT, total_edges=400))  # never trims: at most 10 x 36 = 360
    num_trimmed, n = 0, 0
    for c, (fmt, order, pos) in enumerate(itertools.product(ANSWER_FORMATS, EDGE_ORDERS, TARGET_POSITIONS)):
        t_cfg, l_cfg = (c.replace(answer_format=fmt, edge_order=order, targets_position=pos) for c in (tight, loose))
        for i in range(12 * c, 12 * c + 12):
            n += 1
            s, u = make_sample(t_cfg, 4, i), make_sample(l_cfg, 4, i)
            for x, cfg in ((s, t_cfg), (u, l_cfg)):
                check_sample(x, fmt, order, pos, num_edges=cfg.total_edges.max, num_targets=x.num_targets)
            # The same draws: T, queried shapes, drawn distractor shapes (the loose sample is untrimmed)
            assert s.num_targets == u.num_targets and 6 <= s.num_targets <= 10
            assert [(q["depth"], q["branching"]) for q in s.queries] == [(q["depth"], q["branching"]) for q in u.queries]
            assert not u.trimmed and all(x["drawn_branching"] == x["branching"] for x in u.distractors)
            assert [(x["depth"], x["drawn_branching"]) for x in s.distractors] == \
                   [(x["depth"], x["branching"]) for x in u.distractors]
            assert len(s.distractors) == s.num_targets - len(s.queries)
            # Trimming: only as much as needed; never below one branch
            need = sum(q["depth"] * q["branching"] for q in s.queries) + sum(x["depth"] * x["branching"]
                                                                            for x in s.distractors)
            assert need <= 126 and all(x["branching"] >= 1 for x in s.distractors)
            if s.trimmed:  # trimming stops at the first fit: the last branch removed would not fit back
                num_trimmed += 1
                assert any(need + x["depth"] > 126 for x in s.distractors if x["branching"] < x["drawn_branching"])
    assert 0.2 * n < num_trimmed < n, (num_trimmed, n)


def test_fixed_mode_validation():
    for bad, msg in ((dict(FIXED, num_targets=3), "num_targets must be >= num_queries + 1"),
                     (dict(FIXED, num_distractors=2), "num_distractors is not used"),
                     (dict(FIXED, num_targets=None), "needs num_targets"),
                     (dict(MIXED, num_targets=5), "need targets_mode 'fixed'")):
        try:
            TaskConfig(**bad)
        except ValueError as e:
            assert msg in str(e), str(e)
        else:
            raise AssertionError(f"expected {msg}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TaskConfig(**dict(FIXED, branching=8, num_targets={"range": [4, 9]}))
        assert any("backward search" in str(x.message) for x in w)


def test_check_batch_edges_fixed_worst_case():
    TrainStream(TaskConfig(**FIXED))  # 3 x 24 + 4 x 6 = 96: fits exactly
    try:
        TrainStream(TaskConfig(**dict(FIXED, total_edges={"range": [95, 150]})))
    except ValueError as e:
        assert "a sample can need 96 edges (3 queries x 24 edges at maximum depth and branching + 4 distractors " \
               "as single chains of depth 6)" in str(e), str(e)
    else:
        raise AssertionError("expected the check to refuse")
    train = TaskConfig.load(os.path.join(os.path.dirname(SUITE), "train.json"))
    assert train.fixed and train.max_required_edges() == 4 * 32 + 8 * 8 == train.total_edges.min


def test_fixed_mode_edge_count_independent_of_queries():
    # Per sample, its batch's edge count against its queried shapes; and the queried shapes do not
    # depend on the edge count at all (same draws whatever E is)
    cfg = TaskConfig(**TIGHT)
    n, bs = 3000, 4
    edges, cols, trimmed = [], {"depth": [], "branching": [], "d*b": []}, []
    for j in range(n):
        for x in make_batch(cfg, 12, j, bs):
            edges.append(x.num_edges)
            cols["depth"].append(np.mean([q["depth"] for q in x.queries]))
            cols["branching"].append(np.mean([q["branching"] for q in x.queries]))
            cols["d*b"].append(sum(q["depth"] * q["branching"] for q in x.queries))
            trimmed.append(x.trimmed)
    assert 0.05 < np.mean(trimmed) < 0.95  # trimming happens, and nothing is resampled because of it
    bound = 4 / np.sqrt(len(edges))
    for k, v in cols.items():
        r = np.corrcoef(edges, v)[0, 1]
        assert abs(r) < bound, (k, r, bound)
    for i in range(50):
        shapes = {E: [(q["depth"], q["branching"]) for q in make_sample(cfg, 12, i, total_edges=E).queries]
                  for E in (126, 160, 200)}
        assert shapes[126] == shapes[160] == shapes[200]


def test_fixed_mode_metadata_checked_and_stored():
    cfg = TaskConfig(**dict(TIGHT, total_edges=126))
    s = next(make_sample(cfg, 1, i) for i in range(100) if make_sample(cfg, 1, i).trimmed)
    check_sample(s, cfg.answer_format, num_targets=s.num_targets)
    for field, value in (("branches_removed", s.branches_removed + 1), ("trimmed", False),
                         ("distractors", [dict(x, branching=x["drawn_branching"]) for x in s.distractors]),
                         ("distractors", [dict(x, depth=x["depth"] + 1) for x in s.distractors])):  # shapes only
        bad = dict(vars(s), **{field: value})
        try:
            check_sample(bad, cfg.answer_format)
        except CheckError:
            pass
        else:
            raise AssertionError(f"tampered {field} accepted")
    with tempfile.TemporaryDirectory() as d:
        write_set(d, cfg, 40, seed=1, stream=0, split="debug")
        st = StoredSet(d)
        for i in range(40):
            rec, g = st.get(i), make_sample(cfg, 1, i, "debug")
            assert rec["trimmed"] == g.trimmed and rec["branches_removed"] == g.branches_removed
            assert rec["distractors"] == g.distractors
            check_sample(rec, cfg.answer_format, num_targets=rec["num_targets"])
        b = collate([StoredDataset(d)[i] for i in range(4)])
        assert b["trimmed"].tolist() == [make_sample(cfg, 1, i, "debug").trimmed for i in range(4)]


def test_branching_root_heuristic():
    # Roots: s (2 branches, the answer a2), u and x (chains): branching_root always picks a2
    toks = seq("<bos> s > a1 a1 > a2 s > b1 b1 > b2 u > c1 c1 > c2 x > y y > z T: c2 z a2 Q: s ? a2 <eos>")
    for k in range(5):
        assert audit_mod.heuristic_guesses(toks, [2], np.random.default_rng(k))["branching_root"] == [N["a2"]]


def test_suite_fixed_grid_companions_diag():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        specs = {s.name: s for s in suite.expand(suite.load(SUITE))}
    assert any("companion/d32_b8 skipped" in str(x.message) for x in w)  # 2,056 names > the pool
    grid = {k: v for k, v in specs.items() if k.startswith("grid/")}
    assert len(grid) == 32 and all(v.config.num_targets.spec == 8 and v.config.total_edges.spec == 512
                                   and v.config.num_queries.max == 1 for v in grid.values())
    trimming = sorted(k[5:] for k, v in grid.items() if v.config.max_untrimmed_edges() > 512)
    assert trimming == sorted(["d12_b8", "d16_b8", "d24_b4", "d24_b8", "d32_b4", "d32_b8"])
    comps = sorted(k for k in specs if k.startswith("companion/"))
    assert comps == sorted(f"companion/{c}_{v}" for c in trimming if c != "d32_b8" for v in "AB")
    diag = sorted(k for k in specs if k.startswith("diag/"))
    assert len(diag) == 11 and "diag/b8_t6" not in diag
    assert {specs[k].config.total_edges.spec for k in diag} == {512}  # as the grid
    auto = suite.expand(dict(suite.load(SUITE), diag=dict(suite.load(SUITE)["diag"], total_edges=None)))
    assert {x.config.total_edges.spec for x in auto if x.name.startswith("diag/")} == {8 * 8 + 11 * 8}  # smallest fit
    # Paired companions: same T, edge count and queried component (names and, in random order,
    # edge positions); A untrimmed, B trimmed as the 512-edge grid cell
    A, B = specs["companion/d16_b8_A"], specs["companion/d16_b8_B"]
    assert A.stream == B.stream and A.config.total_edges.spec == B.config.total_edges.spec == 1024
    for i in range(10):
        a, b = (make_sample(x.config, x.seed, i, "eval", x.stream) for x in (A, B))
        check_sample(b, b_fmt := B.config.answer_format, num_edges=1024, num_targets=8)
        check_sample(a, b_fmt, num_edges=1024, num_targets=8)
        assert not a.trimmed and b.trimmed and a.num_edges == b.num_edges and a.num_targets == b.num_targets
        assert a.queries == b.queries  # source, target, path, branch order and token positions
        assert sum(x["depth"] * x["branching"] for x in b.distractors) + 128 <= 512
        pa, pb = parse(a.tokens), parse(b.tokens)
        q_edges = [k for k, (u, v) in enumerate(pa.edges) if u in [a.queries[0]["source"]] + a.queries[0]["path"]
                   or v in a.queries[0]["path"]]
        assert all(pa.edges[k] == pb.edges[k] for k in q_edges)
    # depth_match at chance in grid cells, trimmed or not
    for cell in ("grid/d4_b2", "grid/d16_b8"):
        r = audit_mod.audit(specs[cell].config, 400)
        assert r["depth_match"]["flag"] == "" and abs(r["depth_match"]["chance"] - 1 / 8) < 1e-12, (cell, r["depth_match"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
