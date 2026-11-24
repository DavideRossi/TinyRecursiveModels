"""Unit tests of the independent checker on hand-written graphs (looped_lm/dataset_generation/checker.py).

    python tests/looped_lm/test_checker.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from looped_lm.dataset_generation import formats  # noqa: E402
from looped_lm.dataset_generation.checker import CheckError, analyze, check_sample  # noqa: E402
from looped_lm.dataset_generation import checker  # noqa: E402
from looped_lm.dataset_generation.vocab import ARROW, ASK, BACKTRACK, BOS, CLOSE, EOS, OPEN, QUERY, SEP, TARGETS, VOCAB  # noqa: E402

# Named nodes: s (queried source), its branches s>a1>a2 (target) and s>b1>b2 (dead end);
# distractor u>c1>c2 (target) and u>e1>e2 (dead end).
N = {name: VOCAB.node(i) for i, name in enumerate(
    "s a1 a2 b1 b2 u c1 c2 e1 e2 x y z w".split())}


def seq(text):
    """Token ids from a readable string: names of N, or specials."""
    specials = {">": ARROW, "T:": TARGETS, "Q:": QUERY, "?": ASK, "[": OPEN, "]": CLOSE, "<bt>": BACKTRACK,
                ";": SEP, "<eos>": EOS, "<bos>": BOS}
    return [specials[w] if w in specials else N[w] for w in text.split()]


# Edge order: s>b1 before s>a1, so the canonical order is (b branch, a branch)
EDGES = "u > c1 s > b1 c1 > c2 b1 > b2 u > e1 s > a1 e1 > e2 a1 > a2"
TARGETS_LIST = "T: c2 a2"
ANSWERS = {"direct": "a2", "path": "a1 a2", "frontier": "[ b1 a1 ] [ b2 a2 ] a2", "dfs": "b1 b2 <bt> a1 a2"}


def sample_tokens(fmt, edges=EDGES, targets=TARGETS_LIST, answer=None, query="s"):
    answer = ANSWERS[fmt] if answer is None else answer
    return seq(f"<bos> {edges} {targets} Q: {query} ? {answer} <eos>")


def fails(fn, *args, contains="", **kw):
    try:
        fn(*args, **kw)
    except CheckError as e:
        assert contains in str(e), f"wrong error: {e}"
        return
    raise AssertionError("expected CheckError")


def test_valid_sample_all_formats():
    for fmt in ANSWERS:
        a = analyze(sample_tokens(fmt), fmt)
        q = a["queries"][0]
        assert q["target"] == N["a2"] and q["path"] == [N["a1"], N["a2"]]
        assert q["depth"] == 2 and q["branching"] == 2
        assert q["branch_order"] == [N["b1"], N["a1"]]


def test_canonical_order_follows_first_edges():
    # Same graph, s>a1 now first: the a branch comes first
    edges = "s > a1 u > c1 s > b1 c1 > c2 b1 > b2 u > e1 e1 > e2 a1 > a2"
    analyze(sample_tokens("frontier", edges, answer="[ a1 b1 ] [ a2 b2 ] a2"), "frontier")
    analyze(sample_tokens("dfs", edges, answer="a1 a2"), "dfs")  # target branch first: no backtrack
    fails(analyze, sample_tokens("frontier", edges), "frontier", contains="expected")  # old order rejected
    fails(analyze, sample_tokens("dfs", edges), "dfs", contains="expected")
    # Later edges of a branch don't matter, only its first edge
    edges = "a1 > a2 b1 > b2 s > b1 s > a1 u > c1 c1 > c2 u > e1 e1 > e2"
    analyze(sample_tokens("frontier", edges, answer="[ b1 a1 ] [ b2 a2 ] a2"), "frontier")


def test_wrong_answers_rejected():
    fails(analyze, sample_tokens("direct", answer="c2"), "direct", contains="expected")
    fails(analyze, sample_tokens("path", answer="a2"), "path", contains="expected")
    fails(analyze, sample_tokens("path", answer="b1 a2"), "path", contains="expected")
    fails(analyze, sample_tokens("frontier", answer="[ b1 a1 ] a2"), "frontier", contains="expected")  # level missing
    fails(analyze, sample_tokens("dfs", answer="a1 a2"), "dfs", contains="expected")  # skipped the earlier branch
    fails(analyze, sample_tokens("dfs", answer="b1 b2 a1 a2"), "dfs", contains="expected")  # no backtrack marker


def test_cycle_detected():
    edges = EDGES + " x > y y > z z > x"
    fails(analyze, sample_tokens("direct", edges), "direct", contains="cycle")


def test_merge_detected():
    fails(analyze, sample_tokens("direct", EDGES + " x > a2"), "direct", contains="merge")
    fails(analyze, sample_tokens("direct", EDGES + " e2 > c2"), "direct", contains="merge")


def test_two_reachable_targets_detected():
    fails(analyze, sample_tokens("direct", targets="T: c2 a2 b2"), "direct", contains="targets")


def test_target_without_incoming_path():
    fails(analyze, sample_tokens("direct", targets="T: c2 a2 x"), "direct", contains="no incoming path")


def test_target_not_at_a_path_end():
    fails(analyze, sample_tokens("direct", targets="T: c2 a1"), "direct", contains="not the end")


def test_unequal_branch_lengths():
    edges = EDGES.replace("e1 > e2", "e1 > e2 e2 > w")
    fails(analyze, sample_tokens("direct", edges), "direct", contains="different lengths")


def test_branching_inside_a_path():
    edges = EDGES + " a1 > w"
    fails(analyze, sample_tokens("direct", edges), "direct", contains="branches inside")


def test_too_few_targets_and_bad_queries():
    fails(analyze, sample_tokens("direct", targets="T: a2"), "direct", contains="target(s)")
    fails(analyze, sample_tokens("direct", query="a1", answer="a2"), "direct", contains="not a source")
    tokens = seq(f"<bos> {EDGES} {TARGETS_LIST} Q: s ? a2 ; Q: s ? a2 <eos>")
    fails(analyze, tokens, "direct", contains="queried twice")


def test_malformed_sequences():
    fails(analyze, seq(f"{EDGES} {TARGETS_LIST} Q: s ? a2 <eos>"), "direct", contains="BOS")
    fails(analyze, seq(f"<bos> {EDGES} {TARGETS_LIST} Q: s ? a2"), "direct", contains="not followed")
    fails(analyze, seq(f"<bos> {EDGES} {TARGETS_LIST} Q: s ? a2 ;"), "direct", contains="must end with EOS")
    fails(analyze, seq(f"<bos> {EDGES} {TARGETS_LIST} Q: s ? a2 ; Q: u ? c2 ;"), "direct", contains="must end with EOS")
    fails(analyze, seq(f"<bos> {EDGES} {TARGETS_LIST} Q: s ? a2 <eos> c2"), "direct", contains="after EOS")
    fails(analyze, seq(f"<bos> {EDGES} Q: s ? a2 <eos>"), "direct", contains="no target list")


def test_targets_before_edges_and_two_queries():
    tokens = seq(f"<bos> {TARGETS_LIST} {EDGES} Q: u ? c2 ; Q: s ? a2 <eos>")
    a = analyze(tokens, "direct")
    assert a["parsed"].targets_position == "before"
    assert [q["target"] for q in a["queries"]] == [N["c2"], N["a2"]]


def full_metadata(tokens, fmt):
    """Hand-built metadata of a one-query sample (the expectation the checker must accept)."""
    tokens = np.asarray(tokens)
    L = len(tokens)
    start = int(np.nonzero(tokens == ASK)[0][0]) + 1
    end = L - 1
    roles_by_fmt = {"direct": [formats.ROLE_FINAL], "path": [formats.ROLE_INTERMEDIATE, formats.ROLE_FINAL],
                    "frontier": [3, 2, 2, 3, 3, 2, 2, 3, 1], "dfs": [2, 2, 4, 2, 1]}
    role = np.zeros(L, int)
    role[start:end] = roles_by_fmt[fmt]
    role[end] = formats.ROLE_END
    mask = np.zeros(L, bool)
    mask[start:] = True
    qidx = np.full(L, -1)
    qidx[start - 3:] = 0
    depth, branching = np.zeros(L, int), np.zeros(L, int)
    depth[start - 3:], branching[start - 3:] = 2, 2
    pred = np.full(L, -1)
    pred[end - 1] = end - 2
    query = {"source": N["s"], "target": N["a2"], "path": [N["a1"], N["a2"]], "depth": 2, "branching": 2,
             "branch_order": [N["b1"], N["a1"]], "prompt_pos": start - 3, "answer_start": start, "answer_end": end}
    return {"tokens": tokens, "loss_mask": mask, "query_index": qidx, "depth": depth, "branching": branching,
            "role": role, "pred_pos": pred, "queries": [query]}


def test_metadata_checks():
    for fmt in ANSWERS:
        check_sample(full_metadata(sample_tokens(fmt), fmt), fmt, edge_order="random", targets_position="after",
                     num_edges=8)
    s = full_metadata(sample_tokens("path"), "path")
    bad = dict(s, loss_mask=s["loss_mask"].copy())
    bad["loss_mask"][-1] = False  # end token not in the loss
    fails(check_sample, bad, "path", contains="loss_mask")
    bad = dict(s, loss_mask=s["loss_mask"].copy())
    bad["loss_mask"][1] = True  # an edge token in the loss
    fails(check_sample, bad, "path", contains="loss_mask")
    bad = dict(s, loss_mask=s["loss_mask"].copy())
    bad["loss_mask"][s["queries"][0]["prompt_pos"] + 1] = True  # the query prompt in the loss
    fails(check_sample, bad, "path", contains="loss_mask")
    bad = dict(s, role=s["role"].copy())
    bad["role"][-2] = formats.ROLE_INTERMEDIATE
    fails(check_sample, bad, "path", contains="role")
    bad = dict(s, pred_pos=s["pred_pos"].copy())
    bad["pred_pos"][-2] += 1
    fails(check_sample, bad, "path", contains="pred_pos")
    bad = dict(s, queries=[dict(s["queries"][0], depth=3)])
    fails(check_sample, bad, "path", contains="depth")
    bad = dict(s, queries=[dict(s["queries"][0], branching=1)])
    fails(check_sample, bad, "path", contains="branching")
    bad = dict(s, queries=[dict(s["queries"][0], branch_order=[N["a1"], N["b1"]])])
    fails(check_sample, bad, "path", contains="branch_order")
    fails(check_sample, s, "path", num_edges=9, contains="edges")
    fails(check_sample, s, "path", targets_position="before", contains="target list")


def test_edge_order_checks():
    rev = "c1 > c2 u > c1 b1 > b2 s > b1 e1 > e2 u > e1 a1 > a2 s > a1"
    fwd = "u > c1 c1 > c2 s > b1 b1 > b2 u > e1 e1 > e2 s > a1 a1 > a2"
    for edges, order in ((rev, "reverse"), (fwd, "forward")):
        s = full_metadata(sample_tokens("direct", edges), "direct")
        check_sample(s, "direct", edge_order=order)
        other = "forward" if order == "reverse" else "reverse"
        fails(check_sample, s, "direct", edge_order=other, contains=other)


def test_roles_agree_with_formats():
    assert (checker.OTHER, checker.FINAL, checker.INTERMEDIATE, checker.BRACKET, checker.BACKTRACK_ROLE,
            checker.END) == (formats.ROLE_OTHER, formats.ROLE_FINAL, formats.ROLE_INTERMEDIATE,
                             formats.ROLE_BRACKET, formats.ROLE_BACKTRACK, formats.ROLE_END)


# Fixed-mode structure: the query s (2 branches of depth 2, target a2), a distractor trimmed to a
# single chain u > c1 > c2 (target c2), a target-less component x with 2 branches of depth 1, and a
# target-less chain z > w. Exactly 2 targets.
FIXED_EDGES = "s > a1 x > y a1 > a2 u > c1 s > b1 b1 > b2 c1 > c2 x > e1 z > w"


def test_fixed_structure_accepted():
    a = analyze(sample_tokens("frontier", FIXED_EDGES, answer="[ a1 b1 ] [ a2 b2 ] a2"), "frontier", num_targets=2)
    assert a["distractor_shapes"] == [(2, 1)]  # the trimmed distractor; target-less components excluded
    assert sorted(len(b) for b in a["components"][N["x"]]) == [1, 1]  # target-less, b = 2
    analyze(sample_tokens("dfs", FIXED_EDGES, answer="a1 a2"), "dfs", num_targets=2)


def test_fixed_structure_invariants_rejected():
    def bad(edges, contains, targets=TARGETS_LIST, num_targets=2):
        fails(analyze, sample_tokens("direct", edges, targets), "direct", contains=contains, num_targets=num_targets)
    bad(FIXED_EDGES, "3 targets, expected 2", targets="T: c2 a2 y")           # exactly T targets
    bad(FIXED_EDGES, "2 targets, expected 3", num_targets=3)
    bad(FIXED_EDGES + " u > e2", "paths of different lengths")                # unequal depths in the distractor
    bad(FIXED_EDGES.replace("x > e1", "x > e1 e1 > e2"), "paths of different lengths")  # in a target-less one
    bad(FIXED_EDGES + " c1 > e2", "branches inside a path")                   # branching below the root
    bad(FIXED_EDGES.replace("z > w", "z > w w > e2 y > e2"), "merge")         # a merge


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
