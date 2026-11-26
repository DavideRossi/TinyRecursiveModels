"""Independent checker: validates a sample from its token ids alone.

It parses the edges, the target list and the queries from the tokens, runs its own graph search,
and rebuilds every answer format from the parsed graph and edge order. It imports only the
vocabulary (token ids), never the generator or formats.py, so a bug there can't hide itself.
Optionally it also checks the per-token and per-query metadata against what it derived.
"""
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vocab import ARROW, ASK, BACKTRACK, BOS, CLOSE, EOS, OPEN, PAD, QUERY, SEP, TARGETS, VOCAB, Vocab

# Roles, redefined here on purpose (must agree with formats.ROLE_*; a test compares them)
OTHER, FINAL, INTERMEDIATE, BRACKET, BACKTRACK_ROLE, END = range(6)


class CheckError(Exception):
    pass


class Parsed:
    def __init__(self):
        self.edges: List[Tuple[int, int]] = []
        self.edge_token_pos: List[int] = []  # position of each edge's source token
        self.edge_dest_pos: List[int] = []  # position of each edge's destination token
        self.syntax: Optional[str] = None  # arrow ("x > t", "Q: x ?") | pair ("x t ;", "Q: x"), from the edges
        self.targets: List[int] = []
        self.targets_pos: Optional[int] = None  # position of T:
        self.targets_position: Optional[str] = None  # before | after the edges
        self.queries: List[dict] = []  # prompt_pos, source, answer (ids), answer_start, answer_end, end_token
        self.length = 0  # tokens up to EOS included (trailing PAD excluded)


def parse(tokens: Sequence[int], vocab: Vocab = VOCAB) -> Parsed:
    toks = [int(t) for t in tokens]
    p = Parsed()
    n = len(toks)
    if n == 0 or toks[0] != BOS:
        raise CheckError("sample must start with BOS")
    i = 1
    while i < n and toks[i] != QUERY:
        if toks[i] == TARGETS:
            if p.targets_pos is not None:
                raise CheckError("two target lists")
            p.targets_pos = i
            p.targets_position = "after" if p.edges else "before"
            i += 1
            while i < n and vocab.is_node(toks[i]) and not (i + 1 < n and toks[i + 1] == ARROW):
                p.targets.append(toks[i])
                i += 1
        elif vocab.is_node(toks[i]):
            if p.targets_position == "after":
                raise CheckError(f"edge at position {i} after the target list")
            if i + 2 >= n:
                raise CheckError(f"malformed edge at position {i}")
            if toks[i + 1] == ARROW and vocab.is_node(toks[i + 2]):
                syntax, dest = "arrow", i + 2
            elif vocab.is_node(toks[i + 1]) and toks[i + 2] == SEP:
                syntax, dest = "pair", i + 1
            else:
                raise CheckError(f"malformed edge at position {i}")
            if p.syntax is not None and syntax != p.syntax:
                raise CheckError(f"edge at position {i} in {syntax} syntax after {p.syntax} edges")
            p.syntax = syntax
            p.edges.append((toks[i], toks[dest]))
            p.edge_token_pos.append(i)
            p.edge_dest_pos.append(dest)
            i += 3
        else:
            raise CheckError(f"unexpected token {vocab.tokens[toks[i]]!r} at position {i}")
    if p.targets_pos is None:
        raise CheckError("no target list")
    if not p.edges:
        raise CheckError("no edges")
    answer_tokens = {OPEN, CLOSE, BACKTRACK}
    while i < n:
        if toks[i] != QUERY:
            raise CheckError(f"expected Q: at position {i}")
        prompt_len = 2 if p.syntax == "pair" else 3
        if i + prompt_len >= n or not vocab.is_node(toks[i + 1]) or (prompt_len == 3 and toks[i + 2] != ASK):
            raise CheckError(f"malformed query prompt at position {i}")
        q = {"prompt_pos": i, "source": toks[i + 1], "answer_start": i + prompt_len}
        i += prompt_len
        while i < n and (vocab.is_node(toks[i]) or toks[i] in answer_tokens):
            i += 1
        if i >= n or toks[i] not in (SEP, EOS):
            raise CheckError(f"query at position {q['prompt_pos']}: answer not followed by ; or EOS")
        q["answer_end"] = i
        q["answer"] = toks[q["answer_start"]:i]
        q["end_token"] = toks[i]
        p.queries.append(q)
        i += 1
        if q["end_token"] == EOS:
            break
    if not p.queries:
        raise CheckError("no queries")
    if p.queries[-1]["end_token"] != EOS:
        raise CheckError("the last query must end with EOS")
    if any(t != PAD for t in toks[i:]):
        raise CheckError("tokens after EOS other than padding")
    p.length = i
    return p


class Graph:
    """Directed graph of the parsed edges, with the structural checks of the task."""

    def __init__(self, edges: List[Tuple[int, int]]):
        self.edges = edges
        self.edge_index = {}
        self.out: Dict[int, List[int]] = {}
        self.indeg: Dict[int, int] = {}
        self.parent: Dict[int, int] = {}
        for k, (u, v) in enumerate(edges):
            if u == v:
                raise CheckError(f"self-loop on {u}")
            if (u, v) in self.edge_index:
                raise CheckError(f"duplicate edge {u} > {v}")
            self.edge_index[(u, v)] = k
            self.out.setdefault(u, []).append(v)
            self.out.setdefault(v, [])
            self.indeg[v] = self.indeg.get(v, 0) + 1
            self.indeg.setdefault(u, 0)
            self.parent[v] = u
        self.nodes = list(self.out)

    def check_no_merges(self):
        merges = [v for v, d in self.indeg.items() if d > 1]
        if merges:
            raise CheckError(f"merge: {len(merges)} node(s) with more than one incoming edge")

    def check_acyclic(self):
        indeg = dict(self.indeg)
        queue = deque(v for v, d in indeg.items() if d == 0)
        seen = 0
        while queue:
            u = queue.popleft()
            seen += 1
            for v in self.out[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if seen != len(self.nodes):
            raise CheckError("cycle in the graph")

    def reachable(self, source: int) -> List[int]:
        seen, queue, order = {source}, deque([source]), []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.out[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return order

    def roots(self) -> List[int]:
        return [v for v in self.nodes if self.indeg[v] == 0]

    def branches(self, source: int) -> List[List[int]]:
        """Each out-edge of source followed to its end (requires non-branching paths)."""
        out = []
        for first in self.out[source]:
            br, u = [first], first
            while self.out[u]:
                if len(self.out[u]) > 1:
                    raise CheckError(f"node {u} branches inside a path of source {source}")
                u = self.out[u][0]
                br.append(u)
            out.append(br)
        return out


def expected_answer(fmt: str, branches: List[List[int]], target_branch: int,
                    source: int = -1) -> Tuple[List[int], List[int]]:
    """Answer tokens and roles, rebuilt independently of formats.py."""
    path = branches[target_branch]
    t = path[-1]
    if fmt == "copy":
        return [source], [FINAL]
    if fmt == "direct":
        return [t], [FINAL]
    if fmt == "path":
        return list(path), [INTERMEDIATE] * (len(path) - 1) + [FINAL]
    if fmt == "frontier":
        ids, roles = [], []
        for level in range(len(path)):
            ids.append(OPEN)
            roles.append(BRACKET)
            for br in branches:
                ids.append(br[level])
                roles.append(INTERMEDIATE)
            ids.append(CLOSE)
            roles.append(BRACKET)
        return ids + [t], roles + [FINAL]
    if fmt == "dfs":
        ids, roles = [], []
        for k, br in enumerate(branches):
            ids.extend(br)
            if k == target_branch:
                return ids, roles + [INTERMEDIATE] * (len(br) - 1) + [FINAL]
            ids.append(BACKTRACK)
            roles += [INTERMEDIATE] * len(br) + [BACKTRACK_ROLE]
    raise CheckError(f"unknown answer format {fmt!r}")


def analyze(tokens: Sequence[int], fmt: str, vocab: Vocab = VOCAB, num_targets: Optional[int] = None) -> dict:
    """Parse and check the graph and the answers. Returns what was derived; raises CheckError.

    Invariants: no self-loops, duplicate edges, merges or cycles; every component is a root whose
    branches all have the same depth and never branch (branching only at roots); at most one target
    per component, each target at the end of a branch; at least 2 targets (exactly num_targets if
    given). Components without a target are allowed."""
    p = parse(tokens, vocab)
    g = Graph(p.edges)
    g.check_no_merges()
    g.check_acyclic()
    target_set = set(p.targets)
    if len(target_set) != len(p.targets):
        raise CheckError("duplicate target in the target list")
    if len(p.targets) < 2:
        raise CheckError(f"only {len(p.targets)} target(s)")
    if num_targets is not None and len(p.targets) != num_targets:
        raise CheckError(f"{len(p.targets)} targets, expected {num_targets}")
    for t in p.targets:
        if t not in g.indeg or g.indeg[t] != 1:
            raise CheckError(f"target {vocab.tokens[t]} has no incoming path")
        if g.out[t]:
            raise CheckError(f"target {vocab.tokens[t]} is not the end of a path")
    # Every component: a root with equal-length non-branching paths and at most one target
    reached_by: Dict[int, int] = {}
    components = {}
    for r in g.roots():
        brs = g.branches(r)
        if len({len(b) for b in brs}) != 1:
            raise CheckError(f"source {vocab.tokens[r]}: paths of different lengths")
        ends = [b[-1] for b in brs if b[-1] in target_set]
        if len(ends) > 1:
            raise CheckError(f"source {vocab.tokens[r]} reaches {len(ends)} targets")
        for t in ends:
            reached_by[t] = reached_by.get(t, 0) + 1
        components[r] = brs
    for t in p.targets:
        if reached_by.get(t, 0) != 1:
            raise CheckError(f"target {vocab.tokens[t]} reached from {reached_by.get(t, 0)} sources")
    target_root = {}
    for r, brs in components.items():
        for b in brs:
            if b[-1] in target_set:
                target_root[b[-1]] = r
    # Queries
    queries = []
    sources = [q["source"] for q in p.queries]
    if len(set(sources)) != len(sources):
        raise CheckError("the same source is queried twice")
    for qi, q in enumerate(p.queries):
        s = q["source"]
        if s not in components:
            raise CheckError(f"query {qi}: {vocab.tokens[s]} is not a source")
        reach = [v for v in g.reachable(s) if v in target_set]
        if len(reach) != 1:
            raise CheckError(f"query {qi}: {len(reach)} targets reachable from {vocab.tokens[s]}")
        t = reach[0]
        path = [t]
        while g.parent[path[-1]] != s:
            path.append(g.parent[path[-1]])
        path = path[::-1]
        brs = sorted(components[s], key=lambda b: g.edge_index[(s, b[0])])
        tb = [b[-1] for b in brs].index(t)
        ids, roles = expected_answer(fmt, brs, tb, s)
        if q["answer"] != ids:
            raise CheckError(f"query {qi}: answer {vocab.decode(q['answer'])} != expected {vocab.decode(ids)} ({fmt})")
        queries.append({"source": s, "target": t, "path": path, "depth": len(path), "branching": len(brs),
                        "branch_order": [b[0] for b in brs], "prompt_pos": q["prompt_pos"],
                        "answer_start": q["answer_start"], "answer_end": q["answer_end"], "roles": roles,
                        "end_token": q["end_token"]})
    # Shapes (depth, branching) of the unqueried components with a target
    queried = set(sources)
    distractor_shapes = sorted((len(components[r][0]), len(components[r])) for t, r in target_root.items()
                               if r not in queried)
    return {"parsed": p, "graph": g, "queries": queries, "components": components,
            "distractor_shapes": distractor_shapes}


def check_edge_order(p: Parsed, g: Graph, edge_order: str) -> None:
    if edge_order == "random":
        return
    for k, (u, v) in enumerate(p.edges):
        for w in g.out[v]:
            later = g.edge_index[(v, w)]
            if edge_order == "reverse" and later > k:
                raise CheckError(f"reverse order: edge {u}>{v} appears before its successor")
            if edge_order == "forward" and later < k:
                raise CheckError(f"forward order: edge {u}>{v} appears after its successor")


def check_sample(sample, fmt: str, edge_order: Optional[str] = None, targets_position: Optional[str] = None,
                 num_edges: Optional[int] = None, num_targets: Optional[int] = None, vocab: Vocab = VOCAB,
                 syntax: Optional[str] = None) -> dict:
    """Full check of a sample (generate.Sample, or a dict with the same fields as arrays).

    Checks the graph and the answers from the tokens, then the metadata against them. Raises
    CheckError with the first problem found; returns the analysis otherwise. num_targets: the
    exact number of targets required (fixed mode); syntax: the edge and prompt syntax required
    (arrow | pair; the parser reads it from the edges)."""
    get = (lambda k: sample[k]) if isinstance(sample, dict) else (lambda k: getattr(sample, k))
    tokens = np.asarray(get("tokens"))
    a = analyze(tokens, fmt, vocab, num_targets=num_targets)
    p, g = a["parsed"], a["graph"]
    if edge_order is not None:
        check_edge_order(p, g, edge_order)
    if syntax is not None and p.syntax != syntax:
        raise CheckError(f"{p.syntax} syntax, expected {syntax}")
    if targets_position is not None and p.targets_position != targets_position:
        raise CheckError(f"target list {p.targets_position} the edges, expected {targets_position}")
    if num_edges is not None and len(p.edges) != num_edges:
        raise CheckError(f"{len(p.edges)} edges, expected {num_edges}")

    L = len(tokens)
    exp = {"loss_mask": np.zeros(L, bool), "query_index": np.full(L, -1), "depth": np.zeros(L, int),
           "branching": np.zeros(L, int), "role": np.full(L, OTHER), "pred_pos": np.full(L, -1)}
    for qi, q in enumerate(a["queries"]):
        lo, hi = q["prompt_pos"], q["answer_end"] + 1
        exp["query_index"][lo:hi] = qi
        exp["depth"][lo:hi] = q["depth"]
        exp["branching"][lo:hi] = q["branching"]
        exp["loss_mask"][q["answer_start"]:hi] = True
        exp["role"][q["answer_start"]:q["answer_end"]] = q["roles"]
        exp["role"][q["answer_end"]] = END
        final = q["answer_end"] - 1
        exp["pred_pos"][final] = final - 1
    for k, v in exp.items():
        got = np.asarray(get(k))
        if got.shape != (L,):
            raise CheckError(f"{k}: shape {got.shape}, expected ({L},)")
        bad = np.nonzero(got.astype(np.int64) != v.astype(np.int64))[0]
        if len(bad):
            raise CheckError(f"{k} differs at {len(bad)} position(s), first {bad[0]}: {got[bad[0]]} != {v[bad[0]]}")
    meta = get("queries")
    if len(meta) != len(a["queries"]):
        raise CheckError(f"{len(meta)} query records for {len(a['queries'])} queries")
    for qi, (m, q) in enumerate(zip(meta, a["queries"])):
        for k in ("source", "target", "path", "depth", "branching", "branch_order", "prompt_pos",
                  "answer_start", "answer_end"):
            mv = [int(x) for x in m[k]] if k in ("path", "branch_order") else int(m[k])
            if mv != q[k]:
                raise CheckError(f"query {qi}: metadata {k} = {mv}, graph gives {q[k]}")
    # Distractor shapes and trimming
    try:
        dist = get("distractors")
    except (KeyError, AttributeError):
        dist = None
    if dist is not None:
        shapes = sorted((int(x["depth"]), int(x["branching"])) for x in dist)
        if shapes != a["distractor_shapes"]:
            raise CheckError(f"distractor metadata {shapes} != shapes in the graph {a['distractor_shapes']}")
        removed = 0
        for x in dist:
            if not 1 <= int(x["branching"]) <= int(x["drawn_branching"]):
                raise CheckError(f"distractor branching {x['branching']} outside 1..drawn {x['drawn_branching']}")
            removed += int(x["drawn_branching"]) - int(x["branching"])
        if int(get("branches_removed")) != removed or bool(get("trimmed")) != (removed > 0):
            raise CheckError(f"trimming metadata: branches_removed {get('branches_removed')}, trimmed "
                             f"{get('trimmed')}, distractors give {removed}")
    return a
