"""Answer formats. A component's branches are given in canonical order (by the position of each
branch's first edge in the edge list) as lists of node token ids: branch[k] is the node at level
k + 1, and branch[-1] is its end (the target or a dead end)."""
from typing import List, Sequence, Tuple

from .vocab import BACKTRACK, CLOSE, OPEN

# Token roles (per-token metadata)
ROLE_OTHER, ROLE_FINAL, ROLE_INTERMEDIATE, ROLE_BRACKET, ROLE_BACKTRACK, ROLE_END = range(6)
ROLE_NAMES = ["other", "final", "intermediate", "bracket", "backtrack", "end"]


def answer_tokens(fmt: str, branches: Sequence[Sequence[int]], target_branch: int) -> Tuple[List[int], List[int]]:
    """(token ids, roles) of the answer to a query, without the end token (; or EOS)."""
    path = list(branches[target_branch])
    target = path[-1]
    if fmt == "direct":
        ids = [target]
        roles = [ROLE_FINAL]
    elif fmt == "path":
        ids = path
        roles = [ROLE_INTERMEDIATE] * (len(path) - 1) + [ROLE_FINAL]
    elif fmt == "frontier":
        ids, roles = [], []
        for level in range(len(path)):
            ids += [OPEN] + [b[level] for b in branches] + [CLOSE]
            roles += [ROLE_BRACKET] + [ROLE_INTERMEDIATE] * len(branches) + [ROLE_BRACKET]
        ids.append(target)
        roles.append(ROLE_FINAL)
    elif fmt == "dfs":
        ids, roles = [], []
        for k, b in enumerate(branches):
            if k == target_branch:
                ids += list(b)
                roles += [ROLE_INTERMEDIATE] * (len(b) - 1) + [ROLE_FINAL]
                break
            ids += list(b) + [BACKTRACK]
            roles += [ROLE_INTERMEDIATE] * len(b) + [ROLE_BACKTRACK]
    else:
        raise ValueError(f"unknown answer format {fmt!r}")
    return ids, roles


def max_answer_length(fmt: str, depth: int, branching: int) -> int:
    """Answer tokens (without the end token) in the worst case for one query."""
    return {"direct": 1, "path": depth, "frontier": depth * (branching + 2) + 1,
            "dfs": branching * depth + branching - 1}[fmt]
