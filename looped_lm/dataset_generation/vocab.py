"""The shared vocabulary: special tokens, then one token per node name.

Every set (training stream, evaluation sets) uses the same vocabulary of NODE_POOL node names, so
token ids mean the same thing everywhere. A configuration may draw names from a smaller pool (the
first pool_size names), for debugging only.
"""
import hashlib
import json
from typing import Iterable, List, Sequence

PAD, BOS, EOS, ARROW, TARGETS, QUERY, ASK, OPEN, CLOSE, BACKTRACK, SEP = range(11)
SPECIALS = ["<pad>", "<bos>", "<eos>", ">", "T:", "Q:", "?", "[", "]", "<bt>", ";"]
NODE_POOL = 2048


class Vocab:
    def __init__(self, node_pool: int = NODE_POOL):
        if node_pool < 1:
            raise ValueError(f"node_pool must be positive, got {node_pool}")
        self.node_pool = int(node_pool)
        self.tokens: List[str] = SPECIALS + [f"n{i}" for i in range(self.node_pool)]
        self.ids = {t: i for i, t in enumerate(self.tokens)}

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def first_node(self) -> int:
        return len(SPECIALS)

    def node(self, name_index: int) -> int:
        """Token id of node name n<name_index>."""
        if not 0 <= name_index < self.node_pool:
            raise ValueError(f"node name index {name_index} outside the pool of {self.node_pool}")
        return self.first_node + int(name_index)

    def is_node(self, token: int) -> bool:
        return self.first_node <= int(token) < self.size

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.tokens[int(i)] for i in ids]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.ids[t] for t in tokens]

    def to_text(self, ids: Iterable[int], pretty: bool = True) -> str:
        """Readable string. pretty: one line for the edges, the targets and each query."""
        words = self.decode(ids)
        if not pretty:
            return " ".join(words)
        out, line = [], []
        for w in words:
            if w in ("T:", "Q:") and line:
                out.append(" ".join(line))
                line = []
            line.append(w)
            if w in ("<bos>",):
                out.append(" ".join(line))
                line = []
        if line:
            out.append(" ".join(line))
        return "\n".join(out)

    def to_json(self) -> dict:
        return {"node_pool": self.node_pool, "specials": SPECIALS, "tokens": self.tokens}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.tokens).encode()).hexdigest()[:16]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            data = json.load(f)
        vocab = cls(data["node_pool"])
        if data["tokens"] != vocab.tokens:
            raise ValueError(f"{path} does not match the vocabulary of this code version")
        return vocab


VOCAB = Vocab()
