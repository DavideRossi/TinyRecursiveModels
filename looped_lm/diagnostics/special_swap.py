"""Special-swap probe of a phase 1 (induction) checkpoint: which context predicts the token after a
special in a repeat?

A supervised token t after a special s, in a copy of "... x s t ...", can be found by matching the
name x alone (names are unique within a sample) or the pair x s. Only matching x transfers to the
graph task, where the answer after "Q: x ?" sits after "x >" in the edges. The probe makes phase 1
samples as in training, and in every repeat replaces each special token by another special (never
one already seen at that place of the segment, in the fresh segment or an earlier copy), keeping
the token after it. A model that matches x still predicts t; one that matches x s finds no match.

    python -m looped_lm.diagnostics.special_swap CKPT [CKPT ...] [--num 1000] [--seed 0]

prints, per checkpoint, the accuracy at the tokens after the swapped specials, and at the same
positions without the swap (the control). No training; CPU is fine.
"""
import argparse
import json
from typing import Callable, List

import numpy as np
import torch

from ..dataset_generation.vocab import BOS, PAD, VOCAB
from ..model import LoopedLM, ModelConfig
from .induction import SPECIAL_TOKENS, InductionConfig, make_sample

DEFAULT_CONFIG = "looped_lm/diagnostics/configs/induction.json"


def _special(tok) -> bool:
    return int(tok) != BOS and int(tok) != PAD and int(tok) < VOCAB.first_node


def swap_sample(s: dict, rng: np.random.Generator):
    """(swapped tokens, scored positions): every special of a repeat replaced by a special not seen
    before at the same (segment, offset); the scored positions are the supervised tokens right after
    a swapped special."""
    t, m = s["tokens"].copy(), s["loss_mask"]
    seen = {}  # (segment, offset) -> specials seen there so far
    pos = []
    for q in range(1, len(t)):
        key = (int(s["segment"][q]), int(s["offset"][q]))
        if m[q] and _special(t[q]):
            choices = [x for x in SPECIAL_TOKENS if x not in seen[key]]
            if choices:  # else every special was already seen here: leave it, and don't score
                t[q] = choices[int(rng.integers(len(choices)))]
                if q + 1 < len(t) and m[q + 1]:
                    pos.append(q + 1)
        seen.setdefault(key, set()).add(int(t[q]))
    return t, np.array(pos, dtype=np.int64)


def make_probe(cfg: InductionConfig, num: int, seed: int = 0) -> List[dict]:
    rng = np.random.default_rng(np.random.SeedSequence([4000, seed]))
    out = []
    for _ in range(num):
        s = make_sample(cfg, rng)
        swapped, pos = swap_sample(s, rng)
        if len(pos):
            out.append({"original": s["tokens"], "swapped": swapped, "pos": pos})
    return out


def score(predict: Callable, probe: List[dict], batch_size: int = 32) -> dict:
    """predict(tokens [B, L] long, positions list of arrays) -> list of arrays of predicted tokens
    (the prediction for the token at each position, from the positions before it)."""
    res = {}
    for kind in ("swapped", "original"):
        correct = n = 0
        for i in range(0, len(probe), batch_size):
            chunk = probe[i:i + batch_size]
            L = max(len(p[kind]) for p in chunk)
            toks = torch.full((len(chunk), L), PAD, dtype=torch.long)
            for j, p in enumerate(chunk):
                toks[j, :len(p[kind])] = torch.from_numpy(p[kind])
            preds = predict(toks, [p["pos"] for p in chunk])
            for p, pr in zip(chunk, preds):
                correct += int((np.asarray(pr) == p["original"][p["pos"]]).sum())
                n += len(p["pos"])
        res[kind] = correct / max(1, n)
        res["n"] = n
    return {"swapped_accuracy": res["swapped"], "control_accuracy": res["original"], "n": res["n"]}


def model_predictor(model: LoopedLM):
    @torch.no_grad()
    def predict(toks, positions):
        hidden = model(toks)
        out = []
        for j, pos in enumerate(positions):
            idx = torch.from_numpy(pos) - 1
            out.append(model.head(hidden[j, idx]).argmax(-1).numpy())
        return out
    return predict


def matcher_predictor(pair: bool):
    """Symbolic reference: at position p (a name at p - 2, a special at p - 1), find the most recent
    earlier occurrence of the name (pair: of the name followed by the same special) and predict the
    token two after it; -1 if there is none."""
    def predict(toks, positions):
        out = []
        for row, pos in zip(toks.numpy(), positions):
            preds = []
            for p in pos:
                hit = -1
                for q in range(p - 3, 0, -1):
                    if row[q] == row[p - 2] and (not pair or row[q + 1] == row[p - 1]):
                        hit = row[q + 2]
                        break
                preds.append(hit)
            out.append(np.array(preds))
        return out
    return predict


def load_model(path: str) -> LoopedLM:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = LoopedLM(ModelConfig.from_dict(saved["model_config"]))
    model.load_state_dict(saved["model"])
    return model.eval()


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m looped_lm.diagnostics.special_swap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--references", action="store_true", help="also score the two symbolic matchers")
    a = p.parse_args(argv)
    probe = make_probe(InductionConfig.load(a.config), a.num, a.seed)
    if a.references or not a.checkpoints:
        for name, pair in (("reference: match the name", False), ("reference: match name and special", True)):
            print(json.dumps({"checkpoint": name, **score(matcher_predictor(pair), probe)}), flush=True)
    torch.set_grad_enabled(False)
    for path in a.checkpoints:
        print(json.dumps({"checkpoint": path, **score(model_predictor(load_model(path)), probe)}), flush=True)


if __name__ == "__main__":
    main()
