"""Evaluate a checkpoint on the stored evaluation suite (teacher-forced final-answer accuracy).

    python -m looped_lm.evaluate --checkpoint RUN_DIR_OR_FILE [--suite looped_lm/data/eval_suite]
                                 [--only grid/ mixed/] [--skip companion/ diag/] [--num N] [--out DIR]

RUN_DIR evaluates RUN_DIR/ckpt_latest.pt; a file evaluates that checkpoint (ckpt_latest.pt or
model_step_N.pt). Writes <out>/<checkpoint name>_step<N>.json and .csv (default out:
RUN_DIR/eval). Per set: accuracy with a 95% Wilson interval, number of answers, chance (mean
1 / #targets), mean loss on final answers; the same for each sample's first query only
(first_query_*, free of the elimination heuristic); for sets with several queries, by (d, b);
for sets mixing depths or branchings, first queries by depth and by branching (JSON only).
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

from .evaluation import evaluate_set, find_sets, load_set
from .model import LoopedLM, ModelConfig

CSV_FIELDS = ("set", "depth", "branching", "n", "correct", "accuracy", "ci_low", "ci_high", "chance", "loss",
              "num_samples", "tokens_mean", "first_query_n", "first_query_accuracy", "first_query_ci_low",
              "first_query_ci_high", "first_query_chance")


def load_checkpoint(path: str, device) -> tuple:
    if os.path.isdir(path):
        path = os.path.join(path, "ckpt_latest.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LoopedLM(ModelConfig.from_dict(ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, path


def csv_rows(results: dict):
    for name, r in results.items():
        yield {"set": name, "depth": "", "branching": "", **{k: r.get(k, "") for k in CSV_FIELDS[3:]}}
        for key, s in r.get("by_db", {}).items():
            yield {"set": name, "depth": s["depth"], "branching": s["branching"],
                   **{k: s.get(k, "") for k in CSV_FIELDS[3:10]}, **{k: "" for k in CSV_FIELDS[10:]}}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m looped_lm.evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", default="looped_lm/data/eval_suite")
    p.add_argument("--only", nargs="+", help="sets whose name starts with one of these prefixes")
    p.add_argument("--skip", nargs="+", help="leave out sets whose name starts with one of these, e.g. companion/ diag/")
    p.add_argument("--num", type=int, help="the first N samples of each set (default: all)")
    p.add_argument("--max_tokens", type=int, default=131072, help="padded tokens per batch")
    p.add_argument("--out", help="output directory (default: the run directory's eval/)")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    model, ckpt, path = load_checkpoint(a.checkpoint, device)
    names = find_sets(a.suite, a.only, a.skip)
    if not names:
        sys.exit(f"no sets under {a.suite} (with the given --only / --skip)")
    results = {}
    for name in names:
        t0 = time.time()
        r = evaluate_set(model, load_set(os.path.join(a.suite, name), a.num, a.max_tokens), device)
        results[name] = r
        print(f"{name:28s} acc {r['accuracy']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  n {r['n']:6d}  "
              f"chance {r['chance']:.4f}  loss {r['loss']:.4f}  first query {r['first_query_accuracy']:.4f}  "
              f"({time.time() - t0:.1f} s)", flush=True)

    out_dir = a.out or os.path.join(os.path.dirname(path), "eval")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{os.path.splitext(os.path.basename(path))[0]}_step{ckpt['step']}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump({"checkpoint": os.path.abspath(path), "step": ckpt["step"], "suite": os.path.abspath(a.suite),
                   "num": a.num, "model_config": ckpt["model_config"], "train_config": ckpt["train_config"],
                   "sets": results}, f, indent=1)
    with open(os.path.join(out_dir, stem + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_rows(results))
    print(f"wrote {os.path.join(out_dir, stem)}.json and .csv")


if __name__ == "__main__":
    main()
