"""Command line: write the vocabulary and fixed evaluation sets, inspect, audit and check.

    python -m looped_lm.dataset_generation.cli vocab  OUT_DIR
    python -m looped_lm.dataset_generation.cli show   --config CFG.json [--override key=JSON ...] [--num 3]
    python -m looped_lm.dataset_generation.cli stats  (--config CFG.json | --suite SUITE.json) [--num 2000] [--batch-size B]
    python -m looped_lm.dataset_generation.cli audit  (--config CFG.json | --suite SUITE.json) [--num 2000]
    python -m looped_lm.dataset_generation.cli write  --suite SUITE.json [--out DIR] [--num N] [--check] [--skip PREFIX ...]
                                                      [--only PREFIX ...] [--answer_format FMT]
    python -m looped_lm.dataset_generation.cli check  DIR

See looped_lm/dataset_generation/README.md.
"""
import argparse
import json
import os
import shutil
import sys
import time

from . import suite as suite_mod
from .audit import BACKWARD, EXCLUSION, HEURISTICS, audit, stats, summarize, z_threshold
from .checker import CheckError, check_sample
from .config import ANSWER_FORMATS, TaskConfig
from .generate import make_sample
from .storage import StoredSet, write_set
from .vocab import VOCAB


# Default output of `write`: looped_lm/data/<suite name> (ignored by git, as every data/ directory)
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def _config(args) -> TaskConfig:
    with open(args.config) as f:
        d = json.load(f)
    for ov in args.override or []:
        k, _, v = ov.partition("=")
        d[k] = json.loads(v)
    return TaskConfig.from_dict(d)


def _sets(args):
    """[(name, config, stream)] from --config or --suite (with --only / --skip to select sets by prefix)."""
    if args.config:
        cfg = _config(args)
        return [(cfg.name, cfg, 0)]
    specs = _specs(args.suite, args.only, args.skip)
    return [(s.name, s.config, s.stream) for s in specs]


def _specs(path, only=None, skip=None):
    specs = suite_mod.expand(suite_mod.load(path))
    if only:
        specs = [s for s in specs if any(s.name.startswith(p) for p in only)]
    if skip:
        specs = [s for s in specs if not any(s.name.startswith(p) for p in skip)]
    return specs


def cmd_vocab(args):
    os.makedirs(args.out, exist_ok=True)
    VOCAB.save(os.path.join(args.out, "vocab.json"))
    print(f"wrote {os.path.join(args.out, 'vocab.json')} ({VOCAB.size} tokens)")


def cmd_show(args):
    for name, cfg, stream in _sets(args)[:1]:
        for i in range(args.num):
            s = make_sample(cfg, args.seed, i, split="debug", stream=stream)
            print(f"--- {name} sample {i}: {len(s)} tokens, {s.num_edges} edges, {s.num_targets} targets")
            print(s.text())
            for q in s.queries:
                print(f"    query: {VOCAB.tokens[q['source']]} -> {VOCAB.tokens[q['target']]}  d={q['depth']} "
                      f"b={q['branching']}  path {' '.join(VOCAB.decode(q['path']))}")


def cmd_stats(args):
    batched = args.batch_size is not None
    if batched:
        print(f"batches of {args.batch_size} as the training stream makes them (one edge count per batch); "
              f"padding = mean over batches of 1 - sum(lengths) / (B * longest)")
    print("trimmed = share of samples with a trimmed distractor; trim dist = share of distractors trimmed")
    print(f"{'set':28s} {'tokens mean':>11s} {'min':>5s} {'max':>6s} {'edges mean':>10s} {'max':>5s} "
          f"{'targets mean':>12s} {'min':>4s} {'nodes max':>9s} {'answer mean':>11s} {'max':>5s} "
          f"{'trimmed':>8s} {'trim dist':>9s}" + (f" {'padding':>8s}" if batched else ""))
    for name, cfg, stream in _sets(args):
        r = stats(cfg, args.num, seed=args.seed, stream=stream, batch_size=args.batch_size)
        print(f"{name:28s} {r['tokens']['mean']:11.1f} {r['tokens']['min']:5d} {r['tokens']['max']:6d} "
              f"{r['edges']['mean']:10.1f} {r['edges']['max']:5d} {r['targets']['mean']:12.1f} "
              f"{r['targets']['min']:4d} {r['nodes']['max']:9d} {r['answer_tokens']['mean']:11.1f} "
              f"{r['answer_tokens']['max']:5d} {r['trimmed']['mean']:8.3f} {r['trimmed_distractors']['mean']:9.3f}"
              + (f" {r['padding']['mean']:8.3f}" if batched else ""))


def _targets_str(t) -> str:
    return str(t["min"]) if t["min"] == t["max"] else f"{t['min']}-{t['max']} (mean {t['mean']:.1f})"


def _fmt(v, ceiling, width: int) -> str:
    """accuracy, chance, [ceiling], z, queries; ceiling: True (shown), False (blank), None (no column)."""
    if v["queries"] == 0:
        return f"{'(no queries)':>{width}s}"
    c = f" {v['ceiling']:8.4f}" if ceiling else (" " * 9 if ceiling is False else "")
    return f"{v['accuracy']:9.4f} {v['chance']:8.4f}{c} {v['z']:6.1f} {v['queries']:6d}"


def _print_audit(title, r, info):
    print(f"== {title}   targets: {_targets_str(info)}   trimmed samples: {info['trimmed_share']:.3f}")
    print(f"   {'heuristic':14s} {'all: accuracy':>13s} {'chance':>8s} {'ceiling':>8s} {'z':>6s} {'queries':>6s} "
          f"{'flag':12s} | {'untrimmed: acc':>14s} {'chance':>8s} {'z':>6s} {'queries':>6s} {'flag':12s}")
    for h, v in r.items():
        u = v["untrimmed"]
        print(f"   {h:14s} {_fmt(v, h in EXCLUSION, 41)} {v['flag']:12s} | {_fmt(u, None, 32)} {u['flag']:12s}"
              + ("   (backward search, not a shortcut)" if h in BACKWARD else ""))


def cmd_audit(args):
    sets = _sets(args)
    grid = [s for s in sets if s[0].startswith("grid/")]
    pooled = len(grid) > 1
    z = z_threshold(2 * (len(sets) + pooled) * len(HEURISTICS))
    print(f"{len(sets)} set(s){' + the pooled grid' if pooled else ''} x {len(HEURISTICS)} heuristics x "
          f"(all, untrimmed): flagged when |z| > {z:.2f}")
    results, pool, pool_info = [], {}, []
    for name, cfg, stream in sets:
        r, sums, info = audit(cfg, args.num, seed=args.seed, stream=stream, z_flag=z, return_sums=True)
        results.append((name, r, info))
        _print_audit(name, r, info)
        if pooled and name.startswith("grid/"):
            for h, by_subset in sums.items():
                for subset, v in by_subset.items():
                    acc = pool.setdefault(h, {}).setdefault(subset, dict.fromkeys(v, 0))
                    for k in v:
                        acc[k] += v[k]
            pool_info.append(info)
    if pooled:
        r = {}
        for h, by_subset in pool.items():
            r[h] = summarize(by_subset["all"], z)
            r[h]["untrimmed"] = summarize(by_subset["untrimmed"], z)
        t = {"min": min(x["min"] for x in pool_info), "max": max(x["max"] for x in pool_info),
             "mean": sum(x["mean"] for x in pool_info) / len(pool_info),
             "trimmed_share": sum(x["trimmed_share"] for x in pool_info) / len(pool_info)}
        results.append((f"all {len(grid)} grid sets, pooled", r, t))
        _print_audit(results[-1][0], r, t)
    # Exclusion heuristics at a glance: one row per set, with its number of targets
    cols = list(EXCLUSION)
    print(f"\n{'set':34s} {'targets':>18s} {'trimmed':>8s} "
          + " ".join(f"{c + ' acc/chance/ceiling':>30s} {'z':>6s}" for c in cols))

    def cell(v):
        if v["queries"] == 0:
            return f"{'- (no queries with d >= 2)':>30s} {'':>6s}"
        return f"{v['accuracy']:10.4f} {v['chance']:9.4f} {v['ceiling']:9.4f} {v['z']:6.1f}"
    for name, r, t in results:
        print(f"{name:34s} {_targets_str(t):>18s} {t['trimmed_share']:8.3f} " + " ".join(cell(r[c]) for c in cols))
    flagged = sum(v["flag"] == "ABOVE CHANCE" and h not in BACKWARD
                  for _, r, _ in results for h, v in r.items())
    flagged += sum(v["untrimmed"]["flag"] == "ABOVE CHANCE" and h not in BACKWARD and info["trimmed_share"] > 0
                   for _, r, info in results for h, v in r.items())  # without trimming, the subsets coincide
    print(f"\n{flagged} flag(s) above chance (all samples and untrimmed subset), {' and '.join(sorted(BACKWARD))} "
          f"excluded: they cost a backward search from every target (README, \"Backward search\")")


def cmd_write(args):
    suite = suite_mod.load(args.suite)
    specs = _specs(args.suite, args.only, args.skip)
    if args.answer_format:  # the same graphs, with answers in another format, in their own directory
        specs = [s._replace(config=s.config.replace(answer_format=args.answer_format)) for s in specs]
    if args.out is None:
        args.out = os.path.join(DATA_DIR, os.path.splitext(os.path.basename(args.suite))[0]
                                + (f"_{args.answer_format}" if args.answer_format else ""))
    print(f"writing to {args.out}" + (f" (only {', '.join(args.only)})" if args.only else "")
          + (f" (skipping {', '.join(args.skip)})" if args.skip else ""))
    os.makedirs(args.out, exist_ok=True)
    VOCAB.save(os.path.join(args.out, "vocab.json"))
    shutil.copy(args.suite, os.path.join(args.out, "suite.json"))
    readable = int(suite.get("readable_samples", 5))
    for s in specs:
        n = args.num if args.num is not None else s.num_samples
        t0 = time.time()
        meta = write_set(os.path.join(args.out, s.name), s.config, n, s.seed, s.stream, split="eval",
                         readable=readable)
        print(f"{s.name:28s} {n:7d} samples, tokens mean {meta['tokens_mean']:.1f} max {meta['tokens_max']}, "
              f"trimmed {meta['trimmed_share']:.3f} ({time.time() - t0:.1f} s)")
    if args.check:
        _check_dir(args.out)


def _check_dir(root: str, limit=None) -> int:
    dirs = sorted(dp for dp, _, files in os.walk(root) if "meta.json" in files)
    if not dirs:
        sys.exit(f"no sets (meta.json) under {root}")
    failures = 0
    for d in dirs:
        st = StoredSet(d)
        cfg = st.config
        n = len(st) if limit is None else min(limit, len(st))
        bad = 0
        for i in range(n):
            rec = st.get(i)
            try:
                check_sample(rec, cfg.answer_format, cfg.edge_order, cfg.targets_position,
                             num_edges=rec["num_edges"], num_targets=rec["num_targets"], syntax=cfg.syntax)
                if cfg.fixed and rec["num_targets"] not in cfg.num_targets.support:
                    raise CheckError(f"{rec['num_targets']} targets, outside num_targets {cfg.num_targets.spec}")
            except CheckError as e:
                bad += 1
                if bad <= 3:
                    print(f"   {os.path.relpath(d, root)} sample {i}: {e}")
        failures += bad
        print(f"{os.path.relpath(d, root):28s} {n:7d} checked, {bad} failed")
    print("all samples valid" if failures == 0 else f"{failures} invalid samples")
    return failures


def cmd_check(args):
    if _check_dir(args.dir, args.num):
        sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m looped_lm.dataset_generation.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def source(sp, suite=True):
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--config", help="TaskConfig JSON")
        if suite:
            g.add_argument("--suite", help="evaluation suite JSON")
            sp.add_argument("--only", nargs="+", help="with --suite: sets whose name starts with one of these")
            sp.add_argument("--skip", nargs="+", help="with --suite: leave out the sets whose name starts with one "
                            "of these, e.g. companion/ diag/")
        else:
            sp.set_defaults(suite=None, only=None, skip=None)
        sp.add_argument("--override", nargs="+", metavar="KEY=JSON", help="with --config: override fields")
        sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("vocab", help="write vocab.json")
    sp.add_argument("out")
    sp.set_defaults(func=cmd_vocab)
    sp = sub.add_parser("show", help="print readable samples (debug split)")
    source(sp)
    sp.add_argument("--num", type=int, default=3)
    sp.set_defaults(func=cmd_show)
    sp = sub.add_parser("stats", help="length statistics per configuration")
    source(sp)
    sp.add_argument("--num", type=int, default=2000)
    sp.add_argument("--batch-size", type=int, help="make batches as the training stream does and report "
                    "the mean padding fraction per batch")
    sp.set_defaults(func=cmd_stats)
    sp = sub.add_parser("audit", help="shortcut audit per configuration")
    source(sp)
    sp.add_argument("--num", type=int, default=2000)
    sp.set_defaults(func=cmd_audit)
    sp = sub.add_parser("write", help="write the fixed sets of a suite")
    sp.add_argument("--suite", required=True)
    sp.add_argument("--out", help="output directory (default: looped_lm/data/<suite file name without .json>, "
                    "with _<answer_format> appended when --answer_format is given)")
    sp.add_argument("--answer_format", choices=ANSWER_FORMATS,
                    help="override every set's answer format (the graphs stay the same)")
    sp.add_argument("--num", type=int, help="samples per set (overrides the suite)")
    sp.add_argument("--check", action="store_true", help="run the checker on every written sample")
    sp.add_argument("--only", nargs="+", help="only the sets whose name starts with one of these, e.g. cfg/")
    sp.add_argument("--skip", nargs="+", help="leave out the sets whose name starts with one of these, "
                    "e.g. companion/ diag/")
    sp.set_defaults(func=cmd_write)
    sp = sub.add_parser("check", help="run the checker on written sets")
    sp.add_argument("dir")
    sp.add_argument("--num", type=int, help="samples per set (default: all)")
    sp.set_defaults(func=cmd_check)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
