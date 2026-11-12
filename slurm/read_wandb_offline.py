"""Print metrics from an offline wandb run, without syncing.

Usage (inside the container, from the repo root):
    apptainer exec /scratch.hpc/$USER/trm.sif python slurm/read_wandb_offline.py [RUN] [options]

RUN is a wandb/offline-run-* directory or a .wandb file; by default the most
recent wandb/offline-run-* directory is used. Works while the run is still
writing.

By default only evaluation metrics are printed. Options:
    --train         also print training metrics (train/*), as a table with one
                    row per window of training steps, each value being the mean
                    of the logged per-step values in that window. Metrics that
                    pretrain.py averages over the sequences halted in a step
                    (everything but losses, lr and count) are averaged only
                    over the steps in which some sequence halted.
    --window N      training steps per row (default 1000)
    --last K        only the last K rows of the training table
    --keys S [S...] only metrics whose name contains one of the substrings,
                    e.g. --keys halt  or  --keys exact_accuracy q_halt

Example, watching the halting heads of a running job:
    ... python slurm/read_wandb_offline.py --train --keys halt steps --last 10

The .wandb file is a LevelDB-style record log of protobuf `Record`s. wandb no
longer ships a Python reader for it (it moved into the compiled wandb-core),
so the small reader below is self-contained; only the protobuf definitions
are taken from the installed wandb package.
"""
import argparse
import glob
import json
import os
import struct
import sys

from wandb.proto import wandb_internal_pb2 as pb

HEADER_LEN = 7          # file header: "<4sHB" ident, magic, version; record header: "<IHB" crc, length, type
BLOCK_LEN = 32768
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4


def find_wandb_file(arg):
    if arg is None:
        runs = sorted(glob.glob("wandb/offline-run-*"), key=os.path.getmtime)
        if not runs:
            sys.exit("No wandb/offline-run-* directory found (run from the repo root).")
        arg = runs[-1]
    if os.path.isdir(arg):
        files = glob.glob(os.path.join(arg, "*.wandb"))
        if not files:
            sys.exit(f"No .wandb file in {arg}")
        arg = files[0]
    return arg


def read_records(path):
    """Yield raw protobuf payloads. Stops quietly at a truncated tail (run still writing)."""
    with open(path, "rb") as f:
        ident, _magic, _version = struct.unpack("<4sHB", f.read(HEADER_LEN))
        if ident != b":W&B":
            sys.exit(f"{path} is not a wandb transaction log")
        pos = HEADER_LEN
        pending = b""
        while True:
            space_left = BLOCK_LEN - pos % BLOCK_LEN
            if space_left < HEADER_LEN:  # block trailer padding
                f.read(space_left)
                pos += space_left
            header = f.read(HEADER_LEN)
            if len(header) < HEADER_LEN:
                return
            _crc, length, rtype = struct.unpack("<IHB", header)
            data = f.read(length)
            if len(data) < length:
                return
            pos += HEADER_LEN + length
            if rtype == FULL:
                yield data
            elif rtype == FIRST:
                pending = data
            elif rtype == MIDDLE:
                pending += data
            elif rtype == LAST:
                yield pending + data
                pending = b""
            else:
                return  # zero padding / unknown: treat as end of valid data


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}/{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def history_rows(path):
    for data in read_records(path):
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or "/".join(item.nested_key)
            flatten(key, json.loads(item.value_json), row)
        yield row


def wanted(key, substrings):
    return not substrings or any(sub in key for sub in substrings)


def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def per_halted_sequence(key):
    """pretrain.py divides these by the number of sequences halted in the step (0 if none)."""
    return not (key.endswith("loss") or key in ("train/lr", "train/count"))


def print_train_table(train_rows, keys_filter, window, last):
    """train_rows: list of (step, {key: value}) with all train/ keys. Prints per-window means."""
    keys = sorted({k for _, row in train_rows for k in row if wanted(k, keys_filter)})
    if not keys:
        print("No training metrics logged yet." if not train_rows else "No training metrics match --keys.")
        return

    buckets = {}  # bucket index -> {key: [values]}
    for step, row in train_rows:
        b = buckets.setdefault((step - 1) // window, {})
        any_halted = row.get("train/count", 1) > 0
        for k in keys:
            v = row.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if per_halted_sequence(k) and not any_halted:
                continue  # logged as 0 because nothing halted, not a real value
            b.setdefault(k, []).append(v)

    order = sorted(buckets)
    if last is not None:
        order = order[-last:]

    names = [k[len("train/"):] for k in keys]
    widths = [max(len(n), 10) for n in names]
    step_w = len(f"{order[-1] * window + 1}-{(order[-1] + 1) * window}")
    print(f"Training metrics, mean per {window} steps:")
    print("    " + "steps".ljust(step_w) + "  " + "  ".join(n.rjust(w) for n, w in zip(names, widths)))
    for b in order:
        steps = f"{b * window + 1}-{(b + 1) * window}"
        cells = []
        for k, w in zip(keys, widths):
            vals = buckets[b].get(k)
            cells.append((fmt(sum(vals) / len(vals)) if vals else "-").rjust(w))
        print("    " + steps.ljust(step_w) + "  " + "  ".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Print metrics from an offline wandb run.")
    parser.add_argument("run", nargs="?", help="wandb/offline-run-* directory or .wandb file (default: newest)")
    parser.add_argument("--train", action="store_true", help="also print training metrics, averaged per window")
    parser.add_argument("--window", type=int, default=1000, help="training steps per table row (default 1000)")
    parser.add_argument("--last", type=int, default=None, help="only the last K rows of the training table")
    parser.add_argument("--keys", nargs="+", default=[], help="only metrics whose name contains one of these")
    args = parser.parse_args()

    path = find_wandb_file(args.run)
    print(f"Reading {path}\n")
    found = False
    train_rows = []
    for row in history_rows(path):
        step = row.get("_step", "?")
        if args.train:
            train = {k: v for k, v in row.items() if k.startswith("train/")}
            if train and isinstance(step, int):
                train_rows.append((step, train))

        eval_keys = sorted(k for k in row if not k.startswith("train/") and not k.startswith("_") and k != "num_params" and wanted(k, args.keys))
        if not eval_keys:
            continue
        found = True
        print(f"step {step}:")
        for k in eval_keys:
            print(f"    {k:40s} {row[k]}")
    if not found:
        print("No eval metrics logged yet (the first eval happens after eval_interval epochs).")

    if args.train:
        print()
        print_train_table(train_rows, args.keys, args.window, args.last)


if __name__ == "__main__":
    main()
