"""Print eval metrics from an offline wandb run, without syncing.

Usage (inside the container, from the repo root):
    apptainer exec /scratch.hpc/$USER/trm.sif python slurm/read_wandb_offline.py [RUN_DIR_OR_.wandb_FILE]

With no argument, the most recent wandb/offline-run-* directory is used.
"""
import glob
import json
import os
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore


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


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}/{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def history_rows(path):
    ds = DataStore()
    ds.open_for_scan(path)
    while True:
        try:
            data = ds.scan_data()
        except AssertionError:  # truncated tail while the run is still writing
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or "/".join(item.nested_key)
            flatten(key, json.loads(item.value_json), row)
        yield row


def main():
    path = find_wandb_file(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Reading {path}\n")
    found = False
    for row in history_rows(path):
        eval_keys = sorted(k for k in row if not k.startswith("train/") and not k.startswith("_") and k != "num_params")
        if not eval_keys:
            continue
        found = True
        print(f"step {row.get('_step', '?')}:")
        for k in eval_keys:
            print(f"    {k:40s} {row[k]}")
    if not found:
        print("No eval metrics logged yet (the first eval happens after eval_interval epochs).")


if __name__ == "__main__":
    main()
