"""Print eval metrics from an offline wandb run, without syncing.

Usage (inside the container, from the repo root):
    apptainer exec /scratch.hpc/$USER/trm.sif python slurm/read_wandb_offline.py [RUN_DIR_OR_.wandb_FILE]

With no argument, the most recent wandb/offline-run-* directory is used.

The .wandb file is a LevelDB-style record log of protobuf `Record`s. wandb no
longer ships a Python reader for it (it moved into the compiled wandb-core),
so the small reader below is self-contained; only the protobuf definitions
are taken from the installed wandb package.
"""
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
