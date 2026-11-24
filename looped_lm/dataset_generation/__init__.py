"""Synthetic graph-reachability data for a looped causal transformer (see README.md in this directory).

Main entry points:
    TaskConfig, Dist                  configuration (config.py)
    make_sample, Sample               one sample, a pure function of (config, seed, index, split, stream)
    TrainStream                       infinite on-the-fly training stream (IterableDataset)
    GeneratedDataset, StoredDataset   map-style datasets (generated on access / written to disk)
    collate                           padding collate function
    check_sample, CheckError          independent checker
    VOCAB, Vocab                      the shared vocabulary
"""
from .checker import CheckError, check_sample
from .config import Dist, TaskConfig
from .dataset import GeneratedDataset, StoredDataset, TrainStream, collate
from .generate import Sample, make_sample
from .vocab import VOCAB, Vocab
