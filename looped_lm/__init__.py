"""Looped causal transformer on synthetic graph reachability (see looped_lm/README.md).

    dataset_generation/   the data: generator, checker, audit, suites, PyTorch datasets
    model/                the model family (ModelConfig, LoopedLM)
    train.py, evaluate.py training and evaluation (python -m looped_lm.train / looped_lm.evaluate)
    diagnostics/          MQAR and induction pretraining, checks of the training stack
"""
