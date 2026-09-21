"""Load the binary benchmark matrices distributed with the repository."""
from pathlib import Path

import numpy as np


BENCHMARKS = ("avila", "banknote", "compas", "diabetic", "fico", "give",
              "htru2", "letter", "skin", "spambase", "transactions")


def load_benchmark(name: str, directory: str | Path | None = None):
    """Return X, y, feature names, and integer-to-original-label mapping."""
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {', '.join(BENCHMARKS)}")
    root = Path(directory) if directory is not None else Path(__file__).resolve().parents[1] / "datasets"
    path = root / f"{name}.npz"
    with np.load(path, allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        features = data["feature_names"].tolist()
        labels = dict(enumerate(data["label_names"].tolist()))
    return X, y, features, labels
