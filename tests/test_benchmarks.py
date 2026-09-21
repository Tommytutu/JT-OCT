import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jt_oct import BENCHMARKS, load_benchmark


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmark_integrity_and_schema(name):
    root = Path(__file__).resolve().parents[1] / "datasets"
    metadata = {row["name"]: row for row in json.loads((root / "catalog.json").read_text())}[name]
    assert hashlib.sha256((root / metadata["file"]).read_bytes()).hexdigest() == metadata["sha256"]
    X, y, features, labels = load_benchmark(name)
    assert X.shape == (metadata["observations"], metadata["features"])
    assert X.dtype == np.uint8 and np.isin(X, [0, 1]).all()
    assert len(y) == len(X) and len(features) == X.shape[1]
    assert len(labels) == metadata["classes"]
    assert set(np.unique(y)) == set(labels)
