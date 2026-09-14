"""Solve six-bit parity with a tree that must use all six levels."""
from itertools import product

import numpy as np

from jt_oct import make_problem, solve

X = np.asarray(list(product((0, 1), repeat=6)), dtype=np.uint8)
y = np.bitwise_xor.reduce(X, axis=1).astype(np.int32)
problem = make_problem(X, y, depth=6, penalty=0.0)

for method in ("JT-LP", "JT-CG", "JT-MP"):
    result = solve(problem, method=method, time_limit=30, backend="cpu")
    print(f"{method}: {result['status']}, objective={result['UB']:.6f}, "
          f"splits={result['metrics']['split_nodes']}, "
          f"implementation={result['implementation']}")
