"""Train the three paper methods on the bundled binary toy dataset."""
from pathlib import Path

from jt_oct import load_binary_csv, make_problem, named_tree, solve

HERE = Path(__file__).resolve().parent
X, y, features, labels = load_binary_csv(HERE / "toy_binary.csv", label_column=0)
problem = make_problem(X, y, depth=2, penalty=0.01)

for method in ("JT-LP", "JT-CG", "JT-MP"):
    result = solve(problem, method=method, time_limit=30, backend="cpu")
    print(f"{method}: {result['status']}, objective={result['UB']:.6f}, "
          f"time={result['seconds']:.3f}s")
    print(named_tree(result["tree"], features, labels))
