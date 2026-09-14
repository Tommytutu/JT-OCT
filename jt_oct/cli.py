"""Command-line interface for a binary CSV dataset."""
import argparse
import json
from pathlib import Path

from .api import load_binary_csv, make_problem, named_tree, solve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Solve an optimal classification tree")
    parser.add_argument("csv", help="CSV file with one label column and binary predictors")
    parser.add_argument("--label", default="0", help="label column name or zero-based index")
    parser.add_argument("--method", choices=["JT-LP", "JT-CG", "JT-MP"], default="JT-MP")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--min-leaf", type=int, default=0)
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--time-limit", type=float, default=600)
    parser.add_argument("--max-columns", type=int, default=200_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    label = int(args.label) if args.label.isdigit() else args.label
    X, y, features, labels = load_binary_csv(args.csv, label)
    problem = make_problem(X, y, depth=args.depth, penalty=args.penalty,
                           no_repeat=not args.allow_repeat, min_leaf=args.min_leaf)
    result = solve(problem, args.method, time_limit=args.time_limit,
                   max_columns=args.max_columns, backend=args.backend)
    result["readable_tree"] = named_tree(result.get("tree"), features, labels)
    result["feature_names"] = features
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(payload)


if __name__ == "__main__":
    main()
