"""Public interface for the three algorithms reported in the paper."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from .cg import CGOptions, solve_cg
from .domain import Domain
from .problem import Problem, Tree, evaluate, predict
from .solvers import solve_full, solve_jt_dp

METHODS = ("JT-LP", "JT-CG", "JT-MP")


def _gpu_available() -> bool:
    try:
        from .d3_batched import _configure_cupy_runtime
        _configure_cupy_runtime()
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _native(name: str) -> bool:
    return (Path(__file__).parent / "_native" / name).is_file()


def _backend(problem: Problem, requested: str) -> str:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
    if requested == "gpu":
        if not _gpu_available():
            raise RuntimeError("GPU execution requires CuPy and an available CUDA device")
        return "gpu"
    if requested == "cpu":
        return "cpp"
    return "gpu" if _gpu_available() and (problem.F >= 48 or problem.n >= 10_000) else "cpp"


def load_binary_csv(path: str | Path, label_column: int | str = 0):
    """Load a CSV with one label column and binary predictor columns."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if not rows:
        raise ValueError("The CSV contains no observations")
    label_index = header.index(label_column) if isinstance(label_column, str) else int(label_column)
    if not 0 <= label_index < len(header):
        raise ValueError("label_column is outside the CSV header")
    raw_labels = [row[label_index] for row in rows]
    classes = tuple(dict.fromkeys(raw_labels))
    encode = {label: index for index, label in enumerate(classes)}
    y = np.asarray([encode[label] for label in raw_labels], dtype=np.int32)
    feature_names = [name for index, name in enumerate(header) if index != label_index]
    try:
        X = np.asarray([[int(value) for index, value in enumerate(row) if index != label_index]
                        for row in rows], dtype=np.uint8)
    except ValueError as exc:
        raise ValueError("All predictors must be encoded as 0 or 1") from exc
    if not np.isin(X, [0, 1]).all():
        raise ValueError("All predictors must be encoded as 0 or 1")
    return X, y, feature_names, {index: label for label, index in encode.items()}


def make_problem(X, y, *, depth: int, penalty: float = 0.0,
                 no_repeat: bool = True, min_leaf: int = 0,
                 preparation_workers: int = 1) -> Problem:
    """Construct the paper's binary-feature OCT problem."""
    return Problem(X, y, depth=depth, penalty=penalty, early_stop=True,
                   no_repeat=no_repeat, min_leaf=min_leaf,
                   preparation_workers=preparation_workers)


def _paper_options(problem: Problem, backend: str, max_columns: int, *, message: bool):
    from .contract_cg import ContractOptions
    use_gpu = backend == "gpu"
    return ContractOptions(
        cache=True, warm_d3=True, quotient=True,
        oracle_batch=64 if use_gpu else 16,
        resident_gpu=use_gpu, columns_per_block=0, threads=8,
        max_columns=max_columns, exact_columns_only=True, warm_roots=1,
        bundle_pricing=False, message_bound=True, native_metadata=True,
        metadata_cache_entries=4096, gpu_sync_tiles=8,
        rmp_every_batches=(4 if problem.depth == 5 and len(problem.labels) == 2
                           and problem.F >= 100 and problem.n < 100_000 else 1),
        root_order="gain", cost_kernel="shared",
        class_bound=len(problem.labels) > 2,
        shallow_certificate=problem.penalty > 0,
        adaptive_admission=True, tail_depth=3,
        master_mode="message" if message else "cg", cost_mode="lazy")


def solve_jt_lp(problem: Problem, *, time_limit: float = 600,
                max_columns: int = 200_000, backend: str = "auto"):
    """Solve the reduced LP when available, otherwise the general exact LP."""
    selected = _backend(problem, backend)
    if problem.depth == 2 and _native("d2_cg.dll"):
        from .d2_cg import solve_d2_structural_lp
        result = solve_d2_structural_lp(problem, reduction="sc_ee",
            time_limit=time_limit, threads=8, cost_backend=selected)
        implementation = "native_reduced_lp_d2"
    elif problem.depth == 3 and _native("d3_structural.dll"):
        from .d3_structural_native import solve_d3_structural_native
        result = solve_d3_structural_native(problem, reduction="sc_ee",
            coordinator="lp", backend=selected, threads=8,
            time_limit=time_limit, max_columns=max_columns)
        implementation = "native_reduced_lp_d3"
    elif problem.depth in (4, 5) and _native("contract_rmp.dll") and _native("d3_optimized.dll"):
        from dataclasses import replace
        from .contract_cg import solve_contracted_cg
        options = replace(_paper_options(problem, selected, max_columns, message=False),
                          master_mode="full", cost_mode="eager")
        result = solve_contracted_cg(problem, selected, time_limit, options)
        implementation = "contracted_full_lp"
    else:
        result = solve_full(Domain(problem), time_limit=time_limit, max_columns=max_columns)
        implementation = "general_explicit_junction_tree_lp"
        selected = "cpu"
    if result.get("tree") is not None:
        result.setdefault("metrics", evaluate(problem, Tree.from_dict(result["tree"])))
    result.update(method="JT-LP", implementation=implementation, backend=selected,
                  model_depth=problem.depth)
    return result


def solve_jt_cg(problem: Problem, *, time_limit: float = 600,
                max_columns: int = 200_000, backend: str = "auto"):
    """Solve with the paper's column-generation method."""
    selected = _backend(problem, backend)
    if problem.depth == 2 and _native("d2_cg.dll"):
        from .d2_cg import solve_d2_cg
        result = solve_d2_cg(problem, time_limit=time_limit, max_columns=max_columns,
                             cost_backend="gpu" if selected == "gpu" else "cpp")
        implementation = "native_shallow_cg"
    elif problem.depth == 3 and _native("d3_cg.dll"):
        from .d3_cg import solve_d3_cg
        result = solve_d3_cg(problem, time_limit=time_limit, threads=8,
                             max_columns=max_columns,
                             cost_backend="gpu" if selected == "gpu" else "cpp")
        implementation = "native_shallow_cg"
    elif problem.depth in (4, 5) and _native("contract_rmp.dll") and _native("d3_optimized.dll"):
        from .contract_cg import solve_contracted_cg
        result = solve_contracted_cg(problem, selected, time_limit,
                                     _paper_options(problem, selected, max_columns, message=False))
        implementation = "contracted_paper_cg"
    else:
        result = solve_cg(Domain(problem), time_limit,
                          CGOptions(max_columns=max_columns))
        implementation = "general_path_cluster_column_generation"
        selected = "cpu"
    result.update(method="JT-CG", implementation=implementation, backend=selected,
                  model_depth=problem.depth)
    return result


def solve_jt_mp(problem: Problem, *, time_limit: float = 600,
                max_columns: int = 200_000, backend: str = "auto"):
    """Solve with exact min-sum message passing."""
    selected = _backend(problem, backend)
    if problem.depth in (4, 5) and _native("d3_optimized.dll"):
        from .contract_cg import solve_contracted_cg
        result = solve_contracted_cg(problem, selected, time_limit,
                                     _paper_options(problem, selected, max_columns, message=True))
        implementation = "adaptive_contracted_message_passing"
    elif problem.depth in (2, 3) and _native("d3_optimized.dll"):
        from .d3_optimized import solve_jt_dp_shallow_cpp, solve_jt_dp_shallow_gpu
        solver = solve_jt_dp_shallow_gpu if selected == "gpu" else solve_jt_dp_shallow_cpp
        result = solver(problem, time_limit=time_limit)
        implementation = "native_shallow_message_passing"
    else:
        result = solve_jt_dp(Domain(problem), time_limit=time_limit,
                             max_columns=max_columns)
        implementation = "general_streaming_junction_tree_message_passing"
        selected = "cpu"
    result.update(method="JT-MP", implementation=implementation, backend=selected,
                  model_depth=problem.depth)
    return result


def solve(problem: Problem, method: str = "JT-MP", **kwargs):
    """Run JT-LP, JT-CG, or JT-MP on a validated problem."""
    normalized = method.upper().replace("_", "-")
    functions = {"JT-LP": solve_jt_lp, "JT-CG": solve_jt_cg, "JT-MP": solve_jt_mp}
    if normalized not in functions:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    return functions[normalized](problem, **kwargs)


def named_tree(tree: Tree | dict | None, feature_names: Iterable[str],
               label_names: dict[int, str] | None = None):
    """Convert an integer-indexed result tree to a readable nested dictionary."""
    if tree is None:
        return None
    tree = Tree.from_dict(tree) if isinstance(tree, dict) else tree
    features = tuple(feature_names)
    labels = label_names or {}
    if tree.feature is None:
        return {"predict": labels.get(int(tree.label), int(tree.label))}
    return {"split": features[tree.feature],
            "zero": named_tree(tree.left, features, labels),
            "one": named_tree(tree.right, features, labels)}


__all__ = ["METHODS", "Problem", "Tree", "evaluate", "predict", "load_binary_csv",
           "make_problem", "solve", "solve_jt_lp", "solve_jt_cg", "solve_jt_mp",
           "named_tree"]
