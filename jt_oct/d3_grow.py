"""Fast feasible trees obtained by repeatedly growing a frozen prefix with D3.

The heuristic first solves a depth-three tree on the full sample.  Its root is
kept and depth-three subproblems are solved below the two root arcs to obtain a
D4 tree.  For D5, the first two split levels of that D4 tree are kept and each
existing depth-two frontier is replaced by an exact D3 subtree.

Every intermediate tree is feasible.  Replacing a frontier cannot increase the
objective because the previous (at most depth-two) subtree is feasible for the
new depth-three subproblem.  The frozen prefix makes this a heuristic for D4/D5;
only the individual D3 subproblems are solved exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time

from .d3_optimized import D3Options, D3Workspace
from .problem import Deadline, DeadlineExceeded, Tree, evaluate
from .solvers import result_dict


@dataclass(frozen=True)
class D3GrowOptions:
    d3: D3Options = field(default_factory=lambda: D3Options(fast_prepare=True))
    frontier_order: str = "natural"
    root_beam: int = 1


def _subtree(tree, path):
    current = tree
    for branch in path:
        if current.feature is None:
            return None
        current = current.left if branch == 0 else current.right
    return current


def _replace(tree, path, replacement):
    if not path:
        return replacement
    if tree.feature is None:
        raise ValueError("Cannot replace below a stopped prefix")
    branch, rest = path[0], path[1:]
    if branch == 0:
        return Tree(feature=tree.feature, left=_replace(tree.left, rest, replacement),
                    right=tree.right)
    return Tree(feature=tree.feature, left=tree.left,
                right=_replace(tree.right, rest, replacement))


def _prefix_state(p, tree, path):
    rows, used, current = p.all_rows, (), tree
    for branch in path:
        if current.feature is None:
            return None
        feature = current.feature
        rows = p.route(rows, feature, branch)
        used += (feature,)
        current = current.left if branch == 0 else current.right
    return rows, used


def _frontier_paths(tree, depth):
    paths = []

    def visit(current, path):
        if len(path) == depth:
            paths.append(path)
            return
        if current.feature is None:
            return
        visit(current.left, path + (0,))
        visit(current.right, path + (1,))

    visit(tree, ())
    return paths


def solve_d3_grow(p, backend="gpu", time_limit=100, threads=0, options=None,
                  progress=None):
    """Construct a D4/D5 incumbent by successively grafting exact D3 subtrees."""
    if p.depth not in (4, 5) or p._uniform_weight is None:
        raise ValueError("D3-grow requires depth 4/5 and uniform weights")
    if backend not in ("cpp", "gpu"):
        raise ValueError("D3-grow backend must be cpp or gpu")
    opts = options or D3GrowOptions()
    if opts.root_beam != 1:
        return solve_d3_grow_beam(p, backend, time_limit, threads, opts, progress)
    if opts.frontier_order not in ("natural", "largest"):
        raise ValueError("D3-grow frontier order must be natural or largest")

    clock = Deadline(time_limit)
    name = "D3-Grow-" + backend.upper()
    label = p.best_label(p.all_rows)
    current = Tree(label=label)
    best = evaluate(p, current)["objective"]
    trace = []
    stats = {"backend": backend, "threads": int(threads) if backend == "cpp" else None,
             "options": {"d3": asdict(opts.d3), "frontier_order": opts.frontier_order},
             "d3_calls": 0, "d3_optimal_calls": 0, "d3_seconds": 0.0,
             "frontiers_completed": 0, "objective_improvements": 0,
             "constructed_depth": 0}
    engine = None

    def publish(stage, path, candidate, oracle_status):
        nonlocal current, best
        metrics = evaluate(p, candidate)
        value = metrics["objective"]
        if value > best + 1e-9:
            raise AssertionError("An exact D3 graft increased the incumbent objective")
        improved = value < best - 1e-12
        current, best = candidate, value
        stats["objective_improvements"] += int(improved)
        record = {"stage": stage, "path": list(path), "seconds": clock.elapsed(),
                  "objective": value, "misclassified": metrics["misclassified"],
                  "split_nodes": metrics["split_nodes"], "realized_depth": metrics["realized_depth"],
                  "improved": improved, "d3_status": oracle_status}
        trace.append(record)
        if progress is not None:
            progress(result_dict(name, clock, "RUNNING", current, p, 0.0,
                                 stats=dict(stats), incumbent_trace=list(trace),
                                 optimality_scope="feasible_tree_only"))

    def solve_frontier(path, stage):
        nonlocal current
        state = _prefix_state(p, current, path)
        if state is None:
            return True
        rows, used = state
        before = _subtree(current, path)
        tick = time.perf_counter()
        out = engine.solve(rows=rows, node=path, used=used,
                           time_limit=clock.remaining())
        stats["d3_seconds"] += time.perf_counter() - tick
        stats["d3_calls"] += 1
        stats["d3_optimal_calls"] += int(out["status"] == "OPT")
        if out.get("tree") is not None:
            candidate = _replace(current, path, out["tree"])
            # A timed-out oracle may only return STOP, which can be worse than
            # the existing subtree.  Keep the better feasible tree in that case.
            if evaluate(p, candidate)["objective"] <= best + 1e-12:
                publish(stage, path, candidate, out["status"])
            elif out["status"] == "OPT":
                raise AssertionError("Optimal D3 frontier is worse than its old subtree")
        stats["frontiers_completed"] += int(out["status"] == "OPT")
        return out["status"] == "OPT"

    complete = True
    try:
        engine = D3Workspace(p, backend, threads, opts.d3)
        complete = solve_frontier((), "D3-seed")
        stats["constructed_depth"] = 3 if complete else 0
        if not complete:
            raise DeadlineExceeded("Initial D3 seed did not finish")

        root = _subtree(current, ())
        d4_paths = [(0,), (1,)] if root.feature is not None else []
        if opts.frontier_order == "largest":
            d4_paths.sort(key=lambda path: -len(_prefix_state(p, current, path)[0]))
        for path in d4_paths:
            if not solve_frontier(path, "D4-frontier"):
                complete = False
                raise DeadlineExceeded("D4 frontier did not finish")
        stats["constructed_depth"] = 4
        trace.append({"stage": "D4-complete", "path": [], "seconds": clock.elapsed(),
                      "objective": best, **{k: evaluate(p, current)[k]
                      for k in ("misclassified", "split_nodes", "realized_depth")}})

        if p.depth == 5:
            paths = _frontier_paths(current, 2)
            if opts.frontier_order == "largest":
                paths.sort(key=lambda path: -len(_prefix_state(p, current, path)[0]))
            for path in paths:
                if not solve_frontier(path, "D5-frontier"):
                    complete = False
                    raise DeadlineExceeded("D5 frontier did not finish")
            stats["constructed_depth"] = 5
            trace.append({"stage": "D5-complete", "path": [], "seconds": clock.elapsed(),
                          "objective": best, **{k: evaluate(p, current)[k]
                          for k in ("misclassified", "split_nodes", "realized_depth")}})
    except DeadlineExceeded:
        complete = False
    finally:
        if engine is not None:
            engine.close()

    actual = evaluate(p, current)
    if abs(actual["objective"] - best) > 1e-9:
        raise AssertionError("D3-grow objective disagrees with independent routing")
    return result_dict(name, clock, "FEASIBLE" if complete else "TIME", current, p, 0.0,
                       stats=stats, incumbent_trace=trace,
                       construction="D3 seed; D3 below depth-1 frontiers; D3 below depth-2 frontiers",
                       optimality_scope="feasible_tree_only")


def solve_d3_grow_beam(p, backend="gpu", time_limit=100, threads=0, options=None,
                       progress=None):
    """Evaluate a small stump-ranked beam of frozen roots with D3 growth."""
    if p.depth not in (4, 5) or p._uniform_weight is None:
        raise ValueError("D3-grow requires depth 4/5 and uniform weights")
    if backend not in ("cpp", "gpu"):
        raise ValueError("D3-grow backend must be cpp or gpu")
    opts = options or D3GrowOptions(root_beam=32)
    if opts.frontier_order not in ("natural", "largest") or opts.root_beam < 2:
        raise ValueError("Beam D3-grow requires a valid order and root_beam >= 2")
    clock = Deadline(time_limit)
    name = f"D3-Grow-Beam{opts.root_beam}-{backend.upper()}"
    current = Tree(label=p.best_label(p.all_rows))
    best = evaluate(p, current)["objective"]
    trace = []
    stats = {"backend": backend, "threads": int(threads) if backend == "cpp" else None,
             "options": {"d3": asdict(opts.d3), "frontier_order": opts.frontier_order,
                         "root_beam": opts.root_beam},
             "d3_calls": 0, "d3_optimal_calls": 0, "d3_seconds": 0.0,
             "roots_started": 0, "roots_completed": 0,
             "objective_improvements": 0, "constructed_depth": 0}
    engine = None
    complete = True

    def call(rows, node, used):
        tick = time.perf_counter()
        out = engine.solve(rows=rows, node=node, used=used,
                           time_limit=clock.remaining())
        stats["d3_seconds"] += time.perf_counter() - tick
        stats["d3_calls"] += 1
        stats["d3_optimal_calls"] += int(out["status"] == "OPT")
        return out

    def record(stage, root, candidate, rank, candidate_complete=True):
        nonlocal current, best
        metrics = evaluate(p, candidate)
        value = metrics["objective"]
        improved = value < best - 1e-12
        if improved:
            current, best = candidate, value
            stats["objective_improvements"] += 1
        item = {"stage": stage, "root": root, "root_rank": rank,
                "seconds": clock.elapsed(), "objective": best,
                "candidate_objective": value, "misclassified":
                evaluate(p, current)["misclassified"], "candidate_misclassified":
                metrics["misclassified"], "split_nodes":
                evaluate(p, current)["split_nodes"], "improved": improved,
                "candidate_complete": candidate_complete}
        trace.append(item)
        if progress is not None:
            progress(result_dict(name, clock, "RUNNING", current, p, 0.0,
                                 stats=dict(stats), incumbent_trace=list(trace),
                                 optimality_scope="feasible_tree_only"))

    try:
        engine = D3Workspace(p, backend, threads, opts.d3)
        seed = call(p.all_rows, (), ())
        if seed.get("tree") is not None:
            record("D3-seed", seed["tree"].feature, seed["tree"], 0,
                   seed["status"] == "OPT")
        if seed["status"] != "OPT":
            raise DeadlineExceeded("Initial D3 seed did not finish")
        seed_root = seed["tree"].feature
        ranked = []
        for feature in p.features((), ()):
            children = [p.route(p.all_rows, feature, branch) for branch in (0, 1)]
            if any(len(rows) < p.min_leaf for rows in children):
                continue
            score = p.cost((), feature) + sum(p.best_label_and_loss(rows)[1]
                                               for rows in children)
            ranked.append((score, feature))
        ranked.sort()
        roots = [] if seed_root is None else [seed_root]
        roots.extend(feature for _, feature in ranked if feature != seed_root)
        roots = roots[:min(opts.root_beam, len(roots))]
        stats["root_candidates"] = roots

        for rank, root in enumerate(roots, 1):
            clock.check()
            stats["roots_started"] += 1
            children_rows = [p.route(p.all_rows, root, branch) for branch in (0, 1)]
            candidate = Tree(feature=root,
                             left=Tree(label=p.best_label(children_rows[0])),
                             right=Tree(label=p.best_label(children_rows[1])))
            ok = True
            d4_paths = [(0,), (1,)]
            if opts.frontier_order == "largest":
                d4_paths.sort(key=lambda path: -len(_prefix_state(p, candidate, path)[0]))
            for path in d4_paths:
                rows, used = _prefix_state(p, candidate, path)
                out = call(rows, path, used)
                if out.get("tree") is not None:
                    candidate = _replace(candidate, path, out["tree"])
                if out["status"] != "OPT":
                    ok = False
                    break
            record("beam-D4-root", root, candidate, rank, ok)
            if not ok:
                raise DeadlineExceeded("D4 beam root did not finish")
            if p.depth == 5:
                paths = _frontier_paths(candidate, 2)
                if opts.frontier_order == "largest":
                    paths.sort(key=lambda path: -len(_prefix_state(p, candidate, path)[0]))
                for path in paths:
                    rows, used = _prefix_state(p, candidate, path)
                    out = call(rows, path, used)
                    if out.get("tree") is not None:
                        candidate = _replace(candidate, path, out["tree"])
                    if out["status"] != "OPT":
                        ok = False
                        break
                record("beam-D5-root", root, candidate, rank, ok)
                if not ok:
                    raise DeadlineExceeded("D5 beam root did not finish")
            stats["roots_completed"] += 1
        stats["constructed_depth"] = p.depth
    except DeadlineExceeded:
        complete = False
    finally:
        if engine is not None:
            engine.close()

    actual = evaluate(p, current)
    if abs(actual["objective"] - best) > 1e-9:
        raise AssertionError("Beam D3-grow objective disagrees with independent routing")
    return result_dict(name, clock, "FEASIBLE" if complete else "TIME", current, p, 0.0,
                       stats=stats, incumbent_trace=trace,
                       construction="stump-ranked root beam; D3 below depth-1 and depth-2 frontiers",
                       optimality_scope="feasible_tree_only")


def solve_d3_grow_message(p, backend="gpu", time_limit=100, threads=0,
                          grow_options=None, message_options=None, progress=None):
    """Use D3-grow as an incumbent initializer for exact JT message passing."""
    from .terminal_message import PRESETS, solve_terminal_message

    started = time.perf_counter()
    chosen = grow_options or D3GrowOptions()
    heuristic = solve_d3_grow(p, backend, time_limit, threads, chosen, progress)
    elapsed = time.perf_counter() - started
    suffix = "grow" if chosen.root_beam == 1 else f"grow{chosen.root_beam}"
    method = f"JT-MP-D3-{backend.upper()}-balanced-{suffix}"
    if elapsed >= time_limit or heuristic.get("tree") is None:
        heuristic["method"] = method
        heuristic["heuristic_only_due_to_deadline"] = True
        return heuristic

    initial = Tree.from_dict(heuristic["tree"])
    out = solve_terminal_message(p, backend, max(.001, time_limit - elapsed), threads,
                                 message_options or PRESETS["balanced"], progress=progress,
                                 initial_tree=initial)
    for record in out.get("root_trace", []):
        record["seconds"] += elapsed
    for record in out.get("message_trace", []):
        record["seconds"] += elapsed
    terminal_incumbents = list(out.get("incumbent_trace", []))
    for record in terminal_incumbents:
        record["seconds"] += elapsed
    combined = list(heuristic.get("incumbent_trace", []))
    combined.extend({"stage": "JT-MP-" + record["source"], "path": [], **record}
                    for record in terminal_incumbents
                    if record["source"] not in ("majority_stop", "d3_grow"))
    out["method"] = method
    out["seconds"] = time.perf_counter() - started
    out["incumbent_trace"] = combined
    out["d3_grow"] = {"seconds": heuristic["seconds"], "status": heuristic["status"],
                      "UB": heuristic["UB"], "metrics": heuristic["metrics"],
                      "stats": heuristic["stats"],
                      "incumbent_trace": heuristic.get("incumbent_trace", [])}
    out["stats"]["d3_grow_seconds"] = heuristic["seconds"]
    out["stats"]["d3_grow_UB"] = heuristic["UB"]
    out["stats"]["d3_grow_misclassified"] = heuristic["metrics"]["misclassified"]
    out["stats"]["d3_grow_split_nodes"] = heuristic["metrics"]["split_nodes"]
    return out
