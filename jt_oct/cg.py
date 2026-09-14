"""Certified column generation with sparse union rows and optional discovery aids."""
from dataclasses import dataclass
from itertools import islice
import heapq
import math
import time
import numpy as np
from .problem import Tree, Deadline, DeadlineExceeded, evaluate
from .domain import SPLIT, Domain, CapacityExceeded
from .master import solve_rmp, unpack_dual, reduced_cost, chain_path, RestrictedMaster
from .solvers import result_dict
from .conflicts import (RoutedConflictOracle, conflict_feature_order, conflict_lower_bound,
                        feature_relaxation_bound)


@dataclass
class CGOptions:
    sparse_rows: bool = True
    signature_cache: bool = True
    columns_per_block: int = 200
    prefix_bound: bool = True
    early_return: bool = True
    heuristic_first: bool = False
    heuristic_limit: int = 24
    compatible_bundles: bool = False
    stabilize: bool = False
    static_conflict_lb: bool = True
    conflict_ordering: bool = False
    feature_relaxation_lb: bool = False
    conflict_tail_bound: bool = False
    tolerance: float = 1e-8
    max_columns: int = 200000
    max_iterations: int = 100000
    d3_table_backend: str | None = None
    d3_table_policy: str = "best"
    d3_table_bundles: bool = False
    d3_table_threads: int = 0
    incremental_matrix: bool = False
    rmp_method: int = 1
    adaptive_admission: bool = False


def greedy_feasible(p, deadline):
    """Deterministic feasible initializer, without calling exact tree optimization."""
    def rec(node, rows, used):
        deadline.check()
        label, stop_loss = p.best_label_and_loss(rows)
        stop = Tree(label=label) if len(rows) >= p.min_leaf else None
        if len(node) == p.depth:
            return stop
        choices = []
        for f in p.features(node, used):
            children = [p.route(rows, f, b) for b in (0, 1)]
            if any(len(rs) < p.min_leaf for rs in children):
                continue
            score = p.cost(node, f) + sum(p.best_label_and_loss(rs)[1] for rs in children)
            choices.append((score, f, children))
        # For complete trees splitting is mandatory; for sparse trees one-step
        # non-improvement returns a stop. Exact CG subsequently escapes XOR traps.
        if p.early_stop and stop is not None and (not choices or min(choices)[0] >= stop_loss):
            return stop
        for _, f, children in sorted(choices):
            l = rec(node+(0,), children[0], used+(f,))
            r = rec(node+(1,), children[1], used+(f,))
            if l is not None and r is not None:
                return Tree(feature=f, left=l, right=r)
        return stop if p.early_stop else None
    return rec((), p.all_rows, ())


def prefix_lower_bound(domain, i, rho, allocated_cost, alpha, pi):
    """Safe for split/stop/inactive actions: all unallocated primal cost >= 0.

    Resolved separators are exact; unresolved ones take min(0, compatible dual
    contributions). Unlike the split-only paper bound, no compulsory future
    split costs are charged when stopping is possible.
    """
    value = allocated_cost - alpha[i]
    for e, size, sign in domain.incident[i]:
        if size <= len(rho):
            value -= sign * pi.get((e, rho[:size]), 0.0)
        else:
            value += min([0.0] + [-sign*v for (ee, sigma), v in pi.items()
                                  if ee == e and sigma[:len(rho)] == rho])
    return float(value)


def price(domain, i, alpha, pi, options, deadline, conflict_oracle=None):
    """Return valid min(0,r_q) lower bound and up to k negative columns.

    A timeout raises without returning a certificate. Pruned prefixes all have
    nonnegative lower bounds. Heuristic column costs never enter LB.
    """
    minimum, candidates, order = 0.0, [], 0
    def visit(rho, cost, rows, used):
        tail_bound = (conflict_oracle.bound(rows)
                      if conflict_oracle is not None and len(rho) == domain.h else 0.0)
        return prefix_lower_bound(domain, i, rho, cost + tail_bound, alpha, pi) < 0
    visit = visit if options.prefix_bound else None
    if options.signature_cache:
        it = domain.representatives(i, deadline, visit)
    else:
        def generator():
            for state in domain.prefix_states(i, deadline, visit):
                yield from domain.from_state(i, state, deadline)
        it = generator()
    for c in it:
        deadline.check()
        rc = reduced_cost(domain, i, c, alpha, pi)
        minimum = min(minimum, rc)
        if rc < -options.tolerance:
            # The heap root is the least negative retained reduced cost, so a
            # better candidate replaces it in O(log k) instead of sorting k
            # candidates after every negative column.
            item = (-rc, order, c)
            order += 1
            if len(candidates) < options.columns_per_block:
                heapq.heappush(candidates, item)
            elif rc < -candidates[0][0]:
                heapq.heapreplace(candidates, item)
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
    return minimum, [c for _, _, c in ranked]


def discover(domain, i, alpha, pi, options, deadline, existing, conflict_oracle=None):
    """Stream omitted negative columns and stop after k; never fake a bound.

    If the iterator is exhausted, ``minimum`` is an exact pricing bound.  If it
    stops early, the returned columns are valid improving columns but the bound
    is deliberately marked incomplete.
    """
    minimum, found = 0.0, []
    def visit(rho, cost, rows, used):
        tail_bound = (conflict_oracle.bound(rows)
                      if conflict_oracle is not None and len(rho) == domain.h else 0.0)
        return prefix_lower_bound(domain, i, rho, cost + tail_bound, alpha, pi) < 0
    visit = visit if options.prefix_bound else None
    if options.signature_cache:
        iterator = domain.representatives(i, deadline, visit)
    else:
        def generator():
            for state in domain.prefix_states(i, deadline, visit):
                yield from domain.from_state(i, state, deadline)
        iterator = generator()
    for column in iterator:
        deadline.check()
        rc = reduced_cost(domain, i, column, alpha, pi)
        minimum = min(minimum, rc)
        if column not in existing and rc < -options.tolerance:
            found.append(column)
            if len(found) >= options.columns_per_block:
                return minimum, found, False
    return minimum, found, True


def solve_cg(domain, time_limit=600, options=None, initial_tree=None, progress=None, master_factory=None):
    options = options or CGOptions()
    if options.columns_per_block < 1 or options.tolerance <= 0:
        raise ValueError("Positive number of columns and pricing tolerance required")
    if domain.explicit_labels:
        raise ValueError("Uncoupled CG uses optimized private labels; use full IP for global coupling")
    clock, p = Deadline(time_limit), domain.p
    original_feature_order = p._feature_order
    conflict_oracle = None
    best_tree, best_ub, lb = None, math.inf, 0.0
    trace, pool = [], []
    rmp_seconds, pricing_seconds, completion_seconds = 0.0, 0.0, 0.0
    pricing_setup_seconds = 0.0
    previous_dual, universe = None, None
    initial_fallback = False
    last_certificate = None
    master = None
    try:
        tree = initial_tree or greedy_feasible(p, clock)
        if tree is not None:
            evaluate(p, tree)
            cols = domain.tree_columns(tree)
        if tree is None or not all(domain.permitted(i, c) for i, c in enumerate(cols)):
            initial_fallback = True
            universe = domain.materialize(clock, options.max_columns)
            _, cols = chain_path(domain, universe)
            if cols is None:
                return result_dict("JT-CG", clock, "INFEASIBLE", lb=math.inf)
            tree = domain.recover(cols)
        initial_metric = evaluate(p, tree)
        best_tree, best_ub = tree, initial_metric["objective"]
        conflict_stats = {}
        if options.static_conflict_lb or options.feature_relaxation_lb or options.conflict_tail_bound:
            started = time.perf_counter()
            conflict_oracle = RoutedConflictOracle(p) if options.conflict_tail_bound else None
            static_bound = (conflict_oracle.global_bound if conflict_oracle is not None
                            else conflict_lower_bound(p))
            if options.static_conflict_lb or options.feature_relaxation_lb:
                lb = max(lb, static_bound)
            conflict_stats.update({"static_global_LB": static_bound,
                                   "conflict_groups": (len(conflict_oracle.groups)
                                                       if conflict_oracle is not None else None),
                                   "static_LB_enabled": bool(options.static_conflict_lb or
                                                             options.feature_relaxation_lb),
                                   "tail_bound_enabled": bool(options.conflict_tail_bound),
                                   "oracle_seconds": time.perf_counter()-started})
        pool_features = set(initial_metric["features"])
        if options.conflict_ordering or options.feature_relaxation_lb:
            started = time.perf_counter()
            order, profile = conflict_feature_order(p, initial_metric["features"])
            conflict_stats.update({
                "incumbent_features": list(profile.features),
                "restricted_feature_LB": profile.lower_bound,
                "restricted_groups": profile.groups,
                "positive_conflict_scores": int(sum(v > 1e-15 for v in profile.scores.values())),
                "top_conflict_scores": [[int(f), float(profile.scores[f])] for f in
                                        sorted(profile.scores, key=lambda f: (-profile.scores[f], f))[:20]],
                "score_seconds": time.perf_counter()-started,
                "pool_history": [{"iteration": -1, "features": len(profile.features),
                                  "restricted_LB": profile.lower_bound,
                                  "positive_scores": int(sum(v > 1e-15
                                                             for v in profile.scores.values()))}]})
            if options.conflict_ordering:
                p.set_feature_order(order)
                conflict_stats["feature_order"] = list(order)
            if options.feature_relaxation_lb:
                relaxation = feature_relaxation_bound(
                    p, profile.features, 2**p.depth-1, binary=False,
                    time_limit=min(60, clock.remaining()))
                lb = max(lb, relaxation["best_bound"])
                conflict_stats["feature_relaxation"] = relaxation
        if conflict_stats:
            domain.stats["conflict_strategy"] = conflict_stats
        clock.check()
        if progress:
            progress(result_dict("JT-CG",clock,"RUNNING",best_tree,p,lb))
        pool = [[c] for c in cols]
        pool_sets = [set(columns) for columns in pool]
        table_pricer = None
        if options.d3_table_backend is not None:
            from .d3_pricing import D3PricingTable
            start = time.perf_counter()
            table_pricer = D3PricingTable(
                domain, backend=options.d3_table_backend,
                policy=options.d3_table_policy,
                compatible_bundles=options.d3_table_bundles,
                threads=options.d3_table_threads)
            pricing_setup_seconds = time.perf_counter() - start
            table_pricer.activate_pool(pool)
            domain.stats["d3_table_pricing"] = table_pricer.metadata
        if not options.sparse_rows and universe is None:
            universe = domain.materialize(clock, options.max_columns)
        master = (master_factory or RestrictedMaster)(domain, None if options.sparse_rows else universe,
                                  method=options.rmp_method)
        for iteration in range(options.max_iterations):
            clock.check()
            start = time.perf_counter()
            res, matrix = (master.solve_incremental(pool, clock) if options.incremental_matrix
                           else master.solve(pool, clock))
            rmp_seconds += time.perf_counter()-start
            if res is None or res.status != 0:
                if res is not None and res.status == 1:
                    raise DeadlineExceeded("RMP time limit")
                raise AssertionError("Feasible restricted face unexpectedly infeasible or numerically failed")
            A, b, c, flat, keys = matrix
            positive = [[] for _ in range(domain.M)]
            for j, (i, col) in enumerate(flat):
                if res.x[j] > 1e-7:
                    positive[i].append(col)
            value, selected = chain_path(domain, positive)
            if selected is None or abs(value-res.fun) > 1e-7:
                raise AssertionError("Cannot glue optimal RMP support")
            candidate = domain.recover(selected)
            metric = evaluate(p, candidate)
            if abs(metric["objective"]-res.fun) > 1e-7:
                raise AssertionError("RMP objective disagrees with routed tree")
            if metric["objective"] < best_ub + 1e-12:
                best_tree, best_ub = candidate, metric["objective"]
            alpha, pi = unpack_dual(domain, res, keys)
            # Each omitted separator is assigned dual zero by reduced_cost().
            # At any current dual a conservative valid bound can be computed
            # from nonnegative column costs, without completing pricing.
            cheap = [prefix_lower_bound(domain, i, (), 0, alpha, pi) for i in range(domain.M)]
            lb = max(lb, float(sum(alpha)+sum(min(0, x) for x in cheap)))
            seeds, exact, rmins = [], False, None
            start = time.perf_counter()
            try:
                if table_pricer is not None:
                    rmins, seeds = table_pricer.price(
                        alpha, pi, options.columns_per_block, options.tolerance)
                    exact = True
                elif options.early_return:
                    complete, provisional = True, []
                    for i in range(domain.M):
                        rmin, new, block_complete = discover(
                            domain, i, alpha, pi, options, clock, pool_sets[i],
                            conflict_oracle if options.conflict_tail_bound else None)
                        provisional.append(rmin)
                        complete = complete and block_complete
                        seeds.extend((i, col) for col in new)
                    if complete:
                        exact, rmins = True, provisional
                if not seeds and not exact and options.heuristic_first:
                    guide_a, guide_pi = alpha, pi
                    if options.stabilize and previous_dual is not None:
                        old_a, old_pi = previous_dual
                        guide_a = 0.5*(alpha+old_a)
                        guide_pi = {k: 0.5*(pi.get(k, 0)+old_pi.get(k, 0)) for k in pi.keys() | old_pi.keys()}
                    for i in range(domain.M):
                        it = domain.representatives(i, clock) if options.signature_cache else domain.columns(i, clock)
                        ranked = sorted(islice(it, options.heuristic_limit),
                                        key=lambda col: reduced_cost(domain, i, col, guide_a, guide_pi))
                        for col in ranked[:options.columns_per_block]:
                            if col not in pool_sets[i] and reduced_cost(domain, i, col, alpha, pi) < -options.tolerance:
                                seeds.append((i, col))
                if not seeds and not exact:
                    exact, rmins = True, []
                    for i in range(domain.M):
                        rmin, new = price(domain, i, alpha, pi, options, clock,
                                          conflict_oracle if options.conflict_tail_bound else None)
                        rmins.append(rmin)
                        seeds.extend((i, col) for col in new if col not in pool_sets[i])
                if exact:
                    lb = max(lb, float(sum(alpha)+sum(min(0, x) for x in rmins)))
                    last_certificate = {"alpha_sum": float(sum(alpha)), "pricing_lower_bounds": rmins,
                                        "computed_LB": float(sum(alpha)+sum(min(0, x) for x in rmins)),
                                        "max_equality_residual": (
                                            res.max_equality_residual if A is None else
                                            float(np.max(np.abs(A@res.x-b)))),
                                        "minimum_active_reduced_cost": (
                                            res.minimum_active_reduced_cost if A is None else
                                            float(np.min(c-A.T@res.eqlin.marginals))),
                                        "inactive_row_duals": 0.0}
            finally:
                # Preserve partial pricing time when a deadline interrupts a sweep.
                pricing_seconds += time.perf_counter()-start
            previous_dual = alpha.copy(), pi.copy()
            trace.append({"iteration": iteration, "seconds": clock.elapsed(), "RMP": float(res.fun),
                          "LB": lb, "UB": best_ub, "columns": len(flat),
                          "rows": len(keys) if A is None else A.shape[0],
                          "exact_pricing": exact, "pricing_bounds": rmins, "new_columns": len(seeds)})
            if progress:
                progress(result_dict("JT-CG",clock,"RUNNING",best_tree,p,lb,trace=trace))
            if lb > best_ub+1e-7:
                raise AssertionError("Certified lower bound exceeds routed incumbent")
            if exact and min(rmins) >= -options.tolerance:
                return result_dict("JT-CG", clock, "OPT", best_tree, p, min(lb, best_ub),
                                   trace=trace, certificate=last_certificate, stats=domain.stats,
                                   rmp_seconds=rmp_seconds, pricing_seconds=pricing_seconds,
                                   pricing_setup_seconds=pricing_setup_seconds,
                                   completion_seconds=completion_seconds, initialization_fallback=initial_fallback)
            if not seeds:
                raise AssertionError("Negative pricing bound without an omitted negative column")
            if options.compatible_bundles:
                start = time.perf_counter()
                if universe is None:
                    universe = domain.materialize(clock, options.max_columns)
                bundles = []
                for i, seed in seeds:
                    fixed = [list(cs) if j != i else [seed] for j, cs in enumerate(universe)]
                    _, completion = chain_path(domain, fixed)
                    if completion is not None:
                        bundles.extend(enumerate(completion))
                seeds += bundles
                completion_seconds += time.perf_counter()-start
            if options.adaptive_admission:
                room=options.max_columns-sum(map(len,pool))
                if room<=0:
                    keep=domain.tree_columns(best_tree)
                    if len(keep)>=options.max_columns:raise CapacityExceeded('Capacity cannot fit a feasible bundle plus an entering column')
                    removed=sum(map(len,pool))-len(keep)
                    pool=[[c] for c in keep];pool_sets=[set(cs) for cs in pool]
                    master.close()
                    master=(master_factory or RestrictedMaster)(domain,method=options.rmp_method)
                    domain.stats['capacity_evicted_columns']=domain.stats.get('capacity_evicted_columns',0)+removed
                    # Full prefix domain remains available; reprice against fresh duals.
                    continue
                unique={ (i,col):(i,col) for i,col in seeds if col not in pool_sets[i] }
                ordered=sorted(unique.values(),key=lambda pair:reduced_cost(domain,*pair,alpha,pi))
                budget=min(room,len(ordered))
                if sum(map(len,pool))>=.8*options.max_columns:budget=min(budget,max(domain.M,room//2))
                seeds=ordered[:budget]
                domain.stats['capacity_deferred_columns']=domain.stats.get('capacity_deferred_columns',0)+len(ordered)-len(seeds)
            for i, col in seeds:
                if col not in pool_sets[i]:
                    pool[i].append(col)
                    pool_sets[i].add(col)
                    if table_pricer is not None:
                        table_pricer.activate(i, col)
                    if options.conflict_ordering:
                        pool_features.update(domain.split_marginals(i, col).values())
            if options.conflict_ordering and len(pool_features) != conflict_stats["pool_history"][-1]["features"]:
                started = time.perf_counter()
                order, profile = conflict_feature_order(p, sorted(pool_features))
                p.set_feature_order(order)
                conflict_stats["score_seconds"] += time.perf_counter()-started
                conflict_stats["pool_history"].append({
                    "iteration": iteration, "features": len(profile.features),
                    "restricted_LB": profile.lower_bound,
                    "positive_scores": int(sum(v > 1e-15 for v in profile.scores.values()))})
            if sum(map(len, pool)) > options.max_columns:
                raise CapacityExceeded("Restricted column pool limit exceeded")
        status = "ITERATION_LIMIT"
    except (DeadlineExceeded, CapacityExceeded) as exc:
        status = "TIME" if isinstance(exc, DeadlineExceeded) else "RESOURCE"
    finally:
        if master is not None:
            master.close()
        p.set_feature_order(original_feature_order)
    return result_dict("JT-CG", clock, status, best_tree, p, min(lb, best_ub),
                       trace=trace, certificate=last_certificate, stats=domain.stats,
                       rmp_seconds=rmp_seconds, pricing_seconds=pricing_seconds,
                       pricing_setup_seconds=pricing_setup_seconds,
                       completion_seconds=completion_seconds)
