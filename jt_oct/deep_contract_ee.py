"""D6/D7 exact endpoint elimination using the native generic path core.

Python owns only oracle dispatch and result marshaling. Endpoint aggregation,
separator compatibility, path recovery, RMP solves and reduced-cost evaluation
are performed by ``deep_structural_core.dll``.
"""
from collections import defaultdict
import math
import time

import numpy as np

from .deep_structural_native import DeepMaster, eliminate_endpoints, path_opt
from .domain import CapacityExceeded
from .problem import DeadlineExceeded, evaluate
from .solvers import result_dict


def _lift(state, retained):
    retained = [int(s) for s in retained]
    ids = [retained[0]]
    ids.extend((q + 1) * state.P + s for q, s in enumerate(retained))
    ids.append((state.M - 1) * state.P + retained[-1])
    return ids


def _tree(state, retained):
    ids = _lift(state, retained)
    return state.domain.recover([state.column(i) for i in ids]), ids


def _coordinate(costs, state, clock, options, stats, full=False):
    B, P = costs.shape
    active = np.zeros((B, P), dtype=np.uint8)
    tick = time.perf_counter()
    model = DeepMaster(B, state.F, state.K, state.domain.h, first_block=1,
        memory_limit_bytes=options.memory_limit_bytes)
    stats["rmp_setup_seconds"] += time.perf_counter() - tick
    try:
        if full:
            ids = np.flatnonzero(np.isfinite(costs)).astype(np.int32)
            if ids.size > options.max_columns:
                raise CapacityExceeded("Deep EE full-column capacity")
            tick = time.perf_counter()
            model.update(ids, costs.ravel()[ids]); active.flat[ids] = 1
            stats["rmp_update_seconds"] += time.perf_counter() - tick
        else:
            _, seed = path_opt(costs, state.F, state.K, state.domain.h, first_block=1)
            if seed is None:
                return math.inf, None
            ids = np.asarray([q * P + int(s) for q, s in enumerate(seed)], dtype=np.int32)
            model.update(ids, costs.ravel()[ids]); active.flat[ids] = 1
        while True:
            clock.check(); tick = time.perf_counter(); answer = model.solve(clock.remaining())
            stats["rmp_seconds"] += time.perf_counter() - tick; stats["iterations"] += 1
            if answer["status"] in (9, 11):
                raise DeadlineExceeded("Deep EE RMP timeout")
            if answer["status"] != 2:
                raise RuntimeError(f"Deep EE RMP status {answer['status']}")
            stats["peak_active_columns"] = max(stats["peak_active_columns"], answer["columns"])
            stats["peak_rmp_nonzeros"] = max(stats["peak_rmp_nonzeros"], answer["nonzeros"])
            stats["final_columns"], stats["final_rows"] = answer["columns"], answer["rows"]
            chosen = np.asarray(answer["selection"], dtype=np.int64) % P
            if full:
                return float(answer["objective"]), chosen
            tick = time.perf_counter()
            rc, minima = model.price(costs, answer["alpha"], answer["pi"])
            bound = float(sum(answer["alpha"]) + sum(min(0.0, float(x)) for x in minima))
            enter = np.flatnonzero((rc.ravel() < -1e-9) & ~active.ravel())
            if options.columns_per_block:
                admitted = []
                for q in range(B):
                    local = enter[enter // P == q]
                    order = np.argsort(rc.ravel()[local], kind="stable")
                    admitted.extend(local[order[:options.columns_per_block]])
                enter = np.asarray(admitted, dtype=np.int64)
            stats["pricing_seconds"] += time.perf_counter() - tick
            if not enter.size:
                if bound > answer["objective"] + 1e-7:
                    raise AssertionError("Deep EE pricing bound exceeds RMP")
                return min(bound, float(answer["objective"])), chosen
            if int(active.sum()) + enter.size > options.max_columns:
                raise CapacityExceeded("Deep EE active-column capacity")
            model.update(enter.astype(np.int32), costs.ravel()[enter]); active.flat[enter] = 1
    finally:
        model.close()


def solve_full_endpoint_lp(state, clock, options, stats, tree, initial_lb, progress=None):
    if state.M < 4:
        raise ValueError("Full two-endpoint EE requires D5 or deeper")
    tick = time.perf_counter()
    low = eliminate_endpoints(state.high.reshape(state.M, state.P))
    stats["ee_preprocessing_seconds"] += time.perf_counter() - tick
    value, selected = _coordinate(low, state, clock, options, stats, full=True)
    candidate, _ = _tree(state, selected)
    audited = evaluate(state.p, candidate)["objective"]
    if abs(audited - value) > 1e-7:
        raise AssertionError("Deep full EE lift mismatch")
    if audited < evaluate(state.p, tree)["objective"]:
        tree = candidate
    stats.update(ee_original_clusters=state.M, ee_remaining_clusters=state.M - 2,
        ee_original_candidate_slots=state.N, ee_reduced_candidate_slots=(state.M - 2) * state.P,
        exact_terminal_states=len(state.groups), unresolved_terminal_states=0)
    proof = clock.elapsed()
    return result_dict("JT-LP-SC-EE-D3Tail", clock, "OPT", tree, state.p, audited,
        stats=dict(stats), trace=[dict(seconds=proof, LB=audited, UB=audited,
        columns=stats["final_columns"], rows=stats["final_rows"])], proof_seconds=proof,
        certificate=dict(kind="complete_contracted_LP_after_exact_endpoint_elimination",
                         upper_tree_independently_audited=True),
        formulation="Complete D3-contracted JT-LP after eliminating both path endpoints")


def solve_deep_interval_ee(state, clock, options, stats, tree, initial_lb, progress=None):
    if state.M <= 4:
        raise ValueError("Deep interval EE requires at least eight original clusters")
    lb = float(initial_lb); ub = evaluate(state.p, tree)["objective"]
    status = "TIME"; proof = None; trace = []
    for key in ("ee_refined_states", "ee_requested_states", "rmp_seconds", "pricing_seconds",
                "peak_active_columns", "peak_rmp_nonzeros", "iterations"):
        stats.setdefault(key, 0)
    stats.update(ee_original_clusters=state.M, ee_remaining_clusters=state.M - 2,
        ee_original_candidate_slots=state.N, ee_reduced_candidate_slots=(state.M - 2) * state.P,
        ee_coordinator="native_path_cg_after_two_endpoint_eliminations")
    try:
        while True:
            clock.check(); tick = time.perf_counter()
            low = eliminate_endpoints(state.low.reshape(state.M, state.P))
            high = eliminate_endpoints(state.high.reshape(state.M, state.P))
            stats["ee_preprocessing_seconds"] += time.perf_counter() - tick
            upper, upper_ids = path_opt(high, state.F, state.K, state.domain.h, first_block=1)
            if upper_ids is not None and math.isfinite(upper):
                candidate, _ = _tree(state, upper_ids)
                audited = evaluate(state.p, candidate)["objective"]
                if abs(audited - upper) > 1e-7:
                    raise AssertionError("Deep interval EE upper lift mismatch")
                if audited < ub: tree, ub = candidate, audited
            bound, selected = _coordinate(low, state, clock, options, stats)
            if bound > ub + 1e-7:
                raise AssertionError("Deep interval EE lower bound exceeds incumbent")
            lb = max(lb, min(bound, ub))
            trace.append(dict(seconds=clock.elapsed(), LB=lb, UB=ub,
                exact_terminal_states=sum(g["record"].exact for g in state.groups)))
            if progress:
                progress(result_dict("JT-CG-EE", clock, "RUNNING", tree, state.p, lb,
                                     stats=dict(stats), trace=trace))
            if ub - lb <= 1e-7:
                status = "OPT"; proof = clock.elapsed(); break
            gids = []
            for i in _lift(state, selected):
                gid = int(state.private[i])
                if gid >= 0 and not state.groups[gid]["record"].exact and gid not in gids:
                    gids.append(gid)
            if not gids:
                raise RuntimeError("Deep EE interval gap without refinable state")
            gids = gids[:options.oracle_batch]
            tick = time.perf_counter()
            answers = state.engine.terminal_many([(state.groups[g]["rows"], state.groups[g]["node"],
                state.groups[g]["used"]) for g in gids])
            stats["oracle_wall_seconds"] += time.perf_counter() - tick
            stats["ee_requested_states"] += len(gids)
            before = sum(g["record"].exact for g in state.groups)
            for gid, answer in zip(gids, answers): state.update_group(gid, answer)
            stats["ee_refined_states"] += sum(g["record"].exact for g in state.groups) - before
            if not all(a.exact for a in answers):
                raise DeadlineExceeded("Partial private-subtree result")
    except DeadlineExceeded:
        status = "TIME"
    except CapacityExceeded as exc:
        status = "RESOURCE"; stats["resource_reason"] = str(exc)
    stats["exact_terminal_states"] = sum(g["record"].exact for g in state.groups)
    stats["unresolved_terminal_states"] = len(state.groups) - stats["exact_terminal_states"]
    return result_dict("JT-CG-EE-D3Tail", clock, status, tree, state.p, lb,
        stats=dict(stats), trace=trace, proof_seconds=proof,
        certificate=dict(kind="interval_endpoint_elimination", lower_cost_master=True,
                         complete_pricing_domain=True, upper_tree_independently_audited=True),
        formulation="D3-contracted JT-CG after exact elimination of both path endpoints")
