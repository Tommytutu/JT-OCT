"""Reference full JT LP/IP, message passing, tree DP, and hybrid master."""
import math
from itertools import product
import numpy as np
from .gurobi_backend import optimize_linear
from scipy.sparse import hstack, vstack, coo_matrix
from .problem import Tree, Deadline, DeadlineExceeded, evaluate
from .domain import Domain, CapacityExceeded
from .master import solve_rmp, build_matrix, chain_path, chain_path_streaming


def result_dict(method, deadline, status, tree=None, p=None, lb=0.0, **extra):
    metrics = evaluate(p, tree) if tree is not None else {}
    ub = metrics.get("objective", math.inf)
    raw_lb = lb
    lb = max(0.0, lb)  # Loss and split costs are nonnegative by construction.
    if tree is not None:
        if lb > ub + 1e-7:
            raise AssertionError("Certified LB exceeds actual routed UB")
        lb = min(lb, ub)  # Roundoff only; substantial violations raise above.
    return {"method": method, "status": status, "seconds": deadline.elapsed(),
            "LB": lb, "raw_LB": raw_lb, "UB": ub,
            "absolute_gap": max(0.0, ub-lb) if math.isfinite(ub) else None,
            "gap": max(0.0, ub-lb)/max(1e-10, abs(ub)) if math.isfinite(ub) else None,
            "tree": tree.to_dict() if tree else None, "metrics": metrics, **extra}


def solve_jt_dp(domain, time_limit=600, max_columns=200000):
    clock = Deadline(time_limit)
    try:
        val, selected, columns = chain_path_streaming(domain, clock, max_columns)
        if selected is None:
            return result_dict("JT-DP", clock, "INFEASIBLE", lb=math.inf)
        return result_dict("JT-DP", clock, "OPT", domain.recover(selected), domain.p, val,
                           columns=columns, stats=domain.stats,
                           message_passing="streaming_chain_min_sum")
    except (DeadlineExceeded, CapacityExceeded) as exc:
        return result_dict("JT-DP", clock, "TIME" if isinstance(exc, DeadlineExceeded) else "RESOURCE", reason=str(exc))


def solve_tree_dp(p, time_limit=600):
    clock, d = Deadline(time_limit), Domain(p, tail_depth=p.depth)
    try:
        tree, val, _ = d.best_tail((), p.all_rows, (), clock)
        return result_dict("Tree-DP", clock, "OPT" if tree else "INFEASIBLE", tree, p,
                           val, stats=d.stats)
    except DeadlineExceeded:
        return result_dict("Tree-DP", clock, "TIME")


def coupling_matrix(domain, flat, budget=None, feature_budget=None, feature_groups=None,
                    groups=None, parity_tolerance=None):
    n = len(flat)
    groups_of_feature = list(range(domain.p.F)) if feature_groups is None else list(feature_groups)
    if len(groups_of_feature) != domain.p.F:
        raise ValueError("One original-attribute group per predicate required")
    distinct = sorted(set(groups_of_feature)) if feature_budget is not None else []
    uidx = {g: n+j for j, g in enumerate(distinct)}
    entries, upper = [], []
    if budget is not None:
        entries.append({j: c.splits for j, (_, c) in enumerate(flat)})
        upper.append(float(budget))
    if feature_budget is not None:
        # Use one cluster containing each repeated node, avoiding multiplicity.
        maps = [domain.split_marginals(i, c) for i, c in flat]
        owners = {}
        for (i, _), mapping in zip(flat, maps):
            for node in mapping:
                owners[node] = min(owners.get(node, i), i)
        for group in distinct:
            total = {}
            for node, owner in owners.items():
                row = {j: 1.0 for j, ((i, _), mapping) in enumerate(zip(flat, maps))
                       if i == owner and node in mapping and groups_of_feature[mapping[node]] == group}
                for j, v in row.items():
                    total[j] = total.get(j, 0.0) + v
                row[uidx[group]] = -1.0
                entries.append(row); upper.append(0.0)
            row = {j: -v for j, v in total.items()}
            row[uidx[group]] = 1.0
            entries.append(row); upper.append(0.0)
        entries.append({j: 1.0 for j in uidx.values()}); upper.append(float(feature_budget))
    if parity_tolerance is not None:
        if not domain.explicit_labels or not set(domain.p.labels).issubset({0, 1}):
            raise ValueError("Statistical parity requires explicit binary leaf labels")
        g = np.asarray(groups)
        if g.shape != (domain.p.n,) or len(np.unique(g)) < 2:
            raise ValueError("At least two nonempty groups required")
        from itertools import combinations
        for a, b in combinations(np.unique(g), 2):
            coefs = {j: float(domain.local_predictions(i, c)[g == a].mean() -
                              domain.local_predictions(i, c)[g == b].mean())
                     for j, (i, c) in enumerate(flat)}
            entries.extend([coefs, {j: -v for j, v in coefs.items()}])
            upper.extend([float(parity_tolerance)] * 2)
    rr, cc, vv = [], [], []
    for r, row in enumerate(entries):
        for j, v in row.items():
            rr.append(r); cc.append(j); vv.append(v)
    A = coo_matrix((vv, (rr, cc)), shape=(len(entries), n+len(uidx))).tocsr()
    return A, np.asarray(upper), len(uidx)


def solve_full(domain, time_limit=600, max_columns=200000, mode="jt", integer=False, **coupling):
    clock = Deadline(time_limit)
    try:
        pool = domain.materialize(clock, max_columns)
        if any(not cols for cols in pool):
            return result_dict("Full-JT", clock, "INFEASIBLE", lb=math.inf)
        A, b, c, flat, keys = build_matrix(domain, pool, mode)
        U, rhs, extra = coupling_matrix(domain, flat, **coupling)
        A = hstack([A, coo_matrix((A.shape[0], extra))]).tocsr()
        c = np.r_[c, np.zeros(extra)]
        res = optimize_linear(c, clock, A_eq=A, b_eq=b, A_ub=U, b_ub=rhs,
                              integrality=np.ones(len(c)) if integer else None,
                              ub=np.ones(len(c)) if integer else None, name="Full_JT")
        if res.status == 2:
            return result_dict("Full-JT", clock, "INFEASIBLE", lb=math.inf)
        if res.x is None:
            return result_dict("Full-JT", clock, "TIME" if res.status == 1 else "ERROR", reason=res.message)
        tree = None
        if (mode == "jt" or integer) and (integer or not len(rhs)):
            positive = [[] for _ in range(domain.M)]
            for j, (i, col) in enumerate(flat):
                if res.x[j] > (0.5 if integer else 1e-7):
                    positive[i].append(col)
            val, selected = chain_path(domain, positive)
            if selected is None:
                raise AssertionError("LP support cannot be glued")
            tree = domain.recover(selected)
            if abs(evaluate(domain.p, tree)["objective"] - res.fun) > 1e-7:
                raise AssertionError("Tree objective differs from full JT objective")
        lb = float(res.fun) if res.status == 0 else float(getattr(res, "mip_dual_bound", 0.0) or 0.0)
        out = result_dict("Full-JT-IP" if integer else "Full-"+mode.upper()+"-LP", clock,
                          "OPT" if res.status == 0 else "TIME", tree, domain.p, lb,
                          lp_objective=float(res.fun), columns=len(flat), rows=A.shape[0]+U.shape[0],
                          optimality_scope="tree_problem" if tree is not None else "LP_relaxation",
                          nnz=A.nnz+U.nnz, equality_residual=float(np.max(np.abs(A @ res.x-b))))
        return out
    except (DeadlineExceeded, CapacityExceeded) as exc:
        return result_dict("Full-JT", clock, "TIME" if isinstance(exc, DeadlineExceeded) else "RESOURCE", reason=str(exc))


def solve_hybrid(p, tail_depth=2, time_limit=600, max_upper=100000):
    """Exact joint upper-tree simplex + conditional tail links (eqs. 38--40).

    Private tail columns are compressed to an exact conditional DP optimum.
    This is safe only for the uncoupled additive problem.
    """
    clock, d = Deadline(time_limit), Domain(p, tail_depth=tail_depth)
    h = d.h
    def tops(node, rows, used):
        clock.check()
        if len(node) == h:
            yield Tree(label=-1)
            return
        if p.early_stop and len(rows) >= p.min_leaf:
            for k in p.labels:
                yield Tree(label=k)
        for f in p.features(node, used):
            for l in tops(node+(0,), p.route(rows, f, 0), used+(f,)):
                for r in tops(node+(1,), p.route(rows, f, 1), used+(f,)):
                    yield Tree(feature=f, left=l, right=r)
    try:
        upper, tails, top_costs, memberships = [], {}, [], []
        for top in tops((), p.all_rows, ()):
            if len(upper) >= max_upper:
                raise CapacityExceeded("Hybrid upper-tree count limit exceeded")
            cost, contexts, valid = 0.0, [], True
            def walk(t, node, rows, history):
                nonlocal cost, valid
                if len(node) == h:
                    key = node, history
                    if key not in tails:
                        tails[key] = d.best_tail(node, rows, tuple(f for f, _ in history), clock)
                    if tails[key][0] is None:
                        valid = False
                    contexts.append(key)
                elif t.feature is None:
                    cost += p.loss(rows, t.label)
                else:
                    cost += p.cost(node, t.feature)
                    walk(t.left, node+(0,), p.route(rows, t.feature, 0), history+((t.feature, 0),))
                    walk(t.right, node+(1,), p.route(rows, t.feature, 1), history+((t.feature, 1),))
            walk(top, (), p.all_rows, ())
            if valid:
                upper.append(top); top_costs.append(cost); memberships.append(contexts)
        if not upper:
            return result_dict("Hybrid", clock, "INFEASIBLE", lb=math.inf)
        keys = sorted({k for cs in memberships for k in cs})
        pos = {key: i for i, key in enumerate(keys)}
        rr, cc, vv = [0]*len(upper), list(range(len(upper))), [1.0]*len(upper)
        for j, contexts in enumerate(memberships):
            for key in contexts:
                rr.append(1+pos[key]); cc.append(j); vv.append(-1.0)
        for i, key in enumerate(keys):
            rr.append(1+i); cc.append(len(upper)+i); vv.append(1.0)
        A = coo_matrix((vv, (rr, cc)), shape=(1+len(keys), len(upper)+len(keys))).tocsr()
        costs = np.array(top_costs + [tails[k][1] for k in keys])
        b = np.r_[1.0, np.zeros(len(keys))]
        res = optimize_linear(costs, clock, A_eq=A, b_eq=b, name="JT_Hybrid")
        if res.status != 0:
            return result_dict("Hybrid", clock, "TIME" if res.status == 1 else "ERROR")
        idx = int(np.argmax(res.x[:len(upper)]))
        def fill(t, node=(), history=()):
            if len(node) == h:
                return tails[node, history][0]
            if t.feature is None:
                return t
            return Tree(feature=t.feature,
                        left=fill(t.left, node+(0,), history+((t.feature, 0),)),
                        right=fill(t.right, node+(1,), history+((t.feature, 1),)))
        return result_dict("Hybrid", clock, "OPT", fill(upper[idx]), p, float(res.fun),
                           upper_columns=len(upper), tail_contexts=len(keys))
    except (DeadlineExceeded, CapacityExceeded) as exc:
        return result_dict("Hybrid", clock, "TIME" if isinstance(exc, DeadlineExceeded) else "RESOURCE", reason=str(exc))
