"""Lightweight shared initialization, audit and result helpers."""
import math
import numpy as np
from .problem import Tree, evaluate


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


def conflict_representatives(p):
    # Binary row signatures preserve equality while reducing sort-key width 8x.
    _,reps,inverse=np.unique(np.packbits(p.X,axis=1),axis=0,return_index=True,return_inverse=True)
    total=np.bincount(inverse,minlength=len(reps))
    majority=np.zeros(len(reps),dtype=np.int64)
    for label in p.labels:
        count=np.bincount(inverse[p.y==label],minlength=len(reps))
        np.maximum(majority,count,out=majority)
    mass=total-majority
    chosen=mass>0
    return np.ascontiguousarray(np.column_stack((reps[chosen],mass[chosen])),dtype=np.int32)
