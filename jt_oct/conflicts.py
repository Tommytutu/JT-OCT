"""Conflict bounds and safe feature-ordering diagnostics for binary OCT data."""
from dataclasses import dataclass
import time

import numpy as np

from .problem import RowSet, _mask


@dataclass(frozen=True)
class ConflictProfile:
    features: tuple
    lower_bound: float
    groups: int
    scores: dict


class RoutedConflictOracle:
    """Full-feature conflict loss inside any routed observation subset."""
    def __init__(self, problem):
        self.problem = problem
        inverse, group_count = _partition(problem.X, tuple(range(problem.F)))
        groups = []
        for g in range(group_count):
            group_mask = _mask(inverse == g)
            label_masks = tuple(group_mask & problem._label_masks[k]
                                for k in problem.labels)
            positive = sum(bool(mask) for mask in label_masks)
            if positive > 1:
                groups.append(label_masks)
        self.groups = tuple(groups)
        self.global_bound = self.bound(problem.all_rows)

    def bound(self, rows):
        if not isinstance(rows, RowSet) or not rows:
            return 0.0
        loss = 0.0
        for label_masks in self.groups:
            masses = [self.problem.weight_sum(RowSet(rows.mask & mask))
                      for mask in label_masks]
            loss += sum(masses) - max(masses)
        return float(loss)


def _partition(X, features):
    if not features:
        return np.zeros(len(X), dtype=np.int64), 1
    _, inverse = np.unique(X[:, features], axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False), int(inverse.max()) + 1


def _partition_loss(inverse, group_count, y, weights, labels):
    mass = np.vstack([
        np.bincount(inverse, weights=weights * (y == k), minlength=group_count)
        for k in labels])
    return float(np.sum(np.sum(mass, axis=0) - np.max(mass, axis=0)))


def conflict_lower_bound(problem, features=None):
    """Bayes error after observing only ``features``.

    With every available feature this is a valid global OCT lower bound.  For a
    strict subset it is only a lower bound for trees restricted to that subset.
    """
    features = tuple(range(problem.F)) if features is None else tuple(sorted(set(features)))
    inverse, groups = _partition(problem.X, features)
    return _partition_loss(inverse, groups, problem.y, problem.weights, problem.labels)


def conflict_profile(problem, features):
    """Compute one-feature conflict relief scores around a restricted set."""
    base = tuple(sorted(set(int(f) for f in features)))
    if any(f < 0 or f >= problem.F for f in base):
        raise ValueError("Feature id outside the predicate dictionary")
    inverse, groups = _partition(problem.X, base)
    current = _partition_loss(inverse, groups, problem.y, problem.weights, problem.labels)
    scores = {}
    for f in range(problem.F):
        if f in base:
            continue
        refined = inverse * 2 + problem.X[:, f]
        refined_groups = 2 * groups
        after = _partition_loss(refined, refined_groups, problem.y, problem.weights,
                                problem.labels)
        scores[f] = max(0.0, current - after)
    return ConflictProfile(base, current, groups, scores)


def conflict_feature_order(problem, incumbent_features):
    """Keep incumbent predicates first, then rank omitted ones by conflict relief."""
    profile = conflict_profile(problem, incumbent_features)
    active = tuple(int(f) for f in incumbent_features)
    omitted = tuple(sorted(profile.scores, key=lambda f: (-profile.scores[f], f)))
    return active + tuple(f for f in omitted if f not in active), profile


def aggregate_exact_duplicates(X, y, weights=None):
    """Merge identical (x,y) pairs and return exact combined observation weights."""
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y)
    if weights is None:
        weights = np.full(len(y), 1.0 / len(y))
    weights = np.asarray(weights, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or weights.shape != (len(X),):
        raise ValueError("X, y, and weights have incompatible dimensions")
    keys = np.column_stack([X, y])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    combined = np.bincount(inverse, weights=weights, minlength=len(unique))
    # np.unique sorts rows; extracting X/y from unique guarantees that inverse
    # and combined weights refer to the same deterministic row order.
    return (unique[:, :X.shape[1]].astype(np.uint8),
            unique[:, X.shape[1]].astype(y.dtype), combined,
            {"original_rows": int(len(y)), "unique_xy_rows": int(len(unique)),
             "removed_rows": int(len(y) - len(unique)),
             "compression_ratio": float(len(y) / len(unique))})


def feature_relaxation_bound(problem, base_features, split_budget, binary=False,
                             time_limit=60):
    """Optimistic feature-activation bound from the attached strategy note.

    Base predicates are free and an omitted conflict group may become perfectly
    classified as soon as any varying omitted predicate is activated.  This
    enlarges the set of depth-limited trees, so the optimum is a valid global
    lower bound.  It is expected to be weak and is used diagnostically.
    """
    import gurobipy as gp

    started = time.perf_counter()
    base = tuple(sorted(set(int(f) for f in base_features)))
    inverse, group_count = _partition(problem.X, base)
    omitted = tuple(f for f in range(problem.F) if f not in base)
    groups = []
    for g in range(group_count):
        idx = np.flatnonzero(inverse == g)
        if not len(idx):
            continue
        class_mass = [float(problem.weights[idx][problem.y[idx] == k].sum())
                      for k in problem.labels]
        disagreement = float(sum(class_mass) - max(class_mass))
        if disagreement <= 1e-15:
            continue
        varying = tuple(f for f in omitted
                        if np.any(problem.X[idx, f] != problem.X[idx[0], f]))
        groups.append((disagreement, varying))
    model = gp.Model("feature_relaxation_bound")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(.001, float(time_limit))
    vtype = gp.GRB.BINARY if binary else gp.GRB.CONTINUOUS
    yf = {f: model.addVar(lb=0, ub=1, vtype=vtype, name=f"y_{f}") for f in omitted}
    eg = []
    for j, (disagreement, varying) in enumerate(groups):
        z = model.addVar(lb=0, ub=1, name=f"z_{j}")
        e = model.addVar(lb=0, name=f"e_{j}")
        model.addConstr(z <= gp.quicksum(yf[f] for f in varying))
        model.addConstr(e >= disagreement * (1-z))
        eg.append(e)
    model.addConstr(gp.quicksum(yf.values()) <= int(split_budget))
    model.setObjective(gp.quicksum(eg) + problem.penalty * gp.quicksum(yf.values()),
                       gp.GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Feature-relaxation solver status {model.Status}")
    return {"bound": float(model.ObjVal), "best_bound": float(model.ObjBound),
            "status": int(model.Status), "binary_y": bool(binary),
            "base_features": list(base), "base_groups": int(group_count),
            "conflict_groups": int(len(groups)), "candidate_features": len(omitted),
            "split_budget": int(split_budget),
            "seconds": time.perf_counter() - started}
