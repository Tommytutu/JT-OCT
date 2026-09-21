"""Data validation, tagged tree actions, and independent routed evaluation."""
from dataclasses import dataclass
from itertools import product
import time
import numpy as np


class DeadlineExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, seconds=600):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Finite nonnegative time limit required")
        self.start = time.perf_counter()
        self.end = self.start + seconds

    def check(self):
        if time.perf_counter() >= self.end:
            raise DeadlineExceeded("wall-clock deadline reached")

    def remaining(self):
        self.check()
        return max(0.001, self.end - time.perf_counter())

    def elapsed(self):
        return time.perf_counter() - self.start


@dataclass(frozen=True)
class Tree:
    # Exactly one of feature and label is present; children are present for splits.
    feature: int | None = None
    label: int | None = None
    left: "Tree | None" = None
    right: "Tree | None" = None

    def __post_init__(self):
        if self.feature is None:
            if self.label is None or self.left is not None or self.right is not None:
                raise ValueError("Invalid leaf")
        elif self.label is not None or self.left is None or self.right is None:
            raise ValueError("Invalid split")

    def to_dict(self):
        if self.feature is None:
            return {"label": int(self.label)}
        return {"feature": int(self.feature), "left": self.left.to_dict(),
                "right": self.right.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(label=d["label"]) if "label" in d else cls(
            feature=d["feature"], left=cls.from_dict(d["left"]), right=cls.from_dict(d["right"]))


@dataclass(frozen=True, slots=True)
class RowSet:
    """Compact immutable observation set backed by a Python integer bitset.

    Iteration is retained for baseline adapters and explicit-label constraints;
    the JT pricing hot path uses bit operations through Problem instead.
    """
    mask: int

    def __len__(self):
        return self.mask.bit_count()

    def __bool__(self):
        return bool(self.mask)

    def __iter__(self):
        remaining = self.mask
        while remaining:
            bit = remaining & -remaining
            yield bit.bit_length() - 1
            remaining ^= bit


def _mask(values):
    """Pack a Boolean vector with observation zero in the least-significant bit."""
    packed = np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


def _binary_matrix(raw):
    """Validate integer data without a full-size membership-search temporary."""
    if raw.ndim != 2 or not raw.shape[0] or not raw.shape[1]:
        return False
    if raw.dtype.kind in "biu":
        return bool(raw.min() >= 0 and raw.max() <= 1)
    return bool(np.isin(raw, [0, 1]).all())


def _column_masks(X, workers=1):
    """Cache-sized row tiles avoid repeatedly striding across a large matrix.

    Row blocks start on byte boundaries. The final packbits byte is zero padded,
    preserving observation zero as the least-significant bit exactly.
    """
    n,F=X.shape
    if workers > 1 and X.size >= 1_000_000:
        from concurrent.futures import ThreadPoolExecutor
        # NumPy packing releases the GIL. Each worker owns a feature slab;
        # unlike process workers this does not duplicate the training matrix.
        slabs = np.array_split(np.arange(F), min(workers, F))
        with ThreadPoolExecutor(max_workers=len(slabs)) as executor:
            parts = list(executor.map(lambda ids: _column_masks(X[:, ids[0]:ids[-1]+1]), slabs))
        return tuple(mask for part in parts for mask in part)
    if X.size < 1_000_000:
        return tuple(_mask(X[:,f]) for f in range(F))
    packed=np.empty((F,(n+7)//8),dtype=np.uint8)
    block=max(8,min(16384,(2*1024**2//F//8)*8))
    for first in range(0,n,block):
        tile=np.ascontiguousarray(X[first:first+block].T)
        part=np.packbits(tile,axis=1,bitorder="little")
        packed[:,first//8:first//8+part.shape[1]]=part
    return tuple(int.from_bytes(row.tobytes(),"little") for row in packed)


class Problem:
    def __init__(self, X, y, depth, penalty=0.0, weights=None, early_stop=True,
                 no_repeat=False, min_leaf=0, allowed=None, split_costs=None,
                 preparation_workers=1):
        raw = np.asarray(X)
        if not _binary_matrix(raw):
            raise ValueError("X must be a nonempty binary matrix")
        self.X = raw.astype(np.uint8, order="C")
        self.y = np.asarray(y)
        if self.y.ndim != 1 or len(self.y) != len(self.X):
            raise ValueError("y length must match X")
        if not np.issubdtype(self.y.dtype, np.integer) or np.any(self.y < 0):
            raise ValueError("Labels must be nonnegative integers; encode external labels first")
        if not isinstance(depth, int) or depth < 1 or not np.isfinite(penalty) or penalty < 0:
            raise ValueError("depth >= 1 and nonnegative penalty required")
        self.depth, self.penalty = depth, float(penalty)
        self.n, self.F = self.X.shape
        self.labels = tuple(int(k) for k in np.unique(self.y))
        self.weights = np.full(self.n, 1 / self.n) if weights is None else np.asarray(weights, dtype=float)
        if self.weights.shape != (self.n,) or not np.isfinite(self.weights).all() or np.any(self.weights <= 0):
            raise ValueError("All observation weights must be positive and finite")
        self.early_stop, self.no_repeat = early_stop, no_repeat
        self.min_leaf = int(min_leaf)
        if self.min_leaf < 0 or self.min_leaf != min_leaf:
            raise ValueError("min_leaf must be nonnegative")
        self.allowed = allowed or {}
        if any(not isinstance(f,(int,np.integer)) or not 0 <= f < self.F for fs in self.allowed.values() for f in fs):
            raise ValueError("Allowed predicate ids must index X")
        self.split_costs = split_costs or {}
        if any(c < 0 or not np.isfinite(c) for c in self.split_costs.values()):
            raise ValueError("All split costs must be finite and nonnegative")
        self._all_mask = (1 << self.n) - 1
        self.all_rows = RowSet(self._all_mask)
        self.empty_rows = RowSet(0)
        if not isinstance(preparation_workers, int) or preparation_workers < 1:
            raise ValueError("preparation_workers must be a positive integer")
        one = _column_masks(self.X, preparation_workers)
        self._feature_masks = tuple((self._all_mask ^ m, m) for m in one)
        self._label_masks = {k: _mask(self.y == k) for k in self.labels}
        self._label_mask_items = tuple(self._label_masks.items())
        self._uniform_weight = float(self.weights[0]) if np.all(self.weights == self.weights[0]) else None
        self._weight_unit, self._weight_planes, self._weight_sparse = None, None, None
        if self._uniform_weight is None:
            # Exact duplicate aggregation produces small integer multiplicities
            # times one common unit.  Bit planes retain the popcount hot path
            # instead of falling back to Python iteration over every routed row.
            unit = float(np.min(self.weights))
            multiplicity = np.rint(self.weights / unit).astype(np.int64)
            reconstructed = multiplicity.astype(float) * unit
            if (multiplicity.min() >= 1 and multiplicity.max() <= 2**31 and
                    np.allclose(self.weights, reconstructed, rtol=1e-12, atol=1e-15)):
                self._weight_unit = unit
                changed = np.flatnonzero(multiplicity != 1)
                bits = int(multiplicity.max()).bit_length()
                if len(changed) <= bits:
                    self._weight_sparse = tuple((1 << int(i), int(multiplicity[i]-1))
                                                for i in changed)
                else:
                    self._weight_planes = tuple(
                        _mask((multiplicity & (1 << bit)) != 0)
                        for bit in range(bits))
        self._feature_order = None

    def set_feature_order(self, order=None):
        """Set a deterministic pricing enumeration order; feasibility is unchanged."""
        if order is None:
            self._feature_order = None
            return
        order = tuple(int(f) for f in order)
        if len(order) != self.F or set(order) != set(range(self.F)):
            raise ValueError("Feature order must be a permutation of all predicate ids")
        self._feature_order = order

    def features(self, node, used=()):
        fs = self.allowed.get(node, range(self.F))
        available = tuple(f for f in fs if not self.no_repeat or f not in used)
        if self._feature_order is None:
            return available
        allowed = set(available)
        return tuple(f for f in self._feature_order if f in allowed)

    def cost(self, node, f):
        return self.penalty + self.split_costs.get((node, f), 0.0)

    def route(self, rows, f, b):
        if isinstance(rows, RowSet):
            return RowSet(rows.mask & self._feature_masks[f][b])
        return tuple(i for i in rows if self.X[i, f] == b)

    def weight_sum(self, rows):
        """Weighted mass of a RowSet, with an exact popcount path when possible."""
        mask = rows.mask if isinstance(rows, RowSet) else sum(1 << i for i in rows)
        if self._uniform_weight is not None:
            return float(mask.bit_count() * self._uniform_weight)
        if self._weight_sparse is not None:
            units = mask.bit_count() + sum(extra for bit, extra in self._weight_sparse
                                           if mask & bit)
            return float(units * self._weight_unit)
        if self._weight_planes is not None:
            units = sum((1 << bit) * (mask & plane).bit_count()
                        for bit, plane in enumerate(self._weight_planes))
            return float(units * self._weight_unit)
        return float(sum(self.weights[i] for i in RowSet(mask)))

    def loss(self, rows, label):
        if isinstance(rows, RowSet):
            wrong = rows.mask & (self._all_mask ^ self._label_masks.get(label, 0))
            return self.weight_sum(RowSet(wrong))
        return float(sum(self.weights[i] for i in rows if self.y[i] != label))

    def best_label_and_loss(self, rows):
        """Return the deterministic majority label and its weighted loss once."""
        if isinstance(rows, RowSet) and (self._uniform_weight is not None or
                                         self._weight_planes is not None or
                                         self._weight_sparse is not None):
            label, label_mask = self._label_mask_items[0]
            best_mass = self.weight_sum(RowSet(rows.mask & label_mask))
            for candidate, candidate_mask in self._label_mask_items[1:]:
                mass = self.weight_sum(RowSet(rows.mask & candidate_mask))
                if mass > best_mass:
                    label, best_mass = candidate, mass
            return label, float(self.weight_sum(rows) - best_mass)
        correct = {k: 0.0 for k in self.labels}
        total = 0.0
        for i in rows:
            weight = float(self.weights[i])
            total += weight
            correct[int(self.y[i])] += weight
        label = min(self.labels, key=lambda k: (-correct[k], k))
        return label, float(total - correct[label])

    def best_label(self, rows):
        return self.best_label_and_loss(rows)[0]


def predict(tree, X):
    # Independent prediction still traverses X, but routes a vector of sample
    # indices at each node rather than running a Python loop for every sample.
    X = np.asarray(X)
    out = np.empty(len(X), dtype=int)
    def visit(t, ids):
        if not len(ids): return
        if t.feature is None:
            out[ids] = t.label
        else:
            left = X[ids, t.feature] == 0
            visit(t.left, ids[left]); visit(t.right, ids[~left])
    visit(tree, np.arange(len(X)))
    return out


def evaluate(p, tree, validate=True):
    """Traverse the actual tree, without using column costs or cached statistics."""
    branches, leaves, used_features, split_cost = 0, 0, set(), 0.0
    maximum_depth = 0

    def walk(t, node, used, rows):
        nonlocal branches, leaves, split_cost, maximum_depth
        maximum_depth = max(maximum_depth, len(node))
        if t.feature is None:
            leaves += 1
            if validate and (t.label not in p.labels or len(rows) < p.min_leaf or
                             (not p.early_stop and len(node) != p.depth)):
                raise ValueError("Infeasible leaf")
            return
        if validate and (len(node) >= p.depth or t.feature not in p.features(node, used)):
            raise ValueError("Infeasible split")
        branches += 1
        used_features.add(t.feature)
        split_cost += p.cost(node, t.feature)
        walk(t.left, node + (0,), used + (t.feature,), p.route(rows, t.feature, 0))
        walk(t.right, node + (1,), used + (t.feature,), p.route(rows, t.feature, 1))

    walk(tree, (), (), p.all_rows)
    yp = predict(tree, p.X)
    loss = float(np.dot(p.weights, yp != p.y))
    return {"objective": loss + split_cost, "loss": loss,
            "misclassified": int(np.count_nonzero(yp != p.y)), "split_cost": split_cost,
            "split_nodes": branches, "leaves": leaves, "realized_depth": maximum_depth,
            "features": sorted(used_features), "prediction": yp.tolist()}


def enumerate_trees(p, deadline=None, explicit_labels=False):
    """Independent recursive global enumerator: reference oracle, exponential in nodes."""
    def rec(node, rows, used):
        if deadline:
            deadline.check()
        if len(rows) >= p.min_leaf and (p.early_stop or len(node) == p.depth):
            labels = p.labels if explicit_labels else (p.best_label(rows),)
            for k in labels:
                yield Tree(label=k)
        if len(node) == p.depth:
            return
        for f in p.features(node, used):
            for left in rec(node + (0,), p.route(rows, f, 0), used + (f,)):
                for right in rec(node + (1,), p.route(rows, f, 1), used + (f,)):
                    yield Tree(feature=f, left=left, right=right)
    yield from rec((), p.all_rows, ())
