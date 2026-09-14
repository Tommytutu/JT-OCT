"""Lazy path / connected-subtree configuration domains (paper Sections 2--4)."""
from dataclasses import dataclass
from itertools import product
from .problem import Tree

SPLIT, STOP, INACTIVE = 0, 1, 2
DEAD = (INACTIVE, -1)


class CapacityExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Column:
    prefix: tuple
    tail: Tree | None
    cost: float
    splits: float


class Domain:
    def __init__(self, problem, tail_depth=1, explicit_labels=False, support_filter=None):
        self.p = problem
        if not 1 <= tail_depth <= problem.depth:
            raise ValueError("tail_depth must be between 1 and D")
        self.r = tail_depth
        self.h = problem.depth - tail_depth
        self.roots = tuple(product((0, 1), repeat=self.h))
        self.M = len(self.roots)
        self.explicit_labels = explicit_labels
        self.support_filter = support_filter
        self.separators = []
        for left, right in zip(self.roots, self.roots[1:]):
            common = 0
            while common < self.h and left[common] == right[common]:
                common += 1
            self.separators.append(common + 1)
        self.incident = {i: [] for i in range(self.M)}
        for e, size in enumerate(self.separators):
            self.incident[e].append((e, size, 1))
            self.incident[e + 1].append((e, size, -1))
        self._tail_cache = {}
        self._signature_cache = {}
        self.stats = {"tail_states": 0, "signature_hits": 0, "signature_evaluations": 0,
                      "stump_candidates": 0, "columns_evaluated": 0,
                      "prefixes_visited": 0, "prefixes_pruned": 0}

    def permitted(self, i, col):
        return self.support_filter is None or self.support_filter(i, col)

    def prefix_states(self, i, deadline=None, visit=None):
        """Yield prefix, routed rows, used features, allocated cost and split count."""
        q, p = self.roots[i], self.p

        def rec(rho, rows, used, cost, splits, alive):
            if deadline:
                deadline.check()
            self.stats["prefixes_visited"] += 1
            if visit and not visit(rho, cost, rows, used):
                self.stats["prefixes_pruned"] += 1
                return
            j = len(rho)
            if j == self.h:
                yield rho, rows, used, cost, splits
                return
            if not alive:
                yield from rec(rho + (DEAD,), p.empty_rows, used, cost, splits, False)
                return
            node, mult = q[:j], 2 ** (self.h - j)
            if p.early_stop and len(rows) >= p.min_leaf:
                for k in p.labels:
                    yield from rec(rho + ((STOP, k),), (), used,
                                   cost + p.loss(rows, k) / mult, splits, False)
            for f in p.features(node, used):
                yield from rec(rho + ((SPLIT, f),), p.route(rows, f, q[j]), used + (f,),
                               cost + p.cost(node, f) / mult, splits + 1 / mult, True)

        yield from rec((), p.all_rows, (), 0.0, 0.0, True)

    def tails(self, node, rows, used, deadline=None):
        p = self.p
        if deadline:
            deadline.check()
        if len(rows) >= p.min_leaf and (p.early_stop or len(node) == p.depth):
            if self.explicit_labels:
                for k in p.labels:
                    yield Tree(label=k), p.loss(rows, k), 0
            else:
                k, loss = p.best_label_and_loss(rows)
                yield Tree(label=k), loss, 0
        if len(node) == p.depth:
            return
        for f in p.features(node, used):
            for l, lc, lb in self.tails(node + (0,), p.route(rows, f, 0), used + (f,), deadline):
                for r, rc, rb in self.tails(node + (1,), p.route(rows, f, 1), used + (f,), deadline):
                    yield Tree(feature=f, left=l, right=r), p.cost(node, f) + lc + rc, 1 + lb + rb

    def best_tail(self, node, rows, used, deadline=None):
        """Exact conditional tail DP; terminal observations are not retained.

        Terminal states have no descendants and are normally reached once for a
        given restriction history.  Returning them directly avoids pinning large
        observation sets in a cache without changing the recurrence.
        """
        if deadline:
            deadline.check()
        p = self.p
        if len(node) == p.depth:
            self.stats["tail_states"] += 1
            if len(rows) < p.min_leaf:
                return None, float("inf"), 0
            k, loss = p.best_label_and_loss(rows)
            return Tree(label=k), loss, 0
        key = node, rows, used if self.p.no_repeat else ()
        if key in self._tail_cache:
            return self._tail_cache[key]
        self.stats["tail_states"] += 1
        best = (None, float("inf"), 0)
        if len(rows) >= p.min_leaf and (p.early_stop or len(node) == p.depth):
            k, loss = p.best_label_and_loss(rows)
            best = Tree(label=k), loss, 0
        if len(node) == p.depth - 1:
            # Specialized exact stump solver.  It avoids two recursive calls and
            # two short-lived Tree objects for every candidate predicate.
            for f in p.features(node, used):
                left_rows = p.route(rows, f, 0)
                right_rows = p.route(rows, f, 1)
                self.stats["stump_candidates"] += 1
                if len(left_rows) < p.min_leaf or len(right_rows) < p.min_leaf:
                    continue
                lk, lc = p.best_label_and_loss(left_rows)
                rk, rc = p.best_label_and_loss(right_rows)
                cost, size = p.cost(node, f) + lc + rc, 1
                if (cost, size) < (best[1], best[2]):
                    best = (Tree(feature=f, left=Tree(label=lk), right=Tree(label=rk)),
                            cost, size)
            self._tail_cache[key] = best
            return best
        if len(node) < p.depth:
            for f in p.features(node, used):
                l, lc, lb = self.best_tail(node + (0,), p.route(rows, f, 0), used + (f,), deadline)
                r, rc, rb = self.best_tail(node + (1,), p.route(rows, f, 1), used + (f,), deadline)
                cost, size = p.cost(node, f) + lc + rc, 1 + lb + rb
                if l is not None and r is not None and (cost, size) < (best[1], best[2]):
                    best = Tree(feature=f, left=l, right=r), cost, size
        self._tail_cache[key] = best
        return best

    def from_state(self, i, state, deadline=None, best_only=False):
        rho, rows, used, pc, ps = state
        if any(a[0] != SPLIT for a in rho):
            c = Column(rho, None, pc, ps)
            if self.permitted(i, c):
                yield c
            return
        tail_iter = (self.best_tail(self.roots[i], rows, used, deadline),) if best_only and self.support_filter is None else self.tails(self.roots[i], rows, used, deadline)
        for t, c, s in tail_iter:
            if t is not None:
                col = Column(rho, t, pc + c, ps + s)
                self.stats["columns_evaluated"] += 1
                if self.permitted(i, col):
                    yield col

    def columns(self, i, deadline=None):
        for state in self.prefix_states(i, deadline):
            yield from self.from_state(i, state, deadline)

    def representatives(self, i, deadline=None, visit=None):
        """Cheapest private configuration per complete incident-separator signature."""
        if self.explicit_labels:
            raise ValueError("Label compression cannot be used with coupling constraints")
        for state in self.prefix_states(i, deadline, visit):
            key = i, state[0]
            if key in self._signature_cache:
                self.stats["signature_hits"] += 1
                best = self._signature_cache[key]
            else:
                self.stats["signature_evaluations"] += 1
                best = min(self.from_state(i, state, deadline, best_only=True),
                           key=lambda c: (c.cost, c.splits), default=None)
                self._signature_cache[key] = best
            if best is not None:
                yield best

    def materialize(self, deadline=None, max_columns=200000):
        pool, count = [], 0
        for i in range(self.M):
            block = []
            for c in self.columns(i, deadline):
                count += 1
                if count > max_columns:
                    raise CapacityExceeded(f"Configuration limit {max_columns} exceeded")
                block.append(c)
            pool.append(block)
        return pool

    def tree_columns(self, tree):
        p, cols = self.p, []
        for i, q in enumerate(self.roots):
            t, rows, rho, pc, ps, alive = tree, p.all_rows, (), 0.0, 0.0, True
            for j, b in enumerate(q):
                mult = 2 ** (self.h - j)
                if not alive:
                    rho += (DEAD,)
                elif t.feature is None:
                    rho += ((STOP, t.label),)
                    pc += p.loss(rows, t.label) / mult
                    alive = False
                else:
                    rho += ((SPLIT, t.feature),)
                    pc += p.cost(q[:j], t.feature) / mult
                    ps += 1 / mult
                    rows = p.route(rows, t.feature, b)
                    t = t.left if b == 0 else t.right
            if alive:
                def value(t, node, rows):
                    if t.feature is None:
                        return p.loss(rows, t.label), 0
                    lc, ls = value(t.left, node + (0,), p.route(rows, t.feature, 0))
                    rc, rs = value(t.right, node + (1,), p.route(rows, t.feature, 1))
                    return p.cost(node, t.feature) + lc + rc, 1 + ls + rs
                tc, ts = value(t, q, rows)
                cols.append(Column(rho, t, pc + tc, ps + ts))
            else:
                cols.append(Column(rho, None, pc, ps))
        return cols

    def recover(self, columns):
        if len(columns) != self.M:
            raise ValueError("One compatible column per block required")
        actions, tails = {}, {}
        for q, c in zip(self.roots, columns):
            for j, action in enumerate(c.prefix):
                node = q[:j]
                if node in actions and actions[node] != action:
                    raise ValueError("Inconsistent shared ancestor actions")
                actions[node] = action
            tails[q] = c.tail

        def rec(node):
            if len(node) == self.h:
                if tails[node] is None:
                    raise ValueError("Reached an inactive tail")
                return tails[node]
            tag, value = actions[node]
            if tag == STOP:
                return Tree(label=value)
            if tag != SPLIT:
                raise ValueError("Reached inactive internal node")
            return Tree(feature=value, left=rec(node + (0,)), right=rec(node + (1,)))
        return rec(())

    def local_predictions(self, i, col):
        """H_{iqw}: allocated positive prediction, used only by explicit-label models."""
        import numpy as np
        p, q = self.p, self.roots[i]
        out = np.zeros(p.n)
        rows = p.all_rows
        for j, (tag, val) in enumerate(col.prefix):
            if tag == STOP:
                if val == 1:
                    out[list(rows)] += 1 / (2 ** (self.h - j))
                return out
            if tag == INACTIVE:
                return out
            rows = p.route(rows, val, q[j])
        def walk(t, rows):
            if t.feature is None:
                if t.label == 1:
                    out[list(rows)] += 1
            else:
                walk(t.left, p.route(rows, t.feature, 0))
                walk(t.right, p.route(rows, t.feature, 1))
        walk(col.tail, rows)
        return out

    def split_marginals(self, i, col):
        q, out = self.roots[i], {}
        for j, (tag, val) in enumerate(col.prefix):
            if tag == SPLIT:
                out[q[:j]] = val
        def walk(t, node):
            if t is not None and t.feature is not None:
                out[node] = t.feature
                walk(t.left, node + (0,))
                walk(t.right, node + (1,))
        walk(col.tail, q)
        return out
