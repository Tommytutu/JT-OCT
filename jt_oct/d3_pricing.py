"""Exact O(F^2) pricing tables for the depth-three JT column generator."""
from __future__ import annotations

import ctypes
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .domain import Column, DEAD, SPLIT, STOP
from .d3_batched import (_D3_KERNEL, _configure_cupy_runtime, _cost_arrays,
                         _pack_words, _pack_words64, _require_supported_problem)
from .problem import Tree


def _compute_cpp(p, threads=0):
    start = time.perf_counter()
    library_path = Path(__file__).parent / "_native" / "d3_kappa.dll"
    if not library_path.exists():
        raise RuntimeError("Native pricing backend is not built; run build_native.ps1")
    library = ctypes.CDLL(str(library_path))
    function = library.d3_kappa_cpu
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [
        ctypes.c_double, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    root, second, third, allowed_root, allowed_second, allowed_third = _cost_arrays(p)
    words = np.ascontiguousarray(np.stack(
        [[_pack_words64(p.X[:, f] == b) for f in range(p.F)] for b in range(2)]))
    positive = np.ascontiguousarray(_pack_words64(p.y == p.labels[1]))
    arrays = [np.ascontiguousarray(x) for x in
              (root, second, third, allowed_root, allowed_second, allowed_third)]
    root, second, third, allowed_root, allowed_second, allowed_third = arrays
    F, W = p.F, words.shape[-1]
    kappa = np.empty((4, F, F), dtype=np.float64)
    h_arg = np.empty((4, F, F), dtype=np.int32)
    setup = time.perf_counter() - start
    core_start = time.perf_counter()
    pointers = [x.ctypes.data_as(ctypes.c_void_p) for x in
                (words, positive, root, second, third,
                 allowed_root, allowed_second, allowed_third)]
    code = function(*pointers, F, W, int(p.no_repeat), int(p.early_stop), p.min_leaf,
                    float(p._uniform_weight), int(threads),
                    kappa.ctypes.data_as(ctypes.c_void_p),
                    h_arg.ctypes.data_as(ctypes.c_void_p))
    if code:
        raise RuntimeError(f"Native pricing backend returned error {code}")
    return kappa, h_arg, {
        "backend": "cpp_openmp", "setup_pack_seconds": setup,
        "cost_and_h_reduction_seconds": time.perf_counter() - core_start,
        "threads": int(threads)}


def _compute_gpu(p):
    start = time.perf_counter()
    _configure_cupy_runtime()
    import cupy as cp

    root, second, third, allowed_root, allowed_second, allowed_third = _cost_arrays(p)
    words = np.stack([[_pack_words(p.X[:, f] == b) for f in range(p.F)]
                      for b in range(2)])
    positive = _pack_words(p.y == p.labels[1])
    F, W = p.F, words.shape[-1]
    setup = time.perf_counter() - start
    compile_start = time.perf_counter()
    kernel = cp.RawKernel(_D3_KERNEL, "d3_kappa", options=("--std=c++11",))
    kernel.compile()
    compile_seconds = time.perf_counter() - compile_start
    transfer_start = time.perf_counter()
    inputs = [cp.asarray(x) for x in
              (words, positive, root, second, third,
               allowed_root, allowed_second, allowed_third)]
    d_kappa = cp.empty((4, F, F), dtype=cp.float64)
    d_arg = cp.empty((4, F, F), dtype=cp.int32)
    cp.cuda.Stream.null.synchronize()
    transfer_seconds = time.perf_counter() - transfer_start
    core_start = time.perf_counter()
    kernel((F * F,), (256,),
           (*inputs, np.int32(F), np.int32(W), np.int32(p.no_repeat),
            np.int32(p.early_stop), np.int32(p.min_leaf),
            np.float64(p._uniform_weight), d_kappa, d_arg))
    cp.cuda.Stream.null.synchronize()
    core_seconds = time.perf_counter() - core_start
    copy_start = time.perf_counter()
    kappa, h_arg = cp.asnumpy(d_kappa), cp.asnumpy(d_arg)
    copy_seconds = time.perf_counter() - copy_start
    return kappa, h_arg, {
        "backend": "cuda", "setup_pack_seconds": setup,
        "kernel_compile_seconds": compile_seconds,
        "host_to_device_seconds": transfer_seconds,
        "cost_and_h_reduction_seconds": core_seconds,
        "device_to_host_seconds": copy_seconds,
        "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}


class D3PricingTable:
    """One cheapest private tail per separator signature, with exact pricing."""

    def __init__(self, domain, backend="cpp", policy="best", compatible_bundles=False,
                 threads=0):
        _require_supported_problem(domain.p)
        if domain.r != 1:
            raise ValueError("D3 pricing table requires tail_depth=1")
        if backend not in {"cpp", "gpu"}:
            raise ValueError("D3 pricing backend must be cpp or gpu")
        if policy not in {"first", "best"}:
            raise ValueError("D3 pricing policy must be first or best")
        self.domain, self.p = domain, domain.p
        self.backend, self.policy = backend, policy
        self.compatible_bundles = bool(compatible_bundles)
        started = time.perf_counter()
        kappa, h_arg, metadata = (_compute_cpp(self.p, threads) if backend == "cpp"
                                  else _compute_gpu(self.p))
        self.columns = self._build_columns(kappa, h_arg)
        self.costs = [np.fromiter((c.cost for c in block), dtype=np.float64,
                                  count=len(block)) for block in self.columns]
        self.index = [{column: j for j, column in enumerate(block)}
                      for block in self.columns]
        self.prefix_index = [{column.prefix: j for j, column in enumerate(block)}
                             for block in self.columns]
        self.active = [np.zeros(len(block), dtype=bool) for block in self.columns]
        self.edge_groups = self._build_edge_groups()
        self.side_groups = self._build_side_groups()
        metadata.update({
            "policy": policy, "compatible_bundles": self.compatible_bundles,
            "representatives": int(sum(map(len, self.columns))),
            "table_build_seconds": time.perf_counter() - started})
        self.metadata = metadata

    def _build_columns(self, kappa, h_arg):
        p, F = self.p, self.p.F
        blocks = []
        for q in range(4):
            a, b = q >> 1, q & 1
            block = []
            if p.early_stop and len(p.all_rows) >= p.min_leaf:
                for label in p.labels:
                    block.append(Column(((STOP, label), DEAD), None,
                                        p.loss(p.all_rows, label) / 4.0, 0.0))
            for f in p.features(()) :
                rows_a = p.route(p.all_rows, f, a)
                root_allocated = p.cost((), f) / 4.0
                if p.early_stop and len(rows_a) >= p.min_leaf:
                    for label in p.labels:
                        block.append(Column(((SPLIT, f), (STOP, label)), None,
                                            root_allocated + p.loss(rows_a, label) / 2.0,
                                            0.25))
                for g in p.features((a,), (f,)):
                    action = int(h_arg[q, f, g])
                    if action == -2 or not math.isfinite(kappa[q, f, g]):
                        continue
                    rows_ab = p.route(rows_a, g, b)
                    if action == -1:
                        tail = Tree(label=p.best_label(rows_ab))
                        tail_splits = 0.0
                    else:
                        tail = Tree(feature=action,
                                    left=Tree(label=p.best_label(p.route(rows_ab, action, 0))),
                                    right=Tree(label=p.best_label(p.route(rows_ab, action, 1))))
                        tail_splits = 1.0
                    block.append(Column(((SPLIT, f), (SPLIT, g)), tail,
                                        float(kappa[q, f, g]), 0.75 + tail_splits))
            blocks.append(block)
        return blocks

    def _build_edge_groups(self):
        grouped = []
        for i, block in enumerate(self.columns):
            incident = []
            for edge, size, sign in self.domain.incident[i]:
                unique, ids, inverse = [], {}, np.empty(len(block), dtype=np.int32)
                for j, column in enumerate(block):
                    key = column.prefix[:size]
                    if key not in ids:
                        ids[key] = len(unique)
                        unique.append(key)
                    inverse[j] = ids[key]
                incident.append((edge, sign, unique, inverse))
            grouped.append(incident)
        return grouped

    def _build_side_groups(self):
        output = []
        for left, right in ((0, 1), (2, 3)):
            common = self.prefix_index[left].keys() & self.prefix_index[right].keys()
            by_root = defaultdict(list)
            for prefix in common:
                by_root[prefix[0]].append(
                    (prefix, self.prefix_index[left][prefix], self.prefix_index[right][prefix]))
            output.append({root: (prefixes,
                                  np.asarray([x[1] for x in entries], dtype=np.int32),
                                  np.asarray([x[2] for x in entries], dtype=np.int32))
                           for root, entries in by_root.items()
                           for prefixes in [[x[0] for x in entries]]})
        return output

    def activate_pool(self, pool):
        for i, columns in enumerate(pool):
            for column in columns:
                self.activate(i, column)

    def activate(self, i, column):
        j = self.index[i].get(column)
        if j is not None:
            self.active[i][j] = True

    def _reduced_costs(self, i, alpha, pi):
        values = self.costs[i].copy()
        values -= alpha[i]
        for edge, sign, unique, inverse in self.edge_groups[i]:
            dual = np.fromiter((pi.get((edge, key), 0.0) for key in unique),
                               dtype=np.float64, count=len(unique))
            values -= sign * dual[inverse]
        return values

    def _best_side_prefix(self, side, root, reduced):
        entry = self.side_groups[side].get(root)
        if entry is None:
            return None
        prefixes, left_ids, right_ids = entry
        offset = 2 * side
        scores = reduced[offset][left_ids] + reduced[offset + 1][right_ids]
        return prefixes[int(np.argmin(scores))]

    def price(self, alpha, pi, columns_per_block, tolerance):
        reduced = [self._reduced_costs(i, alpha, pi) for i in range(4)]
        minima = [min(0.0, float(values.min(initial=0.0))) for values in reduced]
        chosen = []
        for i, values in enumerate(reduced):
            eligible = np.flatnonzero((values < -tolerance) & ~self.active[i])
            if len(eligible) > columns_per_block:
                if self.policy == "first":
                    eligible = eligible[:columns_per_block]
                else:
                    part = np.argpartition(values[eligible], columns_per_block - 1)[:columns_per_block]
                    eligible = eligible[part]
            if self.policy == "best" and len(eligible):
                eligible = eligible[np.lexsort((eligible, values[eligible]))]
            chosen.extend((i, int(j)) for j in eligible)

        if self.compatible_bundles and chosen:
            bundles, best_cache = set(), {}
            for i, j in chosen:
                column = self.columns[i][j]
                root = column.prefix[0]
                fixed_side = 0 if i < 2 else 1
                prefixes = [None, None]
                prefixes[fixed_side] = column.prefix
                other = 1 - fixed_side
                key = other, root
                if key not in best_cache:
                    best_cache[key] = self._best_side_prefix(other, root, reduced)
                prefixes[other] = best_cache[key]
                if prefixes[0] is None or prefixes[1] is None:
                    continue
                for block in range(4):
                    prefix = prefixes[block // 2]
                    candidate = self.prefix_index[block].get(prefix)
                    if candidate is not None and not self.active[block][candidate]:
                        bundles.add((block, candidate))
            chosen.extend(sorted(bundles - set(chosen)))
        return minima, [(i, self.columns[i][j]) for i, j in chosen]
