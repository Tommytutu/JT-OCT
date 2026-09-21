"""Batched min-sum elimination for binary depth-three OCT instances.

The generic JT-DP enumerates path configurations as Python ``Column`` objects.
For D=3, the same variable-elimination order can be evaluated directly:

    kappa[q, f, g] = min(STOP, min_h c[q, f, g, h]),
    side[a, f] = min(STOP, min_g kappa[2a,f,g]+kappa[2a+1,f,g]),
    value = min(STOP, min_f side[0,f]+side[1,f]).

The CUDA backend evaluates local sufficient statistics and the h reductions in
one kernel.  Only O(F^2) values and backpointers are retained; no F^3 tensor is
materialized.  Tagged STOP choices make the recurrence exact for sparse trees.
"""
from __future__ import annotations

import math
import os
import sys
import time
import ctypes
from pathlib import Path

import numpy as np

from .problem import Deadline, DeadlineExceeded, Tree, evaluate
from .solvers import result_dict


_CUDA_DLL_HANDLES = []
_CUDA_CONFIGURED_ROOTS = set()


def _cost_arrays(p):
    """Return node/predicate costs and allowed-predicate indicators."""
    F = p.F
    root = np.array([p.cost((), f) for f in range(F)], dtype=np.float64)
    second = np.array([[p.cost((a,), f) for f in range(F)] for a in range(2)],
                      dtype=np.float64)
    third = np.array([[p.cost((q >> 1, q & 1), f) for f in range(F)]
                      for q in range(4)], dtype=np.float64)

    def permitted(node):
        out = np.zeros(F, dtype=np.uint8)
        out[list(p.allowed.get(node, range(F)))] = 1
        return out

    allowed_root = permitted(())
    allowed_second = np.stack([permitted((a,)) for a in range(2)])
    allowed_third = np.stack([permitted((q >> 1, q & 1)) for q in range(4)])
    return root, second, third, allowed_root, allowed_second, allowed_third


def _require_supported_problem(p):
    if p.depth != 3:
        raise ValueError("Batched JT-DP is specialized to depth 3")
    if len(p.labels) != 2:
        raise ValueError("Batched JT-DP currently requires two classes")
    if p._uniform_weight is None:
        raise ValueError("Batched JT-DP currently requires uniform observation weights")


def _stop_costs(p, root_cost):
    """Compute root/side STOP costs and the corresponding routed row sets."""
    root_rows = p.all_rows
    root_label, root_loss = p.best_label_and_loss(root_rows)
    root_stop = float(root_loss) if p.early_stop and len(root_rows) >= p.min_leaf else math.inf
    side_stop = np.full((2, p.F), math.inf, dtype=np.float64)
    side_rows = {}
    for f in range(p.F):
        if not p.allowed.get((), range(p.F)) or f not in p.allowed.get((), range(p.F)):
            continue
        for a in range(2):
            rows = p.route(root_rows, f, a)
            side_rows[f, a] = rows
            if p.early_stop and len(rows) >= p.min_leaf:
                _, loss = p.best_label_and_loss(rows)
                side_stop[a, f] = 0.5 * root_cost[f] + loss
    return root_stop, root_label, side_stop, side_rows


def _join_and_recover(p, kappa, h_arg, root_cost, allowed_root,
                      side_stop, side_rows):
    """Perform the small CPU g reduction and reconstruct the optimal tree."""
    F = p.F
    side_value = np.full((2, F), math.inf, dtype=np.float64)
    g_arg = np.full((2, F), -2, dtype=np.int32)
    for a in range(2):
        for f in range(F):
            if not allowed_root[f]:
                continue
            best, action = float(side_stop[a, f]), -1
            sums = kappa[2 * a, f] + kappa[2 * a + 1, f]
            g = int(np.argmin(sums))
            if sums[g] < best:
                best, action = float(sums[g]), g
            side_value[a, f], g_arg[a, f] = best, action

    return _recover_from_side(p, h_arg, root_cost, side_value, g_arg, side_rows)


def _recover_from_side(p, h_arg, root_cost, side_value, g_arg, side_rows):
    """Perform the final f comparison and reconstruct from stored argmins."""
    root_stop, _, _, _ = _stop_costs(p, root_cost)
    split_values = side_value[0] + side_value[1]
    f = int(np.argmin(split_values))
    if not math.isfinite(root_stop) and not math.isfinite(split_values[f]):
        return math.inf, None, {
            "root_action": "INFEASIBLE", "side_values": side_value, "g_arg": g_arg}
    if root_stop <= split_values[f]:
        return float(root_stop), Tree(label=p.best_label(p.all_rows)), {
            "root_action": "STOP", "side_values": side_value, "g_arg": g_arg}

    def leaf(rows):
        return Tree(label=p.best_label(rows))

    def make_side(a):
        rows_a = side_rows[f, a]
        g = int(g_arg[a, f])
        if g == -1:
            return leaf(rows_a)
        children = []
        for b in range(2):
            q = 2 * a + b
            rows_ab = p.route(rows_a, g, b)
            h = int(h_arg[q, f, g])
            if h == -1:
                children.append(leaf(rows_ab))
            else:
                children.append(Tree(feature=h,
                                     left=leaf(p.route(rows_ab, h, 0)),
                                     right=leaf(p.route(rows_ab, h, 1))))
        return Tree(feature=g, left=children[0], right=children[1])

    tree = Tree(feature=f, left=make_side(0), right=make_side(1))
    return float(split_values[f]), tree, {
        "root_action": int(f), "side_values": side_value, "g_arg": g_arg}


def solve_jt_dp_d3_batched_cpu(p, time_limit=600):
    """Exact D3 min-sum elimination using Python integer bitsets, without Columns."""
    _require_supported_problem(p)
    clock = Deadline(time_limit)
    timings = {}
    try:
        start = time.perf_counter()
        root_cost, second_cost, third_cost, allowed_root, allowed_second, allowed_third = _cost_arrays(p)
        root_stop, _, side_stop, side_rows = _stop_costs(p, root_cost)
        timings["setup_seconds"] = time.perf_counter() - start
        F, weight = p.F, float(p._uniform_weight)
        positive_mask = p._label_masks[p.labels[1]]
        masks = p._feature_masks
        kappa = np.full((4, F, F), math.inf, dtype=np.float64)
        h_arg = np.full((4, F, F), -2, dtype=np.int32)

        reduce_start = time.perf_counter()
        triples = 0
        for f in range(F):
            clock.check()
            if not allowed_root[f]:
                continue
            for g in range(F):
                if not allowed_second[0, g] and not allowed_second[1, g]:
                    continue
                if p.no_repeat and g == f:
                    continue
                rows = [masks[f][q >> 1] & masks[g][q & 1] for q in range(4)]
                totals = [r.bit_count() for r in rows]
                positives = [(r & positive_mask).bit_count() for r in rows]
                best = [math.inf] * 4
                action = [-2] * 4
                for q in range(4):
                    a = q >> 1
                    if not allowed_second[a, g]:
                        continue
                    if p.early_stop and totals[q] >= p.min_leaf:
                        err = min(positives[q], totals[q] - positives[q])
                        best[q] = 0.25 * root_cost[f] + 0.5 * second_cost[a, g] + weight * err
                        action[q] = -1
                for h in range(F):
                    if p.no_repeat and (h == f or h == g):
                        continue
                    h0 = masks[h][0]
                    for q in range(4):
                        a = q >> 1
                        if not allowed_second[a, g] or not allowed_third[q, h]:
                            continue
                        left = rows[q] & h0
                        n0 = left.bit_count()
                        n1 = totals[q] - n0
                        if n0 < p.min_leaf or n1 < p.min_leaf:
                            continue
                        y10 = (left & positive_mask).bit_count()
                        y11 = positives[q] - y10
                        errors = min(y10, n0 - y10) + min(y11, n1 - y11)
                        value = (0.25 * root_cost[f] + 0.5 * second_cost[a, g] +
                                 third_cost[q, h] + weight * errors)
                        triples += 1
                        if value < best[q]:
                            best[q], action[q] = value, h
                for q in range(4):
                    kappa[q, f, g], h_arg[q, f, g] = best[q], action[q]
        timings["h_reduction_seconds"] = time.perf_counter() - reduce_start
        join_start = time.perf_counter()
        value, tree, join = _join_and_recover(
            p, kappa, h_arg, root_cost, allowed_root, side_stop, side_rows)
        timings["join_traceback_seconds"] = time.perf_counter() - join_start
        if tree is None:
            return result_dict("JT-DP-Batched-CPU", clock, "INFEASIBLE", lb=math.inf,
                               timings=timings)
        actual = evaluate(p, tree)
        if abs(actual["objective"] - value) > 1e-9:
            raise AssertionError("Batched CPU recurrence disagrees with routed objective")
        return result_dict("JT-DP-Batched-CPU", clock, "OPT", tree, p, value,
                           message_passing="d3_batched_min_sum",
                           persistent_complexity="O(F^2)", triples_evaluated=triples,
                           timings=timings)
    except DeadlineExceeded:
        return result_dict("JT-DP-Batched-CPU", clock, "TIME", timings=timings)


_D3_KERNEL = r'''
extern "C" __global__
void d3_kappa(const unsigned int* masks, const unsigned int* positive,
              const double* root_cost, const double* second_cost,
              const double* third_cost, const unsigned char* allowed_root,
              const unsigned char* allowed_second, const unsigned char* allowed_third,
              const int F, const int W, const int no_repeat,
              const int early_stop, const int min_leaf, const double weight,
              double* output, int* argmin_h) {
    const int pair = blockIdx.x;
    const int f = pair / F;
    const int g = pair - f * F;
    const int tid = threadIdx.x;
    const double INF = 1.0e300;
    __shared__ int row_total[4];
    __shared__ int row_positive[4];
    __shared__ double costs[4][256];
    __shared__ int actions[4][256];

    if (tid < 4) {
        const int q = tid;
        const int a = q >> 1;
        const int b = q & 1;
        int total = 0, pos = 0;
        if (allowed_root[f] && allowed_second[a * F + g] && (!no_repeat || f != g)) {
            for (int w = 0; w < W; ++w) {
                unsigned int rows = masks[(a * F + f) * W + w]
                                  & masks[(b * F + g) * W + w];
                total += __popc(rows);
                pos += __popc(rows & positive[w]);
            }
        }
        row_total[q] = total;
        row_positive[q] = pos;
    }
    __syncthreads();

    double local_cost[4] = {INF, INF, INF, INF};
    int local_action[4] = {-2, -2, -2, -2};
    if (allowed_root[f] && (!no_repeat || f != g)) {
        for (int h = tid; h < F; h += blockDim.x) {
            if (no_repeat && (h == f || h == g)) continue;
            for (int q = 0; q < 4; ++q) {
                const int a = q >> 1;
                const int b = q & 1;
                if (!allowed_second[a * F + g] || !allowed_third[q * F + h]) continue;
                int n0 = 0, y10 = 0;
                for (int w = 0; w < W; ++w) {
                    unsigned int left = masks[(a * F + f) * W + w]
                                      & masks[(b * F + g) * W + w]
                                      & masks[(0 * F + h) * W + w];
                    n0 += __popc(left);
                    y10 += __popc(left & positive[w]);
                }
                const int n1 = row_total[q] - n0;
                if (n0 < min_leaf || n1 < min_leaf) continue;
                const int y11 = row_positive[q] - y10;
                const int err0 = y10 < n0 - y10 ? y10 : n0 - y10;
                const int err1 = y11 < n1 - y11 ? y11 : n1 - y11;
                const double value = 0.25 * root_cost[f]
                                   + 0.5 * second_cost[a * F + g]
                                   + third_cost[q * F + h]
                                   + weight * (double)(err0 + err1);
                if (value < local_cost[q] ||
                    (value == local_cost[q] && (local_action[q] < 0 || h < local_action[q]))) {
                    local_cost[q] = value;
                    local_action[q] = h;
                }
            }
        }
    }
    for (int q = 0; q < 4; ++q) {
        costs[q][tid] = local_cost[q];
        actions[q][tid] = local_action[q];
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int q = 0; q < 4; ++q) {
                const double other = costs[q][tid + stride];
                const int other_action = actions[q][tid + stride];
                if (other < costs[q][tid] ||
                    (other == costs[q][tid] && other_action >= 0 &&
                     (actions[q][tid] < 0 || other_action < actions[q][tid]))) {
                    costs[q][tid] = other;
                    actions[q][tid] = other_action;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int q = 0; q < 4; ++q) {
            const int a = q >> 1;
            double best = costs[q][0];
            int action = actions[q][0];
            if (allowed_root[f] && allowed_second[a * F + g] &&
                (!no_repeat || f != g) && early_stop && row_total[q] >= min_leaf) {
                const int err = row_positive[q] < row_total[q] - row_positive[q]
                              ? row_positive[q] : row_total[q] - row_positive[q];
                const double stop = 0.25 * root_cost[f]
                                  + 0.5 * second_cost[a * F + g]
                                  + weight * (double)err;
                if (stop <= best) {
                    best = stop;
                    action = -1;
                }
            }
            const int out = (q * F + f) * F + g;
            output[out] = best;
            argmin_h[out] = action;
        }
    }
}
'''


_JOIN_KERNEL = r'''
extern "C" __global__
void d3_join(const double* kappa, const double* side_stop, const int F,
             double* side_value, int* argmin_g) {
    const int item = blockIdx.x;
    const int side = item / F;
    const int f = item - side * F;
    const int tid = threadIdx.x;
    const double INF = 1.0e300;
    __shared__ double costs[256];
    __shared__ int actions[256];
    double best = INF;
    int action = -2;
    for (int g = tid; g < F; g += blockDim.x) {
        const double value = kappa[((2 * side) * F + f) * F + g]
                           + kappa[((2 * side + 1) * F + f) * F + g];
        if (value < best || (value == best && (action < 0 || g < action))) {
            best = value;
            action = g;
        }
    }
    costs[tid] = best;
    actions[tid] = action;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double other = costs[tid + stride];
            const int other_action = actions[tid + stride];
            if (other < costs[tid] ||
                (other == costs[tid] && other_action >= 0 &&
                 (actions[tid] < 0 || other_action < actions[tid]))) {
                costs[tid] = other;
                actions[tid] = other_action;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        const double stop = side_stop[side * F + f];
        if (stop <= costs[0]) {
            costs[0] = stop;
            actions[0] = -1;
        }
        side_value[side * F + f] = costs[0];
        argmin_g[side * F + f] = actions[0];
    }
}
'''


def _pack_words(values):
    packed = np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little")
    packed = np.pad(packed, (0, (-len(packed)) % 4))
    return np.frombuffer(packed.tobytes(), dtype="<u4").copy()


def _pack_words64(values):
    packed = np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little")
    packed = np.pad(packed, (0, (-len(packed)) % 8))
    return np.frombuffer(packed.tobytes(), dtype="<u8").copy()


def solve_jt_dp_d3_cpp(p, time_limit=600, threads=0):
    """Exact C++/OpenMP bitset implementation of the D3 min-sum recurrence."""
    _require_supported_problem(p)
    clock = Deadline(time_limit)
    timings = {}
    try:
        setup_start = time.perf_counter()
        library_path = Path(__file__).parent / "_native" / "d3_kappa.dll"
        if not library_path.exists():
            raise RuntimeError("Native backend is not built; run build_native.ps1")
        library = ctypes.CDLL(str(library_path))
        function = library.d3_kappa_cpu
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 5 + [ctypes.c_double,
                             ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        root_cost, second_cost, third_cost, allowed_root, allowed_second, allowed_third = _cost_arrays(p)
        _, _, side_stop, side_rows = _stop_costs(p, root_cost)
        words = np.ascontiguousarray(np.stack(
            [[_pack_words64(p.X[:, f] == b) for f in range(p.F)] for b in range(2)]))
        positive = np.ascontiguousarray(_pack_words64(p.y == p.labels[1]))
        arrays = [np.ascontiguousarray(x) for x in
                  (root_cost, second_cost, third_cost, allowed_root,
                   allowed_second, allowed_third)]
        root_cost, second_cost, third_cost, allowed_root, allowed_second, allowed_third = arrays
        F, W = p.F, words.shape[-1]
        kappa = np.empty((4, F, F), dtype=np.float64)
        h_arg = np.empty((4, F, F), dtype=np.int32)
        timings["setup_pack_seconds"] = time.perf_counter() - setup_start
        reduction_start = time.perf_counter()
        pointers = [x.ctypes.data_as(ctypes.c_void_p) for x in
                    (words, positive, root_cost, second_cost, third_cost,
                     allowed_root, allowed_second, allowed_third)]
        code = function(*pointers, F, W, int(p.no_repeat), int(p.early_stop),
                        p.min_leaf, float(p._uniform_weight), int(threads),
                        kappa.ctypes.data_as(ctypes.c_void_p),
                        h_arg.ctypes.data_as(ctypes.c_void_p))
        if code != 0:
            raise RuntimeError(f"Native backend returned error {code}")
        timings["cpp_cost_and_h_reduction_seconds"] = time.perf_counter() - reduction_start
        clock.check()
        join_start = time.perf_counter()
        value, tree, _ = _join_and_recover(
            p, kappa, h_arg, root_cost, allowed_root, side_stop, side_rows)
        timings["cpu_g_f_join_traceback_seconds"] = time.perf_counter() - join_start
        if tree is None:
            return result_dict("JT-DP-CPP", clock, "INFEASIBLE", lb=math.inf,
                               timings=timings)
        actual = evaluate(p, tree)
        if abs(actual["objective"] - value) > 1e-9:
            raise AssertionError("C++ recurrence disagrees with independently routed objective")
        triples = 4 * F * (F - 1 if p.no_repeat else F) * (F - 2 if p.no_repeat else F)
        return result_dict("JT-DP-CPP", clock, "OPT", tree, p, value,
                           message_passing="cpp_openmp_d3_batched_min_sum",
                           persistent_complexity="O(F^2)", triples_evaluated=triples,
                           threads=int(threads), timings=timings)
    except DeadlineExceeded:
        return result_dict("JT-DP-CPP", clock, "TIME", timings=timings)


def _configure_cupy_runtime():
    """Configure a CUDA installation selected by environment variables."""
    if os.name != "nt":
        return
    candidate = os.environ.get("JT_OCT_CUDA_ROOT") or os.environ.get("CUDA_PATH")
    if not candidate:
        return
    root = Path(candidate)
    os.environ["CUDA_PATH"] = str(root)
    binary = str(root / "bin")
    key = str(root.resolve()).casefold()
    if key not in _CUDA_CONFIGURED_ROOTS:
        existing = os.environ.get("PATH", "").split(os.pathsep)
        if binary.casefold() not in {part.casefold() for part in existing}:
            os.environ["PATH"] = binary + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory") and (root / "bin").exists():
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(binary))
        _CUDA_CONFIGURED_ROOTS.add(key)


def solve_jt_dp_d3_gpu(p, time_limit=600):
    """Exact CUDA D3 JT-DP with fused bitset statistics and h reduction."""
    _require_supported_problem(p)
    clock = Deadline(time_limit)
    timings = {}
    try:
        setup_start = time.perf_counter()
        _configure_cupy_runtime()
        import cupy as cp

        root_cost, second_cost, third_cost, allowed_root, allowed_second, allowed_third = _cost_arrays(p)
        _, _, side_stop, side_rows = _stop_costs(p, root_cost)
        words = np.stack([[_pack_words(p.X[:, f] == b) for f in range(p.F)]
                          for b in range(2)])
        positive = _pack_words(p.y == p.labels[1])
        F, W = p.F, words.shape[-1]
        timings["setup_pack_seconds"] = time.perf_counter() - setup_start

        compile_start = time.perf_counter()
        kernel = cp.RawKernel(_D3_KERNEL, "d3_kappa", options=("--std=c++11",))
        join_kernel = cp.RawKernel(_JOIN_KERNEL, "d3_join", options=("--std=c++11",))
        kernel.compile()
        join_kernel.compile()
        timings["kernel_compile_seconds"] = time.perf_counter() - compile_start

        transfer_start = time.perf_counter()
        d_words = cp.asarray(words)
        d_positive = cp.asarray(positive)
        d_root = cp.asarray(root_cost)
        d_second = cp.asarray(second_cost)
        d_third = cp.asarray(third_cost)
        d_allowed_root = cp.asarray(allowed_root)
        d_allowed_second = cp.asarray(allowed_second)
        d_allowed_third = cp.asarray(allowed_third)
        d_side_stop = cp.asarray(side_stop)
        d_kappa = cp.empty((4, F, F), dtype=cp.float64)
        d_arg = cp.empty((4, F, F), dtype=cp.int32)
        d_side_value = cp.empty((2, F), dtype=cp.float64)
        d_g_arg = cp.empty((2, F), dtype=cp.int32)
        cp.cuda.Stream.null.synchronize()
        timings["host_to_device_seconds"] = time.perf_counter() - transfer_start

        reduction_start = time.perf_counter()
        kernel((F * F,), (256,),
               (d_words, d_positive, d_root, d_second, d_third,
                d_allowed_root, d_allowed_second, d_allowed_third,
                np.int32(F), np.int32(W), np.int32(p.no_repeat),
                np.int32(p.early_stop), np.int32(p.min_leaf),
                np.float64(p._uniform_weight), d_kappa, d_arg))
        cp.cuda.Stream.null.synchronize()
        timings["gpu_cost_and_h_reduction_seconds"] = time.perf_counter() - reduction_start
        clock.check()

        join_start = time.perf_counter()
        join_kernel((2 * F,), (256,),
                    (d_kappa, d_side_stop, np.int32(F), d_side_value, d_g_arg))
        cp.cuda.Stream.null.synchronize()
        timings["gpu_g_reduction_seconds"] = time.perf_counter() - join_start

        copy_start = time.perf_counter()
        h_arg = cp.asnumpy(d_arg)
        side_value = cp.asnumpy(d_side_value)
        g_arg = cp.asnumpy(d_g_arg)
        timings["device_to_host_seconds"] = time.perf_counter() - copy_start
        traceback_start = time.perf_counter()
        value, tree, join = _recover_from_side(
            p, h_arg, root_cost, side_value, g_arg, side_rows)
        timings["cpu_f_join_traceback_seconds"] = time.perf_counter() - traceback_start
        if tree is None:
            return result_dict("JT-DP-GPU", clock, "INFEASIBLE", lb=math.inf,
                               timings=timings)
        actual = evaluate(p, tree)
        if abs(actual["objective"] - value) > 1e-9:
            raise AssertionError("GPU recurrence disagrees with independently routed objective")

        triples = 4 * F * (F - 1 if p.no_repeat else F) * (F - 2 if p.no_repeat else F)
        memory = int(d_words.nbytes + d_positive.nbytes + d_root.nbytes + d_second.nbytes +
                     d_third.nbytes + d_allowed_root.nbytes + d_allowed_second.nbytes +
                     d_allowed_third.nbytes + d_side_stop.nbytes + d_kappa.nbytes +
                     d_arg.nbytes + d_side_value.nbytes + d_g_arg.nbytes)
        return result_dict("JT-DP-GPU", clock, "OPT", tree, p, value,
                           message_passing="cuda_d3_batched_min_sum",
                           persistent_complexity="O(F^2)", triples_evaluated=triples,
                           persistent_device_bytes=memory,
                           gpu_name=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                           cupy_version=cp.__version__, timings=timings)
    except DeadlineExceeded:
        return result_dict("JT-DP-GPU", clock, "TIME", timings=timings)
