"""Direct min-sum messages on the paper's lexicographic path-cluster chain.

The common root action is conditioned only to tile the separator state space.
Every tile processes all clusters in chain order and explicitly uses the incoming
separator message. No subtree solver or D3 oracle is invoked. Full prefix costs
are streamed; a terminal-private feature is minimized inside each numerical kernel.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import time

import numpy as np

from .cg import greedy_feasible
from .conflicts import conflict_lower_bound
from .d3_batched import _pack_words64, _configure_cupy_runtime
from .domain import Column, Domain, DEAD, SPLIT, STOP
from .problem import Deadline, DeadlineExceeded, RowSet, Tree, evaluate
from .solvers import result_dict

INF = 1.0e300
ROOT = Path(__file__).resolve().parents[1]


def _digits(index, F, D, root):
    features = [root] + [0] * (D - 2)
    for j in range(D - 2, 0, -1):
        features[j] = index % (F + 1)
        index //= F + 1
    return features


def _column(p, q, root, index, h):
    """Reconstruct only a winning path configuration, with independent costs."""
    rho, rows, cost, splits, alive = [], p.all_rows, 0.0, 0.0, True
    for level, feature in enumerate(_digits(index, p.F, p.depth, root)):
        mult = 2 ** (p.depth - 1 - level)
        if not alive:
            rho.append(DEAD)
        elif feature == p.F:
            label = p.best_label(rows)
            rho.append((STOP, label))
            cost += p.loss(rows, label) / mult
            alive = False
        else:
            rho.append((SPLIT, feature))
            cost += p.penalty / mult
            splits += 1.0 / mult
            bit = (q >> (p.depth - 2 - level)) & 1
            rows = p.route(rows, feature, bit)
    tail = None
    if alive:
        if h == -1:
            tail = Tree(label=p.best_label(rows))
            cost += p.loss(rows, tail.label)
        elif h >= 0:
            left, right = p.route(rows, h, 0), p.route(rows, h, 1)
            tail = Tree(feature=h, left=Tree(label=p.best_label(left)),
                        right=Tree(label=p.best_label(right)))
            cost += p.penalty + p.loss(left, tail.left.label) + p.loss(right, tail.right.label)
            splits += 1
        else:
            raise AssertionError("Winning message contains an infeasible private action")
    return Column(tuple(rho), tail, cost, splits)


def _python_costs(p, incoming, root, q, start, count, stride, clock):
    output, tail = np.full(count, INF), np.full(count, -2, dtype=np.int32)
    D, F = p.depth, p.F
    for local in range(count):
        if local % 16 == 0:
            clock.check()
        index = start + local
        previous = float(incoming[index // stride])
        if previous >= INF / 2:
            continue
        features = _digits(index, F, D, root)
        stop = next((j for j, f in enumerate(features) if f == F), D-1)
        active = features[:stop]
        if any(f != F for f in features[stop:]) or (p.no_repeat and len(set(active)) < len(active)):
            continue
        rows = p.all_rows
        allocated = 0.0
        for j, f in enumerate(active):
            allocated += p.penalty / 2**(D-1-j)
            rows = p.route(rows, f, (q >> (D-2-j)) & 1)
        total = len(rows)
        counts = [(rows.mask & p._label_masks[k]).bit_count() for k in p.labels]
        error = total-max(counts)
        if stop < D-1:
            if total >= p.min_leaf:
                output[local] = previous + allocated + error*p._uniform_weight/2**(D-1-stop)
                tail[local] = -3
            continue
        best = error*p._uniform_weight if total >= p.min_leaf else INF
        action = -1 if best < INF/2 else -2
        if best != 0.0:
            for h in range(F):
                if p.no_repeat and h in active:
                    continue
                mask = rows.mask & p._feature_masks[h][0]
                n0 = mask.bit_count()
                left_counts = [(mask & p._label_masks[k]).bit_count() for k in p.labels]
                n1 = total-n0
                if min(n0, n1) < p.min_leaf:
                    continue
                candidate = p.penalty + p._uniform_weight*(total-max(left_counts)-max(c-l for c,l in zip(counts,left_counts)))
                if candidate < best:
                    best, action = candidate, h
        if best < INF/2:
            output[local], tail[local] = previous+allocated+best, action
    return output, tail


class MessageBackend:
    def __init__(self, p, backend, threads):
        self.p, self.backend, self.threads, self.xp = p, backend, threads, np
        if backend == "python":
            return
        masks = np.ascontiguousarray(np.stack([
            [_pack_words64(p.X[:, f] == b) for f in range(p.F)] for b in (0,1)]))
        self.multiclass = len(p.labels) != 2
        if self.multiclass and backend == "gpu":
            raise ValueError("Multiclass direct messages support python and cpp backends")
        positive = np.ascontiguousarray(np.stack([_pack_words64(p.y == k) for k in p.labels]) if self.multiclass else _pack_words64(p.y == p.labels[1]))
        self.W = masks.shape[-1]
        if backend == "cpp":
            self.library = ctypes.CDLL(str(ROOT/"jt_oct"/"_native"/"jt_message.dll"))
            self.fn = self.library.jt_path_costs_multiclass if self.multiclass else self.library.jt_path_costs
            self.fn.restype = ctypes.c_int
            self.fn.argtypes = ([ctypes.c_void_p]*3 + [ctypes.c_int]*(8 if self.multiclass else 7) +
                               [ctypes.c_double]*2 + [ctypes.c_int]*4 +
                               [ctypes.c_void_p]*2)
            self.masks, self.positive = masks, positive
        elif backend == "gpu":
            _configure_cupy_runtime()
            import cupy as cp
            self.xp = cp
            self.kernel = cp.RawKernel((ROOT/"native"/"jt_message.cu").read_text(),
                                       "jt_path_costs_gpu", options=("--std=c++11",))
            self.kernel.compile()
            self.masks, self.positive = cp.asarray(masks), cp.asarray(positive)
            cp.cuda.Stream.null.synchronize()
        else:
            raise ValueError("Message backend must be python, cpp or gpu")

    def costs(self, incoming, root, q, start, count, stride, clock):
        p = self.p
        if self.backend == "python":
            return _python_costs(p, incoming, root, q, start, count, stride, clock)
        output = self.xp.empty(count, dtype=np.float64)
        tail = self.xp.empty(count, dtype=np.int32)
        if self.backend == "cpp":
            ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
            result = self.fn(ptr(self.masks), ptr(self.positive), ptr(incoming),
                             p.F, self.W, *([len(p.labels)] if self.multiclass else []), p.depth, root, q, int(p.no_repeat), p.min_leaf,
                             float(p._uniform_weight), p.penalty,
                             start, count, stride, self.threads, ptr(output), ptr(tail))
            if result:
                raise RuntimeError(f"Native message kernel failed: {result}")
        else:
            self.kernel((count,), (256,),
                        (self.masks, self.positive, incoming,
                         *[np.int32(v) for v in (p.F,self.W,p.depth,root,q,int(p.no_repeat),p.min_leaf)],
                         np.float64(p._uniform_weight), np.float64(p.penalty),
                         np.int32(start),np.int32(count),np.int32(stride),output,tail),
                        shared_mem=self.W*8)
            self.xp.cuda.Stream.null.synchronize()
        return output, tail

    def host(self, a):
        return self.xp.asnumpy(a) if self.backend == "gpu" else np.asarray(a)


def solve_direct_message(p, backend="cpp", time_limit=100, threads=0,
                         tile_contexts=4096, progress=None, audit=None):
    """Direct chain min-sum with root-conditioned tiles; no subtree oracles.

    Only uniform-weight problems and a shared feature dictionary
    are currently supported. Root STOP and ancestor STOP states are explicitly
    retained; private STOP labels are optimized only because costs are separable.
    """
    if not 2 <= p.depth <= 5 or p._uniform_weight is None:
        raise ValueError("Direct numerical messages require D2--D5 and uniform weights")
    if not p.early_stop or p.allowed or p.split_costs:
        raise ValueError("Direct numerical messages require early stopping and shared uniform split costs")
    if tile_contexts < 1:
        raise ValueError("Positive context tile size required")
    clock = Deadline(time_limit)
    domain = Domain(p)
    B, K = p.F + 1, p.depth - 2
    N = B**K
    stats = {"backend":backend, "threads":threads if backend=="cpp" else None,
             "root_tiles_completed":0, "cluster_messages_completed":0,
             "prefix_contexts_processed":0, "prefix_contexts_per_cluster":N,
             "cost_and_private_reduction_seconds":0.0, "message_reduction_seconds":0.0,
             "traceback_seconds":0.0, "setup_seconds":0.0,
             "oracle_calls":0, "max_columns_applies":False,
             "tile_contexts":tile_contexts}
    best_tree = Tree(label=p.best_label(p.all_rows)) if p.n >= p.min_leaf else None
    best = evaluate(p,best_tree)["objective"] if best_tree else float("inf")
    lb = 0.0
    roots, last_progress = [], 0.0

    def publish(force=False):
        nonlocal last_progress
        if progress and (force or clock.elapsed()-last_progress >= 5.0):
            last_progress = clock.elapsed()
            progress(result_dict("JT-MP-"+backend.upper(),clock,"RUNNING",best_tree,p,lb,
                                 stats=dict(stats),root_trace=list(roots)))

    try:
        clock.check()
        lb = conflict_lower_bound(p)
        greedy = greedy_feasible(p,clock)
        if greedy is not None:
            objective = evaluate(p,greedy)["objective"]
            if objective < best:
                best_tree, best = greedy, objective
        publish(True)
        started = time.perf_counter()
        engine = MessageBackend(p,backend,threads)
        stats["setup_seconds"] = time.perf_counter()-started
        xp = engine.xp
        if backend == "gpu":
            stats["gpu_name"] = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        for root in range(p.F):
            clock.check()
            previous, records = xp.zeros(1,dtype=np.float64), []
            for q in range(domain.M):
                free_left = domain.separators[q-1]-1 if q else 0
                free_right = domain.separators[q]-1 if q<domain.M-1 else 0
                incoming_stride, group = B**(K-free_left), B**(K-free_right)
                values, tails = xp.empty(N,dtype=np.float64), xp.empty(N,dtype=np.int32)
                for start in range(0,N,tile_contexts):
                    clock.check()
                    count = min(tile_contexts,N-start)
                    tick = time.perf_counter()
                    try:
                        vals, acts = engine.costs(previous,root,q,start,count,incoming_stride,clock)
                    finally:
                        stats["cost_and_private_reduction_seconds"] += time.perf_counter()-tick
                    values[start:start+count], tails[start:start+count] = vals,acts
                    stats["prefix_contexts_processed"] += count
                    publish()
                tick = time.perf_counter()
                out_count = B**free_right
                grouped = values.reshape(out_count,group)
                winner = xp.argmin(grouped,axis=1)
                prefix = (xp.arange(out_count,dtype=np.int64)*group+winner).astype(np.int32)
                previous = values[prefix].copy()
                selected_tail = tails[prefix]
                records.append((engine.host(prefix),engine.host(selected_tail),incoming_stride))
                if audit is not None:
                    audit(root,q,free_right,engine.host(previous))
                stats["message_reduction_seconds"] += time.perf_counter()-tick
                stats["cluster_messages_completed"] += 1
                del values, tails, grouped, prefix, selected_tail, winner
            value = float(engine.host(previous)[0])
            tick = time.perf_counter()
            if value < INF/2:
                state, chosen = 0, []
                for q in range(domain.M-1,-1,-1):
                    prefixes, acts, stride = records[q]
                    index, h = int(prefixes[state]),int(acts[state])
                    chosen.append(_column(p,q,root,index,h))
                    state = index//stride
                tree = domain.recover(list(reversed(chosen)))
                objective = evaluate(p,tree)["objective"]
                if abs(objective-value) > 1e-8:
                    raise AssertionError(f"Direct messages/routed tree disagree: {value} vs {objective}")
                if objective < best-1e-12:
                    best_tree,best = tree,objective
            stats["traceback_seconds"] += time.perf_counter()-tick
            stats["root_tiles_completed"] += 1
            roots.append({"root":root,"objective":value if value<INF/2 else None,
                          "seconds":clock.elapsed(),"best_UB":best})
            del previous,records
            publish(True)
        if best_tree is None:
            return result_dict("JT-MP-"+backend.upper(),clock,"INFEASIBLE",lb=float("inf"),stats=stats)
        return result_dict("JT-MP-"+backend.upper(),clock,"OPT",best_tree,p,best,
                           stats=stats,root_trace=roots,message_passing="direct_path_cluster_chain",
                           optimality_scope="tree_problem")
    except DeadlineExceeded:
        return result_dict("JT-MP-"+backend.upper(),clock,"TIME",best_tree,p,lb,
                           stats=stats,root_trace=roots,message_passing="direct_path_cluster_chain",
                           optimality_scope="valid_incumbent_and_conflict_LB")
