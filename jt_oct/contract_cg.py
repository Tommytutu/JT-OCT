"""D4/D5 CG with D3 terminal clusters, lazy certified pricing, and native RMP.

Only private subtrees are contracted. Complete ancestor separator actions are
retained. Lower/upper local costs bound every unqueried D3 subtree; an incomplete
pricing pass can therefore give a valid (possibly weak) global dual bound.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import ctypes as ct
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .cg import greedy_feasible
from .domain import Column, Domain, SPLIT, STOP, CapacityExceeded
from .d3_grow import _frontier_paths, _prefix_state, _replace
from .problem import Deadline, DeadlineExceeded, Tree, evaluate
from .solvers import result_dict
from .terminal_message import TerminalMessages, TerminalOptions, Interval

_LIB = None
_DIRS = []


class NativeRMP:
    def __init__(self, blocks, features, classes, method=0):
        global _LIB
        self.handle = None
        if _LIB is None:
            for root in (os.environ.get('GUROBI_HOME'), r'C:\gurobi1300\win64'):
                if root and (Path(root)/'bin/gurobi130.dll').exists():
                    _DIRS.append(os.add_dll_directory(str(Path(root)/'bin')))
            lib = ct.CDLL(str(Path(__file__).parent/'_native/contract_rmp.dll'))
            lib.crmp_create.argtypes = [ct.c_int]*4; lib.crmp_create.restype = ct.c_void_p
            lib.crmp_update.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
            lib.crmp_update.restype = ct.c_int
            lib.crmp_solve.argtypes = [ct.c_void_p, ct.c_double]; lib.crmp_solve.restype = ct.c_char_p
            lib.crmp_error.argtypes = []; lib.crmp_error.restype = ct.c_char_p
            lib.crmp_destroy.argtypes = [ct.c_void_p]; lib.crmp_destroy.restype = None
            _LIB = lib
        self.lib = _LIB
        self.handle = self.lib.crmp_create(blocks, features, classes, method)
        if not self.handle: raise RuntimeError(self.lib.crmp_error().decode())

    def update(self, ids, costs):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        costs = np.ascontiguousarray(costs, dtype=np.float64)
        if ids.shape != costs.shape: raise ValueError('Mismatched native RMP arrays')
        if self.lib.crmp_update(self.handle, len(ids), ids.ctypes.data, costs.ctypes.data):
            raise RuntimeError(self.lib.crmp_error().decode())

    def solve(self, seconds):
        raw = self.lib.crmp_solve(self.handle, seconds)
        if raw is None: raise RuntimeError(self.lib.crmp_error().decode())
        return json.loads(raw)

    def close(self):
        if self.handle: self.lib.crmp_destroy(self.handle); self.handle = None


@dataclass(frozen=True)
class ContractOptions:
    cache: bool = True
    warm_d3: bool = True
    quotient: bool = True
    lookahead: bool = False
    oracle_batch: int = 16
    resident_gpu: bool = True
    columns_per_block: int = 0  # zero: all improving feasible columns
    threads: int = 8
    max_columns: int = 200000
    exact_columns_only: bool = False
    warm_roots: int = 1
    bundle_pricing: bool = False
    message_bound: bool = False
    native_metadata: bool = False
    metadata_cache_entries: int = 0
    gpu_sync_tiles: int = 1
    gpu_pipeline_chunk: int = 0
    rmp_every_batches: int = 1
    rmp_column_cap: int = 0
    dual_smoothing: float = 0.0
    root_order: str = 'gain'
    cost_kernel: str = 'baseline'
    bundle_fraction: float = 1.
    class_bound: bool = False
    shallow_certificate: bool = False
    adaptive_admission: bool = True
    tail_depth: int = 3
    suppress_jt_certificate: bool = False  # disable global JT certificate and its root exclusion
    cost_mode: str = 'lazy'
    master_mode: str = 'cg'
    memory_limit_bytes: int = 48 * 1024**3
    bound_feedback: bool = False
    cutoff_node_budget: int = 1
    gpu_compact_rows: bool = True
    gpu_compact_min_batch: int = 8
    gpu_compact_max_ratio: float = .75
    gpu_pair_tile: int = 4096
    gpu_bucket_min: int = 8
    gpu_fused_join: bool = True
    state_screen: bool = True
    similarity_refs: int = 0


def automatic_contract_configuration(n, features, classes, depth, options=None,
                                     gpu_available=True):
    """Return the measured-profile D4/D5 backend and safe exact options.

    Small problems avoid CUDA startup.  The delayed-RMP policy is limited to
    the two validated high-dimensional binary profiles (FICO/Spambase scale),
    expressed only through problem features rather than dataset identities.
    """
    o=options or ContractOptions()
    use_gpu=bool(gpu_available and (int(features)>=48 or int(n)>=10000))
    delayed=bool(int(depth)==5 and int(classes)==2 and
                 int(features)>=100 and int(n)<100000)
    return dict(backend='gpu' if use_gpu else 'cpp',options=replace(o,
        resident_gpu=use_gpu,oracle_batch=64 if use_gpu else 16,
        rmp_every_batches=4 if delayed else 1,class_bound=int(classes)>2))


def _capacity_batch(ids, reduced_cost, active_count, limit, blocks, block_size):
    """Thin only under capacity pressure; omitted signatures remain priceable."""
    ids=np.asarray(ids,dtype=np.int32)
    room=max(0,limit-active_count)
    if len(ids)<=room and active_count<.8*limit:return ids
    budget=min(len(ids),room,max(blocks,room//2))
    if budget>=len(ids):return ids
    if not budget:return ids[:0]
    # Preserve diversity between JT blocks when many reduced costs tie.
    quota=budget//blocks;chosen=[]
    for q in range(blocks):
        local=ids[ids//block_size==q]
        order=np.lexsort((local,reduced_cost[local]))
        chosen.extend(local[order[:quota]].tolist())
    rest=ids[~np.isin(ids,chosen)]
    order=np.lexsort((rest,reduced_cost[rest]))
    chosen.extend(rest[order[:budget-len(chosen)]].tolist())
    return np.asarray(chosen,dtype=np.int32)


class PricingState:
    """Numeric costs indexed by complete upper-prefix signatures."""
    def __init__(self, p, engine, options, clock):
        self.p, self.engine, self.options = p, engine, options
        self.domain = Domain(p, tail_depth=options.tail_depth)
        self.M, self.F, self.K = self.domain.M, p.F, len(p.labels)
        self.A = self.F + self.K
        counts=[1]
        for _ in range(self.domain.h):counts.append(self.F*counts[-1]+self.K)
        self.counts=counts;self.P=counts[-1]
        self.edge_offsets=[];self.R=0
        for size in self.domain.separators:
            self.edge_offsets.append(self.R);self.R+=counts[size]
        self.N = self.M*self.P
        # This dense exact domain is deliberate for LP-SC and for the current
        # certified deep-CG implementation. Reject before Python/native/Gurobi
        # allocations can cross the campaign's 48 GiB boundary.
        estimated=self.N*49+self.R*24
        if options.memory_limit_bytes and estimated>options.memory_limit_bytes:
            raise CapacityExceeded(f'MEMORY_PRECHECK dense_state_bytes={estimated} limit={options.memory_limit_bytes}')
        self.low = np.full(self.N, np.inf); self.high = self.low.copy()
        self.private = np.full(self.N, -1, dtype=np.int32)
        self.prefix_cost = np.zeros(self.N)
        self.meta = [None]*self.N
        self.groups = []; known = {}
        for q in range(self.M):
            for prefix, rows, used, pc, ps in self.domain.prefix_states(q, clock):
                s = self.signature(prefix); id = q*self.P+s
                self.meta[id] = (prefix, ps)
                self.prefix_cost[id] = pc
                if any(a[0] != SPLIT for a in prefix):
                    self.low[id] = self.high[id] = pc
                    continue
                key = rows.mask if options.cache else (q, s)
                if key in known:
                    gid = known[key]; record = self.groups[gid]['record']
                else:
                    record = engine.initial(rows,terminal_depth=options.tail_depth)
                    gid = len(self.groups); known[key] = gid
                    self.groups.append(dict(rows=rows, node=self.domain.roots[q], used=used,
                                            record=record, ids=[]))
                self.groups[gid]['ids'].append(id); self.private[id] = gid
                self.low[id], self.high[id] = pc+record.lower, pc+record.upper
        for group in self.groups: group['ids'] = np.asarray(group['ids'], dtype=np.int32)
        self.certified = np.isfinite(self.high) & (self.low >= self.high-1e-12)
        if self.domain.h==1:
            self.roots=np.arange(self.P)
        else:
            s=np.arange(self.P);cut=self.F*self.counts[-2]
            self.roots=np.where(s<cut,s//self.counts[-2],self.F+s-cut)

    def signature(self, prefix):
        value=0
        for j,(tag,action) in enumerate(prefix):
            remaining=len(prefix)-j
            if tag==STOP:
                return value+self.F*self.counts[remaining-1]+self.p.labels.index(action)
            if tag!=SPLIT:raise ValueError('INACTIVE cannot precede a signature decision')
            value+=action*self.counts[remaining-1]
        return value

    def column(self, id):
        prefix, splits = self.meta[id]
        gid = self.private[id]
        tail = self.groups[gid]['record'].tree if gid >= 0 else None
        return Column(prefix, tail, float(self.high[id]), splits)

    def inject(self, tree):
        ids = []
        for q, col in enumerate(self.domain.tree_columns(tree)):
            id = q*self.P+self.signature(col.prefix); ids.append(id)
            gid = self.private[id]
            if gid >= 0:
                g = self.groups[gid]; record = g['record']; upper = col.cost-self.prefix_cost[id]
                if upper < record.lower-1e-8: raise AssertionError('Warm subtree violates its lower bound')
                if upper < record.upper:
                    self.update_group(gid, Interval(record.lower, upper, col.tail, record.exact))
        return np.asarray(ids, dtype=np.int32)

    def update_group(self, gid, answer):
        group = self.groups[gid]; previous = group['record']
        if answer.lower > previous.upper+1e-8: raise AssertionError('D3 lower bound exceeds a feasible tail')
        lower = max(previous.lower, answer.lower)
        upper = min(previous.upper, answer.upper)
        tree = answer.tree if answer.upper <= previous.upper else previous.tree
        group['record'] = Interval(lower, upper, tree, answer.exact or previous.exact)
        ids = group['ids']; self.low[ids] = self.prefix_cost[ids]+lower
        self.high[ids] = self.prefix_cost[ids]+upper
        self.certified[ids] = group['record'].exact
        return ids

    def reduced(self, costs, alpha, pi):
        if self.M not in (2,4):
            raise RuntimeError('Deep reduced costs must be computed by the native C++ master')
        values = costs.reshape(self.M,self.P).copy()-np.asarray(alpha)[:,None]
        if self.M == 2:
            values[0] -= pi; values[1] += pi
        else:
            values[0] -= pi[:self.P]
            values[1] += pi[:self.P]-pi[self.P+self.roots]
            values[2] += pi[self.P+self.roots]-pi[self.P+self.A:]
            values[3] += pi[self.P+self.A:]
        return values.reshape(-1)

    def recover(self, support):
        by = [dict() for _ in range(self.M)]
        for id, value in support:
            if value > 1e-9: by[id//self.P][id%self.P] = id
        if self.M == 2:
            common = by[0].keys() & by[1].keys()
            if not common: raise AssertionError('No compatible RMP support')
            s = min(common, key=lambda s: self.high[s]+self.high[self.P+s])
            ids = [s,self.P+s]
        elif self.M==4:
            sides = []
            for a in (0,1):
                found = {}
                for s in by[2*a].keys() & by[2*a+1].keys():
                    r = int(self.roots[s]); value = self.high[2*a*self.P+s]+self.high[(2*a+1)*self.P+s]
                    if r not in found or value < found[r][0]: found[r] = value,s
                sides.append(found)
            common = sides[0].keys() & sides[1].keys()
            if not common: raise AssertionError('No compatible RMP root support')
            r = min(common, key=lambda r: sides[0][r][0]+sides[1][r][0])
            ids = [q*self.P+sides[q//2][r][1] for q in range(4)]
        else:
            raise RuntimeError('Deep RMP recovery must use the native C++ selection')
        tree = self.domain.recover([self.column(id) for id in ids])
        return tree, float(sum(self.high[id] for id in ids))

    def messages(self, costs):
        """Exact upper-level min-sum over complete signatures, with local bounds."""
        c = costs.reshape(self.M,self.P)
        if self.M>4:
            from .deep_structural_native import path_opt
            value,selected=path_opt(c,self.F,self.K,self.domain.h)
            ids=None if selected is None else np.asarray(
                [q*self.P+int(s) for q,s in enumerate(selected)],dtype=np.int32)
            return np.asarray([value]),np.asarray([ids],dtype=object),None,None
        if self.M == 2:
            roots = c[0]+c[1]
            ids = np.stack([np.arange(self.P),self.P+np.arange(self.P)],axis=1)
            return roots,ids,None,None
        pairs = c[::2]+c[1::2]
        shaped = pairs[:,:self.F*self.A].reshape(2,self.F,self.A)
        choice = np.argmin(shaped,axis=2)
        signatures = np.concatenate([np.arange(self.F)[None,:]*self.A+choice,
            np.tile(np.arange(self.F*self.A,self.P),(2,1))],axis=1)
        sides = np.take_along_axis(pairs,signatures,axis=1)
        ids = np.stack([q*self.P+signatures[q//2] for q in range(4)],axis=1)
        return sides.sum(axis=0),ids,sides,pairs

    def min_marginals(self, costs, info=None):
        """Full-domain optimum conditioned on each signature (D4/D5)."""
        info=self.messages(costs) if info is None else info
        if self.M==2:return np.tile(info[0],2)
        if self.M!=4:raise ValueError('State min-marginals require two or four clusters')
        sides,pairs=info[2],info[3]
        # Avoid total-minus-local: infeasible signatures may have infinite cost.
        return np.repeat(pairs+sides[::-1,self.roots],2,axis=0).reshape(-1)

    def compatible_groups(self, limit, incumbent, low_info, high_info):
        """Prioritize sibling pairs for promising compatible roots; no domain removal."""
        root_low,low_ids,side_low,pair_low = low_info
        if self.M>4:
            selected=[];seen=set()
            if low_ids is not None and len(low_ids):
                for id in low_ids[0]:
                    gid=int(self.private[int(id)])
                    if gid>=0 and gid not in seen and not self.groups[gid]['record'].exact:
                        selected.append(gid);seen.add(gid)
                        if len(selected)>=limit:return selected
            return selected
        root_high,_,side_high,pair_high = high_info
        viable = np.flatnonzero(root_low < incumbent-1e-9)
        if self.options.root_order=='bound':
            # Round only the ordering key to avoid arbitrary floating-point ties;
            # neither the certificate nor the exclusion test is rounded.
            roots=viable[np.lexsort((viable,root_high[viable],np.round(root_low[viable],12)))]
        else:
            roots = viable[np.lexsort((viable,root_low[viable],root_high[viable]))]
        selected=[];seen=set()
        def add(ids):
            for id in ids:
                gid=int(self.private[id])
                if gid>=0 and gid not in seen and not self.groups[gid]['record'].exact:
                    selected.append(gid);seen.add(gid)
                    if len(selected)>=limit:return True
            return False
        for r in roots:
            if self.M==2:
                if add(low_ids[r]):break
                continue
            candidates=[]
            for a in (0,1):
                indices=np.flatnonzero(self.roots==r)
                cutoff=min(side_high[a,r],incumbent-side_low[1-a,r])
                eligible=indices[pair_low[a,indices]<cutoff-1e-9]
                eligible=eligible[np.lexsort((eligible,pair_low[a,eligible],pair_high[a,eligible]))]
                candidates.append(eligible)
            for k in range(max(map(len,candidates),default=0)):
                for a in (0,1):
                    if k<len(candidates[a]):
                        s=int(candidates[a][k])
                        if add([(2*a)*self.P+s,(2*a+1)*self.P+s]):return selected
        return selected


def solve_contracted_cg(p, backend='auto', time_limit=60, options=None, progress=None):
    if p.depth not in (4,5,6,7) or p._uniform_weight is None or not p.early_stop or p.allowed or p.split_costs:
        raise ValueError('D3-contracted CG requires D4--D7, uniform weights, STOP and shared node costs/features')
    o = options or ContractOptions()
    automatic = backend=='auto'
    if automatic:
        from .auto_dp import hardware
        selected=automatic_contract_configuration(p.n,p.F,len(p.labels),p.depth,o,
            hardware()['gpu_available'])
        backend,o=selected['backend'],selected['options']
    if backend not in ('cpp','gpu'):raise ValueError('Contracted CG backend must be auto, cpp or gpu')
    if o.gpu_pair_tile<1 or o.gpu_bucket_min<0:raise ValueError('Invalid GPU dispatch options')
    if o.gpu_compact_min_batch<1 or not 0<o.gpu_compact_max_ratio<1:
        raise ValueError('Invalid compact-row profitability thresholds')
    if o.cutoff_node_budget<0:raise ValueError('Nonnegative cutoff node budget required')
    if o.similarity_refs<0:raise ValueError('Nonnegative similarity reference count required')
    screen_states=bool(o.state_screen and p.depth in (4,5) and o.tail_depth==3 and
        o.cost_mode=='lazy' and o.master_mode=='cg' and not o.suppress_jt_certificate)
    if o.similarity_refs and (p.depth not in (4,5) or o.tail_depth!=3 or
            o.cost_mode!='lazy' or o.master_mode!='cg' or o.suppress_jt_certificate):
        raise ValueError('State bounds require D4/D5 lazy CG with D3 tails and JT certificates')
    if o.similarity_refs and p.min_leaf>0:
        raise ValueError('Similarity transfer currently requires min_leaf=0')
    if o.bound_feedback and (o.master_mode!='cg' or o.cost_mode!='lazy' or o.tail_depth!=3):
        raise ValueError('Bound feedback currently requires lazy CG with D3 tails')
    if o.cost_mode not in ('lazy','eager') or o.master_mode not in ('cg','full','message','ee','full_ee'):
        raise ValueError('Invalid ablation mode')
    if o.master_mode in ('full','full_ee') and o.cost_mode!='eager':
        raise ValueError('Full JT-LP requires exact eager costs')
    if o.tail_depth!=3 and (o.cost_mode!='lazy' or o.master_mode!='cg' or o.suppress_jt_certificate):
        raise ValueError('Ablation switches currently require D3 tails')
    if o.tail_depth not in (2,3):raise ValueError('Terminal depth must be 2 or 3')
    if o.tail_depth==2 and p.depth==5:
        from .tail_cg import solve_streamed_tail_cg
        return solve_streamed_tail_cg(p,backend,time_limit,o,progress)
    if o.oracle_batch < 1 or o.columns_per_block < 0 or o.threads < 1 or o.max_columns < 1 or o.warm_roots < 1:
        raise ValueError('Invalid contracted CG options')
    if o.rmp_every_batches<1 or o.rmp_column_cap<0 or not 0<=o.dual_smoothing<1:
        raise ValueError('Invalid RMP scheduling options')
    if o.gpu_sync_tiles<1 or o.gpu_pipeline_chunk<0 or o.metadata_cache_entries<0:
        raise ValueError('Invalid GPU preparation options')
    if o.root_order not in ('gain','bound'):raise ValueError('Invalid root ordering')
    if o.cost_kernel not in ('baseline','fused','shared') or not 0<o.bundle_fraction<=1:
        raise ValueError('Invalid cost kernel or bundle share')
    if o.memory_limit_bytes<1:raise ValueError('Positive memory limit required')
    clock = Deadline(time_limit); name = f'JT-CG-D{o.tail_depth}Tail-'+backend.upper()
    if o.master_mode=='message':
        name=('Exact-cost-MP-' if o.cost_mode=='eager' else 'Adaptive-MP-')+backend.upper()
    stats = defaultdict(int); stats.update(backend=backend, options=asdict(o),automatic_backend=automatic)
    stats['state_screen_active']=screen_states
    similarity_bank={}
    trace, incumbents = [], []; engine = master = state = None
    tree = Tree(label=p.best_label(p.all_rows)) if p.n>=p.min_leaf else None
    best = evaluate(p,tree)['objective'] if tree else math.inf
    lb = 0.; status = 'TIME'; first = clock.elapsed() if tree else None
    proof = None; last_publish = -1.; certificate = None

    def accept(candidate, source):
        nonlocal tree,best,first
        if candidate is None:return
        metric = evaluate(p,candidate)
        if metric['objective'] < best-1e-12:
            tree,best,first = candidate,metric['objective'],clock.elapsed()
            incumbents.append(dict(seconds=first,UB=best,misclassified=metric['misclassified'],
                                   split_nodes=metric['split_nodes'],source=source))

    def publish(force=False):
        nonlocal last_publish
        if progress and (force or clock.elapsed()-last_publish>=1.):
            last_publish = clock.elapsed()
            progress(result_dict(name,clock,'RUNNING',tree,p,lb,stats=dict(stats),
                                 trace=list(trace),incumbent_trace=list(incumbents)))

    try:
        clock.check();accept(greedy_feasible(p,clock),'greedy');publish(True)
        if o.shallow_certificate and p.penalty>0 and best<=3*p.penalty:
            # Trees outside the depth-two domain need at least three splits.
            # A shallow optimum ALONE is not a lower bound for a deeper tree.
            from copy import copy
            from .d2_cg import solve_d2_cg
            shallow=copy(p);shallow.depth=2
            t=time.perf_counter()
            answer=solve_d2_cg(shallow,clock.remaining(),threads=0,cost_backend='auto')
            stats['shallow_certificate_seconds']=time.perf_counter()-t
            stats['shallow_certificate_status']=answer['status']
            stats['shallow_certificate_depth']=2
            stats['shallow_certificate_LB']=float(answer['LB'])
            excluded_lower=3*p.penalty
            lb=max(lb,min(float(answer['LB']),excluded_lower))
            if answer.get('tree'):accept(Tree.from_dict(answer['tree']),'D2_CG_penalty_certificate')
            certificate=dict(kind='depth_exclusion_split_penalty',shallow_depth=2,
                shallow_LB=float(answer['LB']),excluded_min_splits=3,
                excluded_lower_bound=excluded_lower,computed_LB=lb)
            publish(True)
            clock.check()
            if best-lb<=1e-7:
                proof=clock.elapsed()
                stats['shallow_certificate_closed']=1
                return result_dict(name,clock,'OPT',tree,p,min(lb,best),stats=dict(stats),
                    trace=trace,certificate=certificate,incumbent_trace=incumbents,
                    first_final_ub_seconds=first,first_optimal_solution_seconds=first,
                    proof_seconds=proof,formulation='D2 JT-CG LP plus depth-exclusion penalty bound')
        t = time.perf_counter()
        opt = TerminalOptions(cache_entries=20000 if o.cache else 0, fast_prepare=True,
            quotient=o.quotient, warm_d3=False, resident_gpu=o.resident_gpu,
            gpu_batch_size=o.oracle_batch, lookahead_bound=o.lookahead,
            class_capacity_bound=o.class_bound)
        engine = TerminalMessages(p,backend,o.threads,opt,clock,stats)
        engine.workspace.options=replace(engine.workspace.options,
            gpu_native_metadata=o.native_metadata,gpu_metadata_cache_entries=o.metadata_cache_entries,
            gpu_sync_tiles=o.gpu_sync_tiles,gpu_pipeline_chunk=o.gpu_pipeline_chunk,
            gpu_cost_strategy=o.cost_kernel,gpu_compact_rows=o.gpu_compact_rows,
            gpu_compact_min_batch=o.gpu_compact_min_batch,
            gpu_compact_max_ratio=o.gpu_compact_max_ratio,
            gpu_pair_tile=o.gpu_pair_tile,gpu_bucket_min=o.gpu_bucket_min,gpu_fused_join=o.gpu_fused_join)
        stats['setup_seconds'] += time.perf_counter()-t
        lb = max(lb,stats['conflict_lower_bound'])
        if o.warm_d3:
            t = time.perf_counter()
            answer = engine.terminal(p.all_rows,(),(),kind='warm')
            growing = answer.tree
            accept(growing,'D3_seed');publish()
            if growing:
                for level in range(1,p.depth-2):
                    for path in _frontier_paths(growing,level):
                        rows,used = _prefix_state(p,growing,path)
                        answer = engine.terminal(rows,path,used,kind='warm')
                        if answer.tree:
                            candidate = _replace(growing,path,answer.tree)
                            if evaluate(p,candidate)['objective'] <= evaluate(p,growing)['objective']+1e-9:
                                growing = candidate;accept(growing,'D3_grow');publish()
            stats['warm_roots_completed'] = int(growing is not None and growing.feature is not None)
            if o.warm_roots > 1:
                seed_root = growing.feature if growing else None
                ranked = []
                for f in p.features((),()):
                    clock.check()
                    children = [p.route(p.all_rows,f,a) for a in (0,1)]
                    if any(len(rows)<p.min_leaf for rows in children):continue
                    score = p.cost((),f)+sum(p.best_label_and_loss(rows)[1] for rows in children)
                    ranked.append((score,f))
                ranked.sort()
                remaining_roots = o.warm_roots-stats['warm_roots_completed']
                roots = [f for _,f in ranked if f != seed_root][:remaining_roots]
                for f in roots:
                    clock.check()
                    candidate = Tree(feature=f,**{key:Tree(label=p.best_label(p.route(p.all_rows,f,a)))
                        for a,key in enumerate(('left','right'))})
                    for level in range(1,p.depth-2):
                        paths = _frontier_paths(candidate,level)
                        requests = []
                        for path in paths:
                            rows,used = _prefix_state(p,candidate,path)
                            requests.append((rows,path,used))
                        answers = engine.terminal_many(requests,kind='warm')
                        for path,answer in zip(paths,answers):
                            if answer.tree:
                                proposed = _replace(candidate,path,answer.tree)
                                if evaluate(p,proposed)['objective']<=evaluate(p,candidate)['objective']+1e-9:
                                    candidate = proposed
                        accept(candidate,'D3_beam');publish()
                    stats['warm_roots_completed'] += 1
            stats['warm_seconds'] += time.perf_counter()-t
        if o.tail_depth==2:
            old=engine
            engine=TerminalMessages(p,backend,o.threads,opt,clock,stats,terminal_depth=2)
            engine.workspace.options=old.workspace.options
            if backend=='gpu':engine.workspace.share_gpu_data_from(old.workspace)
            old.close()
        stats['terminal_depth']=o.tail_depth
        stats['upper_depth']=p.depth-o.tail_depth
        t = time.perf_counter();state = PricingState(p,engine,o,clock)
        stats['metadata_seconds'] = time.perf_counter()-t
        stats['candidate_slots'] = state.N;stats['terminal_contexts'] = int(np.count_nonzero(state.private>=0))
        stats['unique_terminal_states'] = len(state.groups)
        if tree is None: raise ValueError('No feasible initialization; choose a feasible minimum leaf size')
        initial = state.inject(tree);active = np.zeros(state.N,dtype=bool)
        stats['initial_exact_terminal_states']=sum(g['record'].exact for g in state.groups)
        if o.cost_mode=='eager':
            tick=time.perf_counter()
            pending=[j for j,g in enumerate(state.groups) if not g['record'].exact]
            stats['precompute_requested_states']=len(pending)
            for start in range(0,len(pending),o.oracle_batch):
                clock.check()
                gids=pending[start:start+o.oracle_batch]
                answers=engine.terminal_many([(state.groups[g]['rows'],state.groups[g]['node'],
                    state.groups[g]['used']) for g in gids])
                for gid,answer in zip(gids,answers):state.update_group(gid,answer)
                publish()
            stats['precompute_seconds']=time.perf_counter()-tick
            stats['precompute_complete']=int(all(g['record'].exact for g in state.groups))
            if not stats['precompute_complete']:raise AssertionError('Incomplete eager oracle')
        if o.master_mode=='ee':
            from .contract_ee import solve_interval_ee
            return solve_interval_ee(state,clock,o,stats,tree,lb,progress)
        if o.master_mode=='message':
            # Same state domain, bounds, cache, warm start and cost oracle as CG.
            # No NativeRMP is constructed on this path.
            while True:
                clock.check()
                tick=time.perf_counter()
                lower=state.messages(state.low)
                upper=state.messages(state.high)
                lb=max(lb,float(np.min(lower[0])))
                root=int(np.argmin(upper[0]))
                stats['message_seconds']+=time.perf_counter()-tick
                tick=time.perf_counter()
                if np.isfinite(upper[0][root]):
                    candidate=state.domain.recover([state.column(int(i)) for i in upper[1][root]])
                    metric=evaluate(p,candidate)
                    if abs(metric['objective']-upper[0][root])>1e-7:
                        raise AssertionError('Message recovery objective mismatch')
                    accept(candidate,'message_upper')
                stats['recovery_seconds']+=time.perf_counter()-tick
                if lb>best+1e-7:raise AssertionError('Message LB exceeds incumbent')
                stats['message_passes']+=1
                trace.append(dict(seconds=clock.elapsed(),LB=lb,UB=best,
                    exact_terminal_states=sum(g['record'].exact for g in state.groups)))
                publish()
                if best-lb<=1e-7:
                    status='OPT';proof=clock.elapsed();break
                # Always refine an unresolved state on the minimum-lower-cost
                # assignment first. This guarantees progress on a finite domain.
                r=int(np.argmin(lower[0]));gids=[]
                for i in lower[1][r]:
                    gid=int(state.private[int(i)])
                    if gid>=0 and not state.groups[gid]['record'].exact and gid not in gids:
                        gids.append(gid)
                gids=gids[:o.oracle_batch]
                # Fill spare batch slots using the existing compatible-root
                # scheduling policy. Excluded states stay in both cost arrays.
                for gid in state.compatible_groups(o.oracle_batch,best,lower,upper):
                    if len(gids)>=o.oracle_batch:break
                    if gid not in gids:gids.append(gid)
                if not gids:raise AssertionError('Open message gap without unresolved state')
                tick=time.perf_counter()
                answers=engine.terminal_many([(state.groups[g]['rows'],state.groups[g]['node'],
                    state.groups[g]['used']) for g in gids])
                stats['oracle_wall_seconds']+=time.perf_counter()-tick
                stats['selected_message_states']+=len(gids)
                for gid,answer in zip(gids,answers):state.update_group(gid,answer)
                if not all(answer.exact for answer in answers):
                    # This oracle has no resumable partial solve. Preserve its
                    # intervals and return without claiming optimality.
                    raise DeadlineExceeded('Incomplete private-subtree batch')
            stats.update(final_columns=0,final_rows=0,
                exact_terminal_states=sum(g['record'].exact for g in state.groups),
                unresolved_terminal_states=sum(not g['record'].exact for g in state.groups))
            return result_dict(name,clock,status,tree,p,min(lb,best),stats=dict(stats),trace=trace,
                certificate=dict(kind='exact_cost_messages' if o.cost_mode=='eager' else 'adaptive_cost_messages'),
                incumbent_trace=incumbents,first_final_ub_seconds=first,
                first_optimal_solution_seconds=first,proof_seconds=proof,
                formulation='D3-contracted junction tree solved by direct min-sum messages')
        if o.master_mode=='full_ee':
            from .deep_contract_ee import solve_full_endpoint_lp
            return solve_full_endpoint_lp(state,clock,o,stats,tree,lb,progress)
        def make_master():
            if state.M > 4:
                from .deep_structural_native import DeepMaster
                return DeepMaster(state.M,p.F,len(p.labels),state.domain.h,
                    memory_limit_bytes=o.memory_limit_bytes)
            return NativeRMP(state.M,p.F,len(p.labels))

        t = time.perf_counter()
        master = make_master()
        stats['rmp_setup_seconds'] = time.perf_counter()-t

        def reduced_costs(costs,alpha,pi):
            if state.M<=4:return state.reduced(costs,alpha,pi)
            return master.price(costs,alpha,pi)[0].reshape(-1)

        def recover_master(res):
            if state.M<=4:return state.recover(res['support'])
            ids=[int(i) for i in res['selection']]
            candidate=state.domain.recover([state.column(i) for i in ids])
            return candidate,float(sum(state.high[i] for i in ids))

        def update(ids):
            ids=np.unique(np.asarray(ids,dtype=np.int32))
            if np.count_nonzero(active)+np.count_nonzero(~active[ids])>o.max_columns:
                raise CapacityExceeded('Restricted column limit')
            tick = time.perf_counter();master.update(ids,state.high[ids]);active[ids]=True
            stats['rmp_update_seconds'] += time.perf_counter()-tick
            stats['peak_active_columns']=max(stats['peak_active_columns'],int(np.count_nonzero(active)))

        if o.master_mode=='full':
            ids=np.flatnonzero(np.isfinite(state.high))
            stats['full_feasible_columns']=len(ids)
            update(ids)
            tick=time.perf_counter();res=master.solve(clock.remaining())
            stats['rmp_seconds']+=time.perf_counter()-tick
            if res['status'] in (9,11):raise DeadlineExceeded('Full LP timeout')
            if res['status']!=2:raise AssertionError(f"Full LP status {res['status']}")
            candidate,value=recover_master(res)
            if abs(value-res['objective'])>1e-7:raise AssertionError('Full LP recovery mismatch')
            accept(candidate,'FULL_JT_LP');lb=float(res['objective']);proof=clock.elapsed()
            stats.update(iterations=1,final_columns=len(ids),final_rows=res['rows'],
                exact_terminal_states=sum(g['record'].exact for g in state.groups),unresolved_terminal_states=0)
            return result_dict('JT-LP-D3Tail-'+backend.upper(),clock,'OPT',tree,p,lb,
                stats=dict(stats),trace=[dict(seconds=proof,LB=lb,UB=best,columns=len(ids),rows=res['rows'])],
                certificate=dict(kind='complete_contracted_LP',max_equality_residual=res['max_equality_residual']),
                incumbent_trace=incumbents,first_final_ub_seconds=first,
                first_optimal_solution_seconds=first,proof_seconds=proof,
                formulation='Complete D3-contracted JT-LP with exact private-subtree costs')
        update(initial);dirty=True;iteration=0;batches_since_rmp=0;res=None
        smooth_alpha=smooth_pi=None

        def capacity_messages():
            nonlocal lb
            tick=time.perf_counter()
            lower=state.messages(state.low);upper=state.messages(state.high)
            value=float(np.min(lower[0]))
            if not o.suppress_jt_certificate:
                lb=max(lb,value)
                stats['jt_certificate_updates']+=1
            stats['capacity_message_lower_bound']=max(stats['capacity_message_lower_bound'],value)
            certificate['capacity_message_LB']=value
            root=int(np.argmin(upper[0]))
            if upper[0][root]<best-1e-12:
                accept(state.domain.recover([state.column(int(i)) for i in upper[1][root]]),
                       'capacity_compatible_bundle')
            if lb>best+1e-7:raise AssertionError('Capacity message LB exceeds incumbent')
            lb=min(lb,best)
            stats['capacity_message_seconds']+=time.perf_counter()-tick
            return lower,upper

        while True:
            clock.check()
            if dirty and (res is None or batches_since_rmp>=o.rmp_every_batches):
                t=time.perf_counter();res=master.solve(clock.remaining())
                stats['rmp_seconds']+=time.perf_counter()-t
                if res['status'] in (9,11):raise DeadlineExceeded('Native RMP timeout')
                if res['status']!=2:raise AssertionError(f"Feasible RMP status {res['status']}")
                candidate,value=recover_master(res)
                if abs(value-res['objective'])>1e-7:raise AssertionError('RMP gluing objective mismatch')
                accept(candidate,'CG_RMP');stats['iterations']+=1
                alpha=res['alpha']
                if state.M>4:pi=np.asarray(res['pi'],dtype=float)
                else:
                    pi=np.zeros(state.R)
                    for id,value in res['pi']:pi[id]=value
                if smooth_alpha is None:
                    smooth_alpha=np.asarray(alpha);smooth_pi=pi.copy()
                else:
                    smooth_alpha=o.dual_smoothing*smooth_alpha+(1-o.dual_smoothing)*np.asarray(alpha)
                    smooth_pi=o.dual_smoothing*smooth_pi+(1-o.dual_smoothing)*pi
                dirty=False;batches_since_rmp=0
            t=time.perf_counter();reduced=reduced_costs(state.low,alpha,pi)
            minima=np.minimum(0.,reduced.reshape(state.M,state.P).min(axis=1))
            computed=float(sum(alpha)+sum(minima));lb=max(lb,computed)
            low_info=high_info=None
            if o.message_bound or o.bundle_pricing or screen_states:
                tick=time.perf_counter()
                low_info=state.messages(state.low);high_info=state.messages(state.high)
                message_lower=float(np.min(low_info[0]))
                stats['message_lower_bound']=message_lower
                excludable=(int(np.count_nonzero(low_info[0][:state.F]>=best-1e-9))
                            if state.M<=4 else 0)
                stats['roots_excludable_by_message']=excludable
                stats['roots_excluded_by_message']=0 if o.suppress_jt_certificate else excludable
                if (o.message_bound or screen_states) and not o.suppress_jt_certificate:
                    lb=max(lb,message_lower)
                    stats['jt_certificate_updates']+=1
                if o.bundle_pricing:
                    root=int(np.argmin(high_info[0]))
                    if high_info[0][root]<best-1e-12:
                        accept(state.domain.recover([state.column(int(id)) for id in high_info[1][root]]),'compatible_message')
                stats['message_seconds']+=time.perf_counter()-tick
            if lb>best+1e-7:raise AssertionError('Certified contracted CG LB exceeds incumbent')
            lb=min(lb,best)
            certificate=dict(alpha_sum=float(sum(alpha)),pricing_lower_bounds=minima.tolist(),computed_LB=computed,
                jt_bound_used=bool((o.message_bound or screen_states) and not o.suppress_jt_certificate),
                max_equality_residual=res['max_equality_residual'],minimum_active_reduced_cost=res['minimum_active_reduced_cost'],
                pricing_scope=f'lower bounds for every exact or unresolved private D{o.tail_depth} configuration')
            if low_info is not None:certificate['upper_message_LB']=float(np.min(low_info[0]))
            upper_rc=reduced_costs(state.high,alpha,pi)
            ids=[]
            for q in range(state.M):
                choices=np.flatnonzero((upper_rc[q*state.P:(q+1)*state.P]<-1e-8)&~active[q*state.P:(q+1)*state.P])+q*state.P
                if o.exact_columns_only:
                    choices=choices[state.certified[choices]]
                if o.columns_per_block and len(choices)>o.columns_per_block:
                    choices=choices[np.argsort(upper_rc[choices],kind='stable')[:o.columns_per_block]]
                ids.extend(choices.tolist())
            capacity_pressure=False
            if o.adaptive_admission and ids and best-lb>1e-7:
                active_count=int(np.count_nonzero(active))
                capacity_pressure=active_count+len(ids)>o.max_columns or active_count>=.8*o.max_columns
                if capacity_pressure:low_info,high_info=capacity_messages()
                if active_count>=o.max_columns and best-lb>1e-7:
                    # Recycle only against a freshly solved master. Keep its
                    # positive support and an incumbent bundle, so feasibility
                    # and the current RMP optimum are preserved when possible.
                    if dirty:
                        batches_since_rmp=o.rmp_every_batches
                        iteration+=1
                        continue
                    if state.M>4:
                        # The generic native path core returns the integral
                        # compatible support directly; the legacy D4/D5 RMP
                        # returns sparse primal pairs.
                        support=np.asarray(res['selection'],dtype=np.int32)
                    else:
                        support=np.asarray([int(i) for i,value in res['support'] if value>1e-9],dtype=np.int32)
                    incumbent=state.inject(tree)
                    protected=np.union1d(support,incumbent).astype(np.int32)
                    if len(protected)>=o.max_columns and len(incumbent)<o.max_columns:
                        # The incumbent has objective no worse than this RMP;
                        # a tiny cap can keep that complete feasible bundle.
                        protected=incumbent
                    if len(protected)>=o.max_columns:
                        # A deliberately tiny cap may not fit the feasible
                        # support plus even one entering column.
                        raise CapacityExceeded('Column capacity cannot retain feasible support and an entering column')
                    target=min(o.max_columns-1,max(len(protected),o.max_columns//2))
                    extras=np.flatnonzero(active & ~np.isin(np.arange(state.N),protected))
                    order=np.lexsort((extras,upper_rc[extras]))
                    keep=np.union1d(protected,extras[order[:target-len(protected)]]).astype(np.int32)
                    tick=time.perf_counter()
                    master.close();master=make_master()
                    active[:]=False;update(keep)
                    stats['capacity_rebuilds']+=1
                    stats['capacity_evicted_columns']+=active_count-len(keep)
                    stats['capacity_rebuild_seconds']+=time.perf_counter()-tick
                    stats['last_compaction_iteration']=iteration
                    dirty=True;res=None;batches_since_rmp=0;iteration+=1
                    # Evicted ids are still in state.low/high and the COMPLETE
                    # pricing domain. Recompute prices with the new LP duals.
                    continue
                requested=len(ids)
                ids=_capacity_batch(ids,upper_rc,active_count,o.max_columns,state.M,state.P).tolist()
                stats['column_admission_requested']+=requested
                stats['column_admission_admitted']+=len(ids)
                stats['column_admission_deferred']+=requested-len(ids)
                if len(ids)<requested:stats['capacity_throttled_batches']+=1
            if (o.rmp_column_cap and np.count_nonzero(active)>=o.rmp_column_cap and
                    iteration-stats.get('last_compaction_iteration',-20)>=20):
                # Compact to an incumbent bundle. Removed signatures remain in
                # the complete pricing domain and may be generated again.
                keep=state.inject(tree);master.close();master=make_master()
                active[:]=False;update(keep);dirty=True;res=None
                stats['rmp_compactions']+=1;batches_since_rmp=0
                stats['last_compaction_iteration']=iteration
                continue
            stats['pricing_seconds']+=time.perf_counter()-t
            if dirty or ids or clock.elapsed()-stats.get('last_trace_seconds',-1.)>=1. or best-lb<=1e-7:
                trace.append(dict(iteration=iteration,seconds=clock.elapsed(),LB=lb,UB=best,
                    columns=int(np.count_nonzero(active)),rows=res['rows'],new_columns=len(ids),
                    exact_terminal_states=sum(g['record'].exact for g in state.groups),
                    pricing_lower_bounds=minima.tolist()))
                stats['last_trace_seconds']=clock.elapsed()
            publish()
            if best-lb<=1e-7:status='OPT';proof=clock.elapsed();break
            if ids:
                update(np.asarray(ids,dtype=np.int32));dirty=True
                if o.rmp_every_batches==1 and not capacity_pressure:
                    batches_since_rmp=1;iteration+=1;continue
            candidates=np.flatnonzero((reduced<-1e-8)&(state.private>=0))
            marginal=None
            if screen_states:
                tick=time.perf_counter();marginal=state.min_marginals(state.low,low_info)
                keep=marginal[candidates]<best-1e-9
                stats['state_screened_candidates']+=int(np.count_nonzero(~keep))
                candidates=candidates[keep]
                stats['state_screen_seconds']+=time.perf_counter()-tick
            if o.message_bound and not o.suppress_jt_certificate and state.M<=4:
                root_ids=candidates%state.P if state.M==2 else state.roots[candidates%state.P]
                candidates=candidates[low_info[0][root_ids]<best-1e-9]
            # Choose promising unknown private states, never use an incumbent as LB.
            priority=reduced_costs(state.low,smooth_alpha,smooth_pi) if o.dual_smoothing else reduced
            ranked=candidates[np.argsort(priority[candidates],kind='stable')]
            bundle_limit=max(1,int(o.oracle_batch*o.bundle_fraction))
            gids=state.compatible_groups(o.oracle_batch if capacity_pressure else bundle_limit,
                best,low_info,high_info) if o.bundle_pricing or capacity_pressure else []
            if capacity_pressure:stats['capacity_bundle_states_selected']+=len(gids)
            if marginal is not None:
                gids=[g for g in gids if np.any(marginal[state.groups[g]['ids']]<best-1e-9)]
            stats['bundle_states_selected']+=len(gids)
            seen=set(gids)
            if len(gids)<o.oracle_batch and (not o.bundle_pricing or not gids or o.bundle_fraction<1):
                for id in ranked:
                    gid=int(state.private[id])
                    if gid not in seen and not state.groups[gid]['record'].exact:
                        gids.append(gid);seen.add(gid)
                        if len(gids)>=o.oracle_batch:break
            stats['selected_pricing_states']+=len(gids)
            if not gids:
                if dirty:batches_since_rmp=o.rmp_every_batches;continue
                raise AssertionError('Open pricing gap without an unresolved negative candidate')
            # A bound is reusable across dual epochs; screening against a cutoff
            # is not. Return all newly refined intervals, then rerun complete
            # pricing. Each group gets one bounded attempt before exact fallback.
            if o.similarity_refs:
                tick=time.perf_counter();improved=False
                for gid in gids:
                    g=state.groups[gid]
                    key=tuple(sorted(g['used'])) if p.no_repeat else ()
                    refs=similarity_bank.get(key,[])
                    old=g['record'];lower=old.lower
                    for mask,value in refs:
                        lower=max(lower,value-(mask & ~g['rows'].mask).bit_count()*p._uniform_weight)
                    stats['similarity_comparisons']+=len(refs)
                    if lower>old.lower+1e-10:
                        state.update_group(gid,Interval(lower,old.upper,old.tree,old.exact))
                        stats['similarity_improvements']+=1;improved=True
                stats['similarity_seconds']+=time.perf_counter()-tick
                if improved:
                    iteration+=1;continue # Fresh complete pricing and messages; no exact-cache insertion.
            refine_gids=[g for g in gids if not state.groups[g].get('bound_refined',False)] if o.bound_feedback else []
            if refine_gids:
                changed=[]
                for gid in refine_gids:
                    group=state.groups[gid];before=group['record'].upper
                    # All aliases of this RowSet must be covered, including
                    # signatures outside the currently selected pricing batch.
                    thresholds=group['record'].lower-reduced[group['ids']]
                    if marginal is not None:
                        thresholds=np.minimum(thresholds,best-marginal[group['ids']]+group['record'].lower)
                    cutoff=float(np.max(thresholds))
                    answer=engine.refine(group['rows'],group['node'],group['used'],cutoff,o.cutoff_node_budget)
                    stats['interval_cutoff_closed']+=int(not answer.exact and answer.lower>=cutoff)
                    group['bound_refined']=True
                    affected=state.update_group(gid,answer)
                    if group['record'].upper<before-1e-12:changed.extend(affected[active[affected]].tolist())
                if changed:update(np.asarray(sorted(set(changed)),dtype=np.int32));dirty=True
                stats['bound_feedback_rounds']=stats.get('bound_feedback_rounds',0)+1
                # Old duals still give a valid repaired certificate after cost
                # tightening. Respect the existing master cadence instead of
                # resolving the LP after every improved feasible stump.
                batches_since_rmp+=1
                iteration+=1
                continue
            requests=[(state.groups[g]['rows'],state.groups[g]['node'],state.groups[g]['used']) for g in gids]
            if capacity_pressure:stats['capacity_pricing_interleaves']+=1
            t=time.perf_counter();answers=engine.terminal_many(requests)
            stats['oracle_wall_seconds']+=time.perf_counter()-t
            batches_since_rmp+=1
            changed=[]
            for gid,answer in zip(gids,answers):
                group=state.groups[gid];before=group['record'].upper
                affected=state.update_group(gid,answer)
                if o.similarity_refs and answer.exact:
                    key=tuple(sorted(group['used'])) if p.no_repeat else ()
                    bank=similarity_bank.setdefault(key,[])
                    bank.append((group['rows'].mask,answer.lower))
                    del bank[:-o.similarity_refs]
                if group['record'].upper<before-1e-12:changed.extend(affected[active[affected]].tolist())
            if changed:update(np.asarray(sorted(set(changed)),dtype=np.int32));dirty=True
            if o.rmp_every_batches==1 and dirty:batches_since_rmp=1
            iteration+=1
    except DeadlineExceeded:
        status='TIME'
    except CapacityExceeded as error:
        stats['resource_reason']=str(error)
        status='RESOURCE'
    finally:
        if master:master.close()
        if engine:engine.close()
    if state:
        stats['exact_terminal_states']=sum(g['record'].exact for g in state.groups)
        stats['unresolved_terminal_states']=len(state.groups)-stats['exact_terminal_states']
        stats['final_columns']=int(np.count_nonzero(active)) if 'active' in locals() else 0
    return result_dict(name,clock,status,tree,p,lb,stats=dict(stats),trace=trace,
        certificate=certificate,incumbent_trace=incumbents,first_final_ub_seconds=first,
        first_optimal_solution_seconds=first if status=='OPT' else None,proof_seconds=proof,
        formulation=f'D{o.tail_depth}-contracted JT-LP with complete ancestor separators')
