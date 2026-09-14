"""Batched exact D2/D3 GPU messages with a resident global bitset dictionary.

The owning D3Workspace fixes data, weights and constraints. Each call uploads
only masks/metadata; global features are uploaded once. Sparse mode scans exact
nonzero word indices, never a sample or feature approximation.
"""
from dataclasses import asdict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np

from .d3_batched import _configure_cupy_runtime
from .problem import Deadline, DeadlineExceeded, RowSet, Tree

INF = 1e300


class ResidentGPU:
    def __init__(self, owner):
        self.owner, self.p, self.o = owner, owner.p, owner.options
        if self.o.gpu_cost_strategy not in ('baseline','fused','shared'):
            raise ValueError('Invalid resident GPU cost strategy')
        self.ready = False
        self.buffers = {}
        self.count_cache = OrderedDict()
        self.count_cache_bytes = 0

    def close(self):
        self.buffers.clear()
        self.count_cache.clear()
        self.count_cache_bytes = 0
        for name in ('zero', 'positive'):
            if hasattr(self, name): delattr(self, name)
        self.ready = False

    def prepare(self):
        if self.ready: return
        _configure_cupy_runtime()
        import cupy as cp
        self.cp = cp
        source = (Path(__file__).resolve().parents[1] / 'native/resident_dp.cu').read_text()
        if len(self.p.labels)>2:
            source=(Path(__file__).resolve().parents[1] / 'native/resident_multiclass.cu').read_text().replace('@CLASSES@',str(len(self.p.labels)))
        self.d3 = cp.RawKernel(source, 'resident_d3', options=('--std=c++11',))
        self.d2 = cp.RawKernel(source, 'resident_d2', options=('--std=c++11',))
        self.d3.compile(); self.d2.compile()
        self.fused = None
        if self.o.gpu_cost_strategy != 'baseline':
            extra=(Path(__file__).resolve().parents[1]/'native/resident_fused.cu').read_text().replace('@CLASSES@',str(len(self.p.labels)))
            self.fused=cp.RawKernel(source+'\n'+extra,'resident_d3_fused',options=('--std=c++11',))
            self.fused.compile()
        self.W = self.owner.Wmax
        host = np.frombuffer(b''.join(int(m[0]).to_bytes(self.W*8, 'little')
                                     for m in self.p._feature_masks), dtype=np.uint64)
        self.zero = cp.asarray(host)
        label_bytes=(b''.join(m.to_bytes(self.W*8,'little') for m in self.owner.class_masks)
                     if len(self.p.labels)>2 else self.owner.positive_mask.to_bytes(self.W*8,'little'))
        self.positive = cp.asarray(np.frombuffer(label_bytes, dtype=np.uint64))
        cp.cuda.Stream.null.synchronize()
        self.ready = True

    def share_data_from(self, source):
        """Share read-only GPU inputs across depths/penalties; keep buffers private."""
        if self.ready: raise RuntimeError('Target GPU workspace is already initialized')
        if (self.p.n != source.p.n or self.p.F != source.p.F or
                self.p._feature_masks != source.p._feature_masks or
                self.p.labels != source.p.labels or self.owner.class_masks != source.owner.class_masks):
            raise ValueError('GPU data reuse requires identical binary inputs and label masks')
        source.prepare()
        # Compile this workspace's selected kernel before sharing only inputs.
        if self.o.gpu_cost_strategy != source.o.gpu_cost_strategy:
            self.prepare()
            self.zero,self.positive=source.zero,source.positive
            return
        for name in ('cp', 'W', 'zero', 'positive', 'd2', 'd3','fused'):
            setattr(self, name, getattr(source, name))
        self.ready = True

    def buffer(self, name, shape, dtype, host=None):
        count = int(np.prod(shape))
        key = name, np.dtype(dtype).str
        if key not in self.buffers or self.buffers[key].size < count:
            self.buffers[key] = self.cp.empty(count, dtype=dtype)
        result = self.buffers[key][:count].reshape(shape)
        if host is not None: result.set(np.ascontiguousarray(host))
        return result

    def metadata(self, request, depth):
        p, o, owner = self.p, self.o, self.owner
        rows = request.get('rows', p.all_rows)
        used, node = tuple(request.get('used', ())), tuple(request.get('node', ()))
        if not isinstance(rows, RowSet) or rows.mask < 0 or rows.mask & ~p.all_rows.mask:
            raise ValueError('Invalid routed RowSet')
        if len(node)+depth > p.depth or any(a not in (0, 1) for a in node):
            raise ValueError('Shallow subtree must fit within the parent depth')
        if any(not 0 <= f < p.F for f in used): raise ValueError('Invalid ancestor feature')
        tree, loss = owner.leaf(rows)
        stop = loss if p.early_stop and len(rows) >= p.min_leaf else INF
        stats = dict(backend='gpu', options=asdict(o), input_features=p.F, input_rows=len(rows),
                     classes=len(p.labels), terminal_depth=depth,
                     effective_features=0, equivalent_features_removed=0, kernel_calls=0,
                     gpu_cost_kernel='resident_d'+str(depth), global_input_reused=self.ready,
                     packed_input_reused=self.ready, whole_tree_stop=False, threads=0,
                     global_weight=p._uniform_weight, sparse_words_used=False)
        rec = dict(rows=rows, used=used, node=node, stop=stop, stop_tree=tree,
                   stats=stats, answer=None, depth=depth)
        if len(rows) < p.min_leaf:
            rec['answer'] = ('INFEASIBLE', INF, None)
            return rec
        costs, allowed = owner._metadata(node, used)
        levels = 3 if depth == 2 else 7
        if depth == 2: allowed[3:] = 0
        if o.stop_bounds and stop < INF and stop <= np.min(np.where(allowed[0], costs[0], INF)):
            stats['whole_tree_stop'] = True
            rec['answer'] = ('OPT', stop, tree)
            return rec
        features = np.flatnonzero(np.any(allowed, axis=0))
        counts = self.counts(rows,node,used,stats) if o.gpu_native_metadata else None
        safe = bool(p.early_stop and np.all(allowed[:levels] == allowed[0]) and np.all(costs[:levels] == p.penalty))
        stats['constant_compaction_eligible'] = safe
        if safe and (o.compact_features or o.quotient_features):
            if counts is not None and o.compact_features:
                features=features[(counts[0,features]>0)&(counts[0,features]<len(rows))]
            kept, seen = [], set()
            for f in features:
                left = rows.mask & p._feature_masks[f][0]
                if o.compact_features and left in (0, rows.mask): continue
                key = min(left, rows.mask ^ left)
                if o.quotient_features and key in seen:
                    stats['equivalent_features_removed'] += 1
                    continue
                seen.add(key); kept.append(f)
            features = np.asarray(kept, dtype=np.int32)
        elif o.quotient_features:
            raise ValueError('Feature quotienting requires STOP and shared uniform-cost choices')
        if not len(features):
            rec['answer'] = ('OPT', stop, tree) if stop < INF else ('INFEASIBLE', INF, None)
            return rec
        costs, allowed = costs[:, features].copy(), allowed[:, features].copy()
        floors = np.min(np.where(allowed, costs, INF), axis=1)
        side_stop = np.full((2, len(features)), INF)
        dominated = np.zeros_like(side_stop, dtype=np.uint8)
        if counts is not None:
            for a,(support,errors) in enumerate(owner._side_losses(rows,counts,features)):
                loss=errors*p._uniform_weight
                valid=(allowed[0]!=0)&(support>=p.min_leaf)
                if p.early_stop:side_stop[a]=np.where(valid,.5*costs[0]+loss,INF)
                dominated[a]=((support<p.min_leaf)|(o.stop_bounds&p.early_stop&(loss<=floors[1+a]))).astype(np.uint8)
        for f, original in (() if counts is not None else enumerate(features)):
            if not allowed[0, f]: continue
            for a in (0, 1):
                child = p.route(rows, int(original), a)
                if len(child) < p.min_leaf:
                    dominated[a, f] = 1; continue
                if p.early_stop:
                    error = owner.leaf(child)[1]
                    side_stop[a, f] = .5*costs[0, f]+error
                    if o.stop_bounds and error <= floors[1+a]: dominated[a, f] = 1
        rec.update(features=features, costs=costs, allowed=allowed, floors=floors,
                   side_stop=side_stop, dominated=dominated)
        stats.update(effective_features=len(features), side_stop_states=int(dominated.sum()))
        return rec

    def counts(self,rows,node,used,stats):
        """Reuse exact all-feature statistics, including parent-minus-sibling."""
        capacity=self.o.gpu_metadata_cache_entries
        def remember(mask,value):
            if not capacity:return
            if mask in self.count_cache:
                previous=self.count_cache.pop(mask)
                self.count_cache_bytes-=previous.nbytes+(mask.bit_length()+7)//8+128
            self.count_cache[mask]=value
            self.count_cache_bytes+=value.nbytes+(mask.bit_length()+7)//8+128
            while len(self.count_cache)>capacity or self.count_cache_bytes>64*1024*1024:
                k,v=self.count_cache.popitem(last=False)
                self.count_cache_bytes-=v.nbytes+(k.bit_length()+7)//8+128
        if rows.mask in self.count_cache:
            self.count_cache.move_to_end(rows.mask);stats['metadata_cache_hits']=1
            return self.count_cache[rows.mask]
        parent=None
        if capacity and len(node)==len(used) and node:
            parent=self.p.all_rows
            for f,a in zip(used[:-1],node[:-1]):parent=self.p.route(parent,f,a)
            # Validate ancestry before using the complement identity.
            if self.p.route(parent,used[-1],node[-1]).mask!=rows.mask:parent=None
        if parent is not None:
            if parent.mask not in self.count_cache:
                remember(parent.mask,self.owner._native_counts(parent))
                stats['metadata_native_calls']=stats.get('metadata_native_calls',0)+1
            sibling=parent.mask^rows.mask
            if parent.mask in self.count_cache and sibling in self.count_cache:
                value=self.count_cache[parent.mask]-self.count_cache[sibling]
                stats['metadata_sibling_reuse']=1
                remember(rows.mask,value)
                return value
        value=self.owner._native_counts(rows)
        stats['metadata_native_calls']=stats.get('metadata_native_calls',0)+1
        remember(rows.mask,value)
        stats['metadata_cache_bytes']=self.count_cache_bytes
        return value

    def solve_many(self,requests,time_limit=100,depth=3,audit=None):
        chunk=self.o.gpu_pipeline_chunk
        if not chunk or len(requests)<=chunk:
            return self._solve_chunk(requests,time_limit,depth,audit)
        clock=Deadline(time_limit)
        groups=[requests[i:i+chunk] for i in range(0,len(requests),chunk)]
        def prepare(group):
            start=time.perf_counter();records=[self.metadata(r,depth) for r in group]
            return records,start,time.perf_counter()
        prepared=prepare(groups[0]);answers=[]
        # One preparation worker owns the statistics cache while the main
        # thread owns CUDA. Only read-only data is shared during GPU execution.
        with ThreadPoolExecutor(max_workers=1) as pool:
            for j,group in enumerate(groups):
                future=pool.submit(prepare,groups[j+1]) if j+1<len(groups) and clock.remaining()>0 else None
                records,start,end=prepared;solve_start=time.perf_counter()
                out=self._solve_chunk(group,max(0,clock.remaining()),depth,audit,(records,end-start))
                solve_end=time.perf_counter();answers.extend(out)
                if future is not None:
                    prepared=future.result();_,a,b=prepared
                    out[0]['stats']['pipeline_overlap_seconds']=max(0,min(b,solve_end)-max(a,solve_start))
                    out[0]['stats']['pipeline_prefetch_batches']=1
                elif j+1<len(groups):
                    # Return valid STOP fallbacks for work not started at expiry.
                    for remaining in groups[j+1:]:
                        for request in remaining:
                            rows=request.get('rows',self.p.all_rows);tree,value=self.owner.leaf(rows)
                            status='INFEASIBLE' if len(rows)<self.p.min_leaf else 'TIME'
                            if not self.p.early_stop or len(rows)<self.p.min_leaf:tree,value=None,INF
                            answers.append(dict(status=status,value=value,tree=tree,
                                stats=dict(kernel_calls=0),timings={}))
                    break
        return answers

    def _solve_chunk(self, requests, time_limit=100, depth=3, audit=None, prepared=None):
        if depth not in (2, 3): raise ValueError('Resident messages support D2 or D3')
        clock = Deadline(time_limit)
        records = [self.metadata(r, depth) for r in requests] if prepared is None else prepared[0]
        times = dict(metadata_seconds=clock.elapsed() if prepared is None else prepared[1], backend_setup_seconds=0., pack_seconds=0.,
                     cost_and_h_reduction_seconds=0., join_seconds=0., recovery_seconds=0.)
        pending = [r for r in records if r['answer'] is None]
        try:
            clock.check()
            if pending:
                tick = time.perf_counter(); self.prepare()
                times['backend_setup_seconds'] = time.perf_counter()-tick
                clock.check()
                tick = time.perf_counter()
                cp, p, W, B = self.cp, self.p, self.W, len(pending)
                F = max(len(r['features']) for r in pending)
                masks = np.empty((B, W), dtype=np.uint64)
                indices = np.zeros((B, W), dtype=np.int32)
                nw = np.full(B, -1, dtype=np.int32)
                features = np.zeros((B, F), dtype=np.int32)
                costs = np.full((B, 7, F), INF)
                allowed = np.zeros((B, 7, F), dtype=np.uint8)
                floors = np.empty((B, 7))
                dominated = np.ones((B, 2, F), dtype=np.uint8)
                stops = np.full((B, 2, F), INF)
                for j, r in enumerate(pending):
                    n = len(r['features'])
                    masks[j] = np.frombuffer(r['rows'].mask.to_bytes(W*8, 'little'), dtype=np.uint64)
                    nz = np.flatnonzero(masks[j])
                    sparse = self.o.sparse_words or (self.o.gpu_adaptive_words and len(nz) < self.o.gpu_sparse_threshold*W)
                    if sparse:
                        indices[j, :len(nz)] = nz; nw[j] = len(nz)
                    r['stats'].update(sparse_words_used=bool(sparse), active_words=len(nz), global_words=W,
                                      batch_size=B, gpu_cost_kernel='resident_d'+str(depth))
                    features[j, :n] = r['features']; costs[j, :, :n] = r['costs']
                    allowed[j, :, :n] = r['allowed']; floors[j] = r['floors']
                    dominated[j, :, :n] = r['dominated']; stops[j, :, :n] = r['side_stop']
                args = [self.zero, self.positive]
                for name, value in [('masks', masks), ('indices', indices), ('nw', nw),
                                    ('features', features), ('costs', costs), ('allowed', allowed)]:
                    args.append(self.buffer(name, value.shape, value.dtype, value))
                if depth == 3:
                    args += [self.buffer('floors', floors.shape, floors.dtype, floors),
                             self.buffer('dominated', dominated.shape, dominated.dtype, dominated)]
                blocks = 4 if depth == 3 else 2
                values = self.buffer('values', (B, blocks, F, F), np.float64)
                tail = self.buffer('tail', values.shape, np.int32)
                reasons = self.buffer('reasons', values.shape, np.uint8)
                cp.cuda.Stream.null.synchronize()
                times['pack_seconds'] = time.perf_counter()-tick
                # Check the shared deadline between bounded batches of feature pairs.
                tile = min(self.o.tile_pairs, 256)
                fused = depth==3 and self.o.gpu_cost_strategy!='baseline'
                max_words=int(np.max(np.where(nw<0,W,nw)))
                shared_bytes=8*max_words if fused and self.o.gpu_cost_strategy=='shared' else 0
                # Keep occupancy reasonable and always support huge input bitsets.
                if shared_bytes>16384:shared_bytes=0
                for r in pending:
                    r['stats'].update(gpu_cost_strategy=self.o.gpu_cost_strategy,
                        pair_cache_bytes=shared_bytes,pair_cache_fallback=bool(fused and self.o.gpu_cost_strategy=='shared' and not shared_bytes))
                tick=time.perf_counter();sync_count=0
                for step,start in enumerate(range(0, F*F, tile)):
                    count = min(tile, F*F-start)
                    ints = [F, W, int(p.no_repeat)]
                    if depth == 3: ints += [int(p.early_stop), p.min_leaf, int(self.o.stop_bounds)]
                    else: ints += [p.min_leaf]
                    ints += [start, count]
                    outputs = [values, tail, reasons] if depth == 3 else [values]
                    kernel = self.fused if fused else self.d3 if depth == 3 else self.d2
                    kernel_args=tuple(args)+tuple(np.int32(x) for x in ints)+(np.float64(p._uniform_weight),)+tuple(outputs)
                    if fused:kernel_args+=(np.int32(bool(shared_bytes)),)
                    kernel((count*blocks, B), (256,), kernel_args,shared_mem=shared_bytes)
                    if (step+1)%self.o.gpu_sync_tiles==0 or start+count==F*F:
                        cp.cuda.Stream.null.synchronize();sync_count+=1
                        times['cost_and_h_reduction_seconds'] += time.perf_counter()-tick
                        clock.check();tick=time.perf_counter()
                    for r in pending: r['stats']['kernel_calls'] += 1/B
                pending[0]['stats']['gpu_sync_calls']=sync_count
                clock.check(); tick = time.perf_counter()
                if audit and depth == 3:
                    for j, r in enumerate(pending):
                        n = len(r['features'])
                        audit(r['features'].copy(), cp.asnumpy(values[j, :, :n, :n]), cp.asnumpy(tail[j, :, :n, :n]))
                sums = values[:, ::2]+values[:, 1::2] if depth == 3 else values
                g = cp.argmin(sums, axis=3)
                side = cp.take_along_axis(sums, g[..., None], axis=3)[..., 0]
                stop_d = cp.asarray(stops)
                g = cp.where(stop_d <= side, -1, g)
                side = cp.minimum(side, stop_d)
                root_values = side.sum(axis=1)
                f = cp.argmin(root_values, axis=1)
                batch = cp.arange(B)
                # Transfer only each winning value and seven actions, in one copy.
                answer = cp.full((B, 8), -1., dtype=cp.float64)
                answer[:, 0] = root_values[batch, f]; answer[:, 1] = f
                for a in (0, 1):
                    ga = g[batch, a, f]; answer[:, 2+a] = ga
                    if depth == 3:
                        for b in (0, 1): answer[:, 4+2*a+b] = tail[batch, 2*a+b, f, cp.maximum(ga, 0)]
                answer = cp.asnumpy(answer)
                times['join_seconds'] = time.perf_counter()-tick
                tick = time.perf_counter()
                for r, raw in zip(pending, answer):
                    value = float(raw[0])
                    if r['stop'] <= value and r['stop'] < INF:
                        value, tree = r['stop'], r['stop_tree']
                    elif value >= INF/2:
                        r['answer'] = ('INFEASIBLE', INF, None); continue
                    else:
                        fs, rows = r['features'], r['rows']
                        f = int(fs[int(raw[1])]); children = []
                        for a in (0, 1):
                            child = p.route(rows, f, a); gi = int(raw[2+a])
                            if gi < 0: children.append(self.owner.leaf(child)[0]); continue
                            g = int(fs[gi]); leaves = []
                            for b in (0, 1):
                                ab = p.route(child, g, b)
                                h = int(raw[4+2*a+b]) if depth == 3 else -1
                                if h == -1: leaves.append(self.owner.leaf(ab)[0])
                                elif h >= 0:
                                    hf = int(fs[h])
                                    leaves.append(Tree(feature=hf, left=self.owner.leaf(p.route(ab, hf, 0))[0], right=self.owner.leaf(p.route(ab, hf, 1))[0]))
                                else: raise AssertionError('Invalid resident backpointer')
                            children.append(Tree(feature=g, left=leaves[0], right=leaves[1]))
                        tree = Tree(feature=f, left=children[0], right=children[1])
                    actual = self.owner._validate(tree, r['rows'], r['node'], r['used'])
                    if abs(actual-value) > 1e-9: raise AssertionError('Resident objective mismatch')
                    r['answer'] = ('OPT', value, tree)
                times['recovery_seconds'] = time.perf_counter()-tick
        except DeadlineExceeded:
            pass
        result = []
        for r in records:
            status, value, tree = r['answer'] or ('TIME', r['stop'], r['stop_tree'] if r['stop'] < INF else None)
            result.append(dict(status=status, value=value, tree=tree, seconds=clock.elapsed(), stats=r['stats'],
                               timings={k: v/max(1, len(records)) for k, v in times.items()}))
        return result
