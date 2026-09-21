"""Exact D3 min-sum with STOP bounds, routed-bitset reuse and reusable workspaces.

D3Workspace is tied to one immutable Problem. It can solve routed D3 subproblems
of a deeper parent without constructing another Problem or renormalizing weights.
The original d3_batched backends remain unchanged as independent baselines.
"""
from dataclasses import asdict, dataclass
import ctypes
from itertools import product
from pathlib import Path
import time

import numpy as np

from .d3_batched import _configure_cupy_runtime, _pack_words64
from .problem import Deadline, DeadlineExceeded, RowSet, Tree
from .solvers import result_dict

ROOT=Path(__file__).resolve().parents[1]
INF=1e300
NODES=[()]+[(a,) for a in (0,1)]+list(product((0,1),repeat=2))


@dataclass(frozen=True)
class D3Options:
    stop_bounds: bool = True
    compact_features: bool = True
    sparse_words: bool = False
    tile_pairs: int = 4096
    fast_prepare: bool = False
    quotient_features: bool = False
    private_screen: bool = False
    gpu_force_word_tiled: bool = False
    gpu_word_tile: int = 4096
    gpu_resident: bool = False
    gpu_adaptive_words: bool = True
    gpu_sparse_threshold: float = 0.5
    gpu_native_metadata: bool = False
    gpu_metadata_cache_entries: int = 0
    gpu_sync_tiles: int = 1
    gpu_pipeline_chunk: int = 0
    gpu_cost_strategy: str = 'baseline'
    gpu_compact_rows: bool = True
    gpu_pair_tile: int = 4096
    gpu_bucket_min: int = 8
    gpu_fused_join: bool = True


PRESETS={'core':D3Options(stop_bounds=False,compact_features=False),
         'bounds':D3Options(compact_features=False),
         'compact':D3Options(),
         'sparse':D3Options(sparse_words=True)}


class D3Workspace:
    """Reusable, sequential D3 oracle; do not mutate its parent Problem."""
    def __init__(self,p,backend='cpp',threads=0,options=None):
        if p.depth<2 or p._uniform_weight is None:
            raise ValueError('Shallow workspace requires depth >=2 and uniform weights')
        if backend not in ('cpp','gpu'):
            raise ValueError('D3 optimized backend must be cpp or gpu')
        self.p,self.backend,self.threads=p,backend,int(threads)
        self.options=options or D3Options()
        if self.options.gpu_pair_tile<1 or self.options.gpu_bucket_min<0:
            raise ValueError('Invalid resident GPU dispatch options')
        if self.options.tile_pairs<1:
            raise ValueError('tile_pairs must be positive')
        if self.options.gpu_word_tile<1:
            raise ValueError('gpu_word_tile must be positive')
        if not 0 <= self.options.gpu_sparse_threshold <= 1:
            raise ValueError('gpu_sparse_threshold must be in [0,1]')
        if self.options.gpu_sync_tiles<1 or self.options.gpu_pipeline_chunk<0 or self.options.gpu_metadata_cache_entries<0:
            raise ValueError('Invalid resident GPU preparation options')
        if self.options.private_screen and not self.options.stop_bounds:
            raise ValueError('Private screening requires valid STOP/split lower bounds')
        self.xp=np;self.handle=None;self.ready=False;self.closed=False
        self.last_pack_key=None
        self.prepare_lib=None
        self.resident=None
        self.Wmax=max(1,(p.n+63)//64)
        self.class_masks=tuple(p._label_masks[k] for k in p.labels)
        self.positive_mask=p._label_masks[p.labels[-1]] if len(p.labels)==2 else 0

    def close(self):
        if self.resident is not None:self.resident.close()
        if self.handle:
            self.lib.d3_opt_free(self.handle);self.handle=None
        if self.backend=='gpu' and self.ready:
            for name,value in tuple(vars(self).items()):
                if isinstance(value,self.xp.ndarray):delattr(self,name)
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def _prepare_backend(self):
        if self.ready:return
        p=self.p
        if self.backend=='cpp':
            self.lib=ctypes.CDLL(str(ROOT/'jt_oct/_native/d3_optimized.dll'))
            self.lib.d3_opt_create.argtypes=[ctypes.c_int,ctypes.c_int]
            self.lib.d3_opt_create.restype=ctypes.c_void_p
            self.lib.d3_opt_free.argtypes=[ctypes.c_void_p]
            self.lib.d3_opt_costs.argtypes=[ctypes.c_void_p]*8+[ctypes.c_double]+[ctypes.c_void_p]*3
            self.lib.d3_opt_costs.restype=ctypes.c_int
            for name in ('d3_opt_costs_multiclass','d2_opt_costs_multiclass'):
                fun=getattr(self.lib,name)
                fun.argtypes=self.lib.d3_opt_costs.argtypes
                fun.restype=ctypes.c_int
            self.lib.d3_opt_counts_multiclass.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
            self.lib.d3_opt_counts_multiclass.restype=None
            self.handle=self.lib.d3_opt_create(self.Wmax,self.threads)
            if not self.handle:raise MemoryError('D3 native workspace allocation failed')
        else:
            _configure_cupy_runtime()
            import cupy as cp
            self.xp=cp
            code=(ROOT/'native'/('d3_screened.cu' if self.options.private_screen else 'd3_optimized.cu')).read_text(encoding='utf-8')
            self.kernel=cp.RawKernel(code,'d3_opt_costs',options=('--std=c++11',))
            large_code=(ROOT/'native'/'d3_optimized.cu').read_text(encoding='utf-8')
            self.word_tiled_kernel=cp.RawKernel(large_code,'d3_opt_costs_word_tiled',options=('--std=c++11',))
            self.pack_kernel=cp.RawKernel(code,'d3_opt_pack',options=('--std=c++11',))
            self.kernel.compile();self.word_tiled_kernel.compile();self.pack_kernel.compile()
            self.device_X=cp.asarray(np.ascontiguousarray(p.X.T))
            self.device_y=cp.asarray(np.ascontiguousarray(p.y==p.labels[-1],dtype=np.uint8)
                                     if len(p.labels)==2 else np.zeros(p.n,dtype=np.uint8))
            self.device_ids=cp.empty(p.n,dtype=cp.int32)
            self.device_features=cp.empty(p.F,dtype=cp.int32)
            self.device_costs=cp.empty((7,p.F),dtype=cp.float64)
            self.device_allowed=cp.empty((7,p.F),dtype=cp.uint8)
            self.device_floors=cp.empty(7,dtype=cp.float64)
            self.device_dominated=cp.empty((2,p.F),dtype=cp.uint8)
            self.device_zero=cp.empty(p.F*self.Wmax,dtype=cp.uint64)
            self.device_positive=cp.empty(self.Wmax,dtype=cp.uint64)
        xp=self.xp
        self.output_buffer=xp.empty(4*p.F*p.F,dtype=xp.float64)
        self.tail_buffer=xp.empty(4*p.F*p.F,dtype=xp.int32)
        self.reason_buffer=xp.empty(4*p.F*p.F,dtype=xp.uint8)
        if self.backend=='gpu':xp.cuda.Stream.null.synchronize()
        self.ready=True

    def host(self,a):return self.xp.asnumpy(a) if self.backend=='gpu' else np.asarray(a)

    def leaf(self,rows):
        if len(self.p.labels)>2:
            counts=[(rows.mask&m).bit_count() for m in self.class_masks]
            winner=int(np.argmax(counts))
            return Tree(label=self.p.labels[winner]),(len(rows)-counts[winner])*self.p._uniform_weight
        n=len(rows);pos=(rows.mask&self.positive_mask).bit_count()
        label=self.p.labels[-1] if pos>n-pos else self.p.labels[0]
        return Tree(label=label),min(pos,n-pos)*self.p._uniform_weight

    def _metadata(self,node,used):
        p=self.p
        if self.options.fast_prepare and not p.allowed and not p.split_costs:
            allowed=np.ones((7,p.F),dtype=np.uint8)
            if p.no_repeat and used:allowed[:,list(used)]=0
            return np.full((7,p.F),p.penalty,dtype=np.float64),allowed
        allowed=np.zeros((7,p.F),dtype=np.uint8)
        costs=np.full((7,p.F),p.penalty,dtype=np.float64)
        for k,relative in enumerate(NODES):
            absolute=node+relative
            allowed[k,list(p.features(absolute,used))]=1
            if p.split_costs:
                costs[k]=[p.cost(absolute,f) for f in range(p.F)]
        return costs,allowed

    def _native_counts(self,rows):
        if len(self.p.labels)>2:
            # Preparation is timed as metadata; no binary class approximation.
            if not hasattr(self,'multiclass_count_lib'):
                self.multiclass_count_lib=ctypes.CDLL(str(ROOT/'jt_oct/_native/d3_optimized.dll'))
                self.multiclass_count_lib.d3_opt_counts_multiclass.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
                self.multiclass_count_lib.d3_opt_counts_multiclass.restype=None
                W=self.Wmax
                self.mc_global_zero=np.frombuffer(b''.join(int(m[0]).to_bytes(W*8,'little') for m in self.p._feature_masks),dtype=np.uint64)
                self.mc_global_labels=np.frombuffer(b''.join(m.to_bytes(W*8,'little') for m in self.class_masks),dtype=np.uint64)
            mask=np.frombuffer(rows.mask.to_bytes(self.Wmax*8,'little'),dtype=np.uint64)
            counts=np.empty((1+len(self.p.labels),self.p.F),dtype=np.int32)
            ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
            self.multiclass_count_lib.d3_opt_counts_multiclass(ptr(self.mc_global_zero),ptr(self.mc_global_labels),ptr(mask),
                self.p.F,self.Wmax,len(self.p.labels),ptr(counts))
            return counts
        if self.prepare_lib is None:
            self.prepare_lib=ctypes.CDLL(str(ROOT/'jt_oct/_native/d3_prepare.dll'))
            self.prepare_lib.d3_prepare_counts.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
            self.prepare_lib.d3_prepare_counts.restype=None
            W=self.Wmax
            self.host_global_zero=np.frombuffer(b''.join(int(m[0]).to_bytes(W*8,'little') for m in self.p._feature_masks),dtype=np.uint64)
            self.host_global_positive=np.frombuffer(self.positive_mask.to_bytes(W*8,'little'),dtype=np.uint64)
        mask=np.frombuffer(rows.mask.to_bytes(self.Wmax*8,'little'),dtype=np.uint64)
        counts=np.empty((2,self.p.F),dtype=np.int32)
        ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
        self.prepare_lib.d3_prepare_counts(ptr(self.host_global_zero),ptr(self.host_global_positive),ptr(mask),
                                          self.p.F,self.Wmax,ptr(counts))
        return counts

    def _side_losses(self,rows,counts,features):
        n0=counts[0,features];n1=len(rows)-n0
        if len(self.p.labels)>2:
            totals=np.asarray([(rows.mask&m).bit_count() for m in self.class_masks])[:,None]
            c0=counts[1:,features];c1=totals-c0
            return (n0,n0-c0.max(axis=0)),(n1,n1-c1.max(axis=0))
        y0=counts[1,features];y1=(rows.mask&self.positive_mask).bit_count()-y0
        return (n0,np.minimum(y0,n0-y0)),(n1,np.minimum(y1,n1-y1))

    def _pack(self,rows,features,all_labels=False):
        key=(rows.mask,tuple(features),all_labels)
        n,F=len(rows),len(features);W=max(1,(n+63)//64)
        if key==self.last_pack_key:return self.zero,self.positive,True
        p=self.p
        if rows.mask==p.all_rows.mask:ids=np.arange(p.n,dtype=np.int32)
        elif self.options.fast_prepare:
            bits=np.frombuffer(rows.mask.to_bytes(self.Wmax*8,'little'),dtype=np.uint8)
            ids=np.flatnonzero(np.unpackbits(bits,bitorder='little')).astype(np.int32)
        else:ids=np.fromiter(rows,dtype=np.int32,count=n)
        if self.backend=='cpp':
            packed=np.packbits((p.X[ids][:,features]==0).T,axis=1,bitorder='little')
            packed=np.pad(packed,((0,0),(0,W*8-packed.shape[1])))
            self.zero=np.ascontiguousarray(packed).view(np.uint64).reshape(F,W)
            if len(p.labels)>2 or all_labels:
                self.positive=np.ascontiguousarray(np.stack([
                    _pack_words64(p.y[ids]==k) if n else np.zeros(W,dtype=np.uint64) for k in p.labels]))
            else:
                self.positive=(_pack_words64(p.y[ids]==p.labels[-1]) if n and len(p.labels)==2
                               else np.zeros(W,dtype=np.uint64))
        else:
            self.device_ids[:n].set(ids)
            self.device_features[:F].set(np.asarray(features,dtype=np.int32))
            self.zero=self.device_zero[:F*W].reshape(F,W)
            self.positive=self.device_positive[:W]
            self.pack_kernel(((F*W+7)//8,),(256,),
                (self.device_X,self.device_y,self.device_ids,self.device_features,
                 np.int32(p.n),np.int32(n),np.int32(F),np.int32(W),self.zero,self.positive))
            self.xp.cuda.Stream.null.synchronize()
        self.last_pack_key=key
        return self.zero,self.positive,False

    def solve(self,rows=None,used=(),node=(),time_limit=100,audit=None,depth=3):
        if self.closed:raise RuntimeError('D3 workspace is closed')
        if depth not in (2,3):raise ValueError('Shallow messages require depth 2 or 3')
        if self.backend=='gpu' and (self.options.gpu_resident or len(self.p.labels)>2 or depth==2):
            return self.solve_many([dict(rows=self.p.all_rows if rows is None else rows,used=used,node=node)],
                                   time_limit=time_limit,depth=depth,audit=audit)[0]
        p,o=self.p,self.options
        rows=p.all_rows if rows is None else rows
        if not isinstance(rows,RowSet) or rows.mask<0 or rows.mask&~p.all_rows.mask:
            raise ValueError('rows must be a RowSet from the parent dataset')
        node,used=tuple(node),tuple(used)
        if len(node)+depth>p.depth or any(b not in (0,1) for b in node):
            raise ValueError('Shallow subtree must fit within the parent depth')
        if any(not 0<=f<p.F for f in used):raise ValueError('Invalid ancestor feature')
        clock=Deadline(time_limit)
        stats={'backend':self.backend,'options':asdict(o),'input_features':p.F,'input_rows':len(rows),
               'classes':len(p.labels),'terminal_depth':depth,
               'global_weight':p._uniform_weight,'whole_tree_stop':False,'kernel_calls':0,
               'effective_features':p.F,'packed_input_reused':False,'threads':self.threads}
        times={'metadata_seconds':0.,'backend_setup_seconds':0.,'pack_seconds':0.,
               'cost_and_h_reduction_seconds':0.,'join_seconds':0.,'recovery_seconds':0.}
        stop_tree,stop_value=self.leaf(rows)
        can_stop=p.early_stop and len(rows)>=p.min_leaf
        if not can_stop:stop_value=INF
        def finished(status,value,tree):
            return {'status':status,'value':float(value),'tree':tree,'seconds':clock.elapsed(),
                    'stats':stats,'timings':times}
        try:
            clock.check();tick=time.perf_counter()
            costs,allowed=self._metadata(node,used)
            if depth==2:allowed[3:]=0
            floors=np.min(np.where(allowed,costs,INF),axis=1)
            if o.stop_bounds and can_stop and stop_value<=floors[0]:
                stats['whole_tree_stop']=True;stats['effective_features']=0
                times['metadata_seconds']=time.perf_counter()-tick
                return finished('OPT',stop_value,stop_tree)
            features=np.flatnonzero(np.any(allowed,axis=0))
            # Constant predicates can be removed only if contraction preserves all
            # node-specific choices/costs. General node restrictions disable it.
            levels=3 if depth==2 else 7
            compact_safe=(p.early_stop and np.all(allowed[:levels]==allowed[0]) and np.all(costs[:levels]==p.penalty))
            counts=self._native_counts(rows) if o.fast_prepare else None
            if o.compact_features and compact_safe:
                if counts is not None:features=features[(counts[0,features]>0)&(counts[0,features]<len(rows))].astype(np.int32)
                else:features=np.array([f for f in features if 0<(rows.mask&p._feature_masks[f][0]).bit_count()<len(rows)],dtype=np.int32)
            stats['equivalent_features_removed']=0
            if o.quotient_features:
                if not compact_safe:raise ValueError('Feature quotienting requires STOP and shared uniform-cost choices')
                seen=set();kept=[]
                for f in features:
                    left=rows.mask&p._feature_masks[f][0];key=min(left,rows.mask^left)
                    if key not in seen:seen.add(key);kept.append(f)
                stats['equivalent_features_removed']=len(features)-len(kept)
                features=np.asarray(kept,dtype=np.int32)
            if o.private_screen and counts is not None:
                sides=self._side_losses(rows,counts,features)
                scores=sides[0][1]+sides[1][1]
                features=features[np.argsort(scores,kind='stable')]
            stats['constant_compaction_eligible']=bool(compact_safe)
            stats['effective_features']=len(features)
            if not len(features):
                return finished('OPT' if can_stop else 'INFEASIBLE',stop_value,stop_tree if can_stop else None)
            F=len(features);W=max(1,(len(rows)+63)//64)
            costs=np.ascontiguousarray(costs[:,features]);allowed=np.ascontiguousarray(allowed[:,features])
            floors=np.ascontiguousarray(np.min(np.where(allowed,costs,INF),axis=1))
            side_stop=np.full((2,F),INF,dtype=np.float64)
            dominated=np.zeros((2,F),dtype=np.uint8)
            if counts is not None:
                for a,(support,errors) in enumerate(self._side_losses(rows,counts,features)):
                    loss=errors*p._uniform_weight
                    valid=(allowed[0]!=0)&(support>=p.min_leaf)
                    if p.early_stop:side_stop[a]=np.where(valid,0.5*costs[0]+loss,INF)
                    dominated[a]=((support<p.min_leaf)|(o.stop_bounds&p.early_stop&(loss<=floors[1+a]))).astype(np.uint8)
            for f,original in (() if counts is not None else enumerate(features)):
                if not allowed[0,f]:continue
                for a in (0,1):
                    child=p.route(rows,int(original),a)
                    if len(child)<p.min_leaf:
                        dominated[a,f]=1;continue
                    if p.early_stop:
                        _,loss=self.leaf(child)
                        side_stop[a,f]=0.5*costs[0,f]+loss
                        if o.stop_bounds and loss<=floors[1+a]:dominated[a,f]=1
            stats['side_stop_states']=int(np.count_nonzero(dominated))
            times['metadata_seconds']=time.perf_counter()-tick
            clock.check();tick=time.perf_counter();self._prepare_backend()
            times['backend_setup_seconds']=time.perf_counter()-tick
            clock.check();tick=time.perf_counter()
            zero,positive,reused=self._pack(rows,features,all_labels=depth==2)
            stats['packed_input_reused']=reused
            times['pack_seconds']=time.perf_counter()-tick
            xp=self.xp
            blocks=4 if depth==3 else 2
            kappa=self.output_buffer[:blocks*F*F].reshape(blocks,F,F)
            tail=self.tail_buffer[:blocks*F*F].reshape(blocks,F,F)
            reasons=self.reason_buffer[:blocks*F*F].reshape(blocks,F,F)
            if self.backend=='gpu':
                shared_limit=int(xp.cuda.runtime.getDeviceProperties(0)['sharedMemPerBlock'])
                word_tiled=bool(o.gpu_force_word_tiled or W*12+8192>shared_limit)
                if word_tiled and F>256:
                    return self.solve_many([dict(rows=rows,used=used,node=node)],
                                           time_limit=max(0,clock.end-time.perf_counter()),audit=audit)[0]
                word_tile=min(int(o.gpu_word_tile),max(1,(shared_limit-8192)//8))
                stats['gpu_cost_kernel']='word_tiled' if word_tiled else 'full_shared_rows'
                stats['gpu_word_tile']=int(word_tile) if word_tiled else None
                stats['full_shared_rows_bytes']=int(W*12+8192)
                stats['shared_memory_limit_bytes']=shared_limit
                dc=self.device_costs.ravel()[:7*F].reshape(7,F);dc.set(costs)
                da=self.device_allowed.ravel()[:7*F].reshape(7,F);da.set(allowed)
                self.device_floors.set(floors)
                dd=self.device_dominated.ravel()[:2*F].reshape(2,F);dd.set(dominated)
            for start in range(0,F*F,o.tile_pairs):
                clock.check();count=min(o.tile_pairs,F*F-start);tick=time.perf_counter()
                ip=np.array((F,W,len(rows),int(p.no_repeat),int(p.early_stop),p.min_leaf,
                             int(o.stop_bounds)+(2 if o.private_screen else 0),int(o.sparse_words),start,count),dtype=np.int32)
                if self.backend=='cpp':
                    ptr=lambda x:ctypes.c_void_p(x.ctypes.data)
                    if depth==2 or len(p.labels)>2:
                        ip=np.r_[ip,np.int32(len(p.labels))].astype(np.int32)
                        fun=getattr(self.lib,f'd{depth}_opt_costs_multiclass')
                    else:fun=self.lib.d3_opt_costs
                    rc=fun(self.handle,ptr(zero),ptr(positive),ptr(costs),ptr(allowed),
                        ptr(floors),ptr(dominated),ptr(ip),p._uniform_weight,ptr(kappa),ptr(tail),ptr(reasons))
                    if rc:raise RuntimeError(f'D3 native error {rc}')
                else:
                    if word_tiled:
                        self.word_tiled_kernel((count*4,),(256,),
                            (zero,positive,dc,da,self.device_floors,dd,*ip,np.int32(word_tile),
                             np.float64(p._uniform_weight),kappa,tail,reasons),
                            shared_mem=word_tile*8)
                    else:
                        self.kernel((count*4,),(256,),
                            (zero,positive,dc,da,self.device_floors,dd,*ip,np.float64(p._uniform_weight),kappa,tail,reasons),
                            shared_mem=W*12)
                    xp.cuda.Stream.null.synchronize()
                times['cost_and_h_reduction_seconds']+=time.perf_counter()-tick
                stats['kernel_calls']+=1
            clock.check();tick=time.perf_counter()
            if audit:audit(features.copy(),self.host(kappa).copy(),self.host(tail).copy())
            counts=np.bincount(self.host(reasons).ravel(),minlength=4)
            stats.update(invalid_private_states=int(counts[0]),searched_private_states=int(counts[1]),
                         skipped_by_side_stop=int(counts[2]),private_stop_states=int(counts[3]))
            stats['screened_exact_terminations']=int(counts[4]) if len(counts)>4 else 0
            sums=kappa[::2]+kappa[1::2] if depth==3 else kappa
            winner=xp.argmin(sums,axis=2)
            side_values=xp.take_along_axis(sums,winner[:,:,None],axis=2)[:,:,0]
            side_values=self.host(side_values);garg=self.host(winner).copy()
            choose_stop=side_stop<=side_values
            side_values=np.minimum(side_values,side_stop);garg[choose_stop]=-1
            root_values=side_values.sum(axis=0);f=int(np.argmin(root_values))
            value=float(root_values[f])
            times['join_seconds']=time.perf_counter()-tick
            tick=time.perf_counter()
            if stop_value<=value and can_stop:value,tree=stop_value,stop_tree
            elif value>=INF/2:return finished('INFEASIBLE',INF,None)
            else:
                hargs=self.host(tail[:,f,:])
                sides=[];original_f=int(features[f])
                for a in (0,1):
                    child=p.route(rows,original_f,a);g=int(garg[a,f])
                    if g<0:sides.append(self.leaf(child)[0]);continue
                    original_g=int(features[g]);leaves=[]
                    for b in (0,1):
                        child_ab=p.route(child,original_g,b);h=int(hargs[2*a+b,g]) if depth==3 else -1
                        if h==-1:leaves.append(self.leaf(child_ab)[0])
                        elif h>=0:
                            original_h=int(features[h])
                            leaves.append(Tree(feature=original_h,left=self.leaf(p.route(child_ab,original_h,0))[0],
                                               right=self.leaf(p.route(child_ab,original_h,1))[0]))
                        else:raise AssertionError('Invalid private action in recovered D3 tree')
                    sides.append(Tree(feature=original_g,left=leaves[0],right=leaves[1]))
                tree=Tree(feature=original_f,left=sides[0],right=sides[1])
            actual=self._validate(tree,rows,node,used)
            if abs(actual-value)>1e-9:raise AssertionError(f'D3 objective mismatch {actual} vs {value}')
            times['recovery_seconds']=time.perf_counter()-tick
            return finished('OPT',value,tree)
        except DeadlineExceeded:
            return finished('TIME',stop_value,stop_tree if can_stop else None)

    def _validate(self,tree,rows,node,used):
        p=self.p
        if tree.feature is None:
            if tree.label not in p.labels or len(rows)<p.min_leaf:raise AssertionError('Invalid D3 leaf')
            return p.loss(rows,tree.label)
        f=tree.feature
        if f not in p.features(node,used):raise AssertionError('Invalid D3 split')
        return (p.cost(node,f)+self._validate(tree.left,p.route(rows,f,0),node+(0,),used+(f,))+
                self._validate(tree.right,p.route(rows,f,1),node+(1,),used+(f,)))

    def solve_many(self,requests,time_limit=100,depth=3,audit=None):
        """Solve a bounded batch of independent messages against resident data."""
        if self.closed:raise RuntimeError('D3 workspace is closed')
        if self.backend!='gpu':
            clock=Deadline(time_limit)
            return [self.solve(**r,time_limit=max(0,clock.end-time.perf_counter()),audit=audit,depth=depth) for r in requests]
        if self.resident is None:
            from .resident_table3 import ResidentGPU
            self.resident=ResidentGPU(self)
        return self.resident.solve_many(requests,time_limit,depth,audit)

    def share_gpu_data_from(self,source):
        """Reuse resident bitsets across compatible Problems without mutating them."""
        if self.closed or source.closed:raise RuntimeError('Cannot reuse a closed workspace')
        if self.backend!='gpu' or source.backend!='gpu':raise ValueError('Both workspaces must use GPU')
        from .resident_table3 import ResidentGPU
        if source.resident is None:source.resident=ResidentGPU(source)
        if self.resident is None:self.resident=ResidentGPU(self)
        self.resident.share_data_from(source.resident)
        return self


def solve_jt_dp_shallow_gpu(p,time_limit=100,workspace=None,options=None):
    """Dedicated exact D2 or D3, optionally reusing a caller-owned GPU workspace."""
    if p.depth not in (2,3):raise ValueError('Shallow GPU solver requires depth 2 or 3')
    if workspace is not None and (workspace.p is not p or workspace.backend!='gpu'):
        raise ValueError('Workspace must be a GPU workspace for this exact Problem')
    engine=workspace or D3Workspace(p,'gpu',options=options or D3Options(gpu_resident=True,quotient_features=True))
    clock=Deadline(time_limit)
    try:
        out=engine.solve_many([{}],max(0,clock.end-time.perf_counter()),depth=p.depth)[0]
        bound=out['value'] if out['status']=='OPT' else (float('inf') if out['status']=='INFEASIBLE' else 0.)
        return result_dict('JT-DP-D'+str(p.depth)+'-RESIDENT-GPU',clock,out['status'],out['tree'],p,bound,
                           stats=out['stats'],timings=out['timings'])
    finally:
        if workspace is None:engine.close()


def solve_jt_dp_shallow_cpp(p,time_limit=100,workspace=None,options=None,threads=0):
    """Exact D2/D3 OpenMP messages, including arbitrary class labels."""
    if p.depth not in (2,3):raise ValueError('Shallow C++ solver requires depth 2 or 3')
    if workspace is not None and (workspace.p is not p or workspace.backend!='cpp'):
        raise ValueError('Workspace must be a C++ workspace for this exact Problem')
    clock=Deadline(time_limit)
    engine=workspace or D3Workspace(p,'cpp',threads,options or D3Options(fast_prepare=True,quotient_features=True))
    try:
        out=engine.solve(time_limit=max(0,clock.end-time.perf_counter()),depth=p.depth)
        bound=out['value'] if out['status']=='OPT' else (float('inf') if out['status']=='INFEASIBLE' else 0.)
        return result_dict('JT-DP-D'+str(p.depth)+'-CPP',clock,out['status'],out['tree'],p,bound,
                           stats=out['stats'],timings=out['timings'])
    finally:
        if workspace is None:engine.close()


def solve_jt_dp_d3_optimized(p,time_limit=100,backend='cpp',threads=0,options=None,workspace=None):
    """Standalone D3 interface, with optional caller-owned reusable workspace."""
    if p.depth!=3:raise ValueError('Standalone optimized D3 solver requires depth 3')
    if workspace is not None and workspace.p is not p:raise ValueError('Workspace belongs to another Problem')
    clock=Deadline(time_limit);owned=workspace is None
    engine=workspace or D3Workspace(p,backend,threads,options)
    try:
        out=engine.solve(time_limit=max(0,clock.end-time.perf_counter()))
        bound=out['value'] if out['status']=='OPT' else 0.
        if out['status']=='INFEASIBLE':bound=float('inf')
        return result_dict('JT-DP-OPT-'+engine.backend.upper(),clock,out['status'],out['tree'],p,bound,
                           stats=out['stats'],timings=out['timings'],message_passing='d3_optimized_min_sum',
                           persistent_complexity='O(F^2 + NF) prepared input and message workspace')
    finally:
        if owned:engine.close()
