"""Bound/cache ablations on direct JT chain messages, without subtree oracles.

Only private stump costs are cached by the exact routed bitset. Used ancestor
predicates are constant on that bitset and cannot strictly beat STOP when lambda
is nonnegative. Thus a cached improving private split remains legal on any such
prefix. Forward messages and separator compatibility are never quotient-merged.
"""
from dataclasses import asdict, dataclass
import ctypes
from pathlib import Path
import time

import numpy as np

from .cg import greedy_feasible
from .d3_batched import _pack_words64, _configure_cupy_runtime
from .direct_message import INF, _column
from .domain import Domain
from .problem import Deadline, DeadlineExceeded, Tree, evaluate
from .solvers import result_dict

ROOT=Path(__file__).resolve().parents[1]
COUNTERS=('valid_prefixes','private_states','pure_stop','lambda_stop','conflict_stop',
          'cache_probes','cache_hits','cache_busy','cache_stores','feature_candidates',
          'word_intersections','bound_pruned','nonzero_words','all_words',
          'unreachable_prefixes','invalid_prefixes')


@dataclass(frozen=True)
class MessagePlusOptions:
    bound: int = 2
    sparse_words: bool = False
    cache: bool = False
    prune: bool = False
    root_order: str = 'natural'
    lazy_recovery: bool = False
    tile_contexts: int = 4096
    threads: int = 0
    cache_slots: int = 0  # 0 chooses 1024/thread on CPU or 16384 global on GPU.


PRESETS={
    'bound':MessagePlusOptions(),
    'sparse':MessagePlusOptions(sparse_words=True),
    'cache':MessagePlusOptions(sparse_words=True,cache=True),
    'prune':MessagePlusOptions(sparse_words=True,cache=True,prune=True),
    'gain':MessagePlusOptions(sparse_words=True,cache=True,prune=True,root_order='gain'),
    'small':MessagePlusOptions(sparse_words=True,cache=True,prune=True,root_order='small'),
}


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


def root_information(p,conflict):
    roots=[]
    for f in range(p.F):
        lower,stump,support=p.penalty,p.penalty,[]
        for bit in (0,1):
            rows=p.route(p.all_rows,f,bit)
            support.append(len(rows))
            if len(rows)<p.min_leaf:
                lower=stump=INF
                break
            err=p.loss(rows,p.best_label(rows))
            unavoidable=sum(int(mass) for row,mass in conflict if (rows.mask>>int(row))&1)*p._uniform_weight
            lower+=min(err,p.penalty+unavoidable)
            stump+=err
        roots.append({'root':f,'lower':lower,'stump':stump,'small_support':min(support)})
    return roots


class PlusBackend:
    def __init__(self,p,backend,options,conflict):
        self.p,self.backend,self.o=p,backend,options
        self.xp=np
        masks=np.ascontiguousarray(np.stack([
            [_pack_words64(p.X[:,f]==b) for f in range(p.F)] for b in (0,1)]))
        positive=np.ascontiguousarray(_pack_words64(p.y==p.labels[1]))
        self.W=masks.shape[-1]
        self.nc=len(conflict)
        self.slots=(options.cache_slots or (1024 if backend=='cpp' else 16384)) if options.cache else 0
        self.handle=None
        if backend=='cpp':
            self.lib=ctypes.CDLL(str(ROOT/'jt_oct/_native/jt_message_plus.dll'))
            self.lib.jt_mp_plus_create.argtypes=[ctypes.c_int]*3
            self.lib.jt_mp_plus_create.restype=ctypes.c_void_p
            self.lib.jt_mp_plus_free.argtypes=[ctypes.c_void_p]
            self.lib.jt_mp_plus_stats.argtypes=[ctypes.c_void_p]*2
            self.lib.jt_mp_plus_costs.argtypes=[ctypes.c_void_p]*9
            self.lib.jt_mp_plus_costs.restype=ctypes.c_int
            self.handle=self.lib.jt_mp_plus_create(self.W,options.threads,self.slots)
            if not self.handle:raise MemoryError('Native message workspace allocation failed')
            self.masks,self.positive,self.conflict=masks,positive,conflict
        elif backend=='gpu':
            _configure_cupy_runtime()
            import cupy as cp
            self.xp=cp
            self.kernel=cp.RawKernel((ROOT/'native/jt_message_plus.cu').read_text(),
                                    'jt_mp_plus_gpu',options=('--std=c++11',))
            self.kernel.compile()
            self.masks,self.positive,self.conflict=map(cp.asarray,(masks,positive,conflict))
            slots=max(1,self.slots)
            self.keys=cp.empty((slots,self.W),dtype=cp.uint64)
            self.locks=cp.zeros(slots,dtype=cp.int32)
            self.valid=cp.zeros(slots,dtype=cp.int32)
            self.values=cp.empty(slots,dtype=cp.float64)
            self.actions=cp.empty(slots,dtype=cp.int32)
            self.counters=cp.zeros(16,dtype=cp.uint64)
            cp.cuda.Stream.null.synchronize()
        else:raise ValueError('Optimized messages support cpp or gpu only')

    def close(self):
        if self.handle:
            self.lib.jt_mp_plus_free(self.handle)
            self.handle=None

    def host(self,a):
        return self.xp.asnumpy(a) if self.backend=='gpu' else np.asarray(a)

    def stats(self):
        if self.backend=='gpu':
            counters=self.host(self.counters)
        else:
            counters=np.zeros(16,dtype=np.uint64)
            self.lib.jt_mp_plus_stats(self.handle,ctypes.c_void_p(counters.ctypes.data))
        return {key:int(value) for key,value in zip(COUNTERS,counters)}

    def costs(self,incoming,root,q,start,count,stride,cutoff):
        p,o=self.p,self.o
        ip=np.array((p.F,self.W,p.depth,root,q,int(p.no_repeat),p.min_leaf,start,count,stride,
                     o.bound,int(o.sparse_words),int(o.prune),self.nc),dtype=np.int32)
        dp=np.array((p._uniform_weight,p.penalty,cutoff),dtype=np.float64)
        output=self.xp.empty(count,dtype=np.float64)
        tail=self.xp.empty(count,dtype=np.int32)
        if self.backend=='cpp':
            ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
            result=self.lib.jt_mp_plus_costs(self.handle,ptr(self.masks),ptr(self.positive),
                ptr(self.conflict),ptr(incoming),ptr(ip),ptr(dp),ptr(output),ptr(tail))
            if result:raise RuntimeError(f'Native message-plus error {result}')
        else:
            self.kernel((count,),(256,),
                (self.masks,self.positive,self.conflict,incoming,*ip,*dp,output,tail,self.counters,
                 self.keys,self.locks,self.valid,self.values,self.actions,np.int32(self.slots)),
                shared_mem=self.W*12)
            self.xp.cuda.Stream.null.synchronize()
        return output,tail


def solve_message_plus(p,backend='gpu',time_limit=100,options=None,progress=None,audit=None):
    o=options or MessagePlusOptions()
    if not 2<=p.depth<=5 or len(p.labels)!=2 or p._uniform_weight is None:
        raise ValueError('Message-plus requires D2--D5, binary classes, uniform weights')
    if not p.early_stop or p.allowed or p.split_costs or p.penalty<0:
        raise ValueError('Message-plus requires STOP, shared predicates, and uniform nonnegative split costs')
    if o.bound not in (0,1,2) or o.root_order not in ('natural','gain','small') or o.tile_contexts<1 or o.cache_slots<0:
        raise ValueError('Invalid message-plus options')
    clock=Deadline(time_limit)
    domain=Domain(p)
    B,K=p.F+1,p.depth-2
    N=B**K
    stats={'backend':backend,'options':asdict(o),'root_tiles_completed':0,'roots_skipped_by_bound':0,
           'cluster_messages_completed':0,'prefix_contexts_processed':0,'prefix_contexts_per_cluster':N,
           'cost_and_private_reduction_seconds':0.,'message_reduction_seconds':0.,
           'traceback_seconds':0.,'setup_seconds':0.,'oracle_calls':0,'max_columns_applies':False}
    best_tree=Tree(label=p.best_label(p.all_rows)) if p.n>=p.min_leaf else None
    best=evaluate(p,best_tree)['objective'] if best_tree else float('inf')
    stop_value=best
    root_lbs=np.zeros(p.F)
    lb=0.
    roots=[]
    engine=None
    last_progress=0.
    def refresh():
        if engine:stats.update(engine.stats())
    def publish(force=False):
        nonlocal last_progress
        if progress and (force or clock.elapsed()-last_progress>=5):
            last_progress=clock.elapsed();refresh()
            progress(result_dict('JT-MP-PLUS-'+backend.upper(),clock,'RUNNING',best_tree,p,lb,
                                 stats=dict(stats),root_trace=list(roots)))
    try:
        clock.check()
        tick=time.perf_counter()
        conflicts=conflict_representatives(p)
        conflict_bound=float(conflicts[:,1].sum()*p._uniform_weight)
        stats['conflict_groups']=len(conflicts)
        stats['conflict_lower_bound']=conflict_bound
        root_lbs.fill(conflict_bound)
        infos=root_information(p,conflicts)
        if o.prune:root_lbs=np.array([v['lower'] for v in infos],dtype=float)
        lb=min(stop_value,float(root_lbs.min()))
        greedy=greedy_feasible(p,clock)
        if greedy:
            val=evaluate(p,greedy)['objective']
            if val<best:best_tree,best=greedy,val
        if o.root_order=='gain':infos.sort(key=lambda v:(v['stump'],v['root']))
        elif o.root_order=='small':infos.sort(key=lambda v:(v['lower'],v['small_support'],v['stump'],v['root']))
        stats['root_order']=[v['root'] for v in infos]
        engine=PlusBackend(p,backend,o,conflicts)
        stats['setup_seconds']=time.perf_counter()-tick
        stats['cache_slots']=engine.slots
        xp=engine.xp
        if backend=='gpu':stats['gpu_name']=xp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        publish(True)
        for info in infos:
            clock.check()
            root=info['root'];cutoff=best
            if o.prune and root_lbs[root]>cutoff+1e-12:
                stats['roots_skipped_by_bound']+=1
                roots.append({'root':root,'state':'root_bound','lower':float(root_lbs[root]),
                              'objective':None,'seconds':clock.elapsed(),'best_UB':best})
                stats['root_tiles_completed']+=1
                continue
            previous=xp.zeros(1,dtype=np.float64);records=[]
            for q in range(domain.M):
                free_left=domain.separators[q-1]-1 if q else 0
                free_right=domain.separators[q]-1 if q<domain.M-1 else 0
                stride,group=B**(K-free_left),B**(K-free_right)
                values=xp.empty(N,dtype=np.float64);tails=xp.empty(N,dtype=np.int32)
                for start in range(0,N,o.tile_contexts):
                    clock.check();count=min(o.tile_contexts,N-start)
                    tick=time.perf_counter()
                    vals,acts=engine.costs(previous,root,q,start,count,stride,cutoff)
                    stats['cost_and_private_reduction_seconds']+=time.perf_counter()-tick
                    values[start:start+count]=vals;tails[start:start+count]=acts
                    stats['prefix_contexts_processed']+=count
                    publish()
                tick=time.perf_counter()
                nout=B**free_right
                grouped=values.reshape(nout,group)
                winner=xp.argmin(grouped,axis=1)
                prefix=(xp.arange(nout,dtype=np.int64)*group+winner).astype(np.int32)
                previous=values[prefix].copy()
                records.append((engine.host(prefix),engine.host(tails[prefix]),stride))
                if audit:audit(root,q,free_right,engine.host(previous))
                stats['message_reduction_seconds']+=time.perf_counter()-tick
                stats['cluster_messages_completed']+=1
                del values,tails,grouped,winner,prefix
            value=float(engine.host(previous)[0])
            # With cutoff pruning, missing transitions have a proved cost > cutoff.
            # The exact root optimum is at least min(returned_value, cutoff).
            root_lbs[root]=max(root_lbs[root],min(value,cutoff) if o.prune else value)
            tick=time.perf_counter()
            if value<INF/2 and (not o.lazy_recovery or value<best-1e-12):
                state,chosen=0,[]
                for q in range(domain.M-1,-1,-1):
                    prefixes,actions,stride=records[q]
                    index,h=int(prefixes[state]),int(actions[state])
                    chosen.append(_column(p,q,root,index,h));state=index//stride
                tree=domain.recover(list(reversed(chosen)))
                actual=evaluate(p,tree)['objective']
                if abs(actual-value)>1e-8:raise AssertionError(f'Message-plus tree mismatch {actual} vs {value}')
                if actual<best-1e-12:best_tree,best=tree,actual
            stats['traceback_seconds']+=time.perf_counter()-tick
            lb=min(stop_value,float(root_lbs.min()))
            stats['root_tiles_completed']+=1
            state='exact' if not o.prune or value<cutoff-1e-12 else 'cutoff_bound'
            roots.append({'root':root,'state':state,'objective':value if value<INF/2 else None,
                          'lower':float(root_lbs[root]),'seconds':clock.elapsed(),'best_UB':best})
            del previous,records
            publish(True)
        refresh()
        if best_tree is None:return result_dict('JT-MP-PLUS-'+backend.upper(),clock,'INFEASIBLE',lb=float('inf'),stats=stats)
        if lb<best-1e-8:raise AssertionError('Incomplete lower-bound coverage at claimed optimum')
        return result_dict('JT-MP-PLUS-'+backend.upper(),clock,'OPT',best_tree,p,best,stats=stats,
                           root_trace=roots,message_passing='direct_path_cluster_chain',optimality_scope='tree_problem')
    except DeadlineExceeded:
        refresh()
        return result_dict('JT-MP-PLUS-'+backend.upper(),clock,'TIME',best_tree,p,lb,stats=stats,
                           root_trace=roots,message_passing='direct_path_cluster_chain',optimality_scope='valid_incumbent_and_root_LBs')
    finally:
        if engine:engine.close()
