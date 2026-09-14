"""Stream complete ancestor signatures for D2-tail JT-CG, without an F^3 array.

The RMP is a persistent Gurobi C++ model. Only exact private tail costs enter
pricing certificates; incomplete scans preserve previously certified bounds.
"""
from collections import defaultdict
from dataclasses import asdict,replace
import math
import time
from .cg import CGOptions,greedy_feasible,solve_cg
from .domain import Domain,Column,SPLIT
from .d3_grow import _frontier_paths,_prefix_state,_replace
from .problem import Deadline,DeadlineExceeded,Tree,evaluate
from .solvers import result_dict
from .terminal_message import TerminalMessages,TerminalOptions
from .sparse_tail_rmp import SparseTailMaster


def _splits(tree):
    return 0 if tree is None or tree.feature is None else 1+_splits(tree.left)+_splits(tree.right)


class AcceleratedTailDomain(Domain):
    def __init__(self,p,engine,batch_size,stats,heartbeat=None):
        super().__init__(p,tail_depth=2)
        stats.update(self.stats);self.stats=stats
        self.engine=engine;self.batch_size=batch_size;self.heartbeat=heartbeat

    def representatives(self,i,deadline=None,visit=None):
        pending=[]
        def flush():
            if not pending:return []
            requests=[(s[1],self.roots[i],s[2]) for s in pending]
            tick=time.perf_counter()
            try:answers=self.engine.terminal_many(requests)
            finally:self.stats['oracle_wall_seconds']+=time.perf_counter()-tick
            columns=[]
            for state,answer in zip(pending,answers):
                if not answer.exact:raise DeadlineExceeded('Incomplete D2 pricing; no exact certificate returned')
                if answer.tree is not None:
                    rho,rows,used,pc,ps=state
                    columns.append(Column(rho,answer.tree,pc+answer.upper,ps+_splits(answer.tree)))
            pending.clear()
            if self.heartbeat:self.heartbeat()
            return columns
        for state in self.prefix_states(i,deadline,visit):
            self.stats['stream_prefixes_visited']+=1
            rho,rows,used,pc,ps=state
            if any(a[0]!=SPLIT for a in rho):
                yield Column(rho,None,pc,ps)
                continue
            initial=self.engine.initial(rows,terminal_depth=2)
            if initial.exact:
                self.stats['stream_exact_cache_or_bound_hits']+=1
                if initial.tree is not None:yield Column(rho,initial.tree,pc+initial.upper,ps+_splits(initial.tree))
            else:
                pending.append(state)
                self.stats['stream_pending_peak']=max(self.stats['stream_pending_peak'],len(pending))
                if len(pending)>=self.batch_size:yield from flush()
        yield from flush()


def solve_streamed_tail_cg(p,backend,time_limit,o,progress=None):
    if p.depth not in (4,5) or o.tail_depth!=2:raise ValueError('Streamed tail CG requires D4/D5 with D2 tails')
    if o.oracle_batch<1 or o.threads<1 or o.max_columns<1:raise ValueError('Invalid streamed CG options')
    if o.warm_roots!=1:raise ValueError('Streamed tail experiment supports one D3 initial root')
    clock=Deadline(time_limit);stats=defaultdict(int)
    stats.update(backend=backend,options=asdict(o),terminal_depth=2,upper_depth=p.depth-2,
        terminal_clusters=2**(p.depth-2),prefix_storage='streamed',rmp_backend='gurobi_cpp_sparse',
        upper_message_bound_available=False,pricing_certificate='complete prefix scans only')
    name='JT-CG-D2Tail-STREAM-'+backend.upper()
    best=Tree(label=p.best_label(p.all_rows)) if p.n>=p.min_leaf else None
    best_value=evaluate(p,best)['objective'] if best else math.inf
    first=clock.elapsed() if best else None;incumbents=[];warm=engine=None
    current=None;last_publish=-1.
    def accept(tree,source):
        nonlocal best,best_value,first
        if tree is not None:
            v=evaluate(p,tree)['objective']
            if v<best_value-1e-12:
                best,best_value,first=tree,v,clock.elapsed()
                incumbents.append(dict(seconds=first,UB=v,source=source))
    def publish(raw=None,force=False):
        nonlocal current,last_publish
        if raw is not None:
            current=raw
            if raw.get('tree'):accept(Tree.from_dict(raw['tree']),'CG_RMP')
        if progress and (force or clock.elapsed()-last_publish>=1.):
            last_publish=clock.elapsed()
            lb=max(stats['conflict_lower_bound'],current['LB'] if current else 0.)
            progress(result_dict(name,clock,'RUNNING',best,p,lb,stats=dict(stats),
                trace=list(current.get('trace',[])) if current else [],incumbent_trace=list(incumbents)))
    opt=TerminalOptions(cache_entries=20000 if o.cache else 0,fast_prepare=True,quotient=o.quotient,
        resident_gpu=o.resident_gpu,gpu_batch_size=o.oracle_batch,class_capacity_bound=o.class_bound)
    def tune(e):
        e.workspace.options=replace(e.workspace.options,gpu_native_metadata=o.native_metadata,
            gpu_metadata_cache_entries=o.metadata_cache_entries,gpu_sync_tiles=o.gpu_sync_tiles,
            gpu_pipeline_chunk=o.gpu_pipeline_chunk,gpu_cost_strategy=o.cost_kernel)
        return e
    try:
        clock.check();accept(greedy_feasible(p,clock),'greedy');publish(force=True)
        warm=tune(TerminalMessages(p,backend,o.threads,opt,clock,stats))
        if o.warm_d3:
            tick=time.perf_counter();growing=warm.terminal(p.all_rows,(),(),kind='warm').tree
            accept(growing,'D3_seed');publish()
            if growing:
                for level in range(1,p.depth-2):
                    for path in _frontier_paths(growing,level):
                        rows,used=_prefix_state(p,growing,path)
                        answer=warm.terminal(rows,path,used,kind='warm')
                        if answer.tree:
                            candidate=_replace(growing,path,answer.tree)
                            if evaluate(p,candidate)['objective']<=evaluate(p,growing)['objective']+1e-9:
                                growing=candidate;accept(growing,'D3_grow');publish()
            stats['warm_seconds']=time.perf_counter()-tick
        engine=tune(TerminalMessages(p,backend,o.threads,opt,clock,stats,terminal_depth=2))
        if backend=='gpu':engine.workspace.share_gpu_data_from(warm.workspace)
        warm.close();warm=None
        domain=AcceleratedTailDomain(p,engine,o.oracle_batch,stats,publish)
        opts=CGOptions(incremental_matrix=True,rmp_method=0,early_return=True,
            columns_per_block=o.columns_per_block or o.oracle_batch,
            max_columns=o.max_columns,adaptive_admission=o.adaptive_admission,
            static_conflict_lb=False,conflict_tail_bound=False)
        offset=clock.elapsed()
        out=solve_cg(domain,clock.remaining(),opts,initial_tree=best,progress=publish,master_factory=SparseTailMaster)
        accept(Tree.from_dict(out['tree']) if out.get('tree') else None,'CG_RMP')
        for event in out.get('trace',[]):event['seconds']+=offset
        stats['rmp_seconds']=out.get('rmp_seconds',0.)
        stats['pricing_seconds']=out.get('pricing_seconds',0.)
        stats['iterations']=len(out.get('trace',[]))
        return result_dict(name,clock,out['status'],best,p,max(stats['conflict_lower_bound'],out['LB']),
            stats=dict(stats),trace=out.get('trace',[]),certificate=out.get('certificate'),
            incumbent_trace=incumbents,first_final_ub_seconds=first,
            first_optimal_solution_seconds=first if out['status']=='OPT' else None,
            proof_seconds=clock.elapsed() if out['status']=='OPT' else None,
            formulation='D2-tail JT-LP, streamed complete three-ancestor separators; C++ Gurobi RMP')
    except DeadlineExceeded:
        return result_dict(name,clock,'TIME',best,p,max(stats['conflict_lower_bound'],current['LB'] if current else 0.),
            stats=dict(stats),incumbent_trace=incumbents,first_final_ub_seconds=first,
            first_optimal_solution_seconds=None,proof_seconds=None)
    finally:
        if engine:engine.close()
        if warm:warm.close()
