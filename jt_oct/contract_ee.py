"""Experiment C: interval-safe endpoint elimination on the h=3 master.

D4 eliminates one of two clusters (single-block minimization, not CG).
D5 eliminates original endpoints 0 and 3; CG coordinates blocks (0+1),(2+3).
Lower-cost master columns are optimistic potentials, never feasible incumbents.
Only upper-cost backpointers reconstruct feasible trees.
"""
import math
import time
import numpy as np

from .problem import DeadlineExceeded, evaluate
from .domain import CapacityExceeded
from .solvers import result_dict
from .contract_ee_native import EERMP, sum_pairs, best_join, best_single, admit


def reduced_costs(state, values):
    c=np.asarray(values).reshape(state.M,state.P)
    if state.M==2:
        # A one-block D4 reduction is also performed by the native kernel;
        # duplicate the two inputs only to use its contiguous four-block ABI.
        return sum_pairs(np.vstack((c,c)),state.P)[:1]
    return sum_pairs(c,state.P)


def lift_ids(state, selected):
    if state.M==2:
        s=int(selected[0]);return [s,state.P+s]
    left,right=map(int,selected)
    if state.roots[left]!=state.roots[right]:raise AssertionError('Incompatible reduced root')
    return [left,state.P+left,2*state.P+right,3*state.P+right]


def upper_selection(state, values):
    """Small upper-cost join; each chosen private subtree has a feasible tree."""
    return best_single(values[0]) if state.M==2 else best_join(values,state.roots,state.A)


def lower_cg(state,costs,clock,options,stats):
    """Two-block continuous RMP and complete lower-cost pricing.

    Full-domain coefficient arrays are O(P); no private costs are precomputed.
    A fresh RMP is built after interval refinement, so stale coefficients/duals
    can never be used. Rebuild overhead is included in the reported time.
    """
    roots=state.roots
    finite=[np.flatnonzero(np.isfinite(costs[q])) for q in range(2)]
    common=set(roots[finite[0]]) & set(roots[finite[1]])
    if not common:return math.inf,None
    active=np.zeros((2,costs.shape[1]),dtype=np.uint8)
    root=min(common)
    pending=[(q,int(next(s for s in finite[q] if roots[s]==root))) for q in range(2)]
    P=costs.shape[1]
    started=time.perf_counter(); model=EERMP(P,state.A,roots,method=1)
    try:
        stats['ee_rmp_setup_seconds']+=time.perf_counter()-started
        stats['ee_rmp_rebuilds']+=1
        while True:
            clock.check()
            if int(active.sum())+len(pending)>options.max_columns:
                raise CapacityExceeded('EE active-column capacity')
            tick=time.perf_counter()
            ids=np.asarray([q*P+s for q,s in pending],dtype=np.int32)
            vals=np.asarray([costs[q,s] for q,s in pending],dtype=np.float64)
            model.update(ids,vals)
            active.flat[ids]=1
            stats['ee_rmp_update_seconds']+=time.perf_counter()-tick
            tick=time.perf_counter(); solution=model.solve(clock.remaining())
            stats['rmp_seconds']+=time.perf_counter()-tick;stats['iterations']+=1
            if solution['status']==9:raise DeadlineExceeded('EE RMP timeout')
            if solution['status']!=2:raise RuntimeError(f"EE RMP status {solution['status']}")
            stats['peak_active_columns']=max(stats['peak_active_columns'],int(solution['columns']))
            stats['peak_rmp_nonzeros']=max(stats['peak_rmp_nonzeros'],int(solution['nonzeros']))
            stats['final_columns']=int(solution['columns']);stats['final_rows']=int(solution['rows'])
            tick=time.perf_counter()
            alpha=float(solution['alpha']);pi=np.asarray(solution['pi'])
            rc,minimum=model.price(costs,alpha,pi)
            bound=alpha+sum(min(0.,float(minimum[q])) for q in range(2))
            restricted=np.where(active,costs,np.inf)
            value,chosen=upper_selection(state,restricted)
            if abs(value-solution['objective'])>1e-7:raise AssertionError('Reduced RMP gluing failed')
            if bound>value+1e-7:raise AssertionError('Reduced pricing bound exceeds primal')
            admitted=admit(rc,active,options.columns_per_block)
            pending=[divmod(int(id),P) for id in admitted]
            stats['pricing_seconds']+=time.perf_counter()-tick
            stats['ee_minimum_full_rc']=float(np.min(minimum))
            if value-bound<=1e-8:return min(bound,value),chosen
            if not pending:raise RuntimeError('Reduced pricing stalled without a certificate')
    finally:
        model.close()


def solve_interval_ee(state,clock,options,stats,tree,initial_lb,progress=None):
    if state.M>4:
        from .deep_contract_ee import solve_deep_interval_ee
        return solve_deep_interval_ee(state,clock,options,stats,tree,initial_lb,progress)
    p=state.p;lb=float(initial_lb);ub=evaluate(p,tree)['objective'];trace=[]
    status='TIME';proof=None
    for key in ('ee_refined_states','ee_requested_states','ee_rmp_rebuilds','rmp_seconds',
                'pricing_seconds','peak_active_columns','peak_rmp_nonzeros','iterations'):
        stats.setdefault(key,0)
    stats['ee_original_clusters']=state.M
    stats['ee_remaining_clusters']=state.M//2
    stats['ee_coordinator']='single_block_minimum' if state.M==2 else 'two_block_lower_cost_cg'
    stats['ee_original_candidate_slots']=state.N
    stats['ee_reduced_candidate_slots']=state.N//2
    stats['ee_initial_exact_states']=sum(g['record'].exact for g in state.groups)
    try:
        while True:
            clock.check();tick=time.perf_counter()
            low=reduced_costs(state,state.low);high=reduced_costs(state,state.high)
            stats['ee_preprocessing_seconds']+=time.perf_counter()-tick
            value,choice=upper_selection(state,high)
            if choice is not None and math.isfinite(value):
                candidate=state.domain.recover([state.column(i) for i in lift_ids(state,choice)])
                audited=evaluate(p,candidate)['objective']
                if abs(audited-value)>1e-7:raise AssertionError('EE upper-cost lift mismatch')
                if audited<ub:tree,ub=candidate,audited
            if state.M==2:
                bound,selected=best_single(low[0])
                stats['final_columns']=0;stats['final_rows']=0
            else:
                bound,selected=lower_cg(state,low,clock,options,stats)
            if bound>ub+1e-7:raise AssertionError('EE lower bound exceeds incumbent')
            lb=max(lb,min(bound,ub))
            trace.append(dict(seconds=clock.elapsed(),LB=lb,UB=ub,
                exact_terminal_states=sum(g['record'].exact for g in state.groups)))
            if progress:progress(result_dict('JT-EE',clock,'RUNNING',tree,p,lb,stats=dict(stats),trace=trace))
            if ub-lb<=1e-7:status='OPT';proof=clock.elapsed();break
            gids=[]
            for i in lift_ids(state,selected):
                gid=int(state.private[i])
                if gid>=0 and not state.groups[gid]['record'].exact and gid not in gids:gids.append(gid)
            if not gids:raise RuntimeError('EE interval gap without refinable selected state')
            # Prioritize the lower assignment, then fill the same batch using
            # the existing engine's compatible-state policy for CPU/GPU throughput.
            for gid in state.compatible_groups(options.oracle_batch,ub,
                                                state.messages(state.low),state.messages(state.high)):
                if len(gids)>=options.oracle_batch:break
                if gid not in gids:gids.append(gid)
            gids=gids[:options.oracle_batch]
            before=sum(g['record'].exact for g in state.groups)
            tick=time.perf_counter()
            answers=state.engine.terminal_many([(state.groups[g]['rows'],state.groups[g]['node'],
                                                 state.groups[g]['used']) for g in gids])
            stats['oracle_wall_seconds']+=time.perf_counter()-tick
            stats['ee_requested_states']+=len(gids)
            if len(answers)!=len(gids):raise AssertionError('Incomplete oracle batch response')
            for gid,answer in zip(gids,answers):state.update_group(gid,answer)
            stats['ee_refined_states']+=sum(g['record'].exact for g in state.groups)-before
            if not all(a.exact for a in answers):raise DeadlineExceeded('Partial private subtree result')
    except DeadlineExceeded:status='TIME'
    except CapacityExceeded as exc:status='RESOURCE';stats['resource_reason']=str(exc)
    stats['exact_terminal_states']=sum(g['record'].exact for g in state.groups)
    stats['unresolved_terminal_states']=len(state.groups)-stats['exact_terminal_states']
    return result_dict('JT-EE-D4' if state.M==2 else 'JT-CG-EE-D5',clock,status,tree,p,lb,
        stats=dict(stats),trace=trace,proof_seconds=proof,
        certificate=dict(kind='interval_endpoint_elimination',lower_cost_master=True,
                         complete_pricing_domain=True,upper_tree_independently_audited=True),
        formulation='h=3 interval EE; D4 single block, D5 two-block lower-cost CG')
