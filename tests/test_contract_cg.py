import itertools
from dataclasses import replace
from collections import defaultdict

import numpy as np
import pytest

from jt_oct.solvers import solve_tree_dp
from jt_oct.domain import Domain
from jt_oct import Problem
from jt_oct.contract_cg import ContractOptions, PricingState, solve_contracted_cg
from jt_oct.problem import Deadline
from jt_oct.terminal_message import TerminalMessages, TerminalOptions


@pytest.mark.parametrize('depth,k,cache,warm', list(itertools.product([4,5],[2,4],[False,True],[False,True])))
def test_contracted_lp_exact(depth,k,cache,warm):
    rng=np.random.default_rng(841+k)
    X=rng.integers(0,2,(32,4),dtype=np.uint8); y=np.arange(32)%k*7+2
    p=Problem(X,y,depth,.01,no_repeat=True)
    expected=solve_tree_dp(p,30)
    out=solve_contracted_cg(p,'cpp',30,ContractOptions(cache=cache,warm_d3=warm,threads=1,oracle_batch=4))
    assert out['status']==expected['status']=='OPT'
    assert out['UB']==pytest.approx(expected['UB'],abs=1e-9)
    assert out['LB']==pytest.approx(expected['LB'],abs=1e-9)
    for event in out['trace']:
        assert event['LB']<=expected['UB']+1e-8<=event['UB']+1e-8


@pytest.mark.parametrize('depth,backend,lookahead', [(4,'cpp',False),(5,'cpp',True),(4,'gpu',False),(5,'gpu',True)])
def test_parity_needs_full_depth(depth,backend,lookahead):
    X=np.asarray(list(itertools.product([0,1],repeat=depth)),dtype=np.uint8)
    p=Problem(X,X.sum(axis=1)%2,depth,.001,no_repeat=True)
    out=solve_contracted_cg(p,backend,30,ContractOptions(threads=1,oracle_batch=4,lookahead=lookahead))
    assert out['status']=='OPT'
    assert out['UB']==pytest.approx((2**depth-1)*.001,abs=1e-9)
    assert out['LB']==pytest.approx(out['UB'],abs=1e-9)
    assert out['metrics']['realized_depth']==depth
    assert out['stats']['iterations']>=2
    assert out['first_optimal_solution_seconds']<=out['proof_seconds']<=out['seconds']


@pytest.mark.parametrize('depth', [4,5])
def test_initial_intervals_and_dual_formula(depth):
    rng=np.random.default_rng(77)
    X=rng.integers(0,2,(19,3),dtype=np.uint8)
    p=Problem(X,rng.integers(0,3,19),depth,.01,no_repeat=False,min_leaf=1)
    clock=Deadline(30);stats=defaultdict(int)
    engine=TerminalMessages(p,'cpp',1,TerminalOptions(fast_prepare=True),clock,stats)
    try:
        state=PricingState(p,engine,ContractOptions(warm_d3=False,threads=1),clock)
        domain=Domain(p,tail_depth=3)
        alpha=rng.normal(size=state.M);pi=rng.normal(size=state.R)
        reduced=state.reduced(state.low,alpha,pi)
        for q in range(state.M):
            for c in domain.representatives(q):
                s=state.signature(c.prefix);id=q*state.P+s
                assert state.low[id]<=c.cost+1e-9<=state.high[id]+1e-9
                value=state.low[id]-alpha[q]
                for e,size,sign in domain.incident[q]:
                    if state.M==2:index=s
                    else:index=s if e==0 else state.P+int(state.roots[s]) if e==1 else state.P+state.A+s
                    value-=sign*pi[index]
                assert reduced[id]==pytest.approx(value,abs=1e-9)
        opt=solve_tree_dp(p,30)['UB']
        lower=sum(alpha)+sum(np.minimum(0,reduced.reshape(state.M,state.P).min(axis=1)))
        assert lower<=opt+1e-9
    finally:engine.close()


def test_timeout_resource_and_scope():
    X=np.asarray(list(itertools.product([0,1],repeat=4)),dtype=np.uint8)
    p=Problem(X,X.sum(axis=1)%2,4,.01,no_repeat=True)
    out=solve_contracted_cg(p,'cpp',0)
    assert out['status']=='TIME' and out['proof_seconds'] is None
    assert out['LB']<=solve_tree_dp(p)['UB']<=out['UB']
    out=solve_contracted_cg(p,'cpp',30,ContractOptions(warm_d3=False,threads=1,max_columns=1))
    assert out['status']=='RESOURCE' and out['proof_seconds'] is None
    with pytest.raises(ValueError):
        solve_contracted_cg(Problem(X,p.y,4,weights=np.arange(1,17)))


@pytest.mark.parametrize('depth,backend,beam',list(itertools.product([4,5],['cpp','gpu'],[1,4])))
def test_exact_column_admission_and_beam(depth,backend,beam):
    X=np.asarray(list(itertools.product([0,1],repeat=5)),dtype=np.uint8)
    y=(X[:,0]+X[:,1]+2*X[:,2]+X[:,3]*X[:,4])%3
    p=Problem(X,y,depth,.007,no_repeat=True)
    expected=solve_tree_dp(p,30)
    out=solve_contracted_cg(p,backend,30,ContractOptions(threads=1,
        exact_columns_only=True,warm_roots=beam,oracle_batch=4))
    assert out['status']==expected['status']=='OPT'
    assert out['UB']==pytest.approx(expected['UB'],abs=1e-9)
    assert out['LB']==pytest.approx(expected['LB'],abs=1e-9)
    for event in out['trace']:
        assert event['LB']<=expected['UB']+1e-8<=event['UB']+1e-8
    assert out['stats']['warm_roots_completed']<=beam
