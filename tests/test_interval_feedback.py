from dataclasses import replace
import itertools
import numpy as np
import pytest
from jt_oct.solvers import solve_tree_dp
from jt_oct import Problem
from jt_oct.problem import evaluate
from jt_oct.interval_oracle import IntervalOracle
from jt_oct.contract_cg import ContractOptions, solve_contracted_cg


@pytest.mark.parametrize('k,penalty,minleaf',list(itertools.product([2,4],[0.,.03],[0,3])))
def test_native_intervals(k,penalty,minleaf):
    rng=np.random.default_rng(131+k)
    X=rng.integers(0,2,(33,5),dtype=np.uint8)
    p=Problem(X,rng.integers(0,k,33)*7+2,3,penalty,min_leaf=minleaf)
    expected=solve_tree_dp(p,30)['UB'];oracle=IntervalOracle(p)
    try:
        for budget,cutoff in itertools.product([0,1,8,10000],[expected-.01,expected,expected+.01]):
            lo,hi,tree,exact,stats=oracle.solve(p.all_rows,3,cutoff,budget,30)
            assert lo<=expected+1e-9 and expected<=hi+1e-9
            assert evaluate(p,tree)['objective']==pytest.approx(hi,abs=1e-9)
            if exact:assert hi==pytest.approx(expected,abs=1e-9)
            if budget==10000 and cutoff>expected:assert exact
        lo,hi,tree,exact,stats=oracle.solve(p.all_rows,3,1.,10000,0.)
        assert lo<=expected+1e-9 and expected<=hi+1e-9
        assert stats[2] or exact
    finally:oracle.close()


@pytest.mark.parametrize('depth,backend,budget,cache',list(itertools.product([4,5],['cpp','gpu'],[0,1,8],[False,True])))
def test_feedback_matches_independent_dp(depth,backend,budget,cache):
    X=np.asarray(list(itertools.product([0,1],repeat=depth)),dtype=np.uint8)
    p=Problem(X,X.sum(axis=1)%2,depth,.001,no_repeat=True)
    expected=solve_tree_dp(p,30)['UB']
    out=solve_contracted_cg(p,backend,30,ContractOptions(threads=1,warm_d3=False,
        oracle_batch=4,cache=cache,bound_feedback=True,cutoff_node_budget=budget,message_bound=True))
    assert out['status']=='OPT'
    assert out['UB']==pytest.approx(expected,abs=1e-9)
    for step in out['trace']:assert step['LB']<=expected+1e-8<=step['UB']+1e-8
    assert out['stats']['interval_calls']>0


def test_feedback_validation():
    p=Problem(np.array([[0],[1]],dtype=np.uint8),np.array([0,1]),4,.01)
    with pytest.raises(ValueError):solve_contracted_cg(p,options=ContractOptions(cutoff_node_budget=-1))
    with pytest.raises(ValueError):solve_contracted_cg(p,options=ContractOptions(bound_feedback=True,master_mode='ee'))
