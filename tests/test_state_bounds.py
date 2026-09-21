from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from jt_oct.solvers import solve_tree_dp
from jt_oct import Problem
from jt_oct.contract_cg import ContractOptions, PricingState, solve_contracted_cg


@pytest.mark.parametrize('blocks',[2,4])
def test_min_marginals_match_exhaustive_compatible_configurations(blocks):
    s=PricingState.__new__(PricingState)
    s.M=blocks;s.F=3;s.K=2;s.A=5;s.P=5 if blocks==2 else 17
    s.roots=np.arange(s.P) if blocks==2 else np.r_[np.repeat(np.arange(3),5),[3,4]]
    rng=np.random.default_rng(12)
    c=rng.integers(0,20,(blocks,s.P)).astype(float)
    c[0,0]=np.inf;c[-1,-1]=np.inf
    expected=np.full_like(c,np.inf)
    if blocks==2:
        configurations=[(r,r) for r in range(s.P)]
    else:
        configurations=[(a,a,b,b) for a in range(s.P) for b in range(s.P) if s.roots[a]==s.roots[b]]
    for config in configurations:
        value=sum(c[q,a] for q,a in enumerate(config))
        for q,a in enumerate(config):expected[q,a]=min(expected[q,a],value)
    np.testing.assert_array_equal(s.min_marginals(c.reshape(-1)),expected.reshape(-1))


@pytest.mark.parametrize('depth,penalty,no_repeat,variant',list(product([4,5],[0.,.01],[False,True],['screen','similarity','both','cutoff'])))
def test_state_bounds_match_independent_dp(depth,penalty,no_repeat,variant):
    rng=np.random.default_rng(715)
    X=rng.integers(0,2,(51,5),dtype=np.uint8)
    p=Problem(X,rng.integers(0,2,len(X)),depth,penalty,no_repeat=no_repeat)
    expected=solve_tree_dp(p,30)['UB']
    o=ContractOptions(threads=1,warm_d3=False,oracle_batch=4,message_bound=True,
        state_screen=variant!='similarity',similarity_refs=8 if variant in ('both','similarity') else 0,
        bound_feedback=variant=='cutoff',cutoff_node_budget=1)
    out=solve_contracted_cg(p,'gpu',30,o)
    assert out['status']=='OPT'
    assert out['UB']==pytest.approx(expected,abs=1e-9)
    for t in out['trace']:
        assert t['LB']<=expected+1e-8
        assert expected<=t['UB']+1e-8


def test_similarity_rejects_changed_feasible_domain():
    p=Problem(np.array([[0],[1]],dtype=np.uint8),np.array([0,1]),4,.01,min_leaf=1)
    with pytest.raises(ValueError,match='min_leaf'):solve_contracted_cg(p,options=ContractOptions(similarity_refs=8))
    with pytest.raises(ValueError,match='certificates'):solve_contracted_cg(p,options=ContractOptions(similarity_refs=8,suppress_jt_certificate=True))


def test_default_screen_respects_certificate_ablation():
    p=Problem(np.array([[0],[1]],dtype=np.uint8),np.array([0,1]),4,.01)
    out=solve_contracted_cg(p,'cpp',30,ContractOptions(warm_d3=False,suppress_jt_certificate=True))
    assert out['status']=='OPT'
    assert not out['stats']['state_screen_active']


def test_every_transferred_interval_against_exact_cpu_oracle(monkeypatch):
    from jt_oct.d3_optimized import D3Workspace
    rng=np.random.default_rng(816)
    X=rng.integers(0,2,(80,5),dtype=np.uint8)
    p=Problem(X,rng.integers(0,3,len(X)),5,.01)
    original=PricingState.update_group;checked=[]
    with D3Workspace(p,'cpp',1) as oracle:
        def audited(state,gid,answer):
            group=state.groups[gid]
            ref=oracle.solve(rows=group['rows'],node=group['node'],used=group['used'])['value']
            assert answer.lower<=ref+1e-9
            assert ref<=answer.upper+1e-9
            checked.append(gid)
            return original(state,gid,answer)
        monkeypatch.setattr(PricingState,'update_group',audited)
        out=solve_contracted_cg(p,'gpu',30,ContractOptions(warm_d3=False,threads=1,
            oracle_batch=2,similarity_refs=32,state_screen=True))
    assert out['status']=='OPT' and checked
    assert out['stats']['similarity_comparisons']>0
