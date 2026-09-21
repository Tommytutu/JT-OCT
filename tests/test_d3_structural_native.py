from itertools import product
import numpy as np
import pytest

from jt_oct.solvers import solve_tree_dp
from jt_oct import Problem
from jt_oct.d3_structural_native import solve_d3_structural_native


@pytest.mark.parametrize('backend',['cpp','gpu'])
@pytest.mark.parametrize('early,penalty',[(False,0.),(True,0.),(True,.01)])
def test_native_structural_variants_match_tree_dp(backend,early,penalty):
    X=np.array(list(product((0,1),repeat=4)),dtype=np.uint8)
    y=(X.sum(axis=1)+X[:,0])%3
    p=Problem(X,y,3,penalty,early_stop=early,no_repeat=True,min_leaf=1 if not early else 0)
    expected=solve_tree_dp(p,30)['UB'];coordinator='cg' if early else 'lp'
    outputs=[solve_d3_structural_native(p,r,coordinator,backend,4,30,200000,200000,16)
             for r in ('base','sc','sc_ee')]
    assert all(o['status']=='OPT' for o in outputs)
    assert all(o['UB']==pytest.approx(expected,abs=1e-8) for o in outputs)
    assert all(o['LB']==pytest.approx(expected,abs=1e-7) for o in outputs)
    assert outputs[0]['master_domain_columns']>=outputs[1]['master_domain_columns']
    assert outputs[1]['master_domain_columns']==2*outputs[2]['master_domain_columns']
    assert outputs[2]['remaining_clusters']==2


def test_complete_tree_filtered_size_and_rows():
    X=np.array(list(product((0,1),repeat=4)),dtype=np.uint8)
    p=Problem(X,X.sum(axis=1)%2,3,0.,early_stop=False,no_repeat=True,min_leaf=1)
    base=solve_d3_structural_native(p,'base','lp','cpp',2,30,200000,200000,16)
    sc=solve_d3_structural_native(p,'sc','lp','cpp',2,30,200000,200000,16)
    ee=solve_d3_structural_native(p,'sc_ee','lp','cpp',2,30,200000,200000,16)
    assert base['master_domain_columns']==4*4*3*2
    assert sc['master_domain_columns']==4*4*3
    assert ee['master_domain_columns']==2*4*3
    assert ee['submitted_rows']==1+p.F+len(p.labels)


def test_native_capacity_and_timeout_are_not_opt():
    X=np.array(list(product((0,1),repeat=4)),dtype=np.uint8)
    p=Problem(X,X.sum(axis=1)%2,3,0.,early_stop=False,no_repeat=True,min_leaf=1)
    assert solve_d3_structural_native(p,'base','lp','cpp',2,30,10,10,4)['status']=='RESOURCE'
    assert solve_d3_structural_native(p,'base','lp','cpp',2,0,200000,200000,4)['status']=='TIME'


def test_weighted_node_costs_and_allowed_features():
    X=np.array(list(product((0,1),repeat=4)),dtype=np.uint8);y=(2*X[:,0]+X[:,1])%3
    weights=np.arange(1,len(X)+1,dtype=float);weights/=weights.sum()
    allowed={(0,):(0,2,3),(1,):(1,2,3),(0,0):(1,2,3),(0,1):(0,2,3),
             (1,0):(0,1,3),(1,1):(0,1,2)}
    costs={(0,2):.003,(1,1):.002,(0,0,3):.004}
    p=Problem(X,y,3,.01,weights=weights,no_repeat=True,allowed=allowed,split_costs=costs)
    expected=solve_tree_dp(p,30)['UB']
    for reduction in ('base','sc','sc_ee'):
        out=solve_d3_structural_native(p,reduction,'cg','cpp',4,30,200000,200000,16)
        assert out['status']=='OPT' and out['UB']==pytest.approx(expected,abs=1e-8)
