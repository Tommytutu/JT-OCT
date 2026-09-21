from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from jt_oct.solvers import solve_tree_dp
from jt_oct import Problem
from jt_oct.contract_cg import ContractOptions, solve_contracted_cg
from jt_oct.d3_optimized import D3Options, D3Workspace


@pytest.mark.parametrize('mode,classes,penalty',list(product(['fused','shared'],[1,2,3,12,26],[0.,.01])))
def test_every_fused_message_and_argmin_matches_independent_kernel(mode,classes,penalty):
    rng=np.random.default_rng(677+classes)
    X=rng.integers(0,2,(159,7),dtype=np.uint8)
    p=Problem(X,np.arange(159)%classes,4,penalty,no_repeat=True,min_leaf=2)
    request=dict(rows=p.route(p.all_rows,0,1),used=(0,),node=(1,))
    o=D3Options(gpu_resident=True,gpu_cost_strategy=mode)
    tables=[]
    def audit(features,values,tail):tables.append((features,values,tail))
    with D3Workspace(p,'gpu',options=replace(o,gpu_cost_strategy='baseline')) as baseline:
        old=baseline.solve_many([request],30,audit=audit)[0]
    with D3Workspace(p,'gpu',options=o) as new:
        out=new.solve_many([request],30,audit=audit)[0]
    assert out['status']==old['status']=='OPT'
    assert out['value']==pytest.approx(old['value'],abs=1e-10)
    if tables:
        assert len(tables)==2
        np.testing.assert_array_equal(tables[0][0],tables[1][0])
        np.testing.assert_allclose(tables[0][1],tables[1][1],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(tables[0][2],tables[1][2])


@pytest.mark.parametrize('depth,mode,penalty',list(product([4,5],['fused','shared'],[0.,.01])))
def test_mixed_bundle_cg_preserves_optimum_and_all_reported_bounds(depth,mode,penalty):
    X=np.asarray(list(product([0,1],repeat=5)),dtype=np.uint8)
    p=Problem(X,(X[:,0]+X[:,1]+X[:,2]*X[:,3])%3,depth,penalty,no_repeat=True)
    opt=solve_tree_dp(p,30)['UB']
    out=solve_contracted_cg(p,'gpu',30,ContractOptions(cost_kernel=mode,
        oracle_batch=8,bundle_pricing=True,message_bound=True,bundle_fraction=.25,
        native_metadata=True,metadata_cache_entries=16,gpu_sync_tiles=8))
    assert out['status']=='OPT'
    assert out['UB']==pytest.approx(opt,abs=1e-9)
    for r in out['trace']:assert r['LB']<=opt+1e-8<=r['UB']+1e-8


def test_shared_kernel_large_words_falls_back_and_more_than_256_features():
    rng=np.random.default_rng(920)
    # Separate shape tests avoid cubic work on the cross-product of both limits.
    for n,f in [(786363,4),(67,315)]:
        X=rng.integers(0,2,(n,f),dtype=np.uint8)
        p=Problem(X,np.arange(n)%3,3,.01,no_repeat=True)
        o=D3Options(gpu_resident=True,gpu_cost_strategy='shared')
        with D3Workspace(p,'gpu',options=o) as new:
            out=new.solve_many([{}],30)[0]
        with D3Workspace(p,'gpu',options=replace(o,gpu_cost_strategy='baseline')) as old:
            ref=old.solve_many([{}],30)[0]
        assert out['status']==ref['status']=='OPT'
        assert out['value']==pytest.approx(ref['value'],abs=1e-10)
        if n>200000:assert out['stats']['pair_cache_fallback']


def test_gpu_input_sharing_respects_kernel_choice_and_deadline():
    rng=np.random.default_rng(94)
    p=Problem(rng.integers(0,2,(90,6)),rng.integers(0,3,90),3,.01)
    with D3Workspace(p,'gpu',options=D3Options(gpu_resident=True)) as old:
        ref=old.solve_many([{}],30)[0]
        with D3Workspace(p,'gpu',options=D3Options(gpu_resident=True,gpu_cost_strategy='shared')) as new:
            new.share_gpu_data_from(old)
            out=new.solve_many([{}],30)[0]
            assert out['value']==pytest.approx(ref['value'],abs=1e-10)
            expired=new.solve_many([{}],0)[0]
            assert expired['status']=='TIME'
            assert expired['value']>=out['value']-1e-10
