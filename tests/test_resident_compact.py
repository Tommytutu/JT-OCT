from dataclasses import replace
import itertools
import numpy as np
import pytest
from jt_oct import Problem
from jt_oct.d3_optimized import D3Options,D3Workspace


@pytest.mark.parametrize('k,depth,strategy',list(itertools.product([2,4],[2,3],['baseline','fused','shared'])))
def test_dense_local_words_equal_original(k,depth,strategy):
    rng=np.random.default_rng(271+k)
    X=rng.integers(0,2,(523,9),dtype=np.uint8)
    p=Problem(X,rng.integers(0,k,len(X))*7+2,depth+2,.001,no_repeat=True,min_leaf=2)
    reqs=[dict(rows=p.route(p.route(p.all_rows,0,a),1,b),node=(a,b),used=(0,1)) for a,b in itertools.product([0,1],repeat=2)]
    opts=D3Options(gpu_resident=True,gpu_cost_strategy=strategy,quotient_features=True,gpu_compact_rows=False)
    with D3Workspace(p,'gpu',1,opts) as ws:expected=ws.solve_many(reqs,30,depth=depth)
    with D3Workspace(p,'gpu',1,replace(opts,gpu_compact_rows=True,gpu_compact_min_batch=1)) as ws:actual=ws.solve_many(reqs,30,depth=depth)
    for old,new in zip(expected,actual):
        assert old['status']==new['status']=='OPT'
        assert new['value']==pytest.approx(old['value'],abs=1e-10)
        assert new['tree']==old['tree']
        assert new['stats']['compact_rows_used']
        assert new['stats']['local_words']<new['stats']['global_words']


def test_compact_respects_node_specific_constraints():
    rng=np.random.default_rng(188)
    X=rng.integers(0,2,(257,6),dtype=np.uint8)
    p=Problem(X,rng.integers(0,2,257),4,.01,no_repeat=True,min_leaf=3,
        allowed={(0,):[1,2],(0,0):[2,3]},split_costs={((0,),1):.03})
    reqs=[dict(rows=p.route(p.all_rows,0,0),node=(0,),used=(0,))]
    results=[]
    for compact in [False,True]:
        with D3Workspace(p,'gpu',1,D3Options(gpu_resident=True,gpu_compact_rows=compact,gpu_compact_min_batch=1)) as ws:
            results.append(ws.solve_many(reqs,30)[0])
    assert results[0]['tree']==results[1]['tree']
    assert results[0]['value']==pytest.approx(results[1]['value'])


def test_buckets_preserve_mixed_request_order_and_timeout():
    rng=np.random.default_rng(219)
    X=rng.integers(0,2,(1027,7),dtype=np.uint8)
    p=Problem(X,rng.integers(0,3,len(X)),6,.001,no_repeat=True)
    paths=[(0,1,0),(1,),(),(1,0),(0,0,1)]
    requests=[]
    for path in paths:
        rows=p.all_rows
        for f,side in enumerate(path):rows=p.route(rows,f,side)
        requests.append(dict(rows=rows,node=path,used=tuple(range(len(path)))))
    with D3Workspace(p,'cpp',1,D3Options()) as ws:expected=ws.solve_many(requests,30)
    with D3Workspace(p,'gpu',1,D3Options(gpu_resident=True,gpu_compact_rows=True,gpu_compact_min_batch=1,gpu_cost_strategy='shared')) as ws:
        actual=ws.solve_many(requests,30)
        expired=ws.solve_many(requests,0)
    for ref,out,late in zip(expected,actual,expired):
        assert out['status']=='OPT'
        assert out['value']==pytest.approx(ref['value'],abs=1e-10)
        assert late['status']=='TIME'
        assert late['value']>=ref['value']-1e-10


def test_compact_profitability_guards_small_batches():
    rng=np.random.default_rng(913)
    X=rng.integers(0,2,(1025,7),dtype=np.uint8)
    p=Problem(X,rng.integers(0,2,len(X)),5,.001,no_repeat=True)
    requests=[]
    for path in [(0,),(1,),(0,1),(1,0)]:
        rows=p.all_rows
        for f,side in enumerate(path):rows=p.route(rows,f,side)
        requests.append(dict(rows=rows,node=path,used=tuple(range(len(path)))))
    options=D3Options(gpu_resident=True,gpu_compact_rows=True,
                      gpu_compact_min_batch=8,gpu_compact_max_ratio=.75)
    with D3Workspace(p,'gpu',1,options) as ws:out=ws.solve_many(requests,30)
    assert all(not r['stats']['compact_rows_used'] for r in out)
    assert all(not r['stats']['compact_batch_eligible'] for r in out)


def test_compact_profitability_guards_weak_compression():
    rng=np.random.default_rng(914)
    X=rng.integers(0,2,(1025,7),dtype=np.uint8)
    p=Problem(X,rng.integers(0,2,len(X)),5,.001,no_repeat=True)
    requests=[dict(rows=p.all_rows,node=(),used=()) for _ in range(8)]
    options=D3Options(gpu_resident=True,gpu_compact_rows=True,
                      gpu_compact_min_batch=8,gpu_compact_max_ratio=.75)
    with D3Workspace(p,'gpu',1,options) as ws:out=ws.solve_many(requests,30)
    assert all(r['stats']['compact_batch_eligible'] for r in out)
    assert all(r['stats']['compact_word_ratio']==pytest.approx(1.) for r in out)
    assert all(not r['stats']['compact_rows_used'] for r in out)
