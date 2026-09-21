from dataclasses import replace
import itertools
import numpy as np
import pytest
from jt_oct import Problem
from jt_oct.d3_optimized import D3Workspace,D3Options
from jt_oct.resident_dp import ResidentGPU


@pytest.mark.parametrize('depth,F',list(itertools.product([2,3],[1,7,32,37,159,315])))
def test_fused_join_matches_all_original_actions(depth,F):
    rng=np.random.default_rng(921+F)
    p=Problem(np.array([[0],[1]],dtype=np.uint8),np.array([0,1]),3,.01)
    with D3Workspace(p,'gpu',1) as ws:
        gpu=ResidentGPU(ws);gpu.prepare();cp=gpu.cp
        # Quantized costs exercise equal f/g minima and STOP ties heavily.
        values=rng.integers(0,7,(5,4 if depth==3 else 2,F,F)).astype(np.float64)/16
        values[0]=1e300;values[1]=0
        tails=rng.integers(-1,F,values.shape,dtype=np.int32)
        stops=rng.integers(0,7,(5,2,F)).astype(np.float64)/16
        stops[0]=1e300;stops[1]=0
        vd,td=cp.asarray(values),cp.asarray(tails)
        gpu.o=replace(gpu.o,gpu_fused_join=False)
        expected=gpu.join(vd,td,stops,depth)
        gpu.o=replace(gpu.o,gpu_fused_join=True)
        actual=gpu.join(vd,td,stops,depth)
        np.testing.assert_array_equal(actual,expected)
        gpu.close()


@pytest.mark.parametrize('depth,k,kernel,bucket',list(itertools.product([2,3],[2,4],['baseline','shared'],[0,8])))
def test_dispatch_preserves_exact_trees_and_deadlines(depth,k,kernel,bucket):
    rng=np.random.default_rng(411+k)
    X=rng.integers(0,2,(577,37),dtype=np.uint8)
    p=Problem(X,rng.integers(0,k,len(X)),depth+3,.002,min_leaf=2,no_repeat=True)
    requests=[]
    for path in [(0,1,0),(1,),(),(1,0),(0,0,1)]:
        rows=p.all_rows
        for f,side in enumerate(path):rows=p.route(rows,f,side)
        requests.append(dict(rows=rows,node=path,used=tuple(range(len(path)))))
    options=D3Options(gpu_resident=True,gpu_cost_strategy=kernel,gpu_compact_rows=True,
                      gpu_compact_min_batch=1,gpu_pair_tile=256,gpu_bucket_min=0,gpu_fused_join=False)
    with D3Workspace(p,'gpu',1,options) as ws:old=ws.solve_many(requests,30,depth=depth)
    with D3Workspace(p,'gpu',1,replace(options,gpu_pair_tile=1024,gpu_bucket_min=bucket,gpu_fused_join=True)) as ws:
        new=ws.solve_many(requests,30,depth=depth)
        expired=ws.solve_many(requests,0,depth=depth)
    assert sum(a['stats']['kernel_calls'] for a in new)<sum(a['stats']['kernel_calls'] for a in old)
    for ref,out,late in zip(old,new,expired):
        assert ref['status']==out['status']=='OPT'
        assert ref['tree']==out['tree']
        assert ref['value']==pytest.approx(out['value'],abs=1e-10)
        assert late['status']=='TIME'
        assert late['value']>=out['value']-1e-10


@pytest.mark.parametrize('flags',[dict(gpu_pair_tile=0),dict(gpu_bucket_min=-1)])
def test_invalid_dispatch_flags(flags):
    p=Problem(np.array([[0],[1]],dtype=np.uint8),np.array([0,1]),3,.01)
    with pytest.raises(ValueError):D3Workspace(p,'gpu',1,D3Options(**flags))
