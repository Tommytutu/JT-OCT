"""Sequential oracle microbenchmark on representative routed Table 3 domains."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from jt_oct import Problem
from jt_oct.d3_optimized import D3Options,D3Workspace

a=argparse.ArgumentParser();a.add_argument('--dataset',required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
d=np.load(Path(__file__).parent/'datasets'/(args.dataset+'.npz'));p=Problem(d['X'],d['y'],5,0)
requests=[]
for f,g in ((0,1),(7,13),(20,31),(45,63),(80,99),(110,120)):
    for x in (0,1):
        for y in (0,1):
            rows=p.route(p.route(p.all_rows,f,x),g,y)
            requests.append(dict(rows=rows,node=(x,y),used=(f,g)))
results=[]
for threads in (1,4,8):
    with D3Workspace(p,'gpu',threads,D3Options(native_accuracy=True)) as w:
        t=time.perf_counter();out=w.solve_many(requests,120,3);wall=time.perf_counter()-t
        row=dict(threads=threads,wall=wall,objectives=[r['value'] for r in out],statuses=[r['status'] for r in out],
                 timings={k:sum(r['timings'].get(k,0.) for r in out) for k in out[0]['timings']})
        results.append(row);print(json.dumps(row),flush=True)
assert all(r['statuses']==['OPT']*len(requests) for r in results)
assert all(np.allclose(r['objectives'],results[0]['objectives'],rtol=0,atol=1e-10) for r in results)
args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
