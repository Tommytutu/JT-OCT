"""Run JT-LP, JT-CG, or JT-MP on a supplied benchmark dataset."""
import argparse
import json
from pathlib import Path
import time


def main():
    root = Path(__file__).resolve().parent
    cg_configs = json.loads((root/'config/cg.json').read_text())
    other_configs = json.loads((root/'config/lp_mp.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('JT-LP','JT-CG','JT-MP'), default='JT-CG')
    parser.add_argument('--dataset', required=True, choices=sorted({c['dataset'] for c in cg_configs}))
    parser.add_argument('--depth', required=True, type=int, choices=(2,3,4,5))
    parser.add_argument('--penalty', type=float, choices=(0.0,0.01), default=0.0)
    parser.add_argument('--seconds', type=float, default=600.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    if args.output.exists():
        parser.error('output already exists; choose a new filename')
    configs = cg_configs if args.method=='JT-CG' else [c for c in other_configs if c['method']==args.method]
    job = next(c for c in configs if (c['dataset'],c['depth'],c['penalty']) ==
               (args.dataset,args.depth,args.penalty))
    import numpy as np
    with np.load(root/'datasets'/(args.dataset+'.npz'), allow_pickle=False) as data:
        x, y = data['X'], data['y'].astype(np.int64)

    if args.method=='JT-LP':
        from jt_oct.revision_native import NativeData
        opts=job['options']
        started=time.perf_counter()
        with NativeData(x,y,args.depth,early=opts.get('early',True),no_repeat=True,
                        min_leaf=opts.get('min_leaf',0),threads=opts['threads']) as native:
            result=native.eager('lp_sc_ee',penalty=args.penalty,backend=job['backend'],
                seconds=max(0.0,args.seconds-(time.perf_counter()-started)),
                memory=opts['memory_bytes'],cap=opts['cap'],presolve=opts['presolve'],
                lp_method=opts['lp_method'],batch=opts['batch'],
                private_depth=opts.get('private_depth'),lazy=opts.get('lazy',False))
        result.setdefault('method','JT-LP-SC-EE')
    else:
        from jt_oct import Problem
        problem=Problem(x,y,args.depth,args.penalty,no_repeat=True,early_stop=True)
        if args.depth<=3:
            from jt_oct.d3_table3 import D3Options, solve_jt_dp_shallow_cpp, solve_jt_dp_shallow_gpu
            if args.method=='JT-CG':
                cfg=job['effective'];backend=cfg['backend'];threads=cfg['threads'];opts=cfg['d3']
            else:
                backend=job['backend'];threads=job['threads'];opts=job['options']
            options=D3Options(**opts)
            started=time.perf_counter()
            if backend=='gpu':
                result=solve_jt_dp_shallow_gpu(problem,args.seconds,options=options)
            else:
                result=solve_jt_dp_shallow_cpp(problem,args.seconds,options=options,threads=threads)
        else:
            from jt_oct.contract_cg import ContractOptions, solve_contracted_cg
            options=ContractOptions(**job['options'])
            backend='auto' if args.method=='JT-CG' else job['backend']
            if args.method=='JT-MP':
                assert options.master_mode=='message' and options.cost_mode=='lazy'
            started=time.perf_counter()
            result=solve_contracted_cg(problem,backend=backend,time_limit=args.seconds,options=options)

    result.update(requested_method=args.method,dataset=args.dataset,depth=args.depth,
                  penalty=args.penalty,time_limit=args.seconds,
                  call_wall_seconds=time.perf_counter()-started,configuration=job)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in
                     ('requested_method','method','status','LB','UB','call_wall_seconds')}))


if __name__=='__main__':
    main()
