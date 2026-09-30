"""Sequential process-isolated FICO comparison; never overwrites saved results."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--general-root',type=Path,required=True)
    ap.add_argument('--legacy-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--depths',type=int,nargs='+',choices=(4,5),default=[4,5])
    ap.add_argument('--models',nargs='+',choices=['legacy','native','cc_f1','f1'],default=['legacy','native','cc_f1','f1'])
    args=ap.parse_args();root=Path(__file__).resolve().parent
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    raw=out/'raw';raw.mkdir()
    roots=dict(legacy=args.legacy_root.resolve(),native=root,cc_f1=args.general_root.resolve(),f1=args.general_root.resolve())
    data={k:sha(r/'datasets/fico.npz') for k,r in roots.items()}
    if len(set(data.values()))!=1:raise AssertionError('Different FICO matrices')
    sources={}
    for model,r in roots.items():
        names=(['run.py','jt_oct/contract_cg.py','jt_oct/d3_optimized.py','jt_oct/_native/contract_rmp.dll'] if model in ('native','legacy') else
               ['run_f1_cc.py','run_cm_cpp.py','jt_oct/_native/cm_cc_f1.dll','jt_oct/_native/cm_generic_d4.dll','jt_oct/_native/cm_generic_d5.dll'])
        if model=='native':names+=['jt_oct/accuracy_oracle.py','native/accuracy_oracle.cpp','jt_oct/_native/accuracy_oracle.dll']
        sources[model]={name:sha(r/name) for name in names if (r/name).is_file()}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    manifest=dict(python=sys.executable,python_version=sys.version,platform=platform.platform(),gpu=gpu,
                  dataset_sha256=data,sources=sources,threads=8,time_limit=600,repeats=args.repeats,
                  timing='Fresh process wall time from launch through exit; sequential, no overlapping solvers; existing driver disk caches allowed',
                  min_leaf=dict(legacy=0,native=0,cc_f1=1,f1=1))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    rows=[]
    expected={(4,0.):.27603021321350013,(4,.01):.3172559518118365,(5,0.):.26493928673869377,(5,.01):.3172559518118365}
    for rep in range(1,args.repeats+1):
        for depth in args.depths:
            for penalty in (0.,.01):
                for model in (args.models if rep%2 else list(reversed(args.models))):
                    if model=='legacy' and rep>1:continue
                    name=f'{model}_d{depth}_p{penalty:g}_r{rep}';dest=raw/(name+'.json')
                    common=['--dataset','fico','--depth',str(depth),'--penalty',str(penalty),'--seconds','600','--output',str(dest)]
                    if model in ('legacy','native'):
                        command=[sys.executable,'run.py','--method','JT-CG']+common
                        if model=='native':command+=['--d3-oracle','native','--threads','8']
                    elif model=='cc_f1':command=[sys.executable,'run_f1_cc.py','--engine','native','--gpu','required','--threads','8','--min-leaf','1']+common
                    else:command=[sys.executable,'run_cm_cpp.py','--cm-engine','generic','--metric','f1','--support-policy','budget','--gpu','required','--threads','8','--min-leaf','1']+common
                    print('START '+name,flush=True)
                    tick=time.perf_counter()
                    with (raw/(name+'.log')).open('w',encoding='utf-8') as log:
                        process=subprocess.run(command,cwd=roots[model],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=660)
                    elapsed=time.perf_counter()-tick
                    if process.returncode:raise RuntimeError(f'{name} exited {process.returncode}; inspect its log')
                    result=json.loads(dest.read_text(encoding='utf-8'))
                    if model in ('legacy','native'):
                        if result['status']!='OPT' or abs(result['UB']-expected[depth,penalty])>1e-9 or result['absolute_gap']>1e-7:
                            raise AssertionError(f'Accuracy certificate failed: {name}')
                    elif result['status']!='OPTIMAL_TOL' or result['absolute_gap']>1e-7:
                        raise AssertionError(f'F1 certificate failed: {name}')
                    stats=result.get('stats',{})
                    record=dict(model=model,depth=depth,penalty=penalty,repeat=rep,status=result['status'],
                        process_seconds=elapsed,call_seconds=result.get('call_wall_seconds',result.get('solver_wall_seconds')),
                        LB=result['LB'],UB=result['UB'],gap=result['absolute_gap'],score=result.get('score'),
                        d3_calls=stats.get('d3_calls'),d3_seconds=stats.get('d3_seconds'),
                        gpu_compute_and_join_seconds=stats.get('d3_cost_and_h_reduction_seconds',0)+stats.get('d3_join_seconds',0),
                        metadata_seconds=stats.get('d3_metadata_seconds'),pack_seconds=stats.get('d3_pack_seconds'),
                        recovery_seconds=stats.get('d3_recovery_seconds'),rmp_seconds=stats.get('rmp_seconds'),rmp_solves=stats.get('iterations'),
                        result=str(dest),command=command)
                    rows.append(record)
                    (out/'runs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                        fields=[k for k in record if k!='command'];writer=csv.DictWriter(stream,fields);writer.writeheader()
                        writer.writerows({k:v for k,v in row.items() if k!='command'} for row in rows)
                    print(json.dumps({k:record[k] for k in ('model','depth','penalty','repeat','status','process_seconds','call_seconds')}),flush=True)


if __name__=='__main__':main()
