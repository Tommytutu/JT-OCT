"""Run one Table 3 case with progress and a complete reproducibility record."""
import argparse
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
import json
from pathlib import Path
import subprocess
import time
from dataclasses import replace
from jt_oct.contract_cg import ContractOptions, solve_contracted_cg
from jt_oct import Problem
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
parser.add_argument('--depth',type=int,default=5)
parser.add_argument('--penalty',type=float,default=0)
parser.add_argument('--seconds',type=float,default=600)
parser.add_argument('--options',default='{}')
parser.add_argument('--backend',default='auto')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent
cfg=next(c for c in json.loads((root/'config/cg.json').read_text())
         if (c['dataset'],c['depth'],c['penalty'])==(args.dataset,args.depth,args.penalty))
options=replace(ContractOptions(**cfg['options']),**json.loads(args.options))
path=root/'datasets'/(args.dataset+'.npz')
manifest=dict(dataset=args.dataset,depth=args.depth,penalty=args.penalty,seconds=args.seconds,
    baseline_configuration=cfg,overrides=json.loads(args.options),data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
manifest['source_hashes']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for pattern in ('jt_oct/*.py','native/*.cpp','native/*.hpp','native/general_jt/*.hpp','jt_oct/_native/*.dll') for p in root.glob(pattern)}
with np.load(path,allow_pickle=False) as d:x,y=d['X'],d['y'].astype(np.int64)
manifest.update(n=len(y),F=x.shape[1],classes=np.unique(y).tolist())
manifest['environment']={k:v for k,v in os.environ.items() if k.startswith(('JT_OCT_','OMP_','MKL_','OPENBLAS_','NUMEXPR_'))}
manifest['started_local']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
started=time.perf_counter()
p=Problem(x,y,args.depth,args.penalty,no_repeat=True,early_stop=True)
log=args.output.with_suffix('.progress.jsonl')
last=-30.
def progress(r):
    global last
    if r['seconds']-last<10:return
    last=r['seconds']
    keys=('d3_calls','d3_seconds','d3_metadata_seconds','d3_cost_and_h_reduction_seconds','d3_join_seconds','rmp_seconds',
          'similarity_seconds','similarity_improvements','state_screened_candidates','selected_pricing_states','roots_excluded_by_message',
          'native_root_candidates','native_root_pruned','native_transfer_exact')
    out={k:r.get(k) for k in ('seconds','LB','UB','gap')}
    out.update({k:r['stats'].get(k,0) for k in keys})
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps(out)+'\n')
    print(json.dumps(out),flush=True)
call=time.perf_counter()
r=solve_contracted_cg(p,backend=args.backend,time_limit=args.seconds,options=options,progress=progress)
r.update(manifest=manifest,call_wall_seconds=time.perf_counter()-call,prepared_wall_seconds=time.perf_counter()-started)
args.output.write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps({k:r.get(k) for k in ('status','LB','UB','gap','call_wall_seconds')}),flush=True)
