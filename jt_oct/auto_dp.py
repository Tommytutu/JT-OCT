"""Measured-profile JT-DP portfolio. Selection never uses a dataset identifier.

The reference policy supplies the default configuration. A frozen
policy can select another action. Training and inference share this module.
"""
from functools import lru_cache
from pathlib import Path
import json
import math
import subprocess
import time

import numpy as np

from .d3_optimized import D3Options, solve_jt_dp_shallow_cpp, solve_jt_dp_shallow_gpu
from .d3_grow import D3GrowOptions, solve_d3_grow
from .terminal_message import TerminalOptions, solve_terminal_message
from .problem import Problem, Tree

ACTIONS=('reference','cpu','gpu','fast_gpu','prune_gpu')
POLICY_PATH=Path(__file__).with_name('auto_dp_policy.json')


def profile_problem(p):
    unique={min(a,b) for a,b in p._feature_masks if a and b}
    ids=np.linspace(0,p.n-1,min(p.n,8192),dtype=int)
    packed=np.packbits(p.X[ids],axis=1)
    return dict(n=p.n,F_input=p.F,F_effective=len(unique),K=len(p.labels),
                bitset_words=(p.n+63)//64,density=float(p.X.mean()),
                duplicate_rate_estimate=1-len(np.unique(packed,axis=0))/len(ids))


def features(profile,depth,penalty):
    n,f,k=(profile[x] for x in ('n','F_effective','K'))
    return dict(depth=depth,log_n=math.log2(max(1,n)),log_F=math.log2(max(1,f)),
                K=k,positive_penalty=int(penalty>0),
                log_work=math.log2(max(1,n*f*k)),
                duplicate_rate=profile.get('duplicate_rate_estimate',0.),
                density=profile.get('density',.5))


def configuration(profile,depth,penalty,action='reference'):
    if action not in ACTIONS:raise ValueError('Unknown automatic-DP action')
    if depth not in (2,3,4,5):raise ValueError('Automatic accelerated DP supports D2-D5')
    n,f,k=(profile[x] for x in ('n','F_effective','K'));work=n*f*k
    gpu=(work>=150_000_000 if depth==2 else
         ((f>=64 and work>=8_000_000) or work>=100_000_000) if depth==3 else
         (f>=96 or work>=8_000_000))
    initial_gpu=gpu
    if k==2 and f>64 and work<5_000_000 and (depth==5 or (depth==4 and penalty==0)):gpu=True
    cfg=dict(backend='gpu' if gpu else 'cpp',threads=32 if work>=5_000_000 else 8,
             fast_prepare=True,quotient_features=True,tile_pairs=256,
             warm_fraction=.25 if depth==5 and gpu and penalty==0 else .1,
             warm_cap_seconds=30.,root_beam=1 if depth==4 else (8 if f>160 else 32),
             schedule='gap',order='gain',lookahead_bound=bool(initial_gpu),root_quotient=True,
             gpu_batch_size=2,cache_bytes=128*1024*1024,similarity_neighbors=0,
             adaptive_words=True,gpu_native_metadata=False,gpu_metadata_cache_entries=0,
             gpu_sync_tiles=1)
    if depth==5 and gpu and penalty==0:cfg['root_beam']=32
    if k==2 and penalty>0 and ((depth==4 and work<100_000_000) or
                (depth==5 and f<=64 and 5_000_000<=work<100_000_000)):
        cfg['lookahead_bound']=False
    if action=='cpu':cfg['backend']='cpp'
    if action in ('gpu','fast_gpu','prune_gpu'):cfg['backend']='gpu'
    # Shallow solves do not execute upper-message/warm options. Canonicalize to
    # avoid counting identical actions as separate performance experiments.
    if depth>=4 and action in ('fast_gpu','prune_gpu'):
        cfg.update(gpu_native_metadata=True,gpu_metadata_cache_entries=4096,gpu_sync_tiles=8)
    if depth>=4 and action=='prune_gpu':
        cfg.update(gpu_batch_size=1,lookahead_bound=False,order='lb',schedule='lb',
                   warm_fraction=.05,root_beam=1)
    return cfg


def make_options(cfg):
    d3=D3Options(fast_prepare=cfg['fast_prepare'],quotient_features=cfg['quotient_features'],
          tile_pairs=cfg['tile_pairs'],gpu_resident=True,gpu_adaptive_words=cfg['adaptive_words'],
          gpu_native_metadata=cfg.get('gpu_native_metadata',False),
          gpu_metadata_cache_entries=cfg.get('gpu_metadata_cache_entries',0),
          gpu_sync_tiles=cfg.get('gpu_sync_tiles',1))
    term=TerminalOptions(fast_prepare=True,quotient=True,prune=True,warm_d3=True,
          schedule=cfg['schedule'],order=cfg['order'],resident_gpu=cfg['backend']=='gpu',
          gpu_batch_size=cfg['gpu_batch_size'],adaptive_words=cfg['adaptive_words'],
          lookahead_bound=cfg['lookahead_bound'],root_quotient=cfg['root_quotient'],
          cache_bytes=cfg['cache_bytes'],similarity_neighbors=cfg['similarity_neighbors'],
          gpu_native_metadata=d3.gpu_native_metadata,
          gpu_metadata_cache_entries=d3.gpu_metadata_cache_entries,gpu_sync_tiles=d3.gpu_sync_tiles)
    return d3,term


def execute(p,cfg,remaining,emit=lambda out:None,set_phase=lambda value:None):
    d3,term=make_options(cfg)
    if p.depth<=3:
        set_phase('shallow_exact')
        if cfg['backend']=='cpp':
            return solve_jt_dp_shallow_cpp(p,remaining(),options=d3,threads=cfg['threads'])
        return solve_jt_dp_shallow_gpu(p,remaining(),options=d3)
    set_phase('D3_grow');tick=time.perf_counter()
    warm=solve_d3_grow(p,cfg['backend'],min(cfg['warm_cap_seconds'],cfg['warm_fraction']*remaining()),
         cfg['threads'],D3GrowOptions(d3=d3,root_beam=cfg['root_beam'],frontier_order='largest'),emit)
    warm_elapsed=time.perf_counter()-tick;emit(warm)
    if remaining()<=.01:
        warm.update(status='TIME',warm_elapsed=warm_elapsed);return warm
    set_phase('exact_messages')
    out=solve_terminal_message(p,cfg['backend'],remaining(),cfg['threads'],term,emit,
              initial_tree=Tree.from_dict(warm['tree']) if warm.get('tree') else None)
    out.update(warm_phase=warm,warm_elapsed=warm_elapsed)
    return out


def policy_action(profile,depth,penalty,policy):
    node=policy['tree'];x=features(profile,depth,penalty)
    while 'action' not in node:
        node=node['left' if x[node['feature']]<=node['threshold'] else 'right']
    action=node['action']
    if action not in ACTIONS:raise ValueError('Invalid frozen policy action')
    return action


@lru_cache(maxsize=1)
def hardware():
    import psutil
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],
             capture_output=True,text=True,timeout=3,
             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        mib=int(r.stdout.strip().splitlines()[0]) if r.returncode==0 else 0
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):mib=0
    return dict(gpu_available=mib>0,gpu_memory_bytes=mib*1024**2,
                cpu_threads=psutil.cpu_count(logical=True) or 1,
                ram_bytes=psutil.virtual_memory().available)


def select(profile,depth,penalty,policy=None,machine=None):
    if policy is None:
        if not POLICY_PATH.exists():raise RuntimeError('Automatic policy is not frozen yet')
        policy=json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    action=policy_action(profile,depth,penalty,policy)
    cfg=configuration(profile,depth,penalty,action);machine=machine or hardware();notes=[]
    cfg['threads']=max(1,min(cfg['threads'],machine['cpu_threads']))
    # Conservative storage envelope: resident bitsets plus simultaneous F²
    # cost/argmin messages. This is a feasibility guard, not a speed predictor.
    f=profile.get('F_input',profile['F_effective']);w=(profile['n']+63)//64
    estimate=8*w*(2*f+profile['K']+8)+64*f*f*max(1,cfg['gpu_batch_size'])+64*1024**2
    if cfg['backend']=='gpu' and (not machine['gpu_available'] or
                   estimate>machine['gpu_memory_bytes']*.5):
        cfg['backend']='cpp';notes.append('CPU fallback: GPU unavailable or storage envelope exceeds budget')
    cfg['cache_bytes']=min(cfg['cache_bytes'],max(0,int(machine['ram_bytes']*.05)))
    return dict(action=action,config=cfg,hardware_adjustments=notes,estimated_gpu_bytes=estimate,
                policy_version=policy.get('version','test'))


def solve_auto_dp(X,y,depth=3,penalty=0.,time_limit=100,progress=None):
    """Default entry: callers supply a modeling task, never acceleration presets.

    File reading is external. Problem construction, profiling, hardware detection,
    selection, GPU initialization and D3 warm trees consume the solve budget.
    """
    if not math.isfinite(time_limit) or time_limit<0:raise ValueError('Invalid time limit')
    start=time.perf_counter();p=Problem(X,y,depth,penalty,no_repeat=True,early_stop=True)
    if time.perf_counter()-start>=time_limit:
        label,loss=p.best_label_and_loss(p.all_rows)
        return dict(status='TIME',LB=0.,UB=loss,tree=Tree(label=label).to_dict(),
                    total_seconds=time.perf_counter()-start,automatic_strategy=None)
    profile=profile_problem(p);chosen=select(profile,depth,penalty)
    if time.perf_counter()-start>=time_limit:
        label,loss=p.best_label_and_loss(p.all_rows)
        return dict(status='TIME',LB=0.,UB=loss,tree=Tree(label=label).to_dict(),
                    total_seconds=time.perf_counter()-start,automatic_strategy=chosen,profile=profile)
    remaining=lambda:max(.001,time_limit-(time.perf_counter()-start))
    out=execute(p,chosen['config'],remaining,progress or (lambda out:None))
    out.update(automatic_strategy=chosen,profile=profile,total_seconds=time.perf_counter()-start)
    return out
