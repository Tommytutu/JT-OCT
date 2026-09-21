"""Thin data/result adapter for the native D3 structural experiment engine."""
import ctypes as ct
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from .problem import Tree, evaluate

_LIB=None;_LOCK=threading.Lock();_DIRS=[]
_BYTE=ct.POINTER(ct.c_uint8);_INT=ct.POINTER(ct.c_int32);_DOUBLE=ct.POINTER(ct.c_double)


def _library():
    global _LIB
    with _LOCK:
        if _LIB is not None:return _LIB
        path=Path(__file__).parent/'_native/d3_structural.dll'
        if not path.exists():raise RuntimeError('Build the native structural engine with build_d3_structural.ps1')
        for root in (os.environ.get('GUROBI_HOME'),r'C:\gurobi1300\win64'):
            if root and (Path(root)/'bin/gurobi130.dll').exists():_DIRS.append(os.add_dll_directory(str(Path(root)/'bin')))
        lib=ct.CDLL(str(path));lib.d3struct_error.restype=ct.c_char_p
        lib.d3struct_create.argtypes=[_BYTE,_INT,_DOUBLE,_BYTE,_DOUBLE]+[ct.c_int]*7;lib.d3struct_create.restype=ct.c_void_p
        lib.d3struct_cost_provider.argtypes=[ct.c_void_p,ct.c_void_p]
        lib.d3struct_solve.argtypes=[ct.c_void_p,ct.c_double,ct.c_double]+[ct.c_int]*7;lib.d3struct_solve.restype=ct.c_char_p
        lib.d3struct_destroy.argtypes=[ct.c_void_p];_LIB=lib;return lib


def _decode_tree(data,labels):
    if data is None:return None
    if 'label_index' in data:return Tree(label=labels[int(data['label_index'])])
    return Tree(feature=int(data['feature']),left=_decode_tree(data['left'],labels),right=_decode_tree(data['right'],labels))


def solve_d3_structural_native(problem,reduction='sc_ee',coordinator='cg',backend='cpp',threads=8,
                               time_limit=600,max_columns=20_000_000,max_active=200_000,
                               pricing_batch=64,rmp_method=1,cost_batch=256):
    if problem.depth!=3:raise ValueError('Native structural engine requires D=3')
    if reduction not in ('base','sc','sc_ee') or coordinator not in ('lp','cg'):
        raise ValueError('Invalid structural mode')
    if backend not in ('cpp','gpu'):raise ValueError('backend must be cpp or gpu')
    if not all(isinstance(x,int) and x>0 for x in (threads,max_columns,max_active,pricing_batch,cost_batch)):
        raise ValueError('Positive integer resource options required')
    labels=problem.labels;encoded=np.searchsorted(labels,problem.y).astype(np.int32)
    allowed=np.zeros((7,problem.F),dtype=np.uint8);extras=np.zeros((7,problem.F),dtype=np.float64)
    nodes=((),(0,),(1,),(0,0),(0,1),(1,0),(1,1))
    for j,node in enumerate(nodes):
        allowed[j,list(problem.features(node,()))]=1
        for f in range(problem.F):extras[j,f]=problem.split_costs.get((node,f),0.)
    arrays=(np.ascontiguousarray(problem.X,dtype=np.uint8),encoded,
            np.ascontiguousarray(problem.weights,dtype=np.float64),allowed,extras)
    lib=_library();handle=lib.d3struct_create(arrays[0].ctypes.data_as(_BYTE),arrays[1].ctypes.data_as(_INT),
        arrays[2].ctypes.data_as(_DOUBLE),arrays[3].ctypes.data_as(_BYTE),arrays[4].ctypes.data_as(_DOUBLE),
        problem.n,problem.F,len(labels),problem.min_leaf,int(problem.early_stop),int(problem.no_repeat),threads)
    if not handle:raise RuntimeError(lib.d3struct_error().decode())
    provider=None;started=time.perf_counter()
    try:
        if backend=='gpu':
            from .d3_cg_gpu import GPUCosts
            provider=GPUCosts(problem);lib.d3struct_cost_provider(handle,ct.cast(provider.callback,ct.c_void_p))
        raw=lib.d3struct_solve(handle,problem.penalty,float(time_limit),
            {'base':0,'sc':1,'sc_ee':2}[reduction],{'lp':0,'cg':1}[coordinator],
            max_columns,max_active,pricing_batch,rmp_method,cost_batch)
        if raw is None:raise RuntimeError(lib.d3struct_error().decode())
        result=json.loads(raw)
    finally:lib.d3struct_destroy(handle)
    result['seconds']=time.perf_counter()-started;result['reduction']=reduction;result['coordinator']=coordinator
    result['requested_backend']=backend;result['interface_seconds']=max(0.,result['seconds']-result.get('native_seconds',0.))
    if provider is not None:
        if provider.error:raise RuntimeError('D3 structural GPU: '+provider.error)
        result['gpu_stats']=provider.stats
    tree=_decode_tree(result.get('tree'),labels);result['tree']=tree.to_dict() if tree else None
    if tree:
        audit=evaluate(problem,tree)
        if result.get('UB') is None or not math.isclose(audit['objective'],result['UB'],abs_tol=1e-7):
            raise AssertionError('Native structural tree objective mismatch')
        result['metrics']=audit;result['UB']=audit['objective']
    if result.get('LB') is not None and result.get('UB') is not None:
        if result['LB']>result['UB']+1e-7:raise AssertionError('Native structural LB exceeds UB')
        result['absolute_gap']=max(0.,result['UB']-result['LB'])
    return result
