"""Python interface to the sparse Gurobi C++ JT master."""
import ctypes as ct
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

_LIB=None
_DIRS=[]

class SparseTailMaster:
    def __init__(self,domain,row_universe=None,method=0):
        global _LIB
        if row_universe is not None:raise ValueError('Only generated union rows are supported')
        self.handle=None
        if _LIB is None:
            for root in (os.environ.get('GUROBI_HOME'),r'C:\gurobi1300\win64'):
                if root and (Path(root)/'bin/gurobi130.dll').exists():
                    _DIRS.append(os.add_dll_directory(str(Path(root)/'bin')))
            lib=ct.CDLL(str(Path(__file__).parent/'_native/sparse_tail_rmp.dll'))
            lib.strmp_create.argtypes=[ct.c_int,ct.c_int];lib.strmp_create.restype=ct.c_void_p
            lib.strmp_update.argtypes=[ct.c_void_p,ct.c_int]+[ct.c_void_p]*4;lib.strmp_update.restype=ct.c_int
            lib.strmp_solve.argtypes=[ct.c_void_p,ct.c_double];lib.strmp_solve.restype=ct.c_char_p
            lib.strmp_destroy.argtypes=[ct.c_void_p];lib.strmp_destroy.restype=None
            lib.strmp_error.argtypes=[];lib.strmp_error.restype=ct.c_char_p
            _LIB=lib
        self.lib=_LIB;self.domain=domain;self.rows={};self.seen=set();self.flat=[]
        self.handle=self.lib.strmp_create(domain.M,method)
        if not self.handle:raise RuntimeError(self.lib.strmp_error().decode())

    def solve_incremental(self,pool,deadline):
        deadline.check();blocks=[];left=[];right=[];cost=[]
        for q,columns in enumerate(pool):
            for c in columns:
                if (q,c) in self.seen:continue
                self.seen.add((q,c));self.flat.append((q,c));blocks.append(q);cost.append(c.cost)
                sides={-1:-1,1:-1}
                for e,size,sign in self.domain.incident[q]:
                    key=('sep',e,c.prefix[:size])
                    if key not in self.rows:self.rows[key]=len(self.rows)
                    sides[sign]=self.rows[key]
                left.append(sides[-1]);right.append(sides[1])
        if cost:
            arrays=[np.ascontiguousarray(x,dtype=np.int32) for x in (blocks,left,right)]
            arrays.append(np.ascontiguousarray(cost,dtype=np.float64))
            if self.lib.strmp_update(self.handle,len(cost),*[x.ctypes.data for x in arrays]):
                raise RuntimeError(self.lib.strmp_error().decode())
        raw=self.lib.strmp_solve(self.handle,deadline.remaining())
        if raw is None:raise RuntimeError(self.lib.strmp_error().decode())
        r=json.loads(raw);status={2:0,9:1,11:1,3:2}.get(r['status'],4)
        x=np.zeros(len(self.flat))
        for i,v in r.get('support',[]):x[i]=v
        res=SimpleNamespace(status=status,fun=r.get('objective'),x=x,
            eqlin=SimpleNamespace(marginals=np.asarray(r.get('alpha',[])+r.get('pi',[]))),
            max_equality_residual=r.get('max_equality_residual'),
            minimum_active_reduced_cost=r.get('minimum_active_reduced_cost'))
        keys=[('norm',q) for q in range(self.domain.M)]+list(self.rows)
        return res,(None,None,np.asarray([c.cost for _,c in self.flat]),list(self.flat),keys)

    solve=solve_incremental

    def close(self):
        if self.handle:self.lib.strmp_destroy(self.handle);self.handle=None
