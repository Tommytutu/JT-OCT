"""Sequential native interval oracle; bounded results never enter the exact cache."""
import ctypes as ct
from pathlib import Path
import numpy as np
from .problem import Tree
from .message_plus import conflict_representatives


class IntervalOracle:
    def __init__(self, p):
        if p._uniform_weight is None or not p.early_stop or p.allowed or p.split_costs:
            raise ValueError('Interval oracle requires uniform costs, weights and STOP')
        self.p=p;self.W=max(1,(p.n+63)//64)
        self.lib=ct.CDLL(str(Path(__file__).resolve().parent/'_native/interval_oracle.dll'))
        self.lib.interval_create.argtypes=[ct.c_int]*4+[ct.c_double]*2+[ct.c_void_p]*4+[ct.c_int]
        self.lib.interval_create.restype=ct.c_void_p
        self.lib.interval_free.argtypes=[ct.c_void_p]
        self.lib.interval_solve.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_double,ct.c_int,ct.c_double]+[ct.c_void_p]*3
        self.lib.interval_solve.restype=ct.c_int
        zero=np.concatenate([self.words(p.route(p.all_rows,f,0).mask) for f in range(p.F)])
        labels=np.concatenate([self.words(p._label_masks[k]) for k in p.labels])
        groups=conflict_representatives(p)
        reps=np.asarray([int(r) for r,m in groups],dtype=np.int32)
        masses=np.asarray([int(m) for r,m in groups],dtype=np.int32)
        self.handle=self.lib.interval_create(p.F,self.W,len(p.labels),p.min_leaf,p._uniform_weight,p.penalty,
            zero.ctypes.data,labels.ctypes.data,reps.ctypes.data,masses.ctypes.data,len(reps))
        if not self.handle:raise MemoryError('Interval oracle allocation failed')

    def words(self, mask):
        return np.frombuffer(int(mask).to_bytes(self.W*8,'little'),dtype=np.uint64)

    def close(self):
        if self.handle:self.lib.interval_free(self.handle);self.handle=None

    def solve(self, rows, depth, cutoff, budget, seconds):
        bounds=np.empty(2,dtype=np.float64);actions=np.empty(15,dtype=np.int32);stats=np.empty(3,dtype=np.int32)
        mask=self.words(rows.mask)
        exact=self.lib.interval_solve(self.handle,mask.ctypes.data,depth,cutoff,budget,seconds,
            bounds.ctypes.data,actions.ctypes.data,stats.ctypes.data)
        if exact<0:raise RuntimeError('Native interval oracle failed')
        def recover(i):
            f=int(actions[i])
            return Tree(label=self.p.labels[-f-1]) if f<0 else Tree(feature=f,left=recover(2*i+1),right=recover(2*i+2))
        tree=recover(0) if np.isfinite(bounds[1]) else None
        return float(bounds[0]),float(bounds[1]),tree,bool(exact),stats
