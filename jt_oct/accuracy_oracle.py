"""Native error-count oracles for binary and multiclass JT-CG pricing."""
import ctypes as ct
from pathlib import Path
import os
import sys
import time

import numpy as np
from .problem import RowSet, Tree


class NativeAccuracyOracle:
    def __init__(self, owner, force_general=False):
        p = owner.p
        if (not 2 <= len(p.labels) <= 32 or p._uniform_weight is None or not p.early_stop
                or p.allowed or p.split_costs):
            raise ValueError('Native accuracy oracle needs 2..32 labels, uniform weights, STOP and shared features/costs')
        self.owner, self.p = owner, p
        self.handle = None
        self.x = np.ascontiguousarray(p.X, dtype=np.uint8)
        self.general = force_general or len(p.labels)>2 or p.n>=200000
        self.y = np.ascontiguousarray(np.searchsorted(p.labels,p.y), dtype=np.uint8)
        roots = [Path(sys.prefix)/'Lib/site-packages/nvidia/cuda_nvrtc/bin',
                 Path.home()/'.cache/jt_oct/cuda12_runtime/bin',
                 Path.home()/'.codex/cuda12_runtime/bin']
        override = os.environ.get('JT_OCT_NVRTC')
        rtc = Path(override) if override else next((f for root in roots for f in root.glob('nvrtc64_*.dll') if '.alt.' not in f.name), None)
        if rtc is None or not rtc.is_file():
            raise RuntimeError('NVRTC not found; set JT_OCT_NVRTC to its DLL path')
        library=Path(__file__).parent/('_native/accuracy_multiclass.dll' if self.general else '_native/accuracy_oracle.dll')
        if self.general and os.environ.get('JT_OCT_ACCURACY_DLL'):library=Path(os.environ['JT_OCT_ACCURACY_DLL'])
        self.lib = ct.CDLL(str(library))
        self.lib.accuracy_oracle_create.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_size_t,ct.c_wchar_p]
        self.lib.accuracy_oracle_create.restype = ct.c_void_p
        if self.general:self.lib.accuracy_oracle_create.argtypes += [ct.c_int]
        self.lib.accuracy_oracle_error.restype = ct.c_char_p
        self.lib.accuracy_oracle_free.argtypes = [ct.c_void_p]
        self.lib.accuracy_oracle_batch.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_double,
                                                  ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.lib.accuracy_oracle_batch.restype = ct.c_int
        self.lib.accuracy_oracle_prefetch.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_double]
        self.lib.accuracy_oracle_prefetch.restype = ct.c_int
        args=[self.x.ctypes.data,self.y.ctypes.data,p.n,p.F,p.min_leaf,max(1,owner.threads),owner.options.native_geometry_bytes,str(rtc)]
        if self.general:args.append(len(p.labels))
        self.handle = self.lib.accuracy_oracle_create(*args)
        if not self.handle:
            raise RuntimeError(self.lib.accuracy_oracle_error().decode())
        self.counters=np.zeros(3,np.uint64)
        if self.general and hasattr(self.lib,'accuracy_oracle_counters'):
            self.lib.accuracy_oracle_counters.argtypes=[ct.c_void_p,ct.c_void_p]

    def close(self):
        if self.handle:
            self.lib.accuracy_oracle_free(self.handle)
            self.handle = None

    def prefetch(self, rows, seconds):
        if self.general:return
        words=(self.p.n+63)//64
        masks=np.frombuffer(b''.join(r.mask.to_bytes(words*8,'little') for r in rows),dtype=np.uint64)
        if not self.lib.accuracy_oracle_prefetch(self.handle,masks.ctypes.data,len(rows),seconds):
            raise RuntimeError(self.lib.accuracy_oracle_error().decode())

    def solve_many(self, requests, time_limit, depth, audit=None):
        if audit is not None:
            raise ValueError('Native oracle does not expose the legacy dense-table audit callback')
        if depth not in (2,3):
            raise ValueError('Native oracle supports terminal depth 2 or 3')
        if not self.handle:
            raise RuntimeError('Native oracle is closed')
        start=time.perf_counter();p=self.p;words=(p.n+63)//64
        domains=[]
        for req in requests:
            rows=req.get('rows',p.all_rows);node=tuple(req.get('node',()));used=tuple(req.get('used',()))
            if not isinstance(rows,RowSet) or rows.mask<0 or rows.mask&~p.all_rows.mask:
                raise ValueError('Invalid routed rows')
            if len(node)+depth>p.depth or any(v not in (0,1) for v in node):
                raise ValueError('Invalid subtree depth')
            # Removing an ancestor is safe only when it is constant on this domain.
            for f in used:
                if not 0<=f<p.F or rows.mask&p._feature_masks[f][0] not in (0,rows.mask):
                    raise ValueError('Native oracle requires constant ancestor predicates')
            domains.append((rows,node,used))
        if not domains:return []
        masks=np.frombuffer(b''.join(r.mask.to_bytes(words*8,'little') for r,_,_ in domains),dtype=np.uint64)
        count=len(domains);status=np.zeros(count,np.int32);values=np.zeros(count)
        actions=np.zeros((count,64),np.int32);timings=np.zeros(5)
        ok=self.lib.accuracy_oracle_batch(self.handle,masks.ctypes.data,count,depth,p.penalty/p._uniform_weight,
            max(0.,time_limit-(time.perf_counter()-start)),status.ctypes.data,values.ctypes.data,actions.ctypes.data,timings.ctypes.data)
        if not ok:raise RuntimeError(self.lib.accuracy_oracle_error().decode())
        # Recover and independently audit with Python integer masks in one walk.
        # This avoids building the full 159-feature choice tuple at every node.
        def recover(a,mask,used,remaining,slot=1):
            f=int(a[slot])
            if -len(p.labels)<=f<=-1:
                if mask.bit_count()<p.min_leaf:raise AssertionError('Native leaf too small')
                label=p.labels[-f-1]
                return Tree(label=label),(mask&~p._label_masks[label]).bit_count(),0
            if not 0<=f<p.F or remaining<=0 or (p.no_repeat and f in used):
                raise AssertionError('Invalid native split')
            leftmask=mask&p._feature_masks[f][0]
            l,le,lk=recover(a,leftmask,used+(f,),remaining-1,2*slot)
            r,re,rk=recover(a,mask^leftmask,used+(f,),remaining-1,2*slot+1)
            return Tree(feature=f,left=l,right=r),le+re,1+lk+rk
        out=[]
        for i,(rows,node,used) in enumerate(domains):
            tree=None;value=values[i]*p._uniform_weight
            if status[i]>=0:
                tree,errors,splits=recover(actions[i],rows.mask,used,depth)
                actual=errors*p._uniform_weight+splits*p.penalty
                if abs(actual-value)>1e-9:raise AssertionError('Native oracle objective mismatch')
            out.append(dict(status='OPT' if status[i]>0 else 'INFEASIBLE' if status[i]<0 else 'TIME',value=float(value),tree=tree,
                stats=dict(kernel_calls=1,native_accuracy_oracle=True,input_features=p.F,compact_rows_used=True),timings={}))
        elapsed=time.perf_counter()-start
        if self.general and hasattr(self.lib,'accuracy_oracle_counters'):
            counts=np.zeros(3,np.uint64)
            self.lib.accuracy_oracle_counters(self.handle,counts.ctypes.data)
            out[0]['stats'].update(zip(('native_root_candidates','native_root_pruned','native_transfer_exact'),map(int,counts-self.counters)))
            self.counters=counts
        for r in out:
            r['timings']=dict(metadata_seconds=float(max(0.,timings[0]-sum(timings[2:])))/count,
                cost_and_h_reduction_seconds=float(timings[2])/count,pack_seconds=float(timings[3])/count,
                join_seconds=float(timings[4])/count,recovery_seconds=max(0.,elapsed-timings[0])/count)
        return out
