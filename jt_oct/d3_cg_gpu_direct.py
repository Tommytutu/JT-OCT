"""Regular full-table GPU batches, transferring only state ids and min-h results."""
import ctypes as ct
from pathlib import Path
import time

import numpy as np

from .d3_cg_gpu import GPUCosts

DIRECT_CALLBACK=ct.CFUNCTYPE(ct.c_int,ct.c_int,*([ct.c_void_p]*7),ct.c_double)


class DirectGPUCosts(GPUCosts):
    def __init__(self, problem):
        super().__init__(problem)
        self.buffers={}
        self.stats.update(gpu_h2d_seconds=0.,gpu_kernel_seconds=0.,gpu_d2h_seconds=0.,
                          gpu_descriptor_bytes=0,gpu_result_bytes=0,gpu_full_strategy='direct')
        self.direct_callback=DIRECT_CALLBACK(self.compute_direct)

    def prepare_direct(self,features,classes,allowed,extras):
        if self.ready:return
        started=time.perf_counter()
        from .d3_batched import _configure_cupy_runtime
        _configure_cupy_runtime()
        import cupy as cp
        self.cp=cp
        root=Path(__file__).resolve().parents[1]/'native'
        source=(root/'d3_cg_costs.cu').read_text()+'\n'+(root/'d3_cg_direct.cu').read_text()
        self.module=cp.RawModule(code=source,options=('--std=c++11',))
        self.module.compile()
        self.parent=self.module.get_function('cg_parent_counts')
        self.stumps=self.module.get_function('cg_stump_costs')
        self.make_rows=self.module.get_function('cg_direct_rows')
        self.reduce=self.module.get_function('cg_direct_min')
        self.features=cp.asarray(self.array(features,self.F*self.W,ct.c_uint64))
        self.classes=cp.asarray(self.array(classes,self.K*self.W,ct.c_uint64))
        self.allowed=cp.asarray(self.array(allowed,7*self.F,ct.c_uint8))
        self.extras=cp.asarray(self.array(extras,7*self.F,ct.c_double))
        cp.cuda.Stream.null.synchronize()
        self.stats['gpu_setup_seconds']+=time.perf_counter()-started
        self.stats['gpu_name']=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        self.ready=True

    def buffer(self,name,shape,dtype):
        size=int(np.prod(shape))
        if name not in self.buffers or self.buffers[name].size<size:
            self.buffers[name]=self.cp.empty(size,dtype=dtype)
        return self.buffers[name][:size].reshape(shape)

    def compute_direct(self,count,features,classes,allowed,extras,ids,losses,actions,seconds):
        started=time.perf_counter()
        try:
            self.prepare_direct(features,classes,allowed,extras)
            if time.perf_counter()-started>=seconds:return -1
            cp=self.cp
            state_ids=self.buffer('ids',(count,),cp.int32)
            rows=self.buffer('rows',(count,self.W),cp.uint64)
            counts=self.buffer('counts',(count,self.K),cp.int32)
            raw_loss=self.buffer('raw_loss',(count,self.F),cp.float64)
            raw_labels=self.buffer('raw_labels',(count,self.F,2),cp.int32)
            best=self.buffer('best',(count,),cp.float64)
            best_action=self.buffer('best_action',(count,3),cp.int32)
            t=time.perf_counter();state_ids.set(self.array(ids,count,ct.c_int32))
            cp.cuda.Stream.null.synchronize();self.stats['gpu_h2d_seconds']+=time.perf_counter()-t
            event_start,event_end=cp.cuda.Event(),cp.cuda.Event();event_start.record()
            self.make_rows(((count*self.W+255)//256,),(256,),
                (state_ids,self.features,np.int32(self.p.n),np.int32(self.W),np.int32(self.F),np.int32(count),rows))
            self.parent((count,self.K),(256,),(rows,self.classes,np.int32(self.W),np.int32(self.K),counts))
            self.stumps((self.F,count),(256,),(rows,self.features,self.classes,counts,
                np.int32(self.W),np.int32(self.F),np.int32(self.K),np.int32(self.p.min_leaf),
                np.float64(self.p._uniform_weight),raw_loss,raw_labels))
            self.reduce((count,),(256,),(state_ids,raw_loss,raw_labels,self.allowed,self.extras,
                np.int32(self.F),np.int32(self.p.no_repeat),best,best_action))
            event_end.record();event_end.synchronize()
            self.stats['gpu_kernel_seconds']+=cp.cuda.get_elapsed_time(event_start,event_end)/1000.
            t=time.perf_counter()
            best.get(out=self.array(losses,count,ct.c_double))
            best_action.get(out=self.array(actions,count*3,ct.c_int32).reshape(count,3))
            self.stats['gpu_d2h_seconds']+=time.perf_counter()-t
            self.stats['gpu_batches']+=1
            self.stats['gpu_peak_batch_rows']=max(self.stats['gpu_peak_batch_rows'],count)
            self.stats['gpu_batch_seconds']+=time.perf_counter()-started
            self.stats['gpu_descriptor_bytes']+=count*4
            self.stats['gpu_result_bytes']+=count*20
            return 1
        except Exception as exc:
            self.error=str(exc);self.stats['gpu_fallback_reason']=self.error
            return 0
