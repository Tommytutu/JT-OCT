"""D2 local costs on CUDA, keeping the Gurobi CG master and certification.

Only compact bitsets are uploaded. Pair statistics are independent of lambda;
both the host cost table and its Gurobi environment survive workspace reuse.
"""
import ctypes as ct
from pathlib import Path
import time
import threading

import numpy as np

COST_CALLBACK = ct.CFUNCTYPE(ct.c_int, *([ct.c_void_p] * 6), ct.c_double)
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()


def runtime_ready():
    return _RUNTIME is not None


def warmup_d2_gpu():
    """Optional explicit GPU startup for a long-lived service; returns its cost.

    This does not prepare or solve a tree and never hides startup in solve timing.
    """
    global _RUNTIME
    started=time.perf_counter()
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            from .d3_batched import _configure_cupy_runtime
            _configure_cupy_runtime()
            import cupy as cp
            source=(Path(__file__).resolve().parents[1]/"native/d2_cg_costs.cu").read_text()
            module=cp.RawModule(code=source,options=("--std=c++11",))
            module.compile()
            kernels={name:module.get_function("d2_"+name) for name in ("marginals","pairs","generic","reduce")}
            cp.cuda.Stream.null.synchronize()
            _RUNTIME=(cp,module,kernels,int(cp.cuda.runtime.getDevice()))
    return time.perf_counter()-started


class D2GPUCosts:
    def __init__(self, problem, strategy="pair", allow_fallback=False):
        if problem._uniform_weight is None:
            raise ValueError("D2 GPU costs require uniform weights; use CPU for nonuniform weights")
        self.p, self.strategy, self.allow_fallback = problem, strategy, allow_fallback
        self.F, self.K, self.W = problem.F, len(problem.labels), (problem.n + 63) // 64
        self.ready = False
        self.error = None
        self.stats = dict(gpu_setup_seconds=0., gpu_h2d_seconds=0., gpu_kernel_seconds=0.,
                          gpu_d2h_seconds=0., gpu_total_seconds=0., gpu_batches=0,
                          gpu_input_bytes=0, gpu_output_bytes=0, gpu_workspace_bytes=0,
                          gpu_fallback_reason=None, gpu_cost_strategy=strategy)
        self.callback = COST_CALLBACK(self.compute)

    @staticmethod
    def array(pointer, size, dtype):
        return np.ctypeslib.as_array(ct.cast(pointer, ct.POINTER(dtype)), shape=(size,))

    def prepare(self, features, classes, allowed, extras):
        if self.ready:
            return
        started = time.perf_counter()
        warmup_d2_gpu()
        cp,self.module,kernels,device = _RUNTIME
        if int(cp.cuda.runtime.getDevice()) != device:
            raise RuntimeError("D2 GPU runtime belongs to another CUDA device")
        self.cp=cp
        for name,kernel in kernels.items():setattr(self,name,kernel)
        self.stats["gpu_setup_seconds"] += time.perf_counter() - started
        self.stats["gpu_name"] = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
        # Bound persistent pair outputs before allocating; no n-by-F array and
        # no F-cubed tensor are created on the device.
        required = (self.F+self.K)*self.W*8 + 32*self.F*self.F + 40*self.F*self.K
        free, _ = cp.cuda.runtime.memGetInfo()
        if required > min(512*1024**2, int(free*.5)):
            raise MemoryError("D2 GPU pair table exceeds the bounded memory budget")
        t = time.perf_counter()
        self.features = cp.asarray(self.array(features, self.F*self.W, ct.c_uint64))
        self.classes = cp.asarray(self.array(classes, self.K*self.W, ct.c_uint64))
        self.allowed = cp.asarray(self.array(allowed, 3*self.F, ct.c_uint8))
        self.extras = cp.asarray(self.array(extras, 3*self.F, ct.c_double))
        cp.cuda.Stream.null.synchronize()
        self.stats["gpu_h2d_seconds"] += time.perf_counter()-t
        self.stats["gpu_input_bytes"] = (self.F+self.K)*self.W*8 + 27*self.F
        self.counts = cp.empty((self.F+1, self.K), dtype=cp.int32)
        self.loss = cp.empty((self.F, self.F, 2), dtype=cp.float64)
        self.labels = cp.empty((self.F, self.F, 2, 2), dtype=cp.int32)
        self.best = cp.empty((2*self.F, 2), dtype=cp.float64)
        self.actions = cp.empty((2*self.F, 4), dtype=cp.int32)
        self.stats["gpu_workspace_bytes"] = sum(x.nbytes for x in (
            self.features, self.classes, self.allowed, self.extras, self.counts,
            self.loss, self.labels, self.best, self.actions))
        self.ready = True

    def compute(self, features, classes, allowed, extras, losses, actions, seconds):
        started = time.perf_counter()
        try:
            self.prepare(features, classes, allowed, extras)
            if time.perf_counter()-started >= seconds:
                return -1
            cp = self.cp
            def launch(kernel, grid, args):
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record(); kernel(grid, (256,), args); end.record(); end.synchronize()
                self.stats["gpu_kernel_seconds"] += cp.cuda.get_elapsed_time(begin,end)/1000.
            dims = (np.int32(self.W), np.int32(self.F), np.int32(self.K))
            launch(self.marginals, (self.F+1, self.K), (self.features,self.classes,*dims,self.counts))
            # Bound work per launch, allowing deadline checks even on long bitsets.
            work_per_root = max(1,self.F*self.W*self.K)
            root_batch = max(1,min(self.F,16_000_000//work_per_root))
            total = self.F if self.strategy == "pair" else 2*self.F
            for first in range(0,total,root_batch):
                if time.perf_counter()-started >= seconds:
                    return -1
                count = min(root_batch,total-first)
                args = (self.features,self.classes,self.counts,*dims,np.int32(self.p.min_leaf),
                        np.float64(self.p._uniform_weight),np.int32(first))
                if self.strategy == "pair":
                    args += (np.int32(count),)
                launch(self.pairs if self.strategy == "pair" else self.generic,
                       (self.F,count),args+(self.loss,self.labels))
                self.stats["gpu_batches"] += 1
            if time.perf_counter()-started >= seconds:
                return -1
            launch(self.reduce,(self.F,2),(self.loss,self.labels,self.counts,self.allowed,self.extras,
                np.int32(self.F),np.int32(self.K),np.int32(self.p.min_leaf),np.int32(self.p.no_repeat),
                np.float64(self.p._uniform_weight),self.best,self.actions))
            t=time.perf_counter()
            self.best.get(out=self.array(losses,4*self.F,ct.c_double).reshape(2*self.F,2))
            self.actions.get(out=self.array(actions,8*self.F,ct.c_int32).reshape(2*self.F,4))
            self.stats["gpu_d2h_seconds"] += time.perf_counter()-t
            self.stats["gpu_output_bytes"] = 64*self.F
            return -1 if time.perf_counter()-started >= seconds else 1
        except Exception as exc:
            self.error = str(exc)
            self.stats["gpu_fallback_reason"] = self.error
            return 2 if self.allow_fallback else 0
        finally:
            self.stats["gpu_total_seconds"] += time.perf_counter()-started
