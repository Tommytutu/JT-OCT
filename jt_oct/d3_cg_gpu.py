"""CUDA statistics callback for C++ JT-CG; no DP or MIP substitution."""
import ctypes as ct
from pathlib import Path
import time

import numpy as np

CALLBACK = ct.CFUNCTYPE(ct.c_int, ct.c_int, *([ct.c_void_p] * 5), ct.c_double)


class GPUCosts:
    def __init__(self, problem, allow_fallback=False):
        if problem._uniform_weight is None:
            raise ValueError("GPU D3 cost statistics currently require uniform observation weights")
        self.p = problem
        self.W = (problem.n + 63) // 64
        self.F, self.K = problem.F, len(problem.labels)
        self.allow_fallback = allow_fallback
        self.error = None
        self.ready = False
        self.stats = dict(gpu_setup_seconds=0., gpu_batch_seconds=0., gpu_batches=0,
                          gpu_peak_batch_rows=0, gpu_fallback_reason=None)
        self.callback = CALLBACK(self.compute)

    @staticmethod
    def array(pointer, size, dtype):
        return np.ctypeslib.as_array(ct.cast(pointer, ct.POINTER(dtype)), shape=(size,))

    def prepare(self, features, classes):
        if self.ready:
            return
        started = time.perf_counter()
        from .d3_batched import _configure_cupy_runtime
        _configure_cupy_runtime()
        import cupy as cp
        self.cp = cp
        source = (Path(__file__).resolve().parents[1] / 'native/d3_cg_costs.cu').read_text()
        self.parent = cp.RawKernel(source, 'cg_parent_counts', options=('--std=c++11',))
        self.stumps = cp.RawKernel(source, 'cg_stump_costs', options=('--std=c++11',))
        self.parent.compile(); self.stumps.compile()
        self.features = cp.asarray(self.array(features, self.F*self.W, ct.c_uint64))
        self.classes = cp.asarray(self.array(classes, self.K*self.W, ct.c_uint64))
        cp.cuda.Stream.null.synchronize()
        self.stats['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        self.stats['gpu_setup_seconds'] += time.perf_counter() - started
        self.ready = True

    def compute(self, count, features, classes, rows, losses, labels, seconds):
        started = time.perf_counter()
        if self.error is not None:
            return 2 if self.allow_fallback else 0
        try:
            self.prepare(features, classes)
            if time.perf_counter() - started >= seconds:
                return -1
            cp = self.cp
            host_rows = self.array(rows, count*self.W, ct.c_uint64)
            device_rows = cp.asarray(host_rows)
            counts = cp.empty((count, self.K), dtype=cp.int32)
            loss = cp.empty((count, self.F), dtype=cp.float64)
            lab = cp.empty((count, self.F, 2), dtype=cp.int32)
            self.parent((count, self.K), (256,),
                (device_rows, self.classes, np.int32(self.W), np.int32(self.K), counts))
            self.stumps((self.F, count), (256,), (device_rows, self.features, self.classes, counts,
                np.int32(self.W), np.int32(self.F), np.int32(self.K), np.int32(self.p.min_leaf),
                np.float64(self.p._uniform_weight), loss, lab))
            loss.get(out=self.array(losses, count*self.F, ct.c_double).reshape(count, self.F))
            lab.get(out=self.array(labels, count*self.F*2, ct.c_int32).reshape(count, self.F, 2))
            cp.cuda.Stream.null.synchronize()
            self.stats['gpu_batches'] += 1
            self.stats['gpu_peak_batch_rows'] = max(self.stats['gpu_peak_batch_rows'], count)
            self.stats['gpu_batch_seconds'] += time.perf_counter() - started
            return 1
        except Exception as exc:
            self.error = str(exc)
            self.stats['gpu_fallback_reason'] = self.error
            return 2 if self.allow_fallback else 0
