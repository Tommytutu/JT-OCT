"""Array/CUDA dispatch interface to the C++ structural campaign.

All preparation, feature quotienting, D3 joins, master/pricing, selection and
tree recovery are native. This module does no tree optimization or state search.
"""
import ctypes as ct
import json
import math
import operator
from pathlib import Path
import time

import numpy as np

from .deep_structural_native import _lib


class _GpuBatch(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in
                ("phase", "B", "F", "W", "K", "globalF", "n", "max_words")] + [
        (name, ct.POINTER(ct.c_uint64)) for name in ("zero", "labels", "masks")] + [
        (name, ct.POINTER(ct.c_int32)) for name in ("indices", "nw", "features")] + [
        (name, ct.POINTER(ct.c_double)) for name in ("costs", "floors")] + [
        (name, ct.POINTER(ct.c_uint8)) for name in ("allowed", "dominated")] + [
        ("output", ct.POINTER(ct.c_double)), ("tail", ct.POINTER(ct.c_int32)),
        ("reasons", ct.POINTER(ct.c_uint8)), ("weight", ct.c_double),
        ("remaining", ct.c_double)]


_CALLBACK = ct.CFUNCTYPE(ct.c_int, ct.POINTER(_GpuBatch))


class _CUDA:
    """Only transfers and fixed-tile launches of the existing resident kernel."""
    def __init__(self):
        self.error = None
        self.kernel_calls = 0
        self.streaming_calls = 0
        self.shared_calls = 0
        self.callback = _CALLBACK(self.dispatch)

    def dispatch(self, pointer):
        try:
            r = pointer.contents
            if r.phase == 0:
                from .d3_batched import _configure_cupy_runtime
                _configure_cupy_runtime()
                import cupy as cp
                self.cp = cp
                root = Path(__file__).resolve().parents[1] / "native"
                source = (root / "resident_multiclass.cu").read_text().replace("@CLASSES@", str(r.K))
                source += "\n" + (root / "resident_fused.cu").read_text().replace("@CLASSES@", str(r.K))
                self.kernel = cp.RawKernel(source, "resident_d3_fused", options=("--std=c++11",))
                self.kernel.compile()
                self.zero = cp.asarray(np.ctypeslib.as_array(r.zero, shape=(r.globalF * r.W,)))
                self.labels = cp.asarray(np.ctypeslib.as_array(r.labels, shape=(r.K * r.W,)))
                cp.cuda.Stream.null.synchronize()
                return 0
            cp = self.cp
            deadline = time.perf_counter() + r.remaining
            def upload(pointer, size):
                return cp.asarray(np.ctypeslib.as_array(pointer, shape=(size,)))
            masks = upload(r.masks, r.B * r.W)
            indices = upload(r.indices, r.B * r.W)
            nw = upload(r.nw, r.B)
            features = upload(r.features, r.B * r.F)
            costs = upload(r.costs, r.B * 7 * r.F)
            floors = upload(r.floors, r.B * 7)
            allowed = upload(r.allowed, r.B * 7 * r.F)
            dominated = upload(r.dominated, r.B * 2 * r.F)
            size = r.B * 4 * r.F * r.F
            output = cp.empty(size, dtype=cp.float64)
            tail = cp.empty(size, dtype=cp.int32)
            reasons = cp.empty(size, dtype=cp.uint8)
            # Dynamic shared memory is optional; use the same resident arithmetic
            # for larger masks without allocating more than the default 48 KiB.
            shared = r.max_words * 8
            cache_pair = int(shared <= 40 * 1024)
            if not cache_pair:
                shared = 0
            for start in range(0, r.F * r.F, 256):
                if time.perf_counter() >= deadline:
                    cp.cuda.Stream.null.synchronize()
                    return 2
                count = min(256, r.F * r.F - start)
                self.kernel((count * 4, r.B), (256,), (
                    self.zero, self.labels, masks, indices, nw, features, costs,
                    allowed, floors, dominated, np.int32(r.F), np.int32(r.W),
                    np.int32(1), np.int32(1), np.int32(0), np.int32(1),
                    np.int32(start), np.int32(count), np.float64(r.weight),
                    output, tail, reasons, np.int32(cache_pair)), shared_mem=shared)
                cp.cuda.Stream.null.synchronize()
                self.kernel_calls += 1
                if cache_pair:
                    self.shared_calls += 1
                else:
                    self.streaming_calls += 1
            output.get(out=np.ctypeslib.as_array(r.output, shape=(size,)))
            tail.get(out=np.ctypeslib.as_array(r.tail, shape=(size,)))
            reasons.get(out=np.ctypeslib.as_array(r.reasons, shape=(size,)))
            cp.cuda.Stream.null.synchronize()
            return 0 if time.perf_counter() < deadline else 2
        except Exception as exc:
            self.error = str(exc)
            return 1

    def close(self):
        # Break the ctypes bound-method callback cycle and release this solve's
        # resident arrays; no callback/native object survives ds_campaign_solve.
        self.callback = None
        for name in ("zero", "labels", "kernel"):
            if hasattr(self, name):
                delattr(self, name)


def solve_deep_campaign(X, y, depth, variant, backend="cpp", *, threads=8,
                        time_limit=600, memory_limit_bytes=48 * 1024**3,
                        oracle_batch=16, max_columns=200000, rmp_column_cap=0,
                        message_bound=True, columns_per_block=0):
    """Solve the campaign's fixed tree domain (lambda=0, D3 tails, STOP).

    X/y conversion and shape checking are interface operations. Label encoding,
    binary checks, packing, all numerical kernels and tree audits run in C++.
    """
    variants = {"jt_cg": (0, 0), "jt_cg_ee": (1, 0),
                "jt_lp_sc": (0, 1), "jt_lp_sc_ee": (1, 1)}
    if variant not in variants or backend not in ("cpp", "gpu"):
        raise ValueError("Invalid campaign variant/backend")
    depth = operator.index(depth)
    if not 5 <= depth <= 7:
        raise ValueError("Native campaign requires D5--D7")
    for name, value, minimum in (("threads",threads,1),("oracle_batch",oracle_batch,1),
            ("max_columns",max_columns,1),("rmp_column_cap",rmp_column_cap,0),
            ("columns_per_block",columns_per_block,0)):
        if not minimum <= operator.index(value) <= 2**31-1:
            raise ValueError("Invalid integer option: " + name)
    if not 1 <= operator.index(memory_limit_bytes) <= 2**64-1:
        raise ValueError("Invalid memory limit")
    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("Invalid time limit")
    raw_x, raw_y = np.asarray(X), np.asarray(y)
    if raw_x.ndim != 2 or raw_y.ndim != 1 or len(raw_x) != len(raw_y):
        raise ValueError("Expected X[n,F] and y[n]")
    # Refuse narrowing casts which could turn invalid inputs into binary data.
    if raw_x.dtype != np.uint8 or raw_y.dtype.kind not in "iu":
        raise ValueError("Campaign interface requires uint8 X and integer labels")
    if raw_y.dtype == np.uint64:
        raise ValueError("Use signed int64 label codes")
    x = np.ascontiguousarray(raw_x)
    labels = np.ascontiguousarray(raw_y, dtype=np.int64)
    lib = _lib()
    lib.ds_campaign_solve.argtypes = [ct.c_void_p, ct.c_void_p] + [ct.c_int] * 11 + [ct.c_uint64, ct.c_double, ct.c_void_p]
    lib.ds_campaign_solve.restype = ct.c_char_p
    provider = _CUDA() if backend == "gpu" else None
    ee, full = variants[variant]
    start = time.perf_counter()
    try:
        raw = lib.ds_campaign_solve(x.ctypes.data, labels.ctypes.data, len(x), x.shape[1],
            int(depth), ee, full, int(threads), int(oracle_batch), int(max_columns),
            int(rmp_column_cap), int(bool(message_bound)), int(columns_per_block),
            int(memory_limit_bytes), float(time_limit),
            ct.cast(provider.callback, ct.c_void_p) if provider else None)
    finally:
        if provider:
            provider.close()
    if provider and provider.error:
        raise RuntimeError("CUDA campaign provider: " + provider.error)
    if raw is None:
        raise RuntimeError(lib.ds_error().decode())
    result = json.loads(raw)
    result["interface_wall_seconds"] = time.perf_counter() - start
    result["backend"] = backend
    result["stats"]["gpu_kernel_calls"] = provider.kernel_calls if provider else 0
    result["stats"]["gpu_shared_launches"] = provider.shared_calls if provider else 0
    result["stats"]["gpu_streaming_launches"] = provider.streaming_calls if provider else 0
    return result
