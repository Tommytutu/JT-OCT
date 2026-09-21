"""Data/serialization interface only; every optimization and audit runs in C++.

No Problem construction, Python dynamic program, NumPy pricing, message reduction,
tree scoring, or candidate enumeration is called by these entry points.
"""
import ctypes as ct
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_LIB = None
_DIRS = []
MEMORY = 48 * 1024**3
INACTIVE = -(2**31)
PROGRESS = ct.CFUNCTYPE(ct.c_int, ct.c_char_p)

def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    for folder in (os.environ.get('GUROBI_HOME'), r'C:\gurobi1300\win64'):
        if folder and (Path(folder)/'bin/gurobi130.dll').exists():
            _DIRS.append(os.add_dll_directory(str(Path(folder)/'bin')))
    lib = ct.CDLL(str(ROOT/'jt_oct/_native/revision_experiments.dll'))
    lib.revision_error.restype = ct.c_char_p
    lib.revision_unpack.argtypes = [ct.c_void_p]*3 + [ct.c_int]*3
    lib.revision_unpack.restype = ct.c_int
    lib.revision_data.argtypes = [ct.c_void_p]*5 + [ct.c_int]*7
    lib.revision_data.restype = ct.c_void_p
    lib.revision_data_free.argtypes = [ct.c_void_p]
    lib.revision_metadata.argtypes = [ct.c_void_p]
    lib.revision_metadata.restype = ct.c_char_p
    lib.revision_reference.argtypes = [ct.c_void_p, ct.c_double]
    lib.revision_reference.restype = ct.c_char_p
    lib.revision_seed.argtypes = [ct.c_void_p, ct.c_double]
    lib.revision_seed.restype = ct.c_char_p
    lib.revision_eager.argtypes = [ct.c_void_p, ct.c_double, ct.c_int, ct.c_double,
        ct.c_uint64, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]
    lib.revision_eager.restype = ct.c_char_p
    lib.revision_solve.argtypes = lib.revision_eager.argtypes + [ct.c_int, ct.c_int]
    lib.revision_solve.restype = ct.c_char_p
    lib.revision_d3.argtypes = [ct.c_void_p, ct.c_double, ct.c_int, ct.c_double,
                               ct.c_uint64, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_int]
    lib.revision_d3.restype = ct.c_char_p
    lib.revision_fixed.argtypes = [ct.c_void_p, ct.c_int, ct.c_double] + [ct.c_void_p]*5 + [ct.c_int]
    lib.revision_fixed.restype = ct.c_void_p
    lib.revision_fixed_free.argtypes = [ct.c_void_p]
    lib.revision_fixed_run.argtypes = [ct.c_void_p, ct.c_int, ct.c_double, ct.c_uint64] + [ct.c_int]*3
    lib.revision_fixed_run.restype = ct.c_char_p
    lib.revision_fixture.argtypes = [ct.c_void_p, ct.c_double]
    lib.revision_fixture.restype = ct.c_void_p
    _LIB = lib
    return lib


def checked_json(raw):
    if raw is None:
        raise RuntimeError(library().revision_error().decode())
    return json.loads(raw)


def pointer(array):
    return None if array is None else array.ctypes.data


def unpack_cache(matrix, threads=8):
    raw = np.asarray(matrix)
    if raw.ndim != 2 or raw.shape[1] < 2 or raw.dtype.kind not in 'iu' or raw.dtype == np.uint64:
        raise ValueError('Frozen cache must be an integer [label, binary features] matrix')
    raw = np.ascontiguousarray(raw, dtype=np.int64)
    X = np.empty((len(raw), raw.shape[1]-1), dtype=np.uint8)
    y = np.empty(len(raw), dtype=np.int64)
    if library().revision_unpack(pointer(raw), pointer(X), pointer(y), len(raw), raw.shape[1], threads):
        raise ValueError(library().revision_error().decode())
    return X, y


class NativeData:
    def __init__(self, X, y, depth, *, early=True, no_repeat=True, min_leaf=0,
                 threads=8, weights=None, allowed=None, extras=None):
        self.lib = library()
        raw_x, raw_y = np.asarray(X), np.asarray(y)
        if raw_x.dtype != np.uint8 or raw_y.dtype.kind not in 'iu' or raw_y.dtype == np.uint64:
            raise ValueError('Use uint8 features and signed integer labels; narrowing is forbidden')
        self.X = np.ascontiguousarray(raw_x)
        self.y = np.ascontiguousarray(raw_y, dtype=np.int64)
        if self.X.ndim != 2 or self.y.shape != (len(self.X),):
            raise ValueError('Data shapes do not match')
        self.n, self.F = self.X.shape
        self.depth = depth
        self.threads = threads
        self.early = early
        self.weights = None if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
        self.allowed = None if allowed is None else np.ascontiguousarray(allowed, dtype=np.uint8)
        self.extras = None if extras is None else np.ascontiguousarray(extras, dtype=np.float64)
        if self.weights is not None and self.weights.shape != (self.n,):
            raise ValueError('Weight shape mismatch')
        for name in ('allowed', 'extras'):
            a = getattr(self, name)
            if a is not None and a.shape != ((1 << depth)-1, self.F):
                raise ValueError(name+' shape mismatch')
        self.handle = self.lib.revision_data(pointer(self.X), pointer(self.y), pointer(self.weights),
            pointer(self.allowed), pointer(self.extras), self.n, self.F, depth, int(early),
            int(no_repeat), min_leaf, threads)
        if not self.handle:
            raise RuntimeError(self.lib.revision_error().decode())
        self.metadata = checked_json(self.lib.revision_metadata(self.handle))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.handle:
            self.lib.revision_data_free(self.handle)
            self.handle = None

    def reference(self, penalty=0.):
        self.require_open()
        return checked_json(self.lib.revision_reference(self.handle, penalty))

    def require_open(self):
        if not self.handle:
            raise ValueError('Native data is closed')

    def seed(self, penalty=0.):
        self.require_open()
        return checked_json(self.lib.revision_seed(self.handle, penalty))

    def eager(self, method='lp_sc_ee', *, penalty=0., backend='cpp', seconds=600,
              memory=MEMORY, cap=2_000_000, presolve=1, lp_method=0, batch=16, checkpoint=None,
              private_depth=None, lazy=False):
        self.require_open()
        if self.depth == 3 and private_depth is None and not lazy and method in ('lp_sc','lp_sc_ee','mp'):
            seed = self.seed(penalty) if self.early else None
            if checkpoint and seed is not None:
                checkpoint(seed)
            result = self.d3(method, penalty=penalty, backend=backend, seconds=seconds,
                memory=memory, cap=cap, presolve=presolve, lp_method=lp_method, batch=batch)
            if seed is not None and result.get('UB') is None and result.get('status') in ('TIME', 'MEM', 'COLUMN_LIMIT'):
                return {**seed, **result, 'UB': seed['UB'], 'tree': seed['tree'],
                    'tree_objective': seed['UB'], 'tree_audit_passed': True, 'certificate_full_domain': False}
            return result
        kind = {'lp_sc': 0, 'lp_sc_ee': 1, 'mp': 2, 'cg': 3, 'cg_ee': 4}[method]
        pd = private_depth if private_depth is not None else (1 if self.depth<=3 else 3)
        provider = None
        if backend == 'gpu':
            if pd == 1:
                from .d3_cg_gpu import GPUCosts
                provider = GPUCosts(SimpleNamespace(n=self.n, F=self.F, labels=range(self.metadata['K']),
                    min_leaf=self.metadata['min_leaf'], _uniform_weight=self.metadata['uniform_weight']))
            else:
                from .deep_campaign import _CUDA
                provider = _CUDA()
        elif backend != 'cpp':
            raise ValueError('Backend must be cpp or gpu')
        callback_error = []
        def emit(raw):
            try:
                checkpoint(json.loads(raw))
                return 0
            except Exception as exc:
                callback_error.append(exc)
                return 1
        cb = PROGRESS(emit) if checkpoint else None
        ptr = None if provider is None else ct.cast(provider.callback, ct.c_void_p)
        try:
            raw = self.lib.revision_solve(self.handle, penalty, kind, seconds, memory, cap,
                presolve, lp_method, ptr if pd != 1 else None, ptr if pd == 1 else None,
                batch, None if cb is None else ct.cast(cb, ct.c_void_p), pd, int(lazy))
            if callback_error:
                raise callback_error[0]
            if provider is not None and provider.error:
                raise RuntimeError(provider.error)
            result = checked_json(raw)
            if provider is not None:
                result['gpu_stats'] = getattr(provider, 'stats', {'kernel_calls': provider.kernel_calls} if hasattr(provider, 'kernel_calls') else {})
            return result
        finally:
            if provider is not None and hasattr(provider, 'close'):
                provider.close()

    def d3(self, method='mp', *, penalty=0., backend='cpp', seconds=600,
           memory=MEMORY, cap=2_000_000, presolve=1, lp_method=0, batch=256):
        self.require_open()
        kind = {'lp_sc': 0, 'lp_sc_ee': 1, 'mp': 2}[method]
        provider = None
        if backend == 'gpu':
            from .d3_cg_gpu import GPUCosts
            meta = self.metadata
            if meta['uniform_weight'] is None:
                raise ValueError('CUDA cost kernel requires uniform weights; no silent fallback')
            provider = GPUCosts(SimpleNamespace(n=self.n, F=self.F, labels=range(meta['K']),
                min_leaf=meta['min_leaf'], _uniform_weight=meta['uniform_weight']))
        elif backend != 'cpp':
            raise ValueError('Backend must be cpp or gpu')
        result = checked_json(self.lib.revision_d3(self.handle, penalty, kind, seconds,
            memory, cap, presolve, lp_method, None if provider is None else ct.cast(provider.callback, ct.c_void_p), batch))
        if provider is not None:
            if provider.error:
                raise RuntimeError(provider.error)
            result['gpu_stats'] = provider.stats
        return result


class FixedTable:
    def __init__(self, data, handle):
        if not handle:
            raise RuntimeError(library().revision_error().decode())
        self.data, self.handle = data, handle

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.handle:
            library().revision_fixed_free(self.handle)
            self.handle = None

    @classmethod
    def fixture(cls, data, penalty=0.):
        data.require_open()
        return cls(data, library().revision_fixture(data.handle, penalty))

    @classmethod
    def from_arrays(cls, data, M, high, low, private_ids, tails, exact, penalty=0.):
        data.require_open()
        # Only dtype/layout marshaling; completeness and dimensions are checked natively.
        arrays = [np.ascontiguousarray(high, dtype=np.float64), np.ascontiguousarray(low, dtype=np.float64),
                  np.ascontiguousarray(private_ids, dtype=np.int32), np.ascontiguousarray(tails, dtype=np.int32),
                  np.ascontiguousarray(exact, dtype=np.uint8)]
        P = data.F+data.metadata['K'] if M == 2 else data.F*(data.F+data.metadata['K'])+data.metadata['K']
        if any(a.shape != (M*P,) for a in arrays[:3]) or arrays[3].shape != (len(arrays[4]), 15):
            raise ValueError('Serialized exact table shape mismatch')
        handle = library().revision_fixed(data.handle, M, penalty, *(pointer(a) for a in arrays), len(arrays[4]))
        return cls(data, handle)

    def run(self, method='ee_lp', seconds=60., memory=MEMORY, cap=2_000_000, presolve=1, lp_method=0):
        self.data.require_open()
        if not self.handle:
            raise ValueError('Native table is closed')
        kind = {'lp': 0, 'cg': 1, 'mp': 2, 'ee_lp': 3, 'ee_cg': 4}[method]
        return checked_json(library().revision_fixed_run(self.handle, kind, seconds, memory, cap, presolve, lp_method))

