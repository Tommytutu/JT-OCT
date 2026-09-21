"""Thin ctypes boundary for the C++ endpoint-elimination kernel.

All O(P) EE operations and the D5 Gurobi restricted master stay in the native
module.  This file deliberately contains no pricing or aggregation loop.
"""
import ctypes as ct
import json
import os
from pathlib import Path
import numpy as np

_LIB = None
_DIRS = []


def _lib():
    global _LIB
    if _LIB is None:
        for root in (os.environ.get('GUROBI_HOME'), r'C:\gurobi1300\win64'):
            if root and (Path(root) / 'bin/gurobi130.dll').exists():
                _DIRS.append(os.add_dll_directory(str(Path(root) / 'bin')))
        lib = ct.CDLL(str(Path(__file__).parent / '_native' / 'contract_ee_rmp.dll'))
        lib.ee_error.restype = ct.c_char_p
        lib.ee_sum_pairs.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p]
        lib.ee_best_single.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p]; lib.ee_best_single.restype = ct.c_int
        lib.ee_best_join.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p]; lib.ee_best_join.restype = ct.c_int
        lib.ee_admit.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]; lib.ee_admit.restype = ct.c_int
        lib.ee_create.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_int]; lib.ee_create.restype = ct.c_void_p
        lib.ee_update.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]; lib.ee_update.restype = ct.c_int
        lib.ee_solve.argtypes = [ct.c_void_p, ct.c_double]; lib.ee_solve.restype = ct.c_char_p
        lib.ee_price.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_void_p, ct.c_void_p, ct.c_void_p]; lib.ee_price.restype = ct.c_int
        lib.ee_destroy.argtypes = [ct.c_void_p]
        _LIB = lib
    return _LIB


def _f64(values, shape=None):
    a = np.ascontiguousarray(values, dtype=np.float64)
    return a.reshape(shape) if shape else a


def sum_pairs(values, P):
    src = _f64(values, (4, P)); out = np.empty((2, P), dtype=np.float64)
    _lib().ee_sum_pairs(P, src.ctypes.data, out.ctypes.data)
    return out


def best_single(values):
    v = _f64(values); chosen = ct.c_int(-1); best = ct.c_double()
    if _lib().ee_best_single(v.size, v.ctypes.data, ct.byref(chosen), ct.byref(best)):
        return float('inf'), None
    return best.value, [chosen.value]


def best_join(values, roots, A):
    v = _f64(values); r = np.ascontiguousarray(roots, dtype=np.int32)
    chosen = np.empty(2, dtype=np.int32); best = ct.c_double()
    if _lib().ee_best_join(v.shape[1], A, r.ctypes.data, v.ctypes.data, chosen.ctypes.data, ct.byref(best)):
        return float('inf'), None
    return best.value, chosen.tolist()


def admit(rc, active, cap):
    rc = _f64(rc); active = np.ascontiguousarray(active, dtype=np.uint8)
    out = np.empty(rc.size, dtype=np.int32)
    n = _lib().ee_admit(rc.shape[1], rc.ctypes.data, active.ctypes.data, cap, out.ctypes.data)
    return out[:n]


class EERMP:
    def __init__(self, P, A, roots, method=1):
        self.lib = _lib(); self.handle = None
        self.roots = np.ascontiguousarray(roots, dtype=np.int32)
        self.handle = self.lib.ee_create(P, A, self.roots.ctypes.data, method)
        if not self.handle: raise RuntimeError(self.lib.ee_error().decode())

    def update(self, ids, costs):
        ids = np.ascontiguousarray(ids, dtype=np.int32); costs = _f64(costs)
        if ids.shape != costs.shape: raise ValueError('Mismatched EE RMP arrays')
        if self.lib.ee_update(self.handle, len(ids), ids.ctypes.data, costs.ctypes.data):
            raise RuntimeError(self.lib.ee_error().decode())

    def solve(self, seconds):
        raw = self.lib.ee_solve(self.handle, seconds)
        if raw is None: raise RuntimeError(self.lib.ee_error().decode())
        return json.loads(raw)

    def price(self, costs, alpha, pi):
        costs = _f64(costs, (2, self.roots.size)); pi = _f64(pi)
        rc = np.empty_like(costs); minimum = np.empty(2, dtype=np.float64)
        if self.lib.ee_price(self.handle, costs.ctypes.data, alpha, pi.ctypes.data, rc.ctypes.data, minimum.ctypes.data) < 0:
            raise RuntimeError(self.lib.ee_error().decode())
        return rc, minimum

    def close(self):
        if self.handle: self.lib.ee_destroy(self.handle); self.handle = None
