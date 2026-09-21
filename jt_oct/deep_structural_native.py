"""Thin ctypes interface to the native D5--D7 structural coordinator."""
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
        for root in (os.environ.get("GUROBI_HOME"), r"C:\gurobi1300\win64"):
            if root and (Path(root) / "bin" / "gurobi130.dll").exists():
                _DIRS.append(os.add_dll_directory(str(Path(root) / "bin")))
        lib = ct.CDLL(str(Path(__file__).parent / "_native" / "deep_structural_core.dll"))
        lib.ds_error.restype = ct.c_char_p
        lib.ds_state_count.argtypes = [ct.c_int] * 3; lib.ds_state_count.restype = ct.c_int64
        lib.ds_project.argtypes = [ct.c_int] * 5; lib.ds_project.restype = ct.c_int
        lib.ds_path_opt.argtypes = [ct.c_int] * 5 + [ct.c_void_p, ct.c_void_p, ct.c_void_p]; lib.ds_path_opt.restype = ct.c_int
        lib.ds_eliminate_endpoints.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]; lib.ds_eliminate_endpoints.restype = ct.c_int
        lib.ds_estimate.argtypes = [ct.c_int] * 5; lib.ds_estimate.restype = ct.c_char_p
        lib.ds_create.argtypes = [ct.c_int] * 6 + [ct.c_uint64]; lib.ds_create.restype = ct.c_void_p
        lib.ds_update.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]; lib.ds_update.restype = ct.c_int
        lib.ds_solve.argtypes = [ct.c_void_p, ct.c_double]; lib.ds_solve.restype = ct.c_char_p
        lib.ds_price.argtypes = [ct.c_void_p] + [ct.c_void_p] * 5; lib.ds_price.restype = ct.c_int
        lib.ds_destroy.argtypes = [ct.c_void_p]
        _LIB = lib
    return _LIB


def state_count(features, classes, upper_depth):
    value = _lib().ds_state_count(features, classes, upper_depth)
    if value < 0:
        raise ValueError(_lib().ds_error().decode())
    return int(value)


def project(features, classes, upper_depth, state, separator_depth):
    value = _lib().ds_project(features, classes, upper_depth, state, separator_depth)
    if value < 0:
        raise ValueError(_lib().ds_error().decode())
    return value


def estimate(blocks, features, classes, upper_depth, eliminate=False):
    raw = _lib().ds_estimate(blocks, features, classes, upper_depth, int(eliminate))
    if raw is None:
        raise ValueError(_lib().ds_error().decode())
    return json.loads(raw)


def path_opt(values, features, classes, upper_depth, first_block=0):
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != state_count(features, classes, upper_depth):
        raise ValueError("Path costs must have the complete signature width")
    selected = np.empty(values.shape[0], dtype=np.int32); objective = ct.c_double()
    code = _lib().ds_path_opt(values.shape[0], features, classes, upper_depth, first_block,
        values.ctypes.data, selected.ctypes.data, ct.byref(objective))
    if code < 0:
        raise RuntimeError(_lib().ds_error().decode())
    return objective.value, None if code else selected


def eliminate_endpoints(values):
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Endpoint elimination requires an M by P array with M >= 3")
    out = np.empty((values.shape[0] - 2, values.shape[1]), dtype=np.float64)
    if _lib().ds_eliminate_endpoints(values.shape[0], values.shape[1], values.ctypes.data, out.ctypes.data):
        raise RuntimeError(_lib().ds_error().decode())
    return out


class DeepMaster:
    def __init__(self, blocks, features, classes, upper_depth, first_block=0,
                 method=1, memory_limit_bytes=48 * 1024**3):
        self.lib = _lib(); self.handle = None
        self.P = state_count(features, classes, upper_depth); self.blocks = blocks
        self.R = estimate(blocks, features, classes, upper_depth)["rows"] - blocks if first_block == 0 else sum(
            state_count(features, classes, upper_depth - ((edge + 1) & -(edge + 1)).bit_length() + 1)
            for edge in range(first_block, first_block + blocks - 1))
        self.handle = self.lib.ds_create(blocks, features, classes, upper_depth,
            first_block, method, int(memory_limit_bytes))
        if not self.handle:
            raise RuntimeError(self.lib.ds_error().decode())

    def update(self, ids, costs):
        raw_ids = np.asarray(ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu" or (raw_ids.size and
                (raw_ids.min() < 0 or raw_ids.max() >= self.blocks * self.P)):
            raise ValueError("Column ids outside native master domain")
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        costs = np.ascontiguousarray(costs, dtype=np.float64)
        if ids.shape != costs.shape:
            raise ValueError("Mismatched native deep RMP arrays")
        if self.lib.ds_update(self.handle, ids.size, ids.ctypes.data, costs.ctypes.data):
            raise RuntimeError(self.lib.ds_error().decode())

    def solve(self, seconds):
        raw = self.lib.ds_solve(self.handle, float(seconds))
        if raw is None:
            raise RuntimeError(self.lib.ds_error().decode())
        return json.loads(raw)

    def price(self, costs, alpha, pi):
        costs = np.ascontiguousarray(costs, dtype=np.float64).reshape(self.blocks, self.P)
        alpha = np.ascontiguousarray(alpha, dtype=np.float64)
        pi = np.ascontiguousarray(pi, dtype=np.float64)
        if alpha.shape != (self.blocks,) or pi.shape != (self.R,):
            raise ValueError("Incorrect dual vector dimensions")
        rc = np.empty_like(costs); minima = np.empty(self.blocks, dtype=np.float64)
        if self.lib.ds_price(self.handle, costs.ctypes.data, alpha.ctypes.data,
                             pi.ctypes.data, rc.ctypes.data, minima.ctypes.data) < 0:
            raise RuntimeError(self.lib.ds_error().decode())
        return rc, minima

    def close(self):
        if self.handle:
            self.lib.ds_destroy(self.handle); self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
