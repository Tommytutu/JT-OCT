"""D3 C++ / Gurobi CG, reusing the validated D2 data/result adapter.

Four numerical column blocks preserve full [2,1,2] separator signatures.
The C ABI has the same call shape as D2; D3 owns a separate native workspace.
"""
import ctypes as ct
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from .d2_cg import D2CGWorkspace, _BYTE, _INT, _DOUBLE, _CALLBACK
from .problem import Deadline

_LIBRARY = None
_LIBRARY_LOCK = threading.Lock()
_DLL_DIRS = []


def native_available():
    return (Path(__file__).parent / "_native" / "d3_cg.dll").is_file()


def _library():
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        path = Path(__file__).parent / "_native" / "d3_cg.dll"
        if not path.exists():
            raise RuntimeError("Build the Gurobi 13 D3 library with build_d3_cg.ps1 first")
        if os.name == "nt":
            for root in [os.environ.get("GUROBI_HOME"), r"C:\gurobi1300\win64"]:
                if root and (Path(root) / "bin" / "gurobi130.dll").exists():
                    _DLL_DIRS.append(os.add_dll_directory(str(Path(root) / "bin")))
        lib = ct.CDLL(str(path))
        lib.d3cg_create.argtypes = [_BYTE, _INT, _DOUBLE, _BYTE, _DOUBLE] + [ct.c_int] * 7
        lib.d3cg_create.restype = ct.c_void_p
        lib.d3cg_destroy.argtypes = [ct.c_void_p]
        lib.d3cg_destroy.restype = None
        lib.d3cg_error.argtypes = []
        lib.d3cg_error.restype = ct.c_char_p
        lib.d3cg_solve.argtypes = [ct.c_void_p, ct.c_double, ct.c_double,
                                  ct.c_int, ct.c_int, ct.c_int, _CALLBACK]
        lib.d3cg_solve.restype = ct.c_char_p
        lib.d3cg_price.argtypes = [ct.c_void_p, ct.c_double, _DOUBLE, _DOUBLE]
        lib.d3cg_price.restype = ct.c_char_p
        lib.d3cg_configure.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
        lib.d3cg_configure.restype = ct.c_int
        lib.d3cg_cost_provider.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.d3cg_cost_provider.restype = None
        lib.d3cg_prepare_costs.argtypes = [ct.c_void_p, ct.c_double]
        lib.d3cg_prepare_costs.restype = ct.c_char_p
        lib.d3cg_direct_cost_provider.argtypes = [ct.c_void_p,ct.c_void_p]
        lib.d3cg_direct_cost_provider.restype = None
        # Adapter compatibility names only: every function here calls d3cg_*,
        # and the D2 DLL/solver are neither loaded nor used by this workspace.
        _LIBRARY = SimpleNamespace(owner=lib, **{
            f"d2cg_{name}": getattr(lib, f"d3cg_{name}")
            for name in ("create", "destroy", "error", "solve", "price")})
        return _LIBRARY


class D3CGWorkspace(D2CGWorkspace):
    """Fixed data/constraints with reusable penalty-independent min_h costs.

    A signature is SPLIT(f), action(g or STOP label), or STOP(root label), DEAD.
    Native numeric ids are f*(F+K)+action, or F*(F+K)+root_label_index.
    """
    _depth = 3
    _method = "JT-CG-D3-CPP"
    _nodes = ((), (0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1))
    _load_library = staticmethod(_library)

    def __init__(self, problem, threads=1, *, cost_mode="auto", cost_batch_size=0,
                 cost_cache_mb=128, cost_backend="auto", full_gpu_strategy="direct"):
        """auto uses lazy costs when F>=128 and estimated dense work is large.

        lazy computes bounded batches and certifies uncomputed costs with lower
        bounds. eager retains the original full-table implementation for audits.
        Cache entries are routed-row statistics; constraints are applied on reuse.
        """
        from .d2_cg import _integer
        if cost_mode not in ("auto", "eager", "lazy"):
            raise ValueError("cost_mode must be auto, eager, or lazy")
        if cost_backend not in ("auto", "cpp", "gpu"):
            raise ValueError("cost_backend must be auto, cpp, or gpu")
        if full_gpu_strategy not in ('direct','cached'):
            raise ValueError('full_gpu_strategy must be direct or cached')
        cost_batch_size = _integer(cost_batch_size, 0, "cost_batch_size")
        cost_cache_mb = _integer(cost_cache_mb, 0, "cost_cache_mb")
        super().__init__(problem, threads=threads)
        if not self._lib.owner.d3cg_configure(self._handle,
                {"eager": 0, "lazy": 1, "auto": 2}[cost_mode], cost_batch_size, cost_cache_mb):
            self.close()
            raise RuntimeError("Invalid native D3 cost configuration")
        self._gpu_costs = None
        self._full_gpu = cost_mode == 'eager' and cost_backend == 'gpu'
        self._full_gpu_ready = False
        self.cost_backend = cost_backend
        large = (problem.F >= 128 and 4.*problem.F**3*((problem.n+63)//64)*
                 max(1, len(problem.labels)-1) > 2e9)
        if cost_backend == "gpu" or (cost_backend == "auto" and cost_mode != "eager"
                                    and large and problem._uniform_weight is not None):
            from .d3_cg_gpu import GPUCosts
            try:
                if self._full_gpu and full_gpu_strategy=='direct':
                    from .d3_cg_gpu_direct import DirectGPUCosts
                    self._gpu_costs = DirectGPUCosts(self._p)
                    self._lib.owner.d3cg_direct_cost_provider(self._handle,
                        ct.cast(self._gpu_costs.direct_callback,ct.c_void_p))
                else:
                    self._gpu_costs = GPUCosts(self._p, allow_fallback=cost_backend == "auto")
                    self._lib.owner.d3cg_cost_provider(self._handle,
                        ct.cast(self._gpu_costs.callback, ct.c_void_p))
                if cost_backend == "gpu" and not self._full_gpu:
                    self._lib.owner.d3cg_configure(self._handle, 1, cost_batch_size, cost_cache_mb)
            except Exception:
                self.close()
                raise

    def solve(self, penalty=None, time_limit=600, batch_size=0, max_columns=200000,
              rmp_method=0, progress=None, *, _clock=None):
        clock = _clock if _clock is not None else Deadline(time_limit)
        prep = None
        try:
            if self._full_gpu:
                import copy
                from .d2_cg import _integer
                from .problem import Tree
                from .solvers import result_dict
                p = copy.copy(self._p)
                p.penalty = p.penalty if penalty is None else float(penalty)
                if not math.isfinite(p.penalty) or p.penalty < 0:
                    raise ValueError('penalty must be finite and nonnegative')
                _integer(batch_size,0,'batch_size');_integer(max_columns,1,'max_columns')
                if rmp_method not in (-1,0,1,2):raise ValueError('Invalid rmp_method')
                seed = None
                if p.early_stop and p.n >= p.min_leaf:
                    label, loss = p.best_label_and_loss(p.all_rows);seed = Tree(label=label)
                    if progress:
                        progress(dict(method=self._method,status='RUNNING',seconds=clock.elapsed(),
                                      LB=0.,UB=loss,tree=seed.to_dict(),phase='full_gpu_costs'))
                first_seed = clock.elapsed()
                with self._lock:
                    if not self._handle:raise RuntimeError('D3 workspace is closed')
                    prep = self._decode(self._lib.owner.d3cg_prepare_costs(self._handle,
                        max(0.,clock.end-time.perf_counter())))
                if prep['status'] == 'TIME':
                    result = result_dict(self._method,clock,'TIME',seed,p,0.,trace=[],certificate=None,
                        first_final_ub_seconds=first_seed if seed else None,first_optimal_solution_seconds=None,
                        proof_seconds=None,native_seconds=0.,pricing_setup_seconds=0.,pricing_seconds=0.,
                        rmp_seconds=0.,environment_seconds=0.,iterations=0,columns=0,rows=0,stats={},
                        backend='gurobi_cpp',pricing_threads=self.threads)
                else:
                    result = super().solve(penalty,time_limit,batch_size,max_columns,rmp_method,progress,_clock=clock)
                result['workspace_reused'] = self._full_gpu_ready
                self._full_gpu_ready = prep['status'] == 'PREPARED'
                result['native_seconds'] += prep['prepare_seconds']
                result['pricing_setup_seconds'] += prep['prepare_seconds']
                result['full_cost_preparation_seconds'] = prep['prepare_seconds']
                result.update({k:v for k,v in prep.items() if k not in ('status','prepare_seconds')})
                result['cost_mode'] = 'eager_gpu'
            else:
                result = super().solve(penalty,time_limit,batch_size,max_columns,rmp_method,progress,_clock=clock)
        except RuntimeError as exc:
            if self._gpu_costs is not None and self._gpu_costs.error:
                raise RuntimeError("D3 GPU costs: " + self._gpu_costs.error) from exc
            raise
        if self._gpu_costs is not None:
            result.update(self._gpu_costs.stats)
        result['stats']['pricing_setup_seconds'] = result['pricing_setup_seconds']
        for key in ("initial_setup_seconds", "cost_evaluation_seconds", "first_rmp_seconds",
                    "cost_states_resolved", "cost_states_total", "cost_cache_hits",
                    "cost_cache_misses", "cost_cache_peak_bytes", "stump_evaluations",
                    "cost_word_visits", "cost_batches", "message_lower_bound",
                    "master_update_seconds", "callback_seconds", "cost_batch_final",
                    "gpu_cost_seconds", "gpu_cost_batches", "gpu_setup_seconds",
                    "gpu_batch_seconds", "gpu_peak_batch_rows", "full_cost_preparation_seconds",
                    "gpu_h2d_seconds", "gpu_kernel_seconds", "gpu_d2h_seconds",
                    "gpu_descriptor_bytes", "gpu_result_bytes"):
            if key in result:
                result["stats"][key] = result[key]
        result.setdefault("cost_mode", "eager")
        result.setdefault("cost_backend", "cpp_openmp")
        result["requested_cost_backend"] = self.cost_backend
        result['interface_seconds'] = max(0.,result['seconds']-result['native_seconds'])
        return result

    def price(self, alpha, separator_duals, penalty=None):
        """Complete pricing audit; dual layout is [left prefix, root, right prefix]."""
        A = self._p.F + len(self._labels)
        P = self._p.F * A + len(self._labels)
        alpha = np.ascontiguousarray(alpha, dtype=np.float64)
        dual = np.ascontiguousarray(separator_duals, dtype=np.float64)
        penalty = self._p.penalty if penalty is None else float(penalty)
        if alpha.shape != (4,) or dual.shape != (2 * P + A,):
            raise ValueError("Incorrect D3 dual dimensions")
        if not np.isfinite(alpha).all() or not np.isfinite(dual).all() or not math.isfinite(penalty) or penalty < 0:
            raise ValueError("Finite duals and nonnegative finite penalty required")
        with self._lock:
            if not self._handle:
                raise RuntimeError("D3 workspace is closed")
            return self._decode(self._lib.d2cg_price(self._handle, penalty,
                alpha.ctypes.data_as(_DOUBLE), dual.ctypes.data_as(_DOUBLE)))


def solve_d3_cg(problem, time_limit=600, *, threads=1, batch_size=0,
                max_columns=200000, rmp_method=0, progress=None, cost_mode="auto",
                cost_batch_size=0, cost_cache_mb=128, cost_backend="auto", full_gpu_strategy="direct"):
    """One-shot D3 CG: preparation and final routed tree audit are timed."""
    clock = Deadline(time_limit)
    with D3CGWorkspace(problem, threads=threads, cost_mode=cost_mode,
                       cost_batch_size=cost_batch_size, cost_cache_mb=cost_cache_mb,
                       cost_backend=cost_backend,full_gpu_strategy=full_gpu_strategy) as workspace:
        result = workspace.solve(time_limit=time_limit, batch_size=batch_size,
            max_columns=max_columns, rmp_method=rmp_method, progress=progress, _clock=clock)
    result["seconds"] = clock.elapsed()
    result["interface_seconds"] = max(0., result["seconds"] - result["native_seconds"])
    return result
