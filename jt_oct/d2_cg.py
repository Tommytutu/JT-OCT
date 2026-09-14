"""Thin data/result adapter for native D2 column generation and Gurobi C++.

All local costs, pricing, column management, LP solves and certification are
native. Python validates input through Problem and independently audits the
returned tree. A workspace caches penalty-independent stump costs and its
Gurobi environment across solves of the same fixed data/constraints.
"""
import copy
import ctypes as ct
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from .problem import Deadline, Tree
from .solvers import result_dict

_DLL = None
_DLL_DIRS = []
_LOAD_LOCK = threading.Lock()
_CALLBACK = ct.CFUNCTYPE(None, ct.c_char_p)
_BYTE = ct.POINTER(ct.c_uint8)
_INT = ct.POINTER(ct.c_int32)
_DOUBLE = ct.POINTER(ct.c_double)
_WORD = ct.POINTER(ct.c_uint64)


def native_available():
    return (Path(__file__).parent / "_native" / "d2_cg.dll").is_file()


def _library():
    global _DLL
    with _LOAD_LOCK:
        if _DLL is not None:
            return _DLL
        path = Path(__file__).parent / "_native" / "d2_cg.dll"
        if not path.exists():
            raise RuntimeError("Build the Gurobi 13 D2 library with build_d2_cg.ps1 first")
        if os.name == "nt":
            for root in [os.environ.get("GUROBI_HOME"), r"C:\gurobi1300\win64"]:
                if root and (Path(root) / "bin" / "gurobi130.dll").exists():
                    _DLL_DIRS.append(os.add_dll_directory(str(Path(root) / "bin")))
        lib = ct.CDLL(str(path))
        lib.d2cg_create.argtypes = [_BYTE, _INT, _DOUBLE, _BYTE, _DOUBLE] + [ct.c_int] * 7
        lib.d2cg_create.restype = ct.c_void_p
        lib.d2cg_destroy.argtypes = [ct.c_void_p]
        lib.d2cg_destroy.restype = None
        lib.d2cg_error.argtypes = []
        lib.d2cg_error.restype = ct.c_char_p
        lib.d2cg_solve.argtypes = [ct.c_void_p, ct.c_double, ct.c_double,
                                  ct.c_int, ct.c_int, ct.c_int, _CALLBACK]
        lib.d2cg_solve.restype = ct.c_char_p
        lib.d2cg_price.argtypes = [ct.c_void_p, ct.c_double, _DOUBLE, _DOUBLE]
        lib.d2cg_price.restype = ct.c_char_p
        lib.d2cg_set_packed.argtypes = [ct.c_void_p, _WORD, _WORD]
        lib.d2cg_set_packed.restype = ct.c_int
        lib.d2cg_cost_provider.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.d2cg_cost_provider.restype = None
        _DLL = lib
        return lib


def _integer(value, minimum, name):
    if not isinstance(value, (int, np.integer)) or not minimum <= value <= 2**31 - 1:
        raise ValueError(f"{name} must be an integer >= {minimum} fitting int32")
    return int(value)


class D2CGWorkspace:
    """Fixed D2 problem; solve repeatedly with different nonnegative penalties.

    The workspace owns snapshots of input arrays. It must be recreated when
    observations, weights, allowed features, or other constraints change.
    threads=0 chooses at most eight CPU cost workers by estimated work;
    Gurobi uses one CPU thread. A resident GPU runtime is eligible in auto mode.
    """

    _depth = 2
    _method = "JT-CG-D2-CPP"
    _nodes = ((), (0,), (1,))
    _load_library = staticmethod(_library)

    def __init__(self, problem, threads=0, *, cost_backend="auto",
                 gpu_strategy="pair", reuse_problem_masks=True):
        construction_started = time.perf_counter()
        if problem.depth != self._depth:
            raise ValueError(f"Native D{self._depth} CG requires depth={self._depth}")
        if cost_backend not in ("auto", "cpp", "gpu"):
            raise ValueError("cost_backend must be auto, cpp, or gpu")
        if gpu_strategy not in ("pair", "generic"):
            raise ValueError("gpu_strategy must be pair or generic")
        threads = _integer(threads, 0, "threads")
        work = problem.F*problem.F*((problem.n+63)//64)*len(problem.labels)
        if threads == 0:
            threads = min(8,os.cpu_count() or 1) if work >= 2_000_000 else 1
        self._handle = None
        self._lock = threading.Lock()
        self._p = copy.copy(problem)
        self._p.X = np.array(problem.X, dtype=np.uint8, order="C", copy=True)
        self._p.y = np.array(problem.y, copy=True)
        self._p.weights = np.array(problem.weights, dtype=np.float64, copy=True)
        self._p.allowed = {node: tuple(fs) for node, fs in problem.allowed.items()}
        self._p.split_costs = dict(problem.split_costs)
        self._p.X.flags.writeable = self._p.y.flags.writeable = self._p.weights.flags.writeable = False
        self._labels = problem.labels
        encoded = np.searchsorted(self._labels, self._p.y).astype(np.int32)
        allowed = np.zeros((len(self._nodes), problem.F), dtype=np.uint8)
        extras = np.zeros((len(self._nodes), problem.F), dtype=np.float64)
        for j, node in enumerate(self._nodes):
            allowed[j, list(self._p.features(node, ()))] = 1
            for f in range(problem.F):
                extras[j, f] = self._p.split_costs.get((node, f), 0.)
        self._arrays = (self._p.X, encoded, self._p.weights, allowed, extras)
        self._lib = self._load_library()
        self._handle = self._lib.d2cg_create(
            self._p.X.ctypes.data_as(_BYTE), encoded.ctypes.data_as(_INT),
            self._p.weights.ctypes.data_as(_DOUBLE), allowed.ctypes.data_as(_BYTE),
            extras.ctypes.data_as(_DOUBLE), problem.n, problem.F, len(self._labels),
            problem.min_leaf, int(problem.early_stop), int(problem.no_repeat), threads)
        if not self._handle:
            raise RuntimeError(self._lib.d2cg_error().decode())
        self.threads = threads
        # D3 subclasses retain their own native cost providers and policy.
        if self._depth == 2:
            self._gpu_costs = None
            self._adapter_stats = dict(mask_export_seconds=0., reuse_problem_masks=bool(reuse_problem_masks))
            if reuse_problem_masks:
                started = time.perf_counter()
                W = (problem.n+63)//64
                # Problem has already packed these masks. Exporting bytes is
                # O((F+K)*W), avoiding another O(n*F) row-major data scan.
                feature_words = np.frombuffer(b"".join(
                    masks[1].to_bytes(W*8,"little") for masks in self._p._feature_masks),dtype="<u8")
                class_words = np.frombuffer(b"".join(
                    self._p._label_masks[k].to_bytes(W*8,"little") for k in self._labels),dtype="<u8")
                if not self._lib.d2cg_set_packed(self._handle,
                        feature_words.ctypes.data_as(_WORD),class_words.ctypes.data_as(_WORD)):
                    self.close()
                    raise RuntimeError(self._lib.d2cg_error().decode())
                self._adapter_stats["mask_export_seconds"] = time.perf_counter()-started
            # Work-class policy, never a dataset-name lookup. Cold CUDA startup
            # is expensive at D2; reuse a resident runtime only when work is ample.
            self._cost_backend_requested = cost_backend
            from .d2_cg_gpu import runtime_ready
            gpu_warm = runtime_ready()
            selected = cost_backend
            if selected == "auto":
                selected = "gpu" if (problem._uniform_weight is not None and
                    ((gpu_warm and work >= 20_000_000) or
                     (not gpu_warm and work >= 10_000_000_000))) else "cpp"
            self._adapter_stats.update(backend_policy="d2-work-v1",estimated_pair_word_class_work=work,
                gpu_runtime_was_ready=gpu_warm)
            self._cost_backend = selected
            if selected == "gpu":
                self._method = "JT-CG-D2-GPU"
                from .d2_cg_gpu import D2GPUCosts
                try:
                    self._gpu_costs = D2GPUCosts(self._p,gpu_strategy,cost_backend == "auto")
                    self._lib.d2cg_cost_provider(self._handle,self._gpu_costs.callback)
                except Exception:
                    self.close()
                    raise
            self._adapter_stats["workspace_construction_seconds"] = time.perf_counter()-construction_started

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.d2cg_destroy(self._handle)
                self._handle = None
            if self._depth == 2 and getattr(self,"_gpu_costs",None) is not None:
                self._gpu_costs.callback=None
                self._gpu_costs=None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()

    def _decode(self, raw):
        if raw is None:
            raise RuntimeError(self._lib.d2cg_error().decode())
        return json.loads(raw)

    def _tree(self, data):
        if data is None:
            return None
        if "label" in data:
            return Tree(label=self._labels[data["label"]])
        return Tree(feature=data["feature"], left=self._tree(data["left"]),
                    right=self._tree(data["right"]))

    def solve(self, penalty=None, time_limit=600, batch_size=0,
              max_columns=200000, rmp_method=0, progress=None, *, _clock=None):
        """batch_size=0 adds all improving signatures per complete pricing pass.

        seconds includes interface and final independent tree validation, but
        input-file loading and Problem construction precede this call. On an
        explicit reusable workspace its construction is also outside the call.
        A time limit can prevent preprocessing/LP certification from finishing.
        """
        clock = _clock if _clock is not None else Deadline(time_limit)
        penalty = self._p.penalty if penalty is None else float(penalty)
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        batch_size = _integer(batch_size, 0, "batch_size")
        max_columns = _integer(max_columns, 1, "max_columns")
        if rmp_method not in (-1, 0, 1, 2):
            raise ValueError("rmp_method must be -1, 0, 1, or 2")
        p = copy.copy(self._p)
        p.penalty = penalty
        errors = []
        offset = clock.elapsed()

        def emit(raw):
            if errors:
                return
            try:
                data = self._decode(raw)
                tree = self._tree(data["tree"])
                data["tree"] = tree.to_dict() if tree else None
                data["seconds"] = clock.elapsed()
                data["method"] = self._method
                progress(data)
            except Exception as exc:
                errors.append(exc)

        callback = _CALLBACK(emit) if progress is not None else _CALLBACK()
        with self._lock:
            if not self._handle:
                raise RuntimeError(f"D{self._depth} workspace is closed")
            gpu_before = dict(self._gpu_costs.stats) if self._depth == 2 and self._gpu_costs is not None else {}
            raw = self._lib.d2cg_solve(self._handle, penalty,
                max(0., clock.end - time.perf_counter()), batch_size,
                max_columns, rmp_method, callback)
            try:
                data = self._decode(raw)
            except RuntimeError as exc:
                if self._depth == 2 and self._gpu_costs is not None and self._gpu_costs.error:
                    raise RuntimeError("D2 GPU costs: " + self._gpu_costs.error) from exc
                raise
        if errors:
            raise errors[0]
        tree = self._tree(data.pop("tree"))
        lb = data.pop("LB")
        lb = (math.inf if data["status"] == "INFEASIBLE" else 0.) if lb is None else lb
        native_ub = data.pop("UB")
        trace = data.pop("trace")
        for event in trace:
            event["seconds"] += offset
        first = data.pop("first_final_ub_seconds")
        proof = data.pop("proof_seconds")
        status = data.pop("status")
        result = result_dict(self._method, clock, status, tree, p, lb,
            trace=trace, certificate=(trace[-1] if trace else None),
            first_final_ub_seconds=(first + offset if first is not None else None),
            first_optimal_solution_seconds=(first + offset if status == "OPT" else None),
            proof_seconds=(proof + offset if proof is not None else None),
            backend="gurobi_cpp", pricing_threads=self.threads, **data)
        if native_ub is not None and abs(result["UB"] - native_ub) > 1e-7:
            raise AssertionError("Native objective disagrees with independent routed tree evaluation")
        if status == "OPT" and (tree is None or result["absolute_gap"] > 1e-7):
            raise AssertionError("Native OPT is missing a feasible tree or complete LP certificate")
        result["interface_seconds"] = max(0., result["seconds"] - result["native_seconds"])
        result["stats"] = {k: result[k] for k in (
            "pricing_setup_seconds", "pricing_seconds", "rmp_seconds", "environment_seconds")}
        if self._depth == 2:
            result.update(self._adapter_stats,cost_backend=self._cost_backend,
                          cost_backend_requested=self._cost_backend_requested)
            if self._gpu_costs is not None:
                result.update(self._gpu_costs.stats)
                result["gpu_lifetime_stats"] = dict(self._gpu_costs.stats)
                for key,previous in gpu_before.items():
                    if key.endswith("_seconds") or key == "gpu_batches":
                        result[key] -= previous
                if self._gpu_costs.error:
                    result["cost_backend"] = "cpp_fallback"
        return result

    def price(self, alpha, separator_duals, penalty=None):
        """Audit complete pricing: signatures are SPLIT[0:F], STOP[labels]."""
        alpha = np.ascontiguousarray(alpha, dtype=np.float64)
        dual = np.ascontiguousarray(separator_duals, dtype=np.float64)
        if alpha.shape != (2,) or dual.shape != (self._p.F + len(self._labels),):
            raise ValueError("Incorrect dual dimensions")
        penalty = self._p.penalty if penalty is None else float(penalty)
        if not np.isfinite(alpha).all() or not np.isfinite(dual).all() or not math.isfinite(penalty) or penalty < 0:
            raise ValueError("Finite duals and nonnegative finite penalty required")
        with self._lock:
            if not self._handle:
                raise RuntimeError("D2 workspace is closed")
            return self._decode(self._lib.d2cg_price(self._handle, penalty,
                alpha.ctypes.data_as(_DOUBLE), dual.ctypes.data_as(_DOUBLE)))


def solve_d2_cg(problem, time_limit=600, *, threads=0, batch_size=0,
                max_columns=200000, rmp_method=0, progress=None,
                cost_backend="auto", gpu_strategy="pair", reuse_problem_masks=True):
    """One-shot entry point; includes workspace preparation in reported time."""
    clock = Deadline(time_limit)
    with D2CGWorkspace(problem, threads=threads, cost_backend=cost_backend,
                       gpu_strategy=gpu_strategy, reuse_problem_masks=reuse_problem_masks) as workspace:
        result = workspace.solve(time_limit=time_limit, batch_size=batch_size,
            max_columns=max_columns, rmp_method=rmp_method, progress=progress, _clock=clock)
    result["seconds"] = clock.elapsed()
    result["interface_seconds"] = max(0., result["seconds"] - result["native_seconds"])
    return result
