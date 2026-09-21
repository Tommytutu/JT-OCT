"""Skip optional native and CUDA tests when their backends are unavailable."""
from pathlib import Path
from functools import lru_cache

import pytest


NATIVE = Path(__file__).resolve().parents[1] / "jt_oct" / "_native"
REQUIRED = {
    "test_contract_cg": ("contract_rmp.dll", "d3_optimized.dll"),
    "test_state_bounds": ("contract_rmp.dll", "d3_optimized.dll", "interval_oracle.dll"),
    "test_interval_feedback": ("contract_rmp.dll", "d3_optimized.dll", "interval_oracle.dll"),
    "test_d2_structural_lp": ("d2_cg.dll",),
    "test_d3_structural_native": ("d3_structural.dll",),
    "test_resident_compact": ("d3_optimized.dll",),
    "test_resident_dispatch": ("d3_optimized.dll",),
    "test_resident_fused": ("d3_optimized.dll",),
}


@lru_cache(maxsize=1)
def gpu_available():
    try:
        from jt_oct.d3_batched import _configure_cupy_runtime
        _configure_cupy_runtime()
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.fixture(autouse=True)
def optional_backends(request):
    module = Path(request.node.path).stem
    missing = [name for name in REQUIRED.get(module, ()) if not (NATIVE / name).is_file()]
    if missing:
        pytest.skip("Build native backends first: " + ", ".join(missing))
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    gpu_test = module.startswith("test_resident_") or parameters.get("backend") == "gpu"
    if gpu_test and not gpu_available():
        pytest.skip("CUDA device and CuPy required")
