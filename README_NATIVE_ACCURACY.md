# Native accuracy pricing for JT-CG

This optional Windows/CUDA service accelerates exact D3 terminal pricing.
The contracted JT formulation, complete ancestor separators, Gurobi restricted
master and global bound checks remain the JT-CG implementation. It does not
optimize F1 and does not call the CC-F1 coordinator.

```powershell
python run.py --method JT-CG --dataset fico --depth 5 --penalty 0 --d3-oracle native --threads 8 --output results/fico_native_d5_p0.json
python run.py --method JT-CG --dataset fico --depth 5 --penalty 0.01 --d3-oracle native --threads 8 --output results/fico_native_d5_p001.json
```

Use depth 4 for D4. The default remains `--d3-oracle legacy` for reproducibility
and for configurations outside the new service's supported domain. Native mode
requires binary labels, uniform positive observation weights, STOP, shared
feature choices and split costs, D4/D5 and D3 tails. Invalid requests fail
explicitly. It reports `JT-CG-D3Tail-GPU-Native` and the effective options.

The objective is still misclassification rate plus lambda times split count.
For uniform weight `w`, the native equivalent is `errors + (lambda/w)*K`.
Native outputs are rescaled by `w`, and every returned tree is independently
routed and audited in Python. With nonnegative split penalties and STOP,
constant predicates can be removed without increasing the objective. Ancestor
features are constant on their routed sample set, so removing them preserves
the no-repeat restriction. Equivalent/complementary predicates only need one
representative when node choices and costs are shared.

The new high-dimensional binary D5 profile uses 256 queries per batch and up
to eight pricing batches per master resolve (formerly 64 and four). It retains
all signatures for subsequent pricing. Reusing an older feasible dual does not
certify missing costs: the existing repaired dual bounds and full-domain
message bounds cover every unresolved state. `OPT` still requires a global gap
at most 1e-7. A deadline-interrupted native query returns a feasible witness,
never a partial exact value.

The geometry cache budget is at most 3 GiB and at most one quarter of the CG
memory setting. This is a cache budget, not a bound on total process memory.
Preparation runs in the background and is cancelled/joined on shutdown.
The Python object is sequential and tied to its immutable parent Problem.
The legacy-compatible `kernel_calls` counter counts service requests in native
mode, including requests resolved without a GPU launch. It is not a physical
CUDA launch counter. Use `d3_calls` to compare terminal-query counts, and combine
kernel-launch and join timers to compare GPU evaluation time.

## Build and test

The packaged DLL requires the Microsoft C++ runtime and a compatible NVIDIA
driver. NVRTC is located in the active Python environment's
`nvidia/cuda_nvrtc/bin`, or through `JT_OCT_NVRTC` pointing to its DLL. The
existing Gurobi installation/license is still required by the CG master.

To rebuild with MSVC and the installed NVIDIA Python packages:

```powershell
.\build_accuracy_oracle.ps1 -CudaPackages 'path\to\site-packages\nvidia'
python -m unittest discover -s tests -v
```

Tests require CUDA and fail rather than silently skip when it is unavailable.
They compare to independent exhaustive split enumeration, test routed domains,
constant/duplicate/complementary features, contradictory samples, minimum leaf
sizes, scaled weights, arbitrary binary labels, deadlines, cache-free execution,
and full D4/D5 CG certificates. A legacy CPU regression is included.

To reproduce the comparison against an unchanged JT-OCT checkout and General JT:

```powershell
python benchmark_native_accuracy.py --general-root 'D:\论文高\General JT\JT-CM' --legacy-root 'path\to\unchanged\JT-OCT' --output reports/fresh_comparison
```

Processes run sequentially with eight workers and matching matrix hashes.
F1 uses its recorded `budget` support policy. Compare complete process times;
the different accuracy and F1 objectives do not have comparable objective
values. The min-leaf defaults 0 and 1 have the same optimum here: empty branches
can be contracted under STOP with shared nonnegative costs.
