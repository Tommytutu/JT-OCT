# JT-OCT

Exact optimization of classification trees with binary features. The objective is

\[
\text{misclassification rate} + \lambda\,\text{number of split nodes}.
\]

JT-OCT provides three methods:

| Method | Approach |
|---|---|
| JT-LP | Linear programming with exact structural reductions |
| JT-CG | Column generation with bounds for optimality certification |
| JT-MP | Message passing with adaptive subtree-cost evaluation |

All methods allow early stopping and minimum leaf support. They return a tree
when one is found, objective bounds, runtime, and a termination status.

## Installation

Python 3.10+, NumPy, SciPy, psutil, and Gurobi 13 with a valid license are required.
From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

The reference implementations run without native compilation. To build the
optimized backends on Windows, install MSVC Build Tools and the Gurobi C++ SDK:

```powershell
.\build_all.ps1 -GurobiRoot "C:\gurobi1300\win64"
```

Optional GPU evaluation requires an NVIDIA CUDA device and CuPy:

```powershell
python -m pip install -e ".[gpu]"
```

Set `CUDA_PATH` to the CUDA toolkit directory, or use `JT_OCT_CUDA_ROOT` to select
a separate runtime installation. On Windows, use ASCII-only paths for both the
toolkit and the Python environment so NVRTC can read their headers.
GPU kernels are compiled at runtime from `native/`.
Use the editable installation above when running from this repository.

## Example

```python
from jt_oct import load_benchmark, make_problem, named_tree, solve

X, y, features, labels = load_benchmark("banknote")
problem = make_problem(X, y, depth=4, penalty=0.01)
result = solve(problem, method="JT-CG", time_limit=600, backend="auto")
print(result["status"], result["LB"], result["UB"])
print(named_tree(result["tree"], features, labels))
```

For a small example without native compilation:

```powershell
python examples/basic_usage.py
```

## Command line

Run a bundled dataset or supply a binary CSV:

```powershell
python -m jt_oct --dataset banknote --method JT-CG --depth 4 --penalty 0.01
python -m jt_oct examples/toy_binary.csv --method JT-MP --depth 2 --backend cpu
```

The first CSV column contains labels unless `--label` specifies another name or
index. Use `--output outputs/result.json` to save a result. The matrix is not
silently discretized; predictors must be encoded as 0 or 1.

## Benchmark datasets

The `datasets/` directory contains all 11 binary benchmark matrices, including
all observations used in the experiments. They are stored as compressed NumPy
archives and loaded with `load_benchmark`. See [datasets/README.md](datasets/README.md)
for dimensions, fields, and preprocessing information.

## Depths and defaults

All three methods accept any integer depth `D >= 1`. Depths two and three use
specialized native solvers when built; depths four and five use conditional
subtrees of private depth three. Other depths use the general exact formulation.
The number of configurations can grow exponentially with depth and the number of
candidate split rules. `examples/arbitrary_depth.py` demonstrates depth six.

| Parameter | Default |
|---|---|
| `method` | `JT-MP` |
| `depth` | Required in `make_problem`; 2 in the CLI |
| `penalty` | 0.0 per split |
| `time_limit` | 600 seconds |
| `max_columns` | 200,000 |
| `backend` | `auto` |
| `min_leaf` | 0 |
| `no_repeat` | `True` |
| `preparation_workers` | 1 |

Automatic selection uses the GPU when available and either `F >= 48` or
`n >= 10,000`; smaller problems use the CPU. Native deep searches use eight CPU
workers. D4/D5 evaluation uses state screening, shared GPU cost kernels, and
fused joins. Local bitsets are compacted only when the batch and compression
ratio meet the configured thresholds. Class-count bounds are enabled for
multiclass problems. CG normally re-solves its master after each batch; the D5
binary profile with `F >= 100` and `n < 100,000` uses four batches. These defaults
are shared by CPU/GPU selection and the corresponding solver options.

## Tests

```powershell
python -m pytest
```

Tests compare objectives and bounds with independent tree search, check native
and GPU evaluation, and verify every dataset's checksum and dimensions. Tests
requiring unavailable native or CUDA backends are skipped.

## Citation

Citation metadata are provided in `CITATION.cff`.
