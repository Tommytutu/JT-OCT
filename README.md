# JT-OCT

JT-OCT is the reference implementation accompanying *A Junction-Tree Linear Programming Model for Optimal Classification Trees*. It trains sparse, bounded-depth classification trees over a fixed binary feature dictionary by minimizing

\[
\text{misclassification rate} + \lambda\,\text{number of split nodes}.
\]

The repository provides three exact methods built on the same junction-tree representation:

| Method | Description | Intended use |
|---|---|---|
| JT-LP | Explicit continuous junction-tree LP | Small configuration spaces and formulation studies |
| JT-CG | Column generation with exact pricing and valid lower bounds | Larger instances |
| JT-MP | Exact min-sum message passing with adaptive cost refinement | Fast coordination for the additive model |

All methods return a feasible tree, its independently evaluated objective, a lower bound, and an optimality status. The predictor matrix must already be binary. Continuous variables should be discretized before calling JT-OCT.

## Requirements

- Python 3.10 or newer
- NumPy and SciPy
- Gurobi 13 with a valid license
- Windows, MSVC Build Tools, and the Gurobi C++ SDK for optimized native backends
- NVIDIA CUDA and CuPy only for optional GPU evaluation

Install the Python package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

The bundled small example runs with the reference implementations without compiling native extensions. To build all optimized CPU backends on Windows, set `GUROBI_HOME` if necessary and run:

```powershell
.\build_all.ps1
```

For GPU execution, also install the optional dependency:

```powershell
python -m pip install -e ".[gpu]"
```

## Python example

```python
from jt_oct import load_binary_csv, make_problem, named_tree, solve

X, y, features, labels = load_binary_csv("examples/toy_binary.csv")
problem = make_problem(X, y, depth=2, penalty=0.01)

result = solve(problem, method="JT-MP", time_limit=30, backend="cpu")
print(result["status"], result["LB"], result["UB"])
print(named_tree(result["tree"], features, labels))
```

Run all three methods on the included data:

```powershell
python examples\basic_usage.py
```

The methods should return the same optimal objective. JT-LP may become too large as depth and the number of binary predicates increase; JT-CG and JT-MP use the optimized contracted implementation at depths four and five when the native extensions are available.

## Depth support

The junction-tree model and all three public solvers accept every integer depth
`D >= 1`; there is no hard cutoff at depth five. The implementations are selected
as follows:

| Depth | Implementation selected when available |
|---|---|
| 1 | General path-cluster formulation |
| 2--3 | Specialized shallow JT-CG/JT-MP kernels |
| 4--5 | Depth-three private-subtree contraction used in the experiments |
| 6 and above | General path-cluster JT-LP, JT-CG, or streaming JT-MP |

The general formulation creates \(2^{D-1}\) path clusters before accounting for
their local configurations. It is exact at every depth, but its running time and
memory can therefore grow exponentially, as expected for the NP-hard OCT problem.
The depth-specific native code changes computational efficiency only; it does not
change the model or its feasible trees.

The following example solves six-bit parity using all three general methods. Zero
training error requires all six levels and 63 split nodes, so this checks actual
depth-six behavior rather than a shallow tree under a loose depth limit:

```powershell
python examples\arbitrary_depth.py
```

## Command line

The first CSV column is treated as the class label by default:

```powershell
jt-oct examples\toy_binary.csv --method JT-CG --depth 2 --penalty 0.01 --backend cpu
```

Use `--label class_name` when the label is not the first column. The command prints JSON; pass `--output outputs/result.json` to save it.

## Default parameters

The public interface minimizes

\[
\frac{1}{n}\sum_{i=1}^n \mathbf 1\{T(x_i)\ne y_i\}
+\lambda |B(T)|,
\]

where \(B(T)\) is the set of split nodes. The defaults are:

| Parameter | Default | Meaning |
|---|---:|---|
| `method` | `JT-MP` | Exact min-sum message passing |
| `depth` | Required by `make_problem`; `2` in the CLI | Maximum tree depth |
| `penalty` | `0.0` | Penalty \(\lambda\) per split node |
| `time_limit` | `600` seconds | Wall-clock limit for one solve |
| `max_columns` | `200000` | Maximum number of configurations or active columns |
| `backend` | `auto` | Automatic computational backend selection |
| `min_leaf` | `0` | Minimum observations reaching a leaf |
| `no_repeat` | `True` | Forbid reuse of a feature on one root-to-leaf path |
| `early_stop` | `True` | Permit prediction before the maximum depth |
| observation weight | `1/n` | Uniform contribution to the misclassification rate |
| CSV label column | first column | Override with `--label` or `label_column` |

With `backend="auto"`, GPU evaluation is selected when a CUDA device is
available and either the number of features is at least 48 or the number of
observations is at least 10,000. Otherwise, the optimized code uses CPU
evaluation with eight OpenMP threads. At depths above five, the solver uses the
general path-cluster implementation; the backend choice does not change that
formulation.

An explicit Python call with the defaults is:

```python
problem = make_problem(
    X,
    y,
    depth=6,          # required in the Python interface
    penalty=0.0,
    no_repeat=True,
    min_leaf=0,
)

result = solve(
    problem,
    method="JT-MP",
    time_limit=600,
    max_columns=200_000,
    backend="auto",
)
```

## Input and output

Input predictors must contain only 0 and 1. Labels may be strings or integers and are encoded internally. By default, a feature cannot be used twice on one root-to-leaf path, early stopping is enabled, and the minimum leaf size is zero.

The result dictionary includes `status`, certified bounds `LB` and `UB`, the recovered `tree`, independently evaluated `metrics`, `model_depth`, and the selected `implementation`.

## Verification

```powershell
python -m pytest
python examples\basic_usage.py
```

Large benchmark datasets and experimental result archives are kept outside this source repository.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated paper when using the software.

## License

No redistribution license has been selected yet. Add the authors' chosen license before making the repository public.
