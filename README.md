# JT-OCT

JT-OCT provides exact optimization methods for bounded-depth classification
trees with binary features. For a tree \(T\), the objective is

\[
J(T)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\{T(x_i)\neq y_i\}
     +\lambda\,|B(T)|,
\]

where \(|B(T)|\) is the number of split nodes and \(\lambda\geq 0\) is the
split penalty. The supplied configurations use a maximum depth, allow a branch
to terminate before that depth, and prohibit reuse of a feature along a
root-to-leaf path.

## Methods

| Method | Description |
|---|---|
| `JT-LP` | Solves the reduced linear programming formulation after structural contraction and endpoint elimination. |
| `JT-CG` | Uses exact shallow JT-DP at depths 2 and 3; at depths 4 and 5 it combines conditional-subtree evaluation with column generation and global lower bounds. |
| `JT-MP` | Uses the same shallow exact evaluator and coordinates depth-4/5 conditional costs by message passing with on-demand evaluation. |

Each method returns a feasible tree when one is available, a lower bound `LB`,
an upper bound `UB`, a termination status, and solver statistics. `OPT` means
that the method has closed the global optimality gap.

## Requirements

- 64-bit Windows and Python
- Microsoft Visual C++ runtime
- Gurobi 13 and a valid Gurobi license
- NVIDIA GPU and compatible driver for GPU configurations

Create an environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU evaluation, install the additional packages and prepare the CUDA
runtime directory:

```powershell
python -m pip install -r requirements-gpu.txt
python setup_gpu_runtime.py
```

The repository includes the native Windows libraries used by the solvers.

## Command-line usage

All three methods use the same entry point:

```powershell
python run.py --method JT-LP --dataset banknote --depth 4 --penalty 0.01 --output results/lp.json
python run.py --method JT-CG --dataset banknote --depth 4 --penalty 0.01 --output results/cg.json
python run.py --method JT-MP --dataset banknote --depth 4 --penalty 0.01 --output results/mp.json
```

The bundled command line uses the recorded configuration for the selected
dataset, depth, penalty, and method.

For reproducible timings, the runner defaults `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to `1` and preloads the depth-4/5
contracted backend before preparing the dataset. Existing environment-variable
values are preserved, so these defaults can be overridden explicitly.

| Argument | Values | Default | Meaning |
|---|---|---:|---|
| `--method` | `JT-LP`, `JT-CG`, `JT-MP` | `JT-CG` | Exact solution method. |
| `--dataset` | one of the 11 bundled names | required | Binary benchmark matrix. |
| `--depth` | `2`, `3`, `4`, `5` | required | Maximum tree depth. |
| `--penalty` | `0`, `0.01` | `0` | Penalty for each split node. |
| `--seconds` | positive number | `600` | Solver time limit in seconds. |
| `--output` | JSON path | required | Result file; an existing file is not overwritten. |

Example with a shorter limit:

```powershell
python run.py --method JT-CG --dataset fico --depth 5 --penalty 0.01 --seconds 120 --output results/fico.json
```

## Model parameters

The core `Problem` object accepts the following modeling parameters:

| Parameter | Meaning | Supplied-run value |
|---|---|---:|
| `X` | Nonempty binary feature matrix. | bundled dataset |
| `y` | Nonnegative integer class labels. | bundled dataset |
| `depth` | Maximum depth, with the root at depth 0. | 2--5 |
| `penalty` | Nonnegative split penalty \(\lambda\). | 0 or 0.01 |
| `weights` | Positive observation weights; omitted values give weight \(1/n\). | uniform |
| `early_stop` | Allow a node to predict before maximum depth. | `True` |
| `no_repeat` | Prohibit reuse of a feature on one root-to-leaf path. | `True` |
| `min_leaf` | Minimum number of observations in a leaf. | 0 |
| `allowed` | Optional restrictions on features permitted at individual nodes. | unrestricted |
| `split_costs` | Optional node-specific split costs. | none |
| `preparation_workers` | CPU workers used to prepare feature bitsets. | 1 |

For direct use in Python:

```python
import numpy as np
from jt_oct import Problem
from jt_oct.contract_cg import ContractOptions, solve_contracted_cg

with np.load("datasets/banknote.npz", allow_pickle=False) as data:
    problem = Problem(
        data["X"],
        data["y"].astype(int),
        depth=4,
        penalty=0.01,
        early_stop=True,
        no_repeat=True,
        min_leaf=0,
    )

options = ContractOptions(threads=8, state_screen=True, gpu_fused_join=True)
result = solve_contracted_cg(problem, backend="auto", time_limit=600, options=options)
```

Set `master_mode="message"` and `cost_mode="lazy"` in `ContractOptions` to use
the deep JT-MP coordination path. The command-line entry already selects the
recorded settings for each method.

## Computational parameters

The configuration files contain one entry for every combination of dataset,
depth, and penalty:

- `config/cg.json`: 88 JT-CG/JT-DP settings.
- `config/lp_mp.json`: 88 JT-LP and 88 JT-MP settings.

The main computational options are:

| Option | Meaning |
|---|---|
| `backend` | `cpp`, `gpu`, or automatic CPU/GPU selection. |
| `threads` | Number of CPU search workers. |
| `memory_bytes`, `memory_limit_bytes` | Memory budget checked before large allocations. |
| `cap`, `max_columns` | Capacity for explicit states or active master columns. |
| `batch`, `oracle_batch` | Number of conditional states evaluated in one batch. |
| `tail_depth` | Depth of each conditional subtree; the deep configurations use 3. |
| `master_mode` | `cg` for JT-CG and `message` for JT-MP. |
| `cost_mode` | `lazy` evaluates conditional costs when needed; `eager` constructs a complete table. |
| `rmp_every_batches` | Number of pricing batches between restricted-master solves. |
| `state_screen` | Excludes states whose valid bound cannot improve the current solution. |
| `class_bound` | Enables the class-count bound for multiclass data. |
| `gpu_fused_join` | Uses the fused GPU reduction and join path. |
| `gpu_compact_rows` | Allows local row-bitset compaction when it is profitable. |
| `gpu_compact_min_batch` | Minimum batch size for row compaction. |
| `gpu_compact_max_ratio` | Maximum local/global word ratio for row compaction. |
| `gpu_pair_tile` | Pair tile size used by the GPU evaluator. |
| `gpu_bucket_min` | Minimum bucket size for grouped GPU dispatch. |
| `gpu_sync_tiles` | Number of GPU tiles submitted before synchronization. |

For depth-4/5 JT-CG, automatic selection uses the GPU when it is available and
either the number of binary features is at least 48 or the number of observations
is at least 10,000. Otherwise it uses the C++ backend. The default number of CPU
workers is 8. The restricted master is normally solved after each batch; the
validated high-dimensional depth-5 binary profile uses four batches. The class
bound is enabled only for multiclass data.

Depth-2/3 settings use the backend and thread count recorded in `config/cg.json`.
The GPU path uses resident data, adaptive word processing, compact row bitsets,
and fused joins where enabled by the configuration.

## Output

The JSON result includes:

| Field | Meaning |
|---|---|
| `status` | `OPT`, `TIME`, `INFEASIBLE`, or an error/capacity status. |
| `LB` | Valid lower bound on the optimal objective. |
| `UB` | Objective of the returned feasible tree. |
| `tree` | Nested split/leaf representation using `feature`, `left`, `right`, and `label`. |
| `call_wall_seconds` | Runtime measured around the solver call. |
| `stats` | Backend, search, pricing, message, and GPU counters. |
| `configuration` | Recorded configuration selected for the run; effective backend options are also reported in `stats`. |

For an optimal run, `LB` and `UB` agree within the solver tolerance. A time-limited
run retains the best feasible tree and the valid lower bound obtained before the
limit.

## Datasets

The repository contains the 11 binary matrices used in the experiments:

| Dataset | Observations | Binary features | Classes |
|---|---:|---:|---:|
| avila | 20,867 | 85 | 12 |
| banknote | 1,372 | 36 | 2 |
| compas | 12,381 | 71 | 2 |
| diabetic | 101,766 | 315 | 3 |
| fico | 10,459 | 159 | 2 |
| give | 150,000 | 57 | 2 |
| htru2 | 17,898 | 72 | 2 |
| letter | 20,000 | 99 | 26 |
| skin | 245,057 | 27 | 2 |
| spambase | 4,601 | 152 | 2 |
| transactions | 786,363 | 131 | 2 |

Each `.npz` file contains `X`, `y`, `feature_names`, and `label_names`. The
predictors are already binarized; the solver does not discretize continuous
features. Provider links available for the datasets are recorded in
`datasets/catalog.json`.

## Source and native libraries

```text
run.py              common command-line entry
config/             recorded method and hardware parameters
datasets/           benchmark matrices
jt_oct/              Python interfaces and solver coordination
jt_oct/_native/      prebuilt Windows libraries
native/              C++ and CUDA sources
build_*.ps1          native build scripts
```

To rebuild a component, install MSVC C++ Build Tools. Gurobi builds also require
its C++ headers and libraries. Examples:

```powershell
.\build_d3_optimized.ps1
.\build_contract_rmp.ps1 -GurobiRoot "C:\gurobi1300\win64"
.\build_revision_experiments.ps1 -GurobiRoot "C:\gurobi1300\win64"
```

Rebuilding is unnecessary when the included libraries are compatible with the
local Windows and Gurobi installation.
