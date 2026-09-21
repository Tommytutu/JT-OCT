# JT-OCT

JT-LP, JT-CG, and JT-MP for bounded-depth classification trees.

## Installation

Requires 64-bit Windows, Python, the Microsoft Visual C++ runtime, and Gurobi 13
with a valid license. GPU evaluation requires an NVIDIA GPU and compatible driver.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt
python setup_gpu_runtime.py
```

## Usage

```powershell
python run.py --method JT-LP --dataset banknote --depth 4 --penalty 0.01 --output results/lp.json
python run.py --method JT-CG --dataset banknote --depth 4 --penalty 0.01 --output results/cg.json
python run.py --method JT-MP --dataset banknote --depth 4 --penalty 0.01 --output results/mp.json
```

Depths 2--5 and penalties 0 and 0.01 are supported. The default time limit is
600 seconds; change it with `--seconds`. Shallow JT-CG uses JT-DP. Output contains
the returned tree, objective bounds, status, and runtime.

`config` contains method parameters. `native` contains the C++/CUDA sources, and
`jt_oct/_native` contains the Windows libraries. To rebuild a library, use its
`build_*.ps1` script with MSVC C++ Build Tools installed. Gurobi builds also
require the Gurobi C++ headers and libraries. For example:

```powershell
.\build_contract_rmp.ps1 -GurobiRoot 'C:\gurobi1300\win64'
```
