# JT-OCT

基于线性规划、列生成和消息传递的最优分类树求解器。输入为二值特征矩阵，目标为误分类率加上分裂节点数乘以惩罚系数。支持提前停止和最小叶支持约束。

## 安装

需要 Python 3.10 及以上版本，以及有效的 Gurobi 13 许可证。在仓库根目录执行：

```powershell
python -m pip install -e ".[test]"
```

Windows 下安装 MSVC Build Tools 和 Gurobi C++ SDK 后，可编译优化内核：

```powershell
.\build_all.ps1 -GurobiRoot "C:\gurobi1300\win64"
```

GPU 为可选功能，需要 NVIDIA CUDA 设备和 CuPy：

```powershell
python -m pip install -e ".[gpu]"
```

通过 `CUDA_PATH` 或 `JT_OCT_CUDA_ROOT` 指定 CUDA 安装目录。Windows 下 CUDA 和 Python 环境的目录均应只包含 ASCII 字符。GPU 源码位于 `native/`，使用上述可编辑安装方式运行。

## 数据与运行

`datasets/` 包含论文使用的全部 11 个二值化数据集，保留完整样本，采用无损压缩的 NumPy 格式。特征名和原始类别名也包含在文件中。

```python
from jt_oct import load_benchmark, make_problem, solve

X, y, features, labels = load_benchmark("banknote")
problem = make_problem(X, y, depth=4, penalty=0.01)
result = solve(problem, method="JT-CG", time_limit=600, backend="auto")
print(result["status"], result["LB"], result["UB"])
```

也可以使用命令行：

```powershell
python -m jt_oct --dataset banknote --method JT-CG --depth 4 --penalty 0.01
python -m jt_oct examples/toy_binary.csv --method JT-MP --depth 2 --backend cpu
```

方法可选 `JT-LP`、`JT-CG`、`JT-MP`。自有 CSV 默认第一列为类别；用 `--label` 指定其他列，用 `--output` 保存结果。

## 默认配置

默认惩罚为 0、求解时限为 600 秒、列数上限为 200,000；允许提前停止，同一路径不重复特征，最小叶支持为 0。`backend="auto"` 在 GPU 可用且 `F >= 48` 或 `n >= 10,000` 时选择 GPU，否则使用 CPU。深层原生搜索使用八个 CPU worker。

D4/D5 默认启用状态筛选、共享 GPU 成本计算和融合连接，仅在压缩收益足够时使用局部位集；类别数下界用于多分类。CG 通常每批重新求解主问题；D5 中满足 `F >= 100` 且 `n < 100,000` 的二分类问题使用四批间隔。

三个方法均接受任意正整数深度。D2/D3 使用浅层优化内核，D4/D5 使用私有深度为三的子树收缩，其余深度使用通用精确实现。`examples/arbitrary_depth.py` 提供深度六示例。

## 验证

```powershell
python -m pytest
```

测试检查求解目标与上下界、CPU/GPU 计算及数据完整性。缺少所需原生库或 CUDA 的测试会跳过。引用信息见 `CITATION.cff`。
