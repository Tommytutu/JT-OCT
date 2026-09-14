# JT-OCT 使用说明

本项目是论文 *A Junction-Tree Linear Programming Model for Optimal Classification Trees* 的参考实现，包含JT-LP、JT-CG和JT-MP三种精确方法。三种方法使用相同的二元特征、提前停止和分裂惩罚定义，目标函数为误分类率加上分裂节点惩罚。

安装Python依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

运行自带示例：

```powershell
python examples\basic_usage.py
```

也可以通过命令行求解自己的二元数据：

```powershell
jt-oct data.csv --label class --method JT-MP --depth 4 --penalty 0.01 --backend auto --output outputs\result.json
```

输入CSV除标签列外必须全部为0或1。连续特征需要在调用前离散化。`backend=auto`在GPU可用且特征数不少于48或样本数不少于10,000时使用GPU，否则使用八线程CPU条件子树计算。

JT-LP显式构建完整配置LP，适合浅层或小规模问题；JT-CG通过列生成减少活跃主问题列；JT-MP使用min-sum消息传递协调局部配置。在深度4和5且原生库可用时，JT-CG和JT-MP使用论文中的深度3私有子树压缩实现。

## 深度支持

三种公开方法均接受任意整数深度 `D >= 1`，模型并不以深度5为上限。深度2和3优先调用浅层专用内核；深度4和5优先调用论文实验使用的深度3私有子树压缩实现；深度1以及深度6以上使用通用的路径簇JT-LP、JT-CG或流式JT-MP实现。

通用实现包含 \(2^{D-1}\) 个路径簇，每个路径簇还对应一组局部配置。因此它在任意深度上保持精确，但时间和内存会随深度呈指数增长，这与OCT问题的NP-hard性质一致。专用内核只改善计算效率，不改变模型的可行树集合。

深度6示例求解六位奇偶校验问题。该问题必须使用全部六层和63个分裂节点才能达到零误分类，因此验证的是实际深度6求解，而不只是允许提前停止的浅层树：

```powershell
python examples\arbitrary_depth.py
```

Windows下可运行以下命令编译优化后的CPU后端：

```powershell
.\build_all.ps1
```

编译需要MSVC Build Tools、Gurobi 13 C++ SDK及有效的Gurobi许可证。GPU版本还需要CUDA和CuPy。

## 默认参数

公开接口求解以下目标：

\[
\min_T\ \frac{1}{n}\sum_{i=1}^n\mathbf 1\{T(x_i)\ne y_i\}
+\lambda |B(T)|,
\]

其中 \(B(T)\) 是分裂节点集合。默认参数如下：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `method` | `JT-MP` | 精确min-sum消息传递 |
| `depth` | Python接口必须指定；命令行为`2` | 分类树最大深度 |
| `penalty` | `0.0` | 每个分裂节点的惩罚 \(\lambda\) |
| `time_limit` | `600`秒 | 单次求解的墙钟时间限制 |
| `max_columns` | `200000` | 配置或活跃列数量上限 |
| `backend` | `auto` | 自动选择计算后端 |
| `min_leaf` | `0` | 叶节点的最小样本数 |
| `no_repeat` | `True` | 同一根到叶路径不重复使用特征 |
| `early_stop` | `True` | 允许在最大深度前停止并预测 |
| 样本权重 | `1/n` | 每个样本对误分类率的贡献相同 |
| CSV标签列 | 第一列 | 可通过`--label`或`label_column`修改 |

当`backend="auto"`时，如果CUDA设备可用，并且特征数不少于48或样本数不少于10,000，则选择GPU计算；否则优化实现使用8个OpenMP线程进行CPU计算。深度大于5时使用通用路径簇实现，后端选择不改变该模型。

对应的Python调用为：

```python
problem = make_problem(
    X,
    y,
    depth=6,          # Python接口必须指定
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
