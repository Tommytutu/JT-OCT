# Benchmark datasets

These are the complete binary matrices used by the JT-OCT experiments. No rows
or predictors are sampled or removed when packaging the matrices.

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

Each `.npz` archive contains `X` (binary uint8 predictors), `y` (integer labels),
`feature_names`, and `label_names`. Label `k` corresponds to `label_names[k]`.
Archives can be opened with `numpy.load(path, allow_pickle=False)` or
`jt_oct.load_benchmark(name)`. They contain arrays only, with no pickled objects.

Numerical predictors use cumulative empirical-decile indicators, with repeated
thresholds removed. Categorical values use equality indicators; feature names
record the resulting rules, including missing or special-value indicators.
The binary matrices and column order are fixed across solvers.

`catalog.json` records dimensions, archive checksums, and checksums of the binary
inputs used in the experiments. Original provider links are retained where
available in the dataset records. Dataset attribution and usage terms remain
those of the original providers.
