import numpy as np
import pytest

from jt_oct import make_problem, solve


def test_three_methods_agree_on_toy_problem():
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.asarray([0, 0, 1, 1], dtype=np.int32)
    problem = make_problem(X, y, depth=2, penalty=0.01)
    results = [solve(problem, method, time_limit=30, backend="cpu")
               for method in ("JT-LP", "JT-CG", "JT-MP")]
    assert all(result["status"] == "OPT" for result in results)
    assert [result["UB"] for result in results] == pytest.approx([0.01] * 3)
    assert all(result["LB"] == pytest.approx(result["UB"], abs=1e-7)
               for result in results)


def test_rejects_nonbinary_predictors():
    with pytest.raises(ValueError, match="binary"):
        make_problem([[0, 2], [1, 0]], [0, 1], depth=2)
