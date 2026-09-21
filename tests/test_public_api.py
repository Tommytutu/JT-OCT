from itertools import product

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


def test_all_methods_support_depth_greater_than_five():
    X = np.asarray(list(product((0, 1), repeat=6)), dtype=np.uint8)
    y = np.bitwise_xor.reduce(X, axis=1).astype(np.int32)
    problem = make_problem(X, y, depth=6, penalty=0.0)
    results = [solve(problem, method, time_limit=30, backend="cpu")
               for method in ("JT-LP", "JT-CG", "JT-MP")]
    assert all(result["status"] == "OPT" for result in results)
    assert [result["UB"] for result in results] == pytest.approx([0.0] * 3)
    assert all(result["metrics"]["split_nodes"] == 63 for result in results)
    assert all(result["model_depth"] == 6 for result in results)
    assert results[1]["implementation"] == "general_path_cluster_column_generation"
    assert results[2]["implementation"] == "general_streaming_junction_tree_message_passing"
    assert all(result["backend"] == "cpu" for result in results)


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_current_solvers_match_independent_tree_search(depth):
    from jt_oct.solvers import solve_tree_dp

    rng = np.random.default_rng(831)
    X = rng.integers(0, 2, (31, 4), dtype=np.uint8)
    y = rng.integers(0, 3, len(X))
    problem = make_problem(X, y, depth=depth, penalty=0.01, min_leaf=2)
    expected = solve_tree_dp(problem, 30)["UB"]
    for method in ("JT-LP", "JT-CG", "JT-MP"):
        result = solve(problem, method, time_limit=30, backend="cpu")
        assert result["status"] == "OPT"
        assert result["UB"] == pytest.approx(expected, abs=1e-8)
        assert result["LB"] == pytest.approx(expected, abs=1e-8)
