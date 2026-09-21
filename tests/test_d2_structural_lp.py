from itertools import product

import numpy as np
import pytest

from jt_oct.solvers import solve_tree_dp
from jt_oct import Problem
from jt_oct.d2_cg import solve_d2_structural_lp


@pytest.mark.parametrize("early,penalty", [(False, 0.0), (True, 0.0), (True, 0.01)])
def test_d2_sc_and_ee_match_exact_dp(early, penalty):
    X = np.array(list(product((0, 1), repeat=4)), dtype=np.uint8)
    y = (X.sum(axis=1) + X[:, 0]) % 3
    p = Problem(X, y, 2, penalty, early_stop=early, no_repeat=True,
                min_leaf=0 if early else 1)
    expected = solve_tree_dp(p, 30)["UB"]
    sc = solve_d2_structural_lp(p, "sc", 30, threads=2, cost_backend="cpp")
    ee = solve_d2_structural_lp(p, "sc_ee", 30, threads=2, cost_backend="cpp")
    assert sc["status"] == ee["status"] == "OPT"
    assert sc["UB"] == pytest.approx(expected, abs=1e-8)
    assert ee["UB"] == pytest.approx(expected, abs=1e-8)
    assert sc["master_domain_columns"] == 2 * ee["master_domain_columns"]
    assert sc["remaining_clusters"] == 2
    assert ee["remaining_clusters"] == 1
    assert ee["submitted_rows"] == 0


def test_d2_structural_weighted_node_costs():
    X = np.array(list(product((0, 1), repeat=4)), dtype=np.uint8)
    y = (2 * X[:, 0] + X[:, 1]) % 3
    weights = np.arange(1, len(X) + 1, dtype=float)
    weights /= weights.sum()
    allowed = {(0,): (0, 2, 3), (1,): (1, 2, 3)}
    costs = {(0, 2): 0.003, (1, 1): 0.002}
    p = Problem(X, y, 2, 0.01, weights=weights, no_repeat=True,
                allowed=allowed, split_costs=costs)
    expected = solve_tree_dp(p, 30)["UB"]
    for reduction in ("sc", "sc_ee"):
        out = solve_d2_structural_lp(p, reduction, 30, threads=2, cost_backend="cpp")
        assert out["status"] == "OPT"
        assert out["UB"] == pytest.approx(expected, abs=1e-8)
