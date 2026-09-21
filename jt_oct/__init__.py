"""Junction-tree optimization for optimal classification trees."""
from .api import (METHODS, Problem, Tree, evaluate, load_binary_csv, make_problem,
                  named_tree, predict, solve, solve_jt_cg, solve_jt_lp, solve_jt_mp)
from .datasets import BENCHMARKS, load_benchmark

__version__ = "0.3.0"

__all__ = ["__version__", "METHODS", "Problem", "Tree", "evaluate", "predict", "load_binary_csv",
           "make_problem", "named_tree", "solve", "solve_jt_lp", "solve_jt_cg",
           "solve_jt_mp", "BENCHMARKS", "load_benchmark"]
