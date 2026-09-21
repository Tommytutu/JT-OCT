"""Gurobi backend for the existing reference solvers, one thread.

SciPy is used only to store sparse matrices. No SciPy/HiGHS optimizer is used.
Duals are exposed only after an optimal continuous solve.
The optional cuPDLPx entry point uses the separate pdlp_backend module.
"""
from types import SimpleNamespace
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB

_ENV = None


def environment():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


def optimize_linear(cost, deadline, A_eq=None, b_eq=None, A_ub=None, b_ub=None,
                    ub=None, integrality=None, name="JT_OCT", start=None):
    deadline.check()
    with gp.Model(name, env=environment()) as model:
        model.Params.Threads = 1
        model.Params.Seed = 20260905
        model.Params.Method = 1
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 1e-10
        model.Params.DualReductions = 0
        cost = np.asarray(cost, dtype=float)
        integer = integrality is not None and np.any(integrality)
        types = np.where(np.asarray(integrality), GRB.INTEGER, GRB.CONTINUOUS) if integer else GRB.CONTINUOUS
        x = model.addMVar(len(cost), lb=0, ub=GRB.INFINITY if ub is None else np.asarray(ub),
                         obj=cost, vtype=types, name="x")
        if start is not None:
            candidate = np.asarray(start, dtype=float)
            if candidate.shape != cost.shape:
                raise ValueError("MIP start must have one value per variable")
            x.Start = candidate
        eq = None
        if A_eq is not None and A_eq.shape[0]:
            eq = model.addMConstr(A_eq, x, "=", np.asarray(b_eq), name="eq")
        if A_ub is not None and A_ub.shape[0]:
            model.addMConstr(A_ub, x, "<", np.asarray(b_ub), name="ub")
        model.Params.TimeLimit = deadline.remaining()
        model.optimize()
        status = {GRB.OPTIMAL: 0, GRB.TIME_LIMIT: 1, GRB.INTERRUPTED: 1,
                  GRB.NODE_LIMIT: 1, GRB.ITERATION_LIMIT: 1,
                  GRB.INFEASIBLE: 2, GRB.UNBOUNDED: 3}.get(model.Status, 4)
        bound = -math.inf
        try:
            bound = float(model.ObjBound)
        except (AttributeError, gp.GurobiError):
            pass
        if status == 0:
            bound = float(model.ObjVal)
        return SimpleNamespace(
            status=status, gurobi_status=model.Status, message=f"Gurobi status {model.Status}",
            x=x.X.copy() if model.SolCount else None,
            fun=float(model.ObjVal) if model.SolCount else None,
            mip_dual_bound=bound,
            eqlin=SimpleNamespace(marginals=eq.Pi.copy() if eq is not None and status == 0 and not integer else None),
            backend="Gurobi", version=".".join(map(str, gp.gurobi.version())),
            runtime=model.Runtime, threads=1,
        )
