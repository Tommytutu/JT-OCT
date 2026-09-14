"""Full and restricted LP matrices, including union-indexed separators."""
from itertools import combinations
import numpy as np
from scipy.sparse import coo_matrix
from .gurobi_backend import optimize_linear
from .gurobi_backend import environment
import gurobipy as gp
from types import SimpleNamespace
from .domain import CapacityExceeded


def build_matrix(domain, pool, mode="jt", row_universe=None):
    if mode not in {"jt", "node", "pair"}:
        raise ValueError("Unknown separator relaxation")
    flat = [(i, c) for i, cols in enumerate(pool) for c in cols]
    row_keys = [("norm", i) for i in range(domain.M)]
    def states(i, c):
        for e, size, sign in domain.incident[i]:
            if mode == "jt":
                yield ("sep", e, c.prefix[:size]), sign
            else:
                for order in range(1, min(size, 2 if mode == "pair" else 1) + 1):
                    for positions in combinations(range(size), order):
                        yield ("marginal", e, positions, tuple(c.prefix[j] for j in positions)), sign
    extras = set()
    for i, c in (flat if row_universe is None else [(i, c) for i, cols in enumerate(row_universe) for c in cols]):
        extras.update(key for key, _ in states(i, c))
    row_keys += sorted(extras)
    index = {key: j for j, key in enumerate(row_keys)}
    rr, cc, vv = [], [], []
    for k, (i, c) in enumerate(flat):
        rr.append(i); cc.append(k); vv.append(1.0)
        for key, sign in states(i, c):
            rr.append(index[key]); cc.append(k); vv.append(float(sign))
    A = coo_matrix((vv, (rr, cc)), shape=(len(row_keys), len(flat))).tocsr()
    b = np.zeros(len(row_keys)); b[:domain.M] = 1
    cost = np.array([c.cost for _, c in flat])
    return A, b, cost, flat, row_keys


def solve_rmp(domain, pool, deadline, mode="jt", row_universe=None):
    A, b, cost, flat, keys = build_matrix(domain, pool, mode, row_universe)
    if any(not cols for cols in pool):
        return None, (A, b, cost, flat, keys)
    result = optimize_linear(cost, deadline, A_eq=A, b_eq=b, name="JT_RMP")
    return result, (A, b, cost, flat, keys)


def unpack_dual(domain, result, keys):
    alpha = result.eqlin.marginals[:domain.M].copy()
    pi = { (key[1], key[2]): float(result.eqlin.marginals[j])
           for j, key in enumerate(keys) if key[0] == "sep" }
    return alpha, pi


def reduced_cost(domain, i, col, alpha, pi):
    return col.cost - alpha[i] - sum(sign * pi.get((e, col.prefix[:size]), 0.0)
                                    for e, size, sign in domain.incident[i])


class RestrictedMaster:
    """Persistent Gurobi RMP, with simultaneous union-row/column activation.

    Existing variables and basis information survive between iterations. A newly
    created row is identically zero on all previous columns by the union rule.
    """
    def __init__(self,domain,row_universe=None,method=1):
        self.domain=domain;self.universe=row_universe
        self.model=gp.Model("JT_incremental_RMP",env=environment())
        for key,value in {"Threads":1,"Seed":20260905,"Method":method,"FeasibilityTol":1e-9,
                          "OptimalityTol":1e-9,"DualReductions":0}.items():
            self.model.setParam(key,value)
        self.rows={};self.variables={}
        self.flat=[]

    def solve(self,pool,deadline):
        deadline.check()
        A,b,c,flat,keys=build_matrix(self.domain,pool,row_universe=self.universe)
        for j,key in enumerate(keys):
            if key not in self.rows:
                self.rows[key]=self.model.addConstr(gp.LinExpr()==float(b[j]))
        self.model.update()
        C=A.tocsc()
        for j,pair in enumerate(flat):
            if pair in self.variables:
                continue
            start,end=C.indptr[j:j+2]
            column=gp.Column(C.data[start:end].tolist(),[self.rows[keys[r]] for r in C.indices[start:end]])
            self.variables[pair]=self.model.addVar(lb=0,obj=float(c[j]),column=column)
        self.model.Params.TimeLimit=deadline.remaining()
        self.model.optimize()
        status={2:0,9:1,3:2}.get(self.model.Status,4)
        result=SimpleNamespace(status=status,fun=self.model.ObjVal if status==0 else None,
                               x=np.array(self.model.getAttr("X",[self.variables[k] for k in flat])) if status==0 else None,
                               eqlin=SimpleNamespace(marginals=np.array(self.model.getAttr("Pi",[self.rows[k] for k in keys])) if status==0 else None))
        return result,(A,b,c,flat,keys)

    def solve_incremental(self, pool, deadline):
        """Add only new union rows/columns and avoid rebuilding a SciPy matrix."""
        deadline.check()
        if any(not columns for columns in pool):
            return None, (None, None, np.empty(0), [], [])
        for i in range(self.domain.M):
            key = ("norm", i)
            if key not in self.rows:
                self.rows[key] = self.model.addConstr(gp.LinExpr() == 1.0)
        new_pairs = []
        for i, columns in enumerate(pool):
            for column in columns:
                pair = (i, column)
                if pair in self.variables:
                    continue
                new_pairs.append(pair)
                for edge, size, _ in self.domain.incident[i]:
                    key = ("sep", edge, column.prefix[:size])
                    if key not in self.rows:
                        self.rows[key] = self.model.addConstr(gp.LinExpr() == 0.0)
        self.model.update()
        for i, column in new_pairs:
            constraints = [self.rows[("norm", i)]]
            coefficients = [1.0]
            for edge, size, sign in self.domain.incident[i]:
                constraints.append(self.rows[("sep", edge, column.prefix[:size])])
                coefficients.append(float(sign))
            variable = self.model.addVar(
                lb=0.0, obj=float(column.cost),
                column=gp.Column(coefficients, constraints))
            self.variables[i, column] = variable
            self.flat.append((i, column))
        self.model.update()
        self.model.Params.TimeLimit = deadline.remaining()
        self.model.optimize()
        status = {2: 0, 9: 1, 3: 2}.get(self.model.Status, 4)
        keys = list(self.rows)
        variables = [self.variables[pair] for pair in self.flat]
        constraints = [self.rows[key] for key in keys]
        result = SimpleNamespace(
            status=status,
            fun=self.model.ObjVal if status == 0 else None,
            x=np.array(self.model.getAttr("X", variables)) if status == 0 else None,
            eqlin=SimpleNamespace(
                marginals=np.array(self.model.getAttr("Pi", constraints)) if status == 0 else None),
            max_equality_residual=(float(max(abs(v) for v in self.model.getAttr("Slack", constraints)))
                                   if status == 0 else None),
            minimum_active_reduced_cost=(float(min(self.model.getAttr("RC", variables)))
                                         if status == 0 else None))
        costs = np.fromiter((column.cost for _, column in self.flat), dtype=float,
                            count=len(self.flat))
        return result, (None, None, costs, list(self.flat), keys)

    def close(self):
        self.model.dispose()


def chain_path(domain, pool):
    """Layered shortest path / chain min-sum messages, including one-sided states."""
    prev, records = {(): 0.0}, []
    for i, cols in enumerate(pool):
        next_values, back = {}, {}
        for col in cols:
            left = () if i == 0 else col.prefix[:domain.separators[i-1]]
            right = () if i == domain.M-1 else col.prefix[:domain.separators[i]]
            if left not in prev:
                continue
            value = prev[left] + col.cost
            if right not in next_values or value < next_values[right]:
                next_values[right] = value
                back[right] = left, col
        prev = next_values
        records.append(back)
    if () not in prev:
        return float("inf"), None
    state, selected = (), []
    for back in reversed(records):
        state, col = back[state]
        selected.append(col)
    return prev[()], list(reversed(selected))


def chain_path_streaming(domain, deadline=None, max_columns=200000):
    """Exact chain min-sum messages without materializing the column universe.

    Only the best predecessor for each right-separator state is retained.  This
    reduces JT-DP memory from all configurations to the separator message and
    its backpointers; ``max_columns`` counts configurations inspected.
    """
    prev, records, processed = {(): 0.0}, [], 0
    for i in range(domain.M):
        next_values, back = {}, {}
        for col in domain.columns(i, deadline):
            processed += 1
            if processed > max_columns:
                raise CapacityExceeded(f"Configuration limit {max_columns} exceeded")
            left = () if i == 0 else col.prefix[:domain.separators[i-1]]
            if left not in prev:
                continue
            right = () if i == domain.M-1 else col.prefix[:domain.separators[i]]
            value = prev[left] + col.cost
            if right not in next_values or value < next_values[right]:
                next_values[right] = value
                back[right] = left, col
        prev = next_values
        records.append(back)
        if not prev:
            return float("inf"), None, processed
    if () not in prev:
        return float("inf"), None, processed
    state, selected = (), []
    for back in reversed(records):
        state, col = back[state]
        selected.append(col)
    return prev[()], list(reversed(selected)), processed
