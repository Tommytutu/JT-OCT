"""Independent exhaustive checks for the optional native accuracy service.

Run: python -m unittest discover -s tests -v
Requires the compiled DLL and a CUDA GPU; deliberately does not silently skip.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
from functools import lru_cache
import unittest
import numpy as np
from jt_oct import Problem
from jt_oct.problem import RowSet, evaluate
from jt_oct.d3_optimized import D3Workspace, D3Options
from jt_oct.contract_cg import ContractOptions, solve_contracted_cg


def exhaustive(p, rows, depth):
    """Enumerate every nonconstant split, including duplicate predicates."""
    @lru_cache(None)
    def visit(mask, remaining):
        ids = [i for i in range(p.n) if mask >> i & 1]
        if len(ids) < p.min_leaf:
            return float('inf')
        best = min(sum(float(p.weights[i]) for i in ids if p.y[i] != label)
                   for label in p.labels)
        if remaining:
            for f in range(p.F):
                left = sum(1 << i for i in ids if p.X[i, f] == 0)
                right = mask ^ left
                if left and right:
                    best = min(best, p.penalty + visit(left, remaining-1) + visit(right, remaining-1))
        return best
    return visit(rows.mask, depth)


class NativeAccuracyTests(unittest.TestCase):
    def data(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, (17, 5), dtype=np.uint8)
        # Duplicates, complements, constants and contradictory observations.
        x = np.column_stack((x, x[:, 1], 1-x[:, 1], np.zeros(17, np.uint8)))
        x[1] = x[0]
        y = rng.integers(0, 2, 17)*5+2
        y[:2] = [2, 7]
        return x, y

    def test_exhaustive_routed_and_whole(self):
        for seed in range(3):
            x, y = self.data(seed)
            for ml in (0, 1, 3):
                for penalty in (0., .01, .19):
                    p = Problem(x, y, 5, penalty, weights=np.full(len(y), .037),
                                no_repeat=True, min_leaf=ml)
                    with D3Workspace(p, 'gpu', 2, D3Options(native_accuracy=True)) as w:
                        requests = [dict(rows=p.all_rows)]
                        for f in (0, 1):
                            for bit in (0, 1):
                                requests.append(dict(rows=p.route(p.all_rows, f, bit), node=(bit,), used=(f,)))
                        requests.append(dict(rows=RowSet(0), node=(0,), used=(0,)))
                        for depth in (2, 3):
                            results = w.solve_many(requests, 30, depth)
                            for req, result in zip(requests, results):
                                expected = exhaustive(p, req['rows'], depth)
                                if np.isinf(expected):
                                    self.assertEqual(result['status'], 'INFEASIBLE')
                                else:
                                    self.assertEqual(result['status'], 'OPT')
                                    self.assertAlmostEqual(result['value'], expected, places=10)

    def test_deadline_and_invalid_ancestor(self):
        x, y = self.data(10)
        p = Problem(x, y, 5, 0, no_repeat=True)
        with D3Workspace(p, 'gpu', 2, D3Options(native_accuracy=True)) as w:
            result = w.solve_many([dict(rows=p.all_rows)], 0, 3)[0]
            self.assertEqual(result['status'], 'TIME')
            self.assertAlmostEqual(result['value'], evaluate(p, result['tree'])['objective'])
            with self.assertRaises(ValueError):
                w.solve_many([dict(rows=p.all_rows, used=(0,), node=(0,))], 10, 3)

    def test_cg_certificates(self):
        x, y = self.data(55)
        for depth in (4, 5):
            for penalty in (0., .01):
                p = Problem(x, y, depth, penalty, no_repeat=True, min_leaf=2)
                expected = exhaustive(p, p.all_rows, depth)
                options = ContractOptions(native_d3=True, threads=2, resident_gpu=True,
                                          warm_d3=True, oracle_batch=16)
                result = solve_contracted_cg(p, 'gpu', 60, options)
                self.assertEqual(result['status'], 'OPT')
                self.assertAlmostEqual(result['LB'], expected, places=8)
                self.assertAlmostEqual(result['UB'], expected, places=8)

    def test_legacy_cpu_regression(self):
        x,y=self.data(8)
        p=Problem(x,y,4,.01,no_repeat=True)
        result=solve_contracted_cg(p,'cpp',60,ContractOptions(threads=2))
        self.assertEqual(result['status'],'OPT')
        self.assertAlmostEqual(result['UB'],exhaustive(p,p.all_rows,4),places=8)

    def test_prefetch_deadline_and_no_cache(self):
        x,y=self.data(9)
        p=Problem(x,y,4,.01,no_repeat=True)
        with D3Workspace(p,'gpu',2,D3Options(native_accuracy=True,native_geometry_bytes=0)) as w:
            first=w.solve(rows=p.all_rows,time_limit=10,depth=3)
            w.native_accuracy.prefetch([p.all_rows],0.)
            second=w.solve(rows=p.all_rows,time_limit=10,depth=3)
            self.assertEqual(second['status'],'OPT')
            self.assertAlmostEqual(first['value'],second['value'],places=10)

    def test_multiple_mask_words(self):
        x,y=self.data(24)
        x=np.tile(x,(5,9));y=np.tile(y,5)
        p=Problem(x,y,4,.013,no_repeat=True,min_leaf=4)
        with D3Workspace(p,'gpu',2,D3Options(native_accuracy=True)) as w:
            req=dict(rows=p.route(p.all_rows,0,1),node=(1,),used=(0,))
            result=w.solve_many([req],30,3)[0]
            self.assertEqual(result['status'],'OPT')
            self.assertAlmostEqual(result['value'],exhaustive(p,req['rows'],3),places=10)


if __name__ == '__main__':
    unittest.main()
