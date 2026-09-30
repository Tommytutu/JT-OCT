import unittest
import numpy as np
from test_accuracy_oracle import exhaustive
from jt_oct import Problem
from jt_oct.accuracy_oracle import NativeAccuracyOracle
from jt_oct.d3_optimized import D3Workspace, D3Options
from jt_oct.contract_cg import ContractOptions, solve_contracted_cg

class MulticlassAccuracyTests(unittest.TestCase):
    def test_binary_histogram_dominance(self):
        rng=np.random.default_rng(913)
        x=(rng.random((10000,70))<.1).astype(np.uint8)
        y=x[:,0]&x[:,1];y[rng.choice(len(x),80,replace=False)]=1
        for flipped in (False,True):
            p=Problem(x,1-y if flipped else y,5,0.)
            with D3Workspace(p,'gpu',2,D3Options(native_accuracy=True)) as w, D3Workspace(p,'cpp',2,D3Options()) as reference:
                w.native_accuracy=NativeAccuracyOracle(w,force_general=True)
                for d in (2,3):
                    got=w.solve_many([dict(rows=p.all_rows)],30,d)[0]
                    expected=reference.solve_many([dict(rows=p.all_rows)],30,d)[0]
                    self.assertEqual(got['status'],'OPT');self.assertEqual(expected['status'],'OPT')
                    self.assertAlmostEqual(got['value'],expected['value'],places=10)

    def test_histogram_kernel(self):
        rng=np.random.default_rng(899)
        x=rng.integers(0,2,(10000,70),dtype=np.uint8)
        # Include nested and mutually exclusive predicates to exercise exact
        # intersection bounds, plus unrestricted triples with nonzero width.
        x[:,60]=x[:,0]&x[:,1];x[:,61]=x[:,0]&(1-x[:,1]);x[:,62]=x[:,60]|x[:,61]
        y=rng.integers(0,3,len(x))
        for penalty in (0.,.01,.3):
            p=Problem(x,y,5,penalty)
            with D3Workspace(p,'gpu',2,D3Options(native_accuracy=True)) as w, D3Workspace(p,'cpp',2,D3Options()) as reference:
                for d in (2,3):
                    got=w.solve_many([dict(rows=p.all_rows)],30,d)[0]
                    expected=reference.solve_many([dict(rows=p.all_rows)],30,d)[0]
                    self.assertEqual(got['status'],'OPT');self.assertEqual(expected['status'],'OPT')
                    self.assertAlmostEqual(got['value'],expected['value'],places=10)

    def test_cg_multiclass_and_global_transfer(self):
        rng=np.random.default_rng(189)
        x=rng.integers(0,2,(39,5),dtype=np.uint8)
        y=rng.integers(0,3,39)*7+4
        x[1]=x[0];y[:2]=[4,18]
        for depth in (4,5):
            for penalty in (0.,.0133):
                p=Problem(x,y,depth,penalty)
                answer=solve_contracted_cg(p,'gpu',30,ContractOptions(native_d3=True,threads=2,
                    similarity_refs=8,similarity_global=True,message_bound=True,native_parent_bounds=depth==5))
                self.assertEqual(answer['status'],'OPT')
                expected=exhaustive(p,p.all_rows,depth)
                self.assertAlmostEqual(answer['LB'],expected,places=8)
                self.assertAlmostEqual(answer['UB'],expected,places=8)

    def test_large_words(self):
        rng=np.random.default_rng(451)
        for k,n in ((2,270000),(3,6000)):
            x=rng.integers(0,2,(n,6),dtype=np.uint8)
            y=rng.choice(np.arange(k),n,p=[.99,.01] if k==2 else [.12,.33,.55])
            for penalty in (0.,.01):
                p=Problem(x,y,5,penalty)
                with D3Workspace(p,'gpu',2,D3Options(native_accuracy=True)) as w, D3Workspace(p,'cpp',2,D3Options()) as reference:
                    requests=[dict(rows=p.all_rows),dict(rows=p.route(p.all_rows,0,1),used=(0,),node=(1,))]
                    for d in (2,3):
                        got=w.solve_many(requests,30,d)
                        expected=reference.solve_many(requests,30,d)
                        for a,b in zip(got,expected):
                            self.assertEqual(a['status'],'OPT')
                            self.assertEqual(b['status'],'OPT')
                            self.assertAlmostEqual(a['value'],b['value'],places=10)

    def test_exhaustive(self):
        for k in (2,3,5,26):
            rng=np.random.default_rng(k)
            x=rng.integers(0,2,(89,6),dtype=np.uint8)
            x=np.column_stack([x,x[:,1],1-x[:,1],np.ones(89,np.uint8)])
            y=rng.integers(0,k,89)*3+7
            for ml in (0,4):
                for penalty in (0.,.01,.17):
                    p=Problem(x,y,5,penalty,min_leaf=ml)
                    with D3Workspace(p,'gpu',2,D3Options(native_accuracy=True)) as w:
                        w.native_accuracy=NativeAccuracyOracle(w,force_general=True)
                        for d in (2,3):
                            requests=[dict(rows=p.all_rows)]+[dict(rows=p.route(p.all_rows,f,s),used=(f,),node=(s,)) for f in (0,1) for s in (0,1)]
                            result=w.solve_many(requests,20,d)
                            for req,z in zip(requests,result):
                                self.assertEqual(z['status'],'OPT')
                                self.assertAlmostEqual(z['value'],exhaustive(p,req['rows'],d),places=10)

if __name__=='__main__':unittest.main()
