"""Shared validation/packing must preserve every bit, including tail padding."""
import numpy as np
import pytest

from jt_oct import Problem
from jt_oct.problem import _column_masks, _mask


@pytest.mark.parametrize("n,F,layout",[(16387,65,"C"),(40003,27,"F"),(20007,57,"slice")])
def test_blocked_masks_equal_column_oracle(n,F,layout):
    rng=np.random.default_rng(n)
    X=rng.integers(0,2,(n,F*(2 if layout=="slice" else 1)),dtype=np.uint8)
    if layout=="F":X=np.asfortranarray(X)
    if layout=="slice":X=X[:,::2]
    assert _column_masks(X)==tuple(_mask(X[:,f]) for f in range(F))


@pytest.mark.parametrize("dtype",[np.int8,np.uint8,np.int16,np.int64,np.float32,np.float64,bool])
def test_valid_binary_types(dtype):
    X=np.array([[0,1],[1,0]],dtype=dtype)
    p=Problem(X,np.array([0,1]),2)
    assert p.X.flags.c_contiguous
    assert p._feature_masks==((1,2),(2,1))


@pytest.mark.parametrize("bad",[-1,2,256,.5,np.nan,np.inf])
def test_invalid_values_not_silently_cast(bad):
    with pytest.raises(ValueError,match="binary"):
        Problem(np.array([[0,bad],[1,0]]),np.array([0,1]),2)
