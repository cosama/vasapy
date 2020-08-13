import numpy as np
import vasapy as vp
import pytest
from definitions import nptypes


@pytest.fixture
def set_0_(dtype):
    return vp.set(dtype), np.dtype(dtype)


@pytest.fixture
def set_1_(dtype):
    elem = np.array([0], dtype=dtype)
    return vp.set(elem), elem.dtype, elem


@pytest.fixture
def set_10_(dtype):
    if dtype == np.bool_ or dtype == np.bool8:
        pytest.xfail("Boolean set can only have 2 elements")
    elem = np.arange(10, dtype=dtype)
    return vp.set(elem), elem.dtype, elem


@pytest.mark.parametrize("dtype", nptypes)
class TestSet:
    def helper_check(self, elem_in, elem_out, sort=True):
        if sort:
            ind_in = np.argsort(elem_in)
            ind_out = np.argsort(elem_out)
            elem_in, elem_out = elem_in[ind_in], elem_out[ind_out]
        assert np.all(np.equal(elem_in, elem_out))

    def test_init_types(self, dtype):
        hs = vp.set(dtype)
        assert np.dtype(dtype) == hs.dtype

    def test_init_dtypes(self, dtype):
        dtype_ = np.dtype(dtype)
        hs = vp.set(dtype_)
        assert dtype_ == hs.dtype

    def test_init_arrays(self, set_10_):
        hs, dtype_, _ = set_10_
        assert dtype_ == hs.dtype

    def test_add(self, set_0_):
        hs, dtype_ = set_0_
        if dtype_ == np.bool_ or dtype_ == np.bool8:
            pytest.xfail("Boolean set can only have 2 elements")
        elem = np.arange(10, dtype=dtype_)
        hs.add(elem)
        assert len(hs) == 10
        self.helper_check(np.asarray(hs), elem, sort=True)

    def test_dischard(self, set_10_):
        hs, dtype_, elem = set_10_
        hs.dischard(np.arange(5, 15, dtype=dtype_))
        assert len(hs) == 5
        self.helper_check(np.arange(5, dtype=dtype_), np.asarray(hs),
                          sort=True)

    def test_len(self, set_0_, set_10_):
        assert len(set_0_[0]) == 0
        assert len(set_10_[0]) == 10

    def test_remove(self, set_10_):
        hs, dtype_, elem = set_10_
        hs.remove(np.arange(5, 10, dtype=dtype_))
        assert len(hs) == 5
        self.helper_check(np.arange(5, dtype=dtype_), np.asarray(hs),
                          sort=True)

    def test_remove_fails(self, set_10_):
        hs, dtype_, elem = set_10_
        with pytest.raises(KeyError):
            hs.remove(np.arange(5, 15, dtype=dtype_))

    def test_clear(self, set_1_):
        hs, _, _ = set_1_
        hs.clear()
        assert len(hs) == 0

    def test_pop(self, set_10_):
        hs, _, elem = set_10_
        data = hs.pop()
        assert len(hs) == 9
        assert data in elem

    def test_update(self, set_0_, set_10_):
        hs1, _ = set_0_
        hs2, _, _ = set_10_
        hs1.update(hs2)
        assert len(hs1) == len(hs2)
        k1 = np.asarray(hs1)
        k2 = np.asarray(hs2)
        self.helper_check(k1, k2, sort=True)
