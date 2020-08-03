import numpy as np
import vasapy as vp
import pytest


# from https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
nptypes = [
    np.bool_,
    np.bool8,
    np.byte,
    np.short,
    np.intc,
    np.int_,
    np.longlong,  # see https://github.com/pybind/pybind11/issues/1908
    np.intp,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.ubyte,
    np.ushort,
    np.uintc,
    np.uint,
    np.ulonglong,  # see https://github.com/pybind/pybind11/issues/1908
    np.uintp,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.half,  # no C type available
    np.single,
    np.double,
    np.float_,
    np.longfloat,
    np.float16,  # no C type available
    np.float32,
    np.float64,
    # np.float96,  # Platform dependent, probably rare
    np.float128,
    np.csingle,
    np.complex_,
    np.clongfloat,
    np.complex64,
    np.complex128,
    # np.complex192,  # Platform dependent, probably rare
    # np.complex256
]


@pytest.fixture
def dict_0_(ktype, dtype):
    ktype_ = np.dtype(ktype)
    dtype_ = np.dtype(dtype)
    return vp.dict(ktype_, dtype_), ktype_, dtype_


@pytest.fixture
def dict_1_(ktype, dtype):
    keys = np.array([0], dtype=ktype)
    data = np.array([1], dtype=dtype)
    return vp.dict(keys, data), np.dtype(ktype), np.dtype(dtype), keys, data


@pytest.fixture
def dict_10_(ktype, dtype):
    if ktype == np.bool_ or ktype == np.bool8:
        pytest.xfail("Boolean dict can only have 2 elements")
    keys = np.arange(10, dtype=ktype)
    data = (np.random.rand(10)*100).astype(dtype)
    return vp.dict(keys, data), np.dtype(ktype), np.dtype(dtype), keys, data


@pytest.mark.parametrize("dtype", nptypes)
@pytest.mark.parametrize("ktype", nptypes)
class TestDict:
    def test_init_dtypes(self, dict_0_):
        hd, ktype_, dtype_ = dict_0_
        assert ktype_ == hd.ktype
        assert dtype_ == hd.dtype

    def test_init_arrays(self, dict_10_):
        hd, ktype_, dtype_, _, _ = dict_10_
        assert ktype_ == hd.ktype
        assert dtype_ == hd.dtype

    def test_len(self, dict_0_, dict_10_):
        assert len(dict_0_[0]) == 0
        assert len(dict_10_[0]) == 10

    def test_keys(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        ind_in = np.argsort(keys_in)
        ind_out = np.argsort(keys_out)
        assert np.all(keys_in[ind_in] == keys_out[ind_out])

    def test_values(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        data_out = hd.values()
        ind_in = np.argsort(keys_in)
        ind_out = np.argsort(keys_out)
        assert np.all(data_in[ind_in] == data_out[ind_out])

    def test_getitem(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        data_out = hd[keys_out]
        ind_in = np.argsort(keys_in)
        ind_out = np.argsort(keys_out)
        assert np.all(data_in[ind_in] == data_out[ind_out])

    def test_setitem(self, dict_0_, dict_10_):
        hd, ktype_, dtype_ = dict_0_
        _, _, _, keys_in, data_in = dict_10_
        hd[keys_in] = data_in
        keys_out = hd.keys()
        data_out = hd[keys_out]
        ind_in = np.argsort(keys_in)
        ind_out = np.argsort(keys_out)
        assert np.all(keys_in[ind_in] == keys_out[ind_out])
        assert np.all(data_in[ind_in] == data_out[ind_out])

    def test_contains(self, dict_10_):
        hd, ktype_, _, keys_in, data_in = dict_10_
        assert np.all(hd.contains(keys_in) == True)
        assert np.all(hd.contains(np.array([101, 102], dtype=ktype_)) == False)

    def test_get(self, dict_10_):
        hd, ktype_, dtype_, _, _ = dict_10_
        fill = dtype_.type(1.0)
        tfill = hd.get(np.array([100], dtype=ktype_), fill)[0]
        assert tfill == fill

    def test_update(self, dict_0_, dict_10_):
        hd1, _, _ = dict_0_
        hd2, _, _, _, _ = dict_10_
        hd1.update(hd2)
        assert len(hd1) == len(hd2)
        k1 = hd1.keys()
        k2 = hd2.keys()
        i1 = np.argsort(k1)
        i2 = np.argsort(k2)
        assert np.all(k1[i1] == k2[i2])
        assert np.all((hd1.values())[i1] == (hd2.values())[i2])

    def test_key_access_fails(self, dict_1_):
        hd, ktype_, dtype_, _, _ = dict_1_
        with pytest.raises(IndexError):
            hd[np.array(np.array([0, 1], dtype=ktype_))]

    def test_key_in(self, dict_1_):
        hd, ktype_, dtype_, keys, _ = dict_1_
        assert keys in hd
        assert np.array([0, 1], dtype=ktype_) not in hd
