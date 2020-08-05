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
    return vp.dict(ktype, dtype), np.dtype(ktype), np.dtype(dtype)


@pytest.fixture
def dict_1_(ktype, dtype):
    keys = np.array([0], dtype=ktype)
    data = np.array([1], dtype=dtype)
    return vp.dict(keys, data), keys.dtype, data.dtype, keys, data


@pytest.fixture
def dict_10_(ktype, dtype):
    if ktype == np.bool_ or ktype == np.bool8:
        pytest.xfail("Boolean dict can only have 2 elements")
    keys = np.arange(10, dtype=ktype)
    data = (np.random.rand(10)*100).astype(dtype)
    return vp.dict(keys, data), keys.dtype, data.dtype, keys, data


@pytest.mark.parametrize("dtype", nptypes)
@pytest.mark.parametrize("ktype", nptypes)
class TestDict:
    def helper_check(self, keys_in, keys_out, data_in=None, data_out=None,
                     sort=True):
        if sort:
            ind_in = np.argsort(keys_in)
            ind_out = np.argsort(keys_out)
            keys_in, keys_out = keys_in[ind_in], keys_out[ind_out]
            if data_in is not None and data_out is not None:
                data_in, data_out = data_in[ind_in], data_out[ind_out]
        assert np.all(np.equal(keys_in, keys_out))
        if data_in is not None and data_out is not None:
            assert np.all(np.equal(data_in, data_out))

    def test_init_types(self, ktype, dtype):
        hd = vp.dict(ktype, dtype)
        assert np.dtype(ktype) == hd.ktype
        assert np.dtype(dtype) == hd.dtype

    def test_init_dtypes(self, ktype, dtype):
        ktype_ = np.dtype(ktype)
        dtype_ = np.dtype(dtype)
        hd = vp.dict(ktype_, dtype_)
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
        self.helper_check(keys_in, keys_out, sort=True)

    def test_values(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        data_out = hd.values()
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)

    def test_items(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out, data_out = hd.items()
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)

    def test_getitem(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        data_out = hd[keys_out]
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)

    def test_setitem(self, dict_0_, dict_10_):
        hd, ktype_, dtype_ = dict_0_
        _, _, _, keys_in, data_in = dict_10_
        hd[keys_in] = data_in
        keys_out = hd.keys()
        data_out = hd[keys_out]
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)

    def test_delitem(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        del hd[keys_in[0:5]]
        assert len(hd) == 5
        keys_out, data_out = hd.items()
        self.helper_check(keys_in[5:], keys_out, data_in[5:], data_out, sort=True)

    def test_pop(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        data_out = hd.pop(keys_in[:5])
        assert len(hd) == 5
        self.helper_check(keys_in[:5], keys_in[:5], data_in[:5], data_out, sort=False)
        keys_out, data_out = hd.items()
        self.helper_check(keys_in[5:], keys_out, data_in[5:], data_out, sort=True)

    def test_clear(self, dict_1_):
        hd, _, _, _, _ = dict_1_
        hd.clear()
        assert len(hd) == 0

    def test_contains(self, dict_10_):
        hd, ktype_, _, keys_in, data_in = dict_10_
        assert np.all(hd.contains(keys_in) == True)
        assert np.all(hd.contains(np.array([101, 102])) == False)

    def test_get(self, dict_10_):
        hd, ktype_, dtype_, keys, data = dict_10_
        value = hd.get(keys[5:6])
        assert value[0] == data[5]
        value = hd.get(np.array([100]))
        assert value[0] == dtype_.type(0)
        fill = 1.0
        value = hd.get(np.array([100]), fill)
        assert value[0] == dtype_.type(fill)

    def test_update(self, dict_0_, dict_10_):
        hd1, _, _ = dict_0_
        hd2, _, _, _, _ = dict_10_
        hd1.update(hd2)
        assert len(hd1) == len(hd2)
        k1, d1 = hd1.items()
        k2, d2 = hd2.items()
        self.helper_check(k1, k2, d1, d2, sort=True)

    def test_key_access_fails(self, dict_1_):
        hd, ktype_, dtype_, _, _ = dict_1_
        with pytest.raises(IndexError):
            hd[np.array(np.array([0, 1]))]

    def test_key_in(self, dict_1_):
        hd, ktype_, dtype_, keys, _ = dict_1_
        assert keys in hd
        assert np.array([0, 1]) not in hd
