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
    # np.longlong,  # see https://github.com/pybind/pybind11/issues/1908
    np.intp,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.ubyte,
    np.ushort,
    np.uintc,
    np.uint,
    # np.ulonglong,  # see https://github.com/pybind/pybind11/issues/1908
    np.uintp,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    # np.half,
    np.single,
    np.double,
    np.float_,
    np.longfloat,
    # np.float16,
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

@pytest.mark.parametrize("dtype", nptypes)
@pytest.mark.parametrize("ktype", nptypes)
def test_dict_dtypes(dtype, ktype):
    if ktype == np.bool_ or ktype == np.bool8:
        return
    fill = dtype(1.0)
    keys = np.arange(100, dtype=ktype)
    data = (np.random.rand(100)*100).astype(dtype)
    print(ktype, dtype)
    hd = vp.dict(keys, data, fill=fill)
    d2 = hd[keys]
    tfill = hd[np.array([101], dtype=ktype)][0]

    assert d2.dtype == dtype, "{} not {}".format(d2.dtype, ktype)

    assert all(data == d2), "{} not {} for dtype {}".format(data, d2, dtype)

    assert tfill == fill, "{} not {}".format(tfill, fill)

    assert all(np.sort(hd.keys()) == np.sort(keys))

    assert len(hd) == 100

    assert np.all(hd.contains(keys[[10, 20, 30, 40, 50]]) == True)

    assert np.all(hd.contains(np.array([101, 102], dtype=ktype)) == False)
