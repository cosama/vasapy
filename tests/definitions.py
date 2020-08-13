import numpy as np

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
    # np.longfloat,  # Issue with pybind11 (80bit?)
    np.float16,  # no C type available
    np.float32,
    np.float64,
    # np.float96,  # Platform dependent, probably rare
    # np.float128,  # Issue with pybind11 (80bit?)
    np.csingle,
    np.complex_,
    # np.clongfloat,  # Issue with pybind11 (80bit?)
    np.complex64,
    # np.complex128,  # Issue with pybind11 (80bit?)
    # np.complex192,  # Platform dependent, probably rare
    # np.complex256  # Issue with pybind11 (80bit?)
]

# extended sizes are not supported (tested) for now, it looks like the
# last 6 bytes are not set for any of them, thus random
# a = np.array([0], dtype=np.float128); print(a[0]); b = a.data.hex()[0:32];
# [b[int(i*2):int((i+1)*2)]for i in range(0, len(b)//2)]
# could just not compare (hash them), but then it would break once they become
# valid.
# https://github.com/numpy/numpy/blob/master/numpy/core/include/numpy/npy_common.h
