import numpy as np
import vasapy as vp

dl = [
    # np.bool_, np.byte, np.ubyte,
    # np.cdouble, #np.clongdouble, np.longdouble,
    np.short, np.ushort, np.intc, np.uintc, np.int_, np.uint, np.longlong,
    np.ulonglong, np.half, np.single, np.double, np.csingle, np.int8, np.int16,
    np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32,
    np.float64]

for k in dl:
    for d in dl:
        keys = np.arange(100, dtype=k)
        data = (np.random.rand(100)*100).astype(d)
        hd = vp.dict(keys, data)
        d2 = hd[keys]
        assert d2.dtype == d, "{} not {}".format(d2.dtype, k)
        assert all(data == d2), "{} not {} for dtype {}".format(data, d2, d)
