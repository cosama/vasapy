import numpy as np
import vasapy as vp

def test_dict_dtypes():
    dl = [
        #np.bool_,
        np.byte, np.ubyte,
        #np.cdouble, np.clongdouble, np.longdouble, np.csingle,
        np.short, np.ushort, np.intc, np.uintc, np.int_, np.uint, np.longlong, np.ulonglong,
        #np.half,
        np.single, np.double,
        np.int8, np.int16, np.uint8, np.int32, np.int64,
        np.uint16, np.uint32,np.uint64,
        np.float32, np.float64]

    for k in dl:
        for d in dl:
            fill = d(1.0)
            keys = np.arange(100, dtype=k)
            data = (np.random.rand(100)*100).astype(d)
            print(k, d)
            hd = vp.dict(keys, data, fill=fill)
            d2 = hd[keys]
            tfill = hd[np.array([101], dtype=k)][0]

            assert d2.dtype == d, "{} not {}".format(d2.dtype, k)

            assert all(data == d2), "{} not {} for dtype {}".format(data, d2, d)

            assert tfill == fill, "{} not {}".format(tfill, fill)

            assert all(np.sort(hd.keys()) == np.sort(keys))

            assert len(hd) == 100

            assert np.all(hd.contains(keys[[10, 20, 30, 40, 50]]) == True)

            assert np.all(hd.contains(np.array([101, 102], dtype=k)) == False)
