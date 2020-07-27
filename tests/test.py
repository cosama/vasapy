import numpy as np
import vasapy as vp

dl = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
      np.uint64, np.float32, np.float64]

for k in dl:
    for d in dl:
        keys = np.arange(100, dtype=k)
        data = np.arange(100, dtype=d)
        hd = vp.dict(keys, data)
        d2 = hd[keys]
        assert all(data == d2), "{} not {}".format(data, d2)
