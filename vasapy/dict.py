from _vasapy import _dict
import numpy as np

class dict(_dict):
    def __init__(self, keys, data):
        if isinstance(keys, np.ndarray):
            ktype = keys.dtype
        else:
            ktype = np.dtype(keys)
            keys = None
        if isinstance(data, np.ndarray):
            dtype = data.dtype
        else:
            dtype = np.dtype(data)
            data = None
        super().__init__(ktype, dtype)
        if keys is not None:
            if data is None:
                data = np.zeros(len(keys), dtype=dtype)
            self[keys] = data

    def __contains__(self, keys):
        return np.all(self.contains(np.asarray(keys, dtype=self.ktype)))

    def __delitem__(self, keys):
        super().__delitem__(np.asarray(keys, dtype=self.ktype))

    def __getitem__(self, keys):
        return super().__getitem__(np.asarray(keys, dtype=self.ktype))

    def __setitem__(self, keys, data):
        super().__setitem__(np.asarray(keys, dtype=self.ktype),
                            np.asarray(data, dtype=self.dtype))

    def contains(self, keys):
        return super().contains(np.asarray(keys, dtype=self.ktype))

    def get(self, keys, default=0):
        default = np.atleast_1d(default)
        assert len(keys) == len(default) or len(default) == 1
        return super().get(np.asarray(keys, dtype=self.ktype),
                           np.asarray(default, dtype=self.dtype))

    def pop(self, keys):
        return super().pop(np.asarray(keys, dtype=self.ktype))
