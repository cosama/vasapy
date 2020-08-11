from _vasapy import _dict
import numpy as np

class dict(_dict):
    def __init__(self, keys, data):
        if isinstance(keys, np.ndarray):
            ktype = keys.dtype
        elif isinstance(keys, (list, tuple)):
            keys = np.asarray(keys)
            ktype = keys.dtype
        else:
            ktype = np.dtype(keys)
            keys = None
        if isinstance(data, np.ndarray):
            dtype = data.dtype
        elif isinstance(data, (list, tuple)):
            data = np.asarray(data)
            dtype = data.dtype
        else:
            dtype = np.dtype(data)
            data = None
        super().__init__(ktype, dtype)
        if keys is not None:
            if data is None:
                data = np.zeros_like(keys, dtype=dtype)
            self[keys] = data

    def __contains__(self, keys):
        ret = np.all(self.contains(np.asarray(keys, dtype=self.ktype)))
        return ret.item() if np.isscalar(keys) else ret

    def __delitem__(self, keys):
        super().__delitem__(np.asarray(keys, dtype=self.ktype))

    def __getitem__(self, keys):
        ret = super().__getitem__(np.asarray(keys, dtype=self.ktype))
        return ret.item() if np.isscalar(keys) else ret

    def __setitem__(self, keys, data):
        super().__setitem__(np.asarray(keys, dtype=self.ktype),
                            np.asarray(data, dtype=self.dtype))

    def contains(self, keys):
        ret = super().contains(np.asarray(keys, dtype=self.ktype))
        return ret.item() if np.isscalar(keys) else ret

    @classmethod
    def fromkeys(cls, keys, values=0):
        keys = np.asarray(keys)
        if np.isscalar(values):
            values = np.full(keys.shape, values)
        else:
            values = np.array(values)
        return cls(keys, values)

    def get(self, keys, defaults=0):
        keys = np.asarray(keys, dtype=self.ktype)
        defaults = np.asarray(defaults, dtype=self.dtype)
        assert keys.size == defaults.size or defaults.size == 1
        ret = super().get(keys, defaults)
        return ret.item() if np.isscalar(keys) else ret

    def pop(self, keys, default=0):
        ret = self.get(keys, default)
        del self[keys]
        return ret.item() if np.isscalar(keys) else ret

    def popitem(self):
        ret = super().popitem()
        return (ret[0].item(), ret[1].item())

    def setdefault(self, keys, default=0):
        key_is_scalar = np.isscalar(keys)
        if key_is_scalar:
            keys = np.atleast_1d(keys)
        keys = np.asarray(keys, dtype=self.ktype)
        if np.isscalar(default):
            ret = np.full_like(keys, default, dtype=self.dtype)
        else:
            ret = np.array(default, dtype=self.dtype)
        mask = self.contains(keys)
        ret[mask] = self[keys[mask]]
        self[keys[~mask]] = ret[~mask]
        return ret[0] if key_is_scalar else ret
