from _vasapy import _set
import numpy as np

class set(_set):
    def __init__(self, elem):
        if isinstance(elem, np.ndarray):
            dtype = elem.dtype
        elif isinstance(elem, (list, tuple)):
            elem = np.asarray(elem)
            dtype = elem.dtype
        else:
            dtype = np.dtype(elem)
            elem = None
        super().__init__(dtype)
        if elem is not None:
            self.add(elem)

    def __contains__(self, elem):
        return np.all(self.contains(np.asarray(elem, dtype=self.dtype)))

    def add(self, elem):
        super().add(np.asarray(elem, dtype=self.dtype))

    def contains(self, elem):
        ret = super().contains(np.asarray(elem, dtype=self.dtype))
        return ret.item() if np.isscalar(elem) else ret

    def dischard(self, elem):
        super().dischard(np.asarray(elem, dtype=self.dtype))

    def pop(self):
        return super().pop().item()

    def remove(self, elem):
        super().remove(np.asarray(elem, dtype=self.dtype))
