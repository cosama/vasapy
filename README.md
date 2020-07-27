vasapy (Vectorized ASsociative Arrays for PYthon)
=================================================

This is a lightweight wrapper of [parallel_hashmap](https://github.com/greg7mdp/parallel-hashmap.git)
using [pybind11](https://github.com/pybind/pybind11) to allow for vectorized
operation on dictionaries (and maybe later sets).

Right now it is limited to insert on creation and getting items but I hope to find
the time to add additional functionality.

```python
import vasapy as vp
import numpy as np

keys = np.arange(100)
data = np.random.rand(100)*100
ind = np.arange(10)

d = vp.dict(keys, data)
print(d[ind])
```
