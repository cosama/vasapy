vasapy (Vectorized ASsociative Arrays for PYthon)
=================================================

This is a lightweight wrapper of [parallel_hashmap](https://github.com/greg7mdp/parallel-hashmap.git)
using [pybind11](https://github.com/pybind/pybind11) to allow for vectorized
operation on dictionaries (and maybe later sets).

The dictionary behave like the python default dictionary with most methods
implemented, but additionally all methods can also use numpy arrays, doing
the iterations over array elements internally in C++ way more efficiently. It
supports most of the numpy types, (128/256 byte types are not tested and might
behave unexpectedly).

```python
import vasapy as vp
import numpy as np

keys = np.arange(100)
data = np.random.rand(100)*100
ind = np.arange(10)

# create a dictionary from arrays
d = vp.dict(keys, data)

# key access with array or integer
print(d[ind])
print(d[0])

# setting elements with arrays or integer
d[ind] = np.zeros(len(ind))
d[101] = 0

# accessing elements with default value
ind = np.arange(0, 200)
print(d.get(ind, 0))
```
