import numpy as np
import vasapy as vp
import pytest
from definitions import nptypes


@pytest.fixture(params=[True, False], ids=lambda p: f"parallel={p}")
def parallel(request):
    return request.param


@pytest.fixture
def dict_0_(ktype, dtype, parallel):
    return vp.dict(ktype, dtype, parallel=parallel), np.dtype(ktype), np.dtype(dtype)


@pytest.fixture
def dict_1_(ktype, dtype, parallel):
    keys = np.array([0], dtype=ktype)
    data = np.array([1], dtype=dtype)
    return vp.dict(keys, data, parallel=parallel), keys.dtype, data.dtype, keys, data


@pytest.fixture
def dict_10_(ktype, dtype, parallel):
    if ktype == np.bool_ or ktype == np.bool8:
        pytest.xfail("Boolean dict can only have 2 elements")
    keys = np.arange(10, dtype=ktype)
    data = (np.random.rand(10)*100).astype(dtype)
    return vp.dict(keys, data, parallel=parallel), keys.dtype, data.dtype, keys, data


@pytest.mark.parametrize("dtype", nptypes)
@pytest.mark.parametrize("ktype", nptypes)
class TestDict:
    def helper_check(self, keys_in, keys_out, data_in=None, data_out=None,
                     sort=True):
        if sort:
            ind_in = np.argsort(keys_in)
            ind_out = np.argsort(keys_out)
            keys_in, keys_out = keys_in[ind_in], keys_out[ind_out]
            if data_in is not None and data_out is not None:
                data_in, data_out = data_in[ind_in], data_out[ind_out]
        assert np.all(np.equal(keys_in, keys_out))
        if data_in is not None and data_out is not None:
            assert np.all(np.equal(data_in, data_out))

    def test_init_types(self, ktype, dtype, parallel):
        hd = vp.dict(ktype, dtype, parallel=parallel)
        assert np.dtype(ktype) == hd.ktype
        assert np.dtype(dtype) == hd.dtype
        assert parallel == hd.parallel

    def test_init_dtypes(self, ktype, dtype, parallel):
        ktype_ = np.dtype(ktype)
        dtype_ = np.dtype(dtype)
        hd = vp.dict(ktype_, dtype_, parallel=parallel)
        assert ktype_ == hd.ktype
        assert dtype_ == hd.dtype
        assert parallel == hd.parallel

    def test_init_arrays(self, dict_10_):
        hd, ktype_, dtype_, _, _ = dict_10_
        assert ktype_ == hd.ktype
        assert dtype_ == hd.dtype

    def test_delitem(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        del hd[keys_in[0:5]]
        assert len(hd) == 5
        keys_out, data_out = hd.items()
        self.helper_check(keys_in[5:], keys_out, data_in[5:], data_out, sort=True)

    def test_getitem(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        data_out = hd[keys_out]
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)

    def test_len(self, dict_0_, dict_10_):
        assert len(dict_0_[0]) == 0
        assert len(dict_10_[0]) == 10

    def test_setitem(self, dict_0_, dict_10_):
        hd, ktype_, dtype_ = dict_0_
        _, _, _, keys_in, data_in = dict_10_
        hd[keys_in] = data_in
        keys_out = hd.keys()
        data_out = hd[keys_out]
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)

    def test_clear(self, dict_1_):
        hd, _, _, _, _ = dict_1_
        hd.clear()
        assert len(hd) == 0

    def test_contains(self, dict_10_):
        hd, ktype_, _, keys_in, data_in = dict_10_
        assert np.all(hd.contains(keys_in) == True)
        assert np.all(hd.contains(np.array([101, 102])) == False)

    def test_fromkeys(self, ktype, dtype, parallel):
        keys = np.array([0], dtype=ktype)
        data = np.array([1], dtype=dtype)
        hd = vp.dict.fromkeys(keys, data)
        assert len(hd) == 1
        assert hd.ktype == ktype
        assert hd.dtype == dtype
        # The fromkeys classmethod won't know about the parallel flag from an instance
        # so we can't test hd.parallel here reliably without more changes.
        hd = vp.dict.fromkeys(keys)
        assert len(hd) == 1
        assert hd.ktype == ktype

    def test_get(self, dict_10_):
        hd, ktype_, dtype_, keys, data = dict_10_
        value = hd.get(keys[5:6])
        assert value[0] == data[5]
        value = hd.get(np.array([100]))
        assert value[0] == dtype_.type(0)
        fill = 1.0
        value = hd.get(np.array([100]), fill)
        assert value[0] == dtype_.type(fill)

    def test_items(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out, data_out = hd.items()
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)

    def test_keys(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        self.helper_check(keys_in, keys_out, sort=True)

    def test_key_access_fails(self, dict_1_):
        hd, ktype_, dtype_, _, _ = dict_1_
        with pytest.raises(IndexError):
            hd[np.array(np.array([0, 1]))]

    def test_key_in(self, dict_1_):
        hd, ktype_, dtype_, keys, _ = dict_1_
        assert keys in hd
        assert np.array([0, 1]) not in hd

    def test_pop(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        data_out = hd.pop(keys_in[:5])
        assert len(hd) == 5
        self.helper_check(keys_in[:5], keys_in[:5], data_in[:5], data_out, sort=False)
        keys_out, data_out = hd.items()
        self.helper_check(keys_in[5:], keys_out, data_in[5:], data_out, sort=True)

    def test_popitem(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        key, data = hd.popitem()
        assert len(hd) == 9
        mask = (key == keys_in)
        assert np.sum(mask) == 1
        assert data == data_in[mask][0]

    def test_setdefault(self, dict_10_):
        hd, ktype_, dtype_, keys_in, data_in = dict_10_
        fill = dtype_.type(100)
        keys_out = np.hstack((keys_in, np.arange(10, 20, dtype=ktype_)))
        data_out = hd.setdefault(keys_out, fill)
        self.helper_check(keys_in, keys_out[:10], data_in, data_out[:10],
                          sort=False)
        assert np.all(data_out[10:] == fill)

    def test_update(self, dict_0_, dict_10_):
        hd1, _, _ = dict_0_
        hd2, _, _, _, _ = dict_10_
        hd1.update(hd2)
        assert len(hd1) == len(hd2)
        k1, d1 = hd1.items()
        k2, d2 = hd2.items()
        self.helper_check(k1, k2, d1, d2, sort=True)

    def test_values(self, dict_10_):
        hd, _, _, keys_in, data_in = dict_10_
        keys_out = hd.keys()
        data_out = hd.values()
        self.helper_check(keys_in, keys_out, data_in, data_out, sort=True)
