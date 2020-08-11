#include <parallel_hashmap/phmap.h>
#include <byte_set.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;


struct dict_ {
  virtual ~dict_() = default;
  virtual void clear() = 0;
  virtual py::array_t<bool> contains(py::array) = 0;
  virtual void delitem(py::array) = 0;
  virtual py::array get(py::array, py::array) = 0;
  virtual py::array getitem(py::array) = 0;
  virtual std::pair<py::array, py::array> items() = 0;
  virtual py::array keys() = 0;
  virtual py::size_t len() = 0;
  virtual std::pair<py::array, py::array> popitem() = 0;
  virtual void setitem(py::array, py::array) = 0;
  virtual void update(dict_ &) = 0;
  virtual py::array values() = 0;

  py::dtype dtype_;
  py::dtype ktype_;
};


template <typename K, typename T> struct dict_typed_: dict_ {

  dict_typed_(py::dtype ktype, py::dtype dtype) {
    ktype_ = ktype;
    dtype_ = dtype;
  };

  void clear() {
    map_.clear();
  };

  py::array_t<bool> contains(py::array keys) {
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    auto ret = py::array_t<bool>({(py::size_t)kinfo.size});
    py::buffer_info rinfo = ret.request();
    bool *rptr = (bool*)(rinfo.ptr);
    auto end = map_.end();
    for(int i = 0; i < kinfo.size; ++i) {
      rptr[i] = map_.find(kptr[i]) != end;
    }
    return ret;
  };

  void delitem(py::array keys) {
    py::buffer_info kinfo = keys.request();
    assert (ktype_.is(py::dtype(kinfo)));
    K *kptr = (K*)(kinfo.ptr);
    for(int i = 0; i < kinfo.size; ++i) { map_.erase(kptr[i]); }
  };

  py::array get(py::array keys, py::array fill) {
    py::buffer_info kinfo = keys.request();
    py::buffer_info finfo = fill.request();
    assert (ktype_.is(py::dtype(kinfo)));
    assert (dtype_.is(py::dtype(finfo)));
    T *fptr = (T*)(finfo.ptr);
    K *kptr = (K*)(kinfo.ptr);
    assert (kinfo.size == finfo.size || finfo.size == 1);
    auto strides = kinfo.strides;
    for(auto& s : strides) {
      double se = s / kinfo.itemsize; s = se * dtype_.itemsize();
    }
    auto data = py::array(dtype_, kinfo.shape, strides);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);
    for(int i = 0; i < kinfo.size; ++i) {
      auto d = map_.find(kptr[i]);
      if(d == map_.end())
        dptr[i] = (finfo.size == 1) ? fptr[0] : fptr[i];
      else
        dptr[i] = d->second;
    }
    return data;
  };

  py::array getitem(py::array keys) {
    py::buffer_info kinfo = keys.request();
    assert (ktype_.is(py::dtype(kinfo)));
    K *kptr = (K*)(kinfo.ptr);
    auto strides = kinfo.strides;
    for(auto& s : strides) {
      double se = s / kinfo.itemsize; s = se * dtype_.itemsize();
    };
    auto data = py::array(dtype_, kinfo.shape, strides);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);
    for(int i = 0; i < kinfo.size; ++i) dptr[i] = map_.at(kptr[i]);
    return data;
  };

  std::pair<py::array, py::array> items() {
    auto keys = py::array(ktype_, {map_.size()}, {ktype_.itemsize()});
    auto data = py::array(dtype_, {map_.size()}, {dtype_.itemsize()});
    py::buffer_info kinfo = keys.request();
    py::buffer_info dinfo = data.request();
    K *kptr = (K*)(kinfo.ptr);
    T *dptr = (T*)(dinfo.ptr);
    for(const auto& p: map_) {
      kptr[0] = p.first; dptr[0] = p.second;
      ++kptr; ++dptr;
    }
    return std::pair<py::array, py::array>(keys, data);
  };

  py::array keys() {
    auto keys = py::array(ktype_, {map_.size()}, {ktype_.itemsize()});
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    for(const auto& p: map_) { kptr[0] = p.first; ++kptr; }
    return keys;
  };

  py::size_t len() {
    return map_.size();
  };

  std::pair<py::array, py::array> popitem() {
    auto keys = py::array(ktype_, {1}, {ktype_.itemsize()});
    auto data = py::array(dtype_, {1}, {dtype_.itemsize()});
    K *kptr = (K*)(keys.request().ptr);
    T *dptr = (T*)(data.request().ptr);
    auto p = map_.begin();
    kptr[0] = p->first;
    dptr[0] = p->second;
    map_.erase(p->first);
    return std::pair<py::array, py::array>(keys, data);
  };

  void setitem(py::array keys, py::array data) {
    py::buffer_info kinfo = keys.request();
    py::buffer_info dinfo = data.request();
    assert (kinfo.size == dinfo.size || dinfo.size == 1);
    assert (ktype_.is(py::dtype(kinfo)));
    assert (dtype_.is(py::dtype(dinfo)));
    K *kptr = (K*)(kinfo.ptr);
    T *dptr = (T*)(dinfo.ptr);
    for(int i = 0; i < kinfo.size; ++i)
      map_[kptr[i]] = (dinfo.size == 1) ? dptr[0] : dptr[i];
  };

  void update(dict_ &other) {
    // dynamic cast is not optimal but probably best we can do here
    auto o = dynamic_cast<dict_typed_*>(&other);
    for(auto &p : o->map_) map_[p.first] = p.second;
  };

  py::array values() {
    auto data = py::array(dtype_, {map_.size()}, {dtype_.itemsize()});
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);
    for(const auto& p: map_) { dptr[0] = p.second; ++dptr; }
    return data;
  };

  phmap::flat_hash_map<K, T> map_;
};


template <int ...> struct IntList {};
template<int ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, IntList<>, IntList<N...>) {
  throw std::invalid_argument("Data type not supported");
};
template<int ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, IntList<N...>, IntList<>) {
  throw std::invalid_argument("Data type not supported");
};
template <int I, int ...N, int J, int ...M>
std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, IntList<I, N...>, IntList<J, M...>) {
  if (I != k.itemsize()) {
    return init_dict_(k, d, IntList<N...>(), IntList<J, M...>());
  }
  if (J != d.itemsize()) {
    return init_dict_(k, d, IntList<I, N...>(), IntList<M...>());
  }
  return std::make_unique<dict_typed_<byte_set<I>, byte_set<J> > >(k, d);
};


PYBIND11_MODULE(_vasapy, m) {
    py::class_<dict_>(m, "_dict")
        .def(py::init(
          [](py::dtype k, py::dtype d) {
            return init_dict_(k, d, IntList<1, 2, 4, 8, 16, 32>(), IntList<1, 2, 4, 8, 16, 32>());
          }
        ), py::arg("keys"), py::arg("data"))
        .def("__delitem__", &dict_::delitem, py::arg("keys"))
        .def("__getitem__", &dict_::getitem, py::arg("keys"))
        .def("__len__", &dict_::len)
        .def("__setitem__", &dict_::setitem, py::arg("keys"), py::arg("data"))
        .def("clear", &dict_::clear)
        .def("contains", &dict_::contains, py::arg("keys"))
        .def("get", &dict_::get, py::arg("keys"), py::arg("default"))
        .def("items", &dict_::items)
        .def("keys", &dict_::keys)
        .def("popitem", &dict_::popitem)
        .def("update", &dict_::update, py::arg("other"))
        .def("values", &dict_::values)
        .def_readonly("ktype", &dict_::ktype_)
        .def_readonly("dtype", &dict_::dtype_);
};
