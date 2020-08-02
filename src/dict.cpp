#include <parallel_hashmap/phmap.h>
#include <byte_set.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

// Function to convert any python object into a byte_set by converting it first
// into a numpy array. Uses lots of python code, couldn't come up with
// something better.
template <int T> byte_set<T> byte_set_from_pyobject(
    const py::object &pyinp, const py::dtype &dtype) {
  py::object np = py::module::import("numpy");
  py::object array = np.attr("array");
  py::object ap = array(pyinp, py::cast(&dtype));
  py::array ac = ap.cast<py::array>();
  py::buffer_info info = ac.request();
  return byte_set<T>(info.ptr);
};

struct dict_ {
  virtual ~dict_() = default;
  virtual py::array getitem(py::array) = 0;
  virtual void setitem(py::array, py::array) = 0;
  virtual py::array keys() = 0;
  virtual py::array_t<bool> contains(py::array) = 0;
  virtual py::size_t len() = 0;
};


template <typename K, typename T> struct dict_typed_: dict_ {

  dict_typed_(py::dtype ktype, py::dtype dtype, T fill) {
    ktype_ = ktype;
    dtype_ = dtype;
    fill_ = fill;
  };

  py::array getitem(py::array keys) {
    py::buffer_info kinfo = keys.request();
    assert (ktype_.is(py::dtype(kinfo)));
    K *kptr = (K*)(kinfo.ptr);
    auto strides = kinfo.strides;
    for(auto& s : strides) {
      double se = s / kinfo.itemsize; s = se * dtype_.itemsize();
    }
    auto data = py::array(dtype_, kinfo.shape, strides);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);
    for(int i = 0; i < kinfo.size; i++) {
      auto d = map_.find(kptr[i]);
      if(d == map_.end())
        dptr[i] = fill_;
      else
        dptr[i] = d->second;
    }
    return data;
  };

  void setitem(py::array keys, py::array data) {
    py::buffer_info kinfo = keys.request();
    py::buffer_info dinfo = data.request();
    assert (kinfo.size == dinfo.size);
    assert (ktype_.is(py::dtype(kinfo)));
    assert (dtype_.is(py::dtype(dinfo)));
    K *kptr = (K*)(kinfo.ptr);
    T *dptr = (T*)(dinfo.ptr);
    for(int i = 0; i < kinfo.size; i++) map_.emplace(kptr[i], dptr[i]);
  };

  py::array keys() {
    auto keys = py::array(ktype_, {map_.size()}, {ktype_.itemsize()});
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    for(const auto& p: map_) { kptr[0] = p.first; ++kptr; }
    return keys;
  };

  py::array_t<bool> contains(py::array keys) {
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    auto ret = py::array_t<bool>({(py::size_t)kinfo.size});
    py::buffer_info rinfo = ret.request();
    bool *rptr = (bool*)(rinfo.ptr);
    auto end = map_.end();
    for(int i = 0; i < kinfo.size; i++) {
      rptr[i] = map_.find(kptr[i]) != end;
    }
    return ret;
  };

  py::size_t len() {
    return map_.size();
  };

  phmap::flat_hash_map<K, T> map_;
  py::dtype ktype_;
  py::dtype dtype_;
  T fill_;
};


template <int ...> struct IntList {};
template<int ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, py::object o, IntList<>, IntList<N...>) {
  throw std::invalid_argument("Data type not supported");
}
template<int ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, py::object o, IntList<N...>, IntList<>) {
  throw std::invalid_argument("Data type not supported");
}
template <int I, int ...N, int J, int ...M>
std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, py::object o, IntList<I, N...>, IntList<J, M...>) {
  if (I != k.itemsize()) {
    return init_dict_(k, d, o, IntList<N...>(), IntList<J, M...>()); }
  if (J != d.itemsize()) {
    return init_dict_(k, d, o, IntList<I, N...>(), IntList<M...>()); }
  byte_set<J> fill = byte_set_from_pyobject<J>(o, d);
  return std::make_unique<dict_typed_<byte_set<I>, byte_set<J> > >(k, d, fill);
};


std::unique_ptr<dict_> init_dict_dtype_(py::dtype k, py::dtype d, py::object o) {
  return init_dict_(k, d, o, IntList<1, 2, 4, 8, 16, 32>(), IntList<1, 2, 4, 8, 16, 32>());
};


std::unique_ptr<dict_> init_dict_array_(py::array k, py::array d, py::object o) {
  py::buffer_info kinfo = k.request();
  py::buffer_info dinfo = d.request();
  auto ret_ = init_dict_dtype_(py::dtype(kinfo), py::dtype(dinfo), o);
  ret_->setitem(k, d);
  return ret_;
}

PYBIND11_MODULE(vasapy, m) {
    py::class_<dict_>(m, "dict")
        .def(py::init(
          [](py::dtype k, py::dtype d, py::object o) {
            return init_dict_dtype_(k, d, o);
          }), py::arg("ktype"), py::arg("dtype"), py::arg("fill") = 0)
          .def(py::init(
            [](py::array k, py::array d, py::object o) {
              return init_dict_array_(k, d, o);
            }), py::arg("keys"), py::arg("data"), py::arg("fill") = 0)
        .def("__len__", &dict_::len)
        .def("__getitem__", &dict_::getitem, py::arg("keys"))
        .def("__setitem__", &dict_::setitem, py::arg("keys"), py::arg("data"))
        .def("contains", &dict_::contains, py::arg("keys"))
        .def("keys", &dict_::keys);
};
