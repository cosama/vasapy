#include <parallel_hashmap/phmap.h>
#include <byte_set.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <complex.h>

namespace py = pybind11;

#define __NP_TYPES__ std::int8_t, std::int16_t, std::int32_t, std::int64_t, \
                     std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, \
                     bool, float, double, long long, unsigned long long, long double
                     //float _Complex, double _Complex, long double _Complex


// Function to convert any python object into a byte_set by converting it first
// into a numpy array. Uses lots of python code, couldn't come up with
// something better.
template <typename T> T type_from_pyobject(
    const py::object &pyinp, const py::dtype &dtype) {
  return (py::cast(&dtype).attr("type")(pyinp)).cast<T>();

};

struct dict_ {
  virtual ~dict_() = default;
  virtual py::array getitem(py::array) = 0;
  virtual py::array keys() = 0;
  virtual py::array_t<bool> contains(py::array) = 0;
  virtual py::size_t len() = 0;
};


template <typename KK, typename TT> struct dict_typed_: dict_ {
  using K = byte_set<sizeof(KK)>;
  using T = byte_set<sizeof(TT)>;

  dict_typed_(py::array keys, py::array data, TT fill) {
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);
    ktype_ = py::dtype(kinfo);
    dtype_ = py::dtype(dinfo);
    fill_ = fill;
    map_.reserve(kinfo.size);
    if (kinfo.size != dinfo.size)
        throw std::runtime_error("Input shapes must match");
    for(int i = 0; i < kinfo.size; i++) map_.emplace(kptr[i], dptr[i]);
  };

  py::array getitem(py::array keys) {
    py::buffer_info kinfo = keys.request();
    if(!py::dtype(kinfo).is(ktype_))
      throw std::runtime_error("Keys dtype doesn't match containers key dtype");
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
        dptr[i] = ((T*)(&fill_))[0];
      else
        dptr[i] = d->second;
    }
    return data;
  };

  py::array keys() {
    auto keys = py::array(ktype_, {map_.size()}, {ktype_.itemsize()});
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    for(const auto& p: map_) {
      kptr[0] = p.first;
      ++kptr;
    }
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
  TT fill_;
};


template <typename ...> struct TypeList {};

template<typename ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::array k, py::array d, py::object o, TypeList<>, TypeList<N...>) {
  throw std::invalid_argument("Data type not supported");
}
template<typename ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::array k, py::array d, py::object o, TypeList<N...>, TypeList<>) {
  throw std::invalid_argument("Data type not supported");
}
template <typename I, typename ...N, typename J, typename ...M>
std::unique_ptr<dict_> init_dict_(
    py::array k, py::array d, py::object o, TypeList<I, N...>, TypeList<J, M...>) {
  py::buffer_info kinfo = k.request();
  py::buffer_info dinfo = d.request();
  if (!py::dtype(kinfo).is(py::dtype::of<I>())) {
    return init_dict_(k, d, o, TypeList<N...>(), TypeList<J, M...>()); }
  if (!py::dtype(dinfo).is(py::dtype::of<J>())) {
    return init_dict_(k, d, o, TypeList<I, N...>(), TypeList<M...>()); }
  J fill = type_from_pyobject<J>(o, py::dtype(dinfo));
  return std::make_unique<dict_typed_<I, J> >(k, d, fill);
};


PYBIND11_MODULE(vasapy, m) {
    py::class_<dict_>(m, "dict")
        .def(py::init(
          [](py::array k, py::array d, py::object o) {
            return init_dict_(k, d, o, TypeList<__NP_TYPES__>(), TypeList<__NP_TYPES__>());
          }), py::arg("keys"), py::arg("data"), py::arg("fill") = 0)
        .def("__len__", &dict_::len)
        .def("__getitem__", &dict_::getitem, py::arg("keys"))
        .def("contains", &dict_::contains, py::arg("keys"))
        .def("keys", &dict_::keys);
};
