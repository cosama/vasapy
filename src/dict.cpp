#include <parallel_hashmap/phmap.h>
#include <byte_set.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;


struct arr_map {
  virtual ~arr_map() = default;
  virtual py::array getitem(py::array) = 0;
};


template <typename K, typename T> struct flat_arr_map: arr_map {

  flat_arr_map(py::array keys, py::array data) {
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);
    ktype_ = py::dtype(kinfo);
    dtype_ = py::dtype(dinfo);
    map_.reserve(kinfo.size);
    if (kinfo.size != dinfo.size)
        throw std::runtime_error("Input shapes must match");
    for(int i=0; i < kinfo.size; i++) map_.emplace(kptr[i], dptr[i]);
  };

  py::array getitem(py::array keys) {
    py::buffer_info kinfo = keys.request();
    if(!py::dtype(kinfo).is(ktype_))
      throw std::runtime_error("Keys dtype doesn't match containers key dtype");
    K *kptr = (K*)(kinfo.ptr);
    auto strides = kinfo.strides;
    for(auto& s : strides){ double se = s / kinfo.itemsize; s = se * dtype_.itemsize(); }
    auto data = py::array(dtype_, kinfo.shape, strides);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);
    for(int i = 0; i < kinfo.size; i++) {
      auto d = map_.find(kptr[i]);
      if(d == map_.end()) {
        throw std::runtime_error("Fill not implemented");
      }
      else
        dptr[i] = d->second;
    }
    return data;
  }

  phmap::flat_hash_map<K, T> map_;
  py::dtype ktype_;
  py::dtype dtype_;
  //T fill_;
};


template <int ...> struct IntList {};


class dict{
private:
  std::unique_ptr<arr_map> m;

  template<int ...N> [[ noreturn ]]std::unique_ptr<arr_map> init_dict(
      py::array k, py::array d, IntList<>, IntList<N...>) {
    throw std::invalid_argument("Data type not supported");
  }
  template<int ...N> [[ noreturn ]]std::unique_ptr<arr_map> init_dict(
      py::array k, py::array d, IntList<N...>, IntList<>) {
    throw std::invalid_argument("Data type not supported");
  }
  template <int I, int ...N, int J, int ...M>
  std::unique_ptr<arr_map> init_dict(
      py::array k, py::array d, IntList<I, N...>, IntList<J, M...>) {
    py::buffer_info kinfo = k.request();
    py::buffer_info dinfo = d.request();
    if (I != kinfo.itemsize) {
      return init_dict(k, d, IntList<N...>(), IntList<J, M...>()); }
    if (J != dinfo.itemsize) {
      return init_dict(k, d, IntList<I, N...>(), IntList<M...>()); }
   return std::make_unique<flat_arr_map<byte_set<I>, byte_set<J> > >(k, d);
  };
public:
  dict(py::array k, py::array d){
    m = init_dict(k, d, IntList<1, 2, 4, 8, 16, 32>(), IntList<1, 2, 4, 8, 16, 32>());
  };

  py::array getitem(py::array k){
    return m->getitem(k);
  };
};


PYBIND11_MODULE(vasapy, m) {
    py::class_<dict>(m, "dict")
        .def(py::init<py::array, py::array>())
        .def("__getitem__", &dict::getitem);
};
