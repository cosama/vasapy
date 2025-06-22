#include <parallel_hashmap/phmap.h>
#include <byte_set.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;


struct set_ {
  virtual ~set_() = default;
  virtual void add(py::array) = 0;
  virtual py::buffer_info buffer() = 0;
  virtual void clear() = 0;
  virtual py::array_t<bool> contains(py::array) = 0;
  virtual void dischard(py::array) = 0;
  virtual py::size_t len() = 0;
  virtual py::array pop() = 0;
  virtual void remove(py::array) = 0;
  virtual void update(set_ &) = 0;
  virtual py::array items() = 0;

  py::dtype dtype_;
  bool parallel_;
};


template <typename T, typename Set> struct set_typed_:set_ {

  set_typed_(py::dtype dtype, bool parallel) {
    dtype_ = dtype;
    parallel_ = parallel;
  };

  void add(py::array elem) {
    py::buffer_info einfo = elem.request();
    if (!dtype_.is(py::dtype(einfo)))
        throw std::invalid_argument("Element array has incorrect dtype");
    auto end = (T*)(einfo.ptr) + einfo.size;
    #pragma omp parallel for if(dict.parallel_)
    for(auto p = (T*)(einfo.ptr); p < end; ++p) map_.emplace(p);
  };

  py::buffer_info buffer() {
    auto ret = py::array(dtype_, {map_.size()}, {dtype_.itemsize()});
    py::buffer_info rinfo = ret.request();
    T *rptr = (T*)(rinfo.ptr);
    #pragma omp parallel for if(dict.parallel_)
    for(const auto& p: map_) { rptr[0] = p; ++rptr; };
    return rinfo;
  };

  void clear() {
    map_.clear();
  };

  py::array_t<bool> contains(py::array elem) {
    py::buffer_info einfo = elem.request();
    T *eptr = (T*)(einfo.ptr);
    auto ret = py::array_t<bool>({(py::ssize_t)einfo.size});
    py::buffer_info rinfo = ret.request();
    bool *rptr = (bool*)(rinfo.ptr);
    auto end = map_.end();
    #pragma omp parallel for if(dict.parallel_)
    for(py::ssize_t i = 0; i < einfo.size; ++i) {
      rptr[i] = map_.find(eptr[i]) != end;
    }
    return ret;
  };

  void dischard(py::array elem) {
    py::buffer_info einfo = elem.request();
    if (!dtype_.is(py::dtype(einfo)))
        throw std::invalid_argument("Element array has incorrect dtype");
    T *eptr = (T*)(einfo.ptr);
    #pragma omp parallel for if(dict.parallel_)
    for(py::ssize_t i = 0; i < einfo.size; ++i) { map_.erase(eptr[i]); }
  };

  py::size_t len() {
    return map_.size();
  };

  py::array pop() {
    auto elem = py::array(dtype_, {1}, {dtype_.itemsize()});
    T *eptr = (T*)(elem.request().ptr);
    auto p = map_.begin();
    eptr[0] = *p;
    map_.erase(p);
    return elem;
  };

  void remove(py::array elem) {
    py::buffer_info einfo = elem.request();
    if (!dtype_.is(py::dtype(einfo)))
        throw std::invalid_argument("Element array has incorrect dtype");
    T *eptr = (T*)(einfo.ptr);
    #pragma omp parallel for if(dict.parallel_)
    for(py::ssize_t i = 0; i < einfo.size; ++i) {
      if(!map_.erase(eptr[i]))
        throw pybind11::key_error("Element not in set");
      }
  };

  void update(set_ &other) {
    py::buffer_info binfo = other.buffer();
    py::array elem(other.dtype_, {(py::ssize_t)binfo.shape[0]}, {(py::ssize_t)binfo.strides[0]}, binfo.ptr);
    add(elem);
  };
  
  py::array items() {
    auto it = py::array(dtype_, {map_.size()}, {dtype_.itemsize()});
    py::buffer_info kinfo = it.request();
    K *kptr = (K*)(kinfo.ptr);
    #pragma omp parallel for if(dict.parallel_)
    for(const auto& p: map_) { kptr[0] = p.first; ++kptr; }
    return it;
  };

  Set map_;
};


template <int ...> struct IntList {};
template<int ...N> [[ noreturn ]]std::unique_ptr<set_> init_set_(
    py::dtype d, bool, IntList<>) {
  throw std::invalid_argument("Element type not supported");
};
template <int I, int ...N>
std::unique_ptr<set_> init_set_(py::dtype d, bool parallel, IntList<I, N...>) {
  if (I != d.itemsize()) {
    return init_set_(d, parallel, IntList<N...>());
  }
  if (parallel) {
    return std::make_unique<set_typed_<byte_set<I>, phmap::parallel_flat_hash_set<byte_set<I>> > >(d, parallel);
  }
  return std::make_unique<set_typed_<byte_set<I>, phmap::flat_hash_set<byte_set<I>> > >(d, parallel);
};


void init_vasapy_set(py::module &m) {
    py::class_<set_>(m, "_set", py::buffer_protocol())
        .def(py::init(
          [](py::dtype d, bool parallel) {
            return init_set_(d, parallel, IntList<1, 2, 4, 8, 16, 32>());
          }
        ), py::arg("elem"), py::arg("parallel") = false)
        .def("__len__", &set_::len)
        .def("add", &set_::add, py::arg("elem"))
        .def("clear", &set_::clear)
        .def("contains", &set_::contains, py::arg("keys"))
        .def("dischard", &set_::dischard, py::arg("elem"))
        .def("pop", &set_::pop)
        .def("remove", &set_::remove, py::arg("elem"))
        .def("update", &set_::update, py::arg("other"))
        .def_readonly("dtype", &set_::dtype_)
        .def_readonly("parallel", &set_::parallel_)
        .def_buffer([](set_ &other) -> py::buffer_info {
          return other.buffer();
        });
};
