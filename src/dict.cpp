#include <parallel_hashmap/phmap.h>
#include <byte_set.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <mutex>
#include <stdexcept>
#include <vector>

#define __NPD_TYPES__ bool, char, \
                     float, double, long double, \
                     std::complex<float>, std::complex<double>, std::complex<long double>, \
                     std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, \
                     std::int32_t, std::uint32_t, std::int64_t, std::uint64_t


namespace py = pybind11;


namespace {
    constexpr int OMP_MAX_THREADS = 4;
}


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
  virtual py::array values() = 0;

  py::dtype dtype_;
  py::dtype ktype_;
};


struct ParallelReadInfo {
    size_t total_size;
    std::vector<size_t> offsets;
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
    auto ret = py::array_t<bool>({static_cast<py::ssize_t>(kinfo.size)});
    py::buffer_info rinfo = ret.request();
    bool *rptr = (bool*)(rinfo.ptr);
    auto end = map_.end();

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for(int i = 0; i < kinfo.size; ++i) {
      rptr[i] = map_.find(kptr[i]) != end;
    }
    return ret;
  };

  void delitem(py::array keys) {
    py::buffer_info kinfo = keys.request();
    if (!ktype_.is(py::dtype(kinfo)))
        throw std::invalid_argument("Key array has incorrect dtype");
    K *kptr = (K*)(kinfo.ptr);

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for(int i = 0; i < kinfo.size; ++i) { map_.erase(kptr[i]); }
  };

  py::array get(py::array keys, py::array fill) {
    py::buffer_info kinfo = keys.request();
    py::buffer_info finfo = fill.request();
    if (!ktype_.is(py::dtype(kinfo)))
        throw std::invalid_argument("Key array has incorrect dtype");
    if (!dtype_.is(py::dtype(finfo)))
        throw std::invalid_argument("Default value array has incorrect dtype");
    K *kptr = (K*)(kinfo.ptr);
    T *fptr = (T*)(finfo.ptr);
    if (kinfo.size != finfo.size && finfo.size != 1)
        throw std::invalid_argument("Key and default arrays have incompatible sizes");
    auto strides = kinfo.strides;
    for(auto& s : strides) {
      double se = s / kinfo.itemsize; s = se * dtype_.itemsize();
    }
    auto data = py::array(dtype_, kinfo.shape, strides);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(OMP_MAX_THREADS)
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
    if (!ktype_.is(py::dtype(kinfo)))
        throw std::invalid_argument("Key array has incorrect dtype");
    K *kptr = (K*)(kinfo.ptr);
    auto strides = kinfo.strides;
    for(auto& s : strides) {
      double se = s / kinfo.itemsize; s = se * dtype_.itemsize();
    };
    auto data = py::array(dtype_, kinfo.shape, strides);
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for(int i = 0; i < kinfo.size; ++i) dptr[i] = map_.at(kptr[i]);
    return data;
  };

  ParallelReadInfo get_parallel_read_info() {
      const size_t num_submaps = map_.subcnt();
      std::vector<size_t> offsets(num_submaps + 1, 0);
      for (size_t i = 0; i < num_submaps; ++i) {
          map_.with_submap(i, [&](const auto& submap) {
              offsets[i + 1] = submap.size();
          });
      }

      for (size_t i = 0; i < num_submaps; ++i) {
          offsets[i + 1] += offsets[i];
      }
      return {offsets.back(), offsets};
  }

  std::pair<py::array, py::array> items() {
    auto info = get_parallel_read_info();
    auto keys = py::array(ktype_, {info.total_size}, {ktype_.itemsize()});
    auto data = py::array(dtype_, {info.total_size}, {ktype_.itemsize()});
    py::buffer_info kinfo = keys.request();
    py::buffer_info dinfo = data.request();
    K *kptr = (K*)(kinfo.ptr);
    T *dptr = (T*)(dinfo.ptr);

    //py::gil_scoped_release release;
    //#pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for (size_t i = 0; i < map_.subcnt(); ++i) {
        map_.with_submap(i, [&](const auto& submap) {
            K* k_out = kptr + info.offsets[i];
            T* d_out = dptr + info.offsets[i];
            for (const auto& p : submap) {
                *k_out++ = p.first;
                *d_out++ = p.second;
            }
        });
    }
    return std::pair<py::array, py::array>(keys, data);
  };

  py::array keys() {
    auto info = get_parallel_read_info();
    auto keys = py::array(ktype_, {info.total_size}, {ktype_.itemsize()});
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);

    //py::gil_scoped_release release;
    //#pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for (size_t i = 0; i < map_.subcnt(); ++i) {
        map_.with_submap(i, [&](const auto& submap) {
            K* k_out = kptr + info.offsets[i];
            for (const auto& p : submap) {
                *k_out++ = p.first;
            }
        });
    }
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
    if (kinfo.size != dinfo.size && dinfo.size != 1)
        throw std::invalid_argument("Key and data arrays have incompatible sizes");
    if (!ktype_.is(py::dtype(kinfo)))
        throw std::invalid_argument("Key array has incorrect dtype");
    if (!dtype_.is(py::dtype(dinfo)))
        throw std::invalid_argument("Data array has incorrect dtype");
    K *kptr = (K*)(kinfo.ptr);
    T *dptr = (T*)(dinfo.ptr);

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for(int i = 0; i < kinfo.size; ++i)
      map_[kptr[i]] = (dinfo.size == 1) ? dptr[0] : dptr[i];
  };

  py::array values() {
    auto info = get_parallel_read_info();
    auto data = py::array(dtype_, {info.total_size}, {dtype_.itemsize()});
    py::buffer_info dinfo = data.request();
    T *dptr = (T*)(dinfo.ptr);

    //py::gil_scoped_release release;
    //#pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for (size_t i = 0; i < map_.subcnt(); ++i) {
        map_.with_submap(i, [&](const auto& submap) {
            T* d_out = dptr + info.offsets[i];
            for (const auto& p : submap) {
                *d_out++ = p.second;
            }
        });
    }
    return data;
  };

  phmap::parallel_flat_hash_map<K, T,
                                phmap::priv::hash_default_hash<K>,
                                phmap::priv::hash_default_eq<K>,
                                std::allocator<std::pair<const K, T>>,
                                4,
                                std::mutex> map_;
};


template <int ...> struct IntList {};
template <typename ...> struct TypeList {};


template<typename ...M, typename O> void inpl_op_(
    dict_ &, py::array, py::array, py::array, IntList<>, TypeList<M...>, O) {
  throw std::invalid_argument("Data type not supported");
};
template<int ...N, typename O> void inpl_op_(
    dict_ &, py::array, py::array, py::array, IntList<N...>, TypeList<>, O) {
  throw std::invalid_argument("Data type not supported");
};
template <int I, int ... N, typename J, typename ...M, typename O>
void inpl_op_(dict_ &dict, py::array keys, py::array data, py::array fill,
              IntList<I, N...>, TypeList<J, M...>, O op) {
  if (I != dict.ktype_.itemsize()) {
    inpl_op_(dict, keys, data, fill, IntList<N...>(), TypeList<J, M...>(), op);
  }
  else if (!py::dtype::of<J>().is(dict.dtype_)) {
    inpl_op_(dict, keys, data, fill, IntList<I, N...>(), TypeList<M...>(), op);
  }
  else {
    py::buffer_info kinfo = keys.request();
    py::buffer_info dinfo = data.request();
    py::buffer_info finfo = fill.request();
    
    if (!dict.ktype_.is(py::dtype(kinfo)))
        throw std::invalid_argument("Key array has incorrect dtype for in-place operation");
    if (!dict.dtype_.is(py::dtype(dinfo)))
        throw std::invalid_argument("Data array has incorrect dtype for in-place operation");
    if (!dict.dtype_.is(py::dtype(finfo)))
        throw std::invalid_argument("Default value array has incorrect dtype for in-place operation");
    if ((kinfo.size != dinfo.size && dinfo.size != 1) || (kinfo.size != finfo.size && finfo.size != 1))
        throw std::invalid_argument("Array sizes are incompatible for in-place operation");

    auto dict_t = dynamic_cast<
      dict_typed_<byte_set<I>, byte_set<sizeof(J)>>*>(&dict);
    auto kptr = (byte_set<I>*)(kinfo.ptr);
    auto dptr = (J*)(dinfo.ptr);
    auto fptr = (byte_set<sizeof(J)>*)(finfo.ptr);

    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(OMP_MAX_THREADS)
    for(int i = 0; i < kinfo.size; ++i) {
        dict_t->map_.lazy_emplace_l(
            kptr[i],
            [&, i](auto& pair) {
                J* val_ptr = (J*)&(pair.second);
                J data_val = (dinfo.size == 1) ? dptr[0] : dptr[i];
                *val_ptr = op(*val_ptr, data_val);
            },
            [&, i](const auto& constructor) {
                J data_val = (dinfo.size == 1) ? dptr[0] : dptr[i];
                J fill_val;
                memcpy(&fill_val, (finfo.size == 1) ? fptr : &fptr[i], sizeof(J));
                J new_val_numeric = op(fill_val, data_val);
                constructor(kptr[i], byte_set<sizeof(J)>(&new_val_numeric));
            }
        );
    };
  };
};


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


void init_vasapy_dict(py::module &m) {
    omp_set_dynamic(1);
    py::class_<dict_>(m, "_dict")
        .def(py::init(
          [](py::dtype k, py::dtype d) {
            return init_dict_(k, d, IntList<1, 2, 4, 8, 16, 32>(),
                              IntList<1, 2, 4, 8, 16, 32>());
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
        .def("values", &dict_::values)
        .def_readonly("ktype", &dict_::ktype_)
        .def_readonly("dtype", &dict_::dtype_);
    m.def("iadd", [](dict_ &dict, py::array k, py::array d, py::array f){
        inpl_op_(dict, k, d, f, IntList<1, 2, 4, 8, 16>(),
                 TypeList<__NPD_TYPES__>(), std::plus());
    });
    m.def("isub", [](dict_ &dict, py::array k, py::array d, py::array f){
        inpl_op_(dict, k, d, f, IntList<1, 2, 4, 8, 16>(),
                 TypeList<__NPD_TYPES__>(), std::minus());
    });
    m.def("imul", [](dict_ &dict, py::array k, py::array d, py::array f){
        inpl_op_(dict, k, d, f, IntList<1, 2, 4, 8, 16>(),
                 TypeList<__NPD_TYPES__>(), std::multiplies());
    });
    m.def("idiv", [](dict_ &dict, py::array k, py::array d, py::array f){
        inpl_op_(dict, k, d, f, IntList<1, 2, 4, 8, 16>(),
                 TypeList<__NPD_TYPES__>(), std::divides());
    });
};
