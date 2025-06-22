#include <omp.h>
#include <vector>
#include <parallel_hashmap/phmap.h>
#include <byte_set.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>


#define __NPD_TYPES__ bool, char, \
                     float, double, long double, \
                     std::complex<float>, std::complex<double>, std::complex<long double>, \
                     std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, \
                     std::int32_t, std::uint32_t, std::int64_t, std::uint64_t


namespace py = pybind11;

// Alias for a parallel map that is correctly configured with a real mutex
template<class K, class V>
using parallel_map_with_mutex = phmap::parallel_flat_hash_map<
    K, V,
    phmap::priv::hash_default_hash<K>,
    phmap::priv::hash_default_eq<K>,
    phmap::priv::Allocator<phmap::priv::Pair<const K, V>>,
    4,
    std::mutex>;

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
  bool parallel_;
};


template <typename K, typename T, typename Map> struct dict_typed_: dict_ {

  dict_typed_(py::dtype ktype, py::dtype dtype, bool parallel) {
    ktype_ = ktype;
    dtype_ = dtype;
    parallel_ = parallel;
  };

  void clear() {
    map_.clear();
  };

  py::array_t<bool> contains(py::array keys) {
    py::buffer_info kinfo = keys.request();
    K *kptr = (K*)(kinfo.ptr);
    auto ret = py::array_t<bool>({(py::ssize_t)kinfo.size});
    py::buffer_info rinfo = ret.request();
    bool *rptr = (bool*)(rinfo.ptr);
    
    py::gil_scoped_release release;
    #pragma omp parallel for if(parallel_)
    for(py::ssize_t i = 0; i < kinfo.size; ++i) {
      rptr[i] = map_.contains(kptr[i]);
    }
    
    return ret;
  };

  void delitem(py::array keys) {
    py::buffer_info kinfo = keys.request();
    if (!ktype_.is(py::dtype(kinfo)))
        throw std::invalid_argument("Key array has incorrect dtype");
    K *kptr = (K*)(kinfo.ptr);

    py::gil_scoped_release release;
    #pragma omp parallel for if(parallel_)
    for(py::ssize_t i = 0; i < kinfo.size; ++i) { map_.erase(kptr[i]); }
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
    #pragma omp parallel for if(parallel_)
    for(py::ssize_t i = 0; i < kinfo.size; ++i) {
      auto it = map_.find(kptr[i]);
      if(it == map_.end())
        dptr[i] = (finfo.size == 1) ? fptr[0] : fptr[i];
      else
        dptr[i] = it->second;
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
    #pragma omp parallel for if(parallel_)
    for(py::ssize_t i = 0; i < kinfo.size; ++i) dptr[i] = map_.at(kptr[i]);
    return data;
  };

  std::pair<py::array, py::array> items() {
    const size_t n = map_.size();
    auto keys = py::array(ktype_, {n}, {ktype_.itemsize()});
    auto data = py::array(dtype_, {n}, {dtype_.itemsize()});
    if (n == 0) return {keys, data};

    K *kptr = (K*)(keys.request().ptr);
    T *dptr = (T*)(data.request().ptr);

    if (parallel_) {
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
        
        py::gil_scoped_release release;
        #pragma omp parallel for
        for (size_t i = 0; i < num_submaps; ++i) {
            map_.with_submap(i, [&](const auto& submap) {
                size_t current_pos = offsets[i];
                for (const auto& p : submap) {
                    kptr[current_pos] = p.first;
                    dptr[current_pos] = p.second;
                    ++current_pos;
                }
            });
        }
    } else {
        for(const auto& p: map_) {
            kptr[0] = p.first; dptr[0] = p.second;
            ++kptr; ++dptr;
        }
    }
    return {keys, data};
  };

  py::array keys() {
    auto keys = py::array(ktype_, {map_.size()}, {ktype_.itemsize()});
    if (map_.empty()) return keys;

    K *kptr = (K*)(keys.request().ptr);

    if (parallel_) {
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

        py::gil_scoped_release release;
        #pragma omp parallel for
        for (size_t i = 0; i < num_submaps; ++i) {
            map_.with_submap(i, [&](const auto& submap) {
                size_t current_pos = offsets[i];
                for (const auto& p : submap) {
                    kptr[current_pos++] = p.first;
                }
            });
        }
    } else {
        for(const auto& p: map_) { kptr[0] = p.first; ++kptr; }
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
    #pragma omp parallel for if(parallel_)
    for(py::ssize_t i = 0; i < kinfo.size; ++i) {
      map_[kptr[i]] = (dinfo.size == 1) ? dptr[0] : dptr[i];
    }
  };

  py::array values() {
    auto data = py::array(dtype_, {map_.size()}, {dtype_.itemsize()});
    if (map_.empty()) return data;
    
    T *dptr = (T*)(data.request().ptr);

    if (parallel_) {
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

        py::gil_scoped_release release;
        #pragma omp parallel for
        for (size_t i = 0; i < num_submaps; ++i) {
            map_.with_submap(i, [&](const auto& submap) {
                size_t current_pos = offsets[i];
                for (const auto& p : submap) {
                    dptr[current_pos++] = p.second;
                }
            });
        }
    } else {
        for(const auto& p: map_) { dptr[0] = p.second; ++dptr; }
    }
    return data;
  };

  Map map_;
};


template <int ...> struct IntList {};
template <typename ...> struct TypeList {};

template<typename DictTyped, typename J, typename O>
void do_inpl_op_parallel_(dict_ &dict, py::array& keys, py::array& data, py::array& fill, O op) {
    py::buffer_info kinfo = keys.request(), dinfo = data.request(), finfo = fill.request();
    auto dict_t = dynamic_cast<DictTyped*>(&dict);
    auto kptr = (typename decltype(dict_t->map_)::key_type*)(kinfo.ptr);
    auto dptr = (J*)(dinfo.ptr);
    auto fptr = (typename decltype(dict_t->map_)::mapped_type*)(finfo.ptr);

    py::gil_scoped_release release;
    #pragma omp parallel for
    for(py::ssize_t i = 0; i < kinfo.size; ++i) {
        dict_t->map_.lazy_emplace_l(
            kptr[i],
            [&](auto& p) { 
              J* val = (J*)(&p.second);
              *val = op(*val, (dinfo.size == 1) ? dptr[0] : dptr[i]); },
            [&](const auto& ctor) { ctor(kptr[i], (finfo.size == 1) ? fptr[0] : fptr[i]);}
        );
    };
}

template<typename DictTyped, typename J, typename O>
void do_inpl_op_flat_(dict_ &dict, py::array keys, py::array data, py::array fill, O op) {
    py::buffer_info kinfo = keys.request(), dinfo = data.request(), finfo = fill.request();
    auto dict_t = dynamic_cast<DictTyped*>(&dict);
    auto kptr = (typename decltype(dict_t->map_)::key_type*)(kinfo.ptr);
    auto dptr = (J*)(dinfo.ptr);
    auto fptr = (typename decltype(dict_t->map_)::mapped_type*)(finfo.ptr);
    J* val;
    for(py::ssize_t i = 0; i < kinfo.size; ++i) {
      auto d = dict_t->map_.find(kptr[i]);
      if(d == dict_t->map_.end()) {
        auto p = dict_t->map_.emplace(kptr[i], ((finfo.size == 1) ? fptr[0] : fptr[i]));
        if (!p.second)
            throw std::runtime_error("In-place operation failed due to logic error during insertion");
        val = (J*)&(p.first->second);
      } else {
        val = (J*)&(d->second);
      };
      *val = op(val[0], ((dinfo.size == 1) ? dptr[0] : dptr[i]));
    };
};

template<typename ...M, typename O>
void inpl_op_(dict_ &, py::array, py::array, py::array, IntList<>, TypeList<M...>, O) {
  throw std::invalid_argument("Key type not supported");
};

template<int ...N, typename O>
void inpl_op_(dict_ &, py::array, py::array, py::array, IntList<N...>, TypeList<>, O) {
  throw std::invalid_argument("Data type not supported");
};

template <int I, int ... N, typename J, typename ...M, typename O>
void inpl_op_(dict_ &dict, py::array keys, py::array data, py::array fill,
              IntList<I, N...>, TypeList<J, M...>, O op) {
  if (I != dict.ktype_.itemsize()) {
    inpl_op_(dict, keys, data, fill, IntList<N...>(), TypeList<J, M...>{}, op);
  }
  else if (!py::dtype::of<J>().is(dict.dtype_)) {
    inpl_op_(dict, keys, data, fill, IntList<I, N...>{}, TypeList<M...>(), op);
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

    if (dict.parallel_) {
        do_inpl_op_parallel_<dict_typed_<byte_set<I>, byte_set<sizeof(J)>, parallel_map_with_mutex<byte_set<I>, byte_set<sizeof(J)>>>, J>(dict, keys, data, fill, op);
    } else {
        do_inpl_op_flat_<dict_typed_<byte_set<I>, byte_set<sizeof(J)>, phmap::flat_hash_map<byte_set<I>, byte_set<sizeof(J)>>>, J>(dict, keys, data, fill, op);
    }
  };
};


template<int ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, bool, IntList<>, IntList<N...>) {
  throw std::invalid_argument("Key type not supported");
};
template<int ...N> [[ noreturn ]]std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, bool, IntList<N...>, IntList<>) {
  throw std::invalid_argument("Data type not supported");
};
template <int I, int ...N, int J, int ...M>
std::unique_ptr<dict_> init_dict_(
    py::dtype k, py::dtype d, bool parallel, IntList<I, N...>, IntList<J, M...>) {
  if (I != k.itemsize()) {
    return init_dict_(k, d, parallel, IntList<N...>(), IntList<J, M...>());
  }
  if (J != d.itemsize()) {
    return init_dict_(k, d, parallel, IntList<I, N...>(), IntList<M...>());
  }
  if (parallel) {
      return std::make_unique<dict_typed_<byte_set<I>, byte_set<J>, parallel_map_with_mutex<byte_set<I>, byte_set<J> > > >(k, d, parallel);
  }
  return std::make_unique<dict_typed_<byte_set<I>, byte_set<J>, phmap::flat_hash_map<byte_set<I>, byte_set<J> > > >(k, d, parallel);
};


void init_vasapy_dict(py::module &m) {
    py::class_<dict_>(m, "_dict")
        .def(py::init(
          [](py::dtype k, py::dtype d, bool parallel) {
            return init_dict_(k, d, parallel, IntList<1, 2, 4, 8, 16, 32>(),
                              IntList<1, 2, 4, 8, 16, 32>());
          }
        ), py::arg("keys"), py::arg("data"), py::arg("parallel") = false)
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
        .def_readonly("dtype", &dict_::dtype_)
        .def_readonly("parallel", &dict_::parallel_);
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
