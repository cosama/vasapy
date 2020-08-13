#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_vasapy_dict(py::module &);
void init_vasapy_unordered_set(py::module &m);
//void init_vasapy_sorted_set(py::module &m);


PYBIND11_MODULE(_vasapy, m) {
  init_vasapy_dict(m);
  init_vasapy_unordered_set(m);
  //init_vasapy_sorted_set(m);
};
