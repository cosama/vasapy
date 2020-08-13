#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_vasapy_dict(py::module &);
void init_vasapy_set(py::module &);


PYBIND11_MODULE(_vasapy, m) {
  init_vasapy_dict(m);
  init_vasapy_set(m);
};
