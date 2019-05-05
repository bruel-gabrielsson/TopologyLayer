#include <torch/extension.h>

#include "cohom.h"
#include "complex.h"

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<SimplicialComplex>(m, "SimplicialComplex")
  .def(py::init<>())
  .def("append", &SimplicialComplex::append)
  .def("printCells", &SimplicialComplex::print);
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}
