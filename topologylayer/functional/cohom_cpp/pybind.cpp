#include <torch/extension.h>

#include "cohom.h"
#include "complex.h"

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<SimplicialComplex>(m, "SimplicialComplex")
  .def(py::init<>())
  .def("append", &SimplicialComplex::append)
  .def("initialize", &SimplicialComplex::initialize)
  .def("extendDouble", &SimplicialComplex::extend)
  .def("sortedOrder", &SimplicialComplex::sortedOrder)
  .def("printFiltration", &SimplicialComplex::printFiltration)
  .def("printCells", &SimplicialComplex::print);
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}

//   .def("extendDouble", &SimplicialComplex::extend<double>)
