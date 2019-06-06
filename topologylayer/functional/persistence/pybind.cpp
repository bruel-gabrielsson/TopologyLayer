#include <torch/extension.h>

#include "hom.h"
#include "cohom.h"
#include "complex.h"

namespace py = pybind11;



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<SimplicialComplex>(m, "SimplicialComplex")
  .def(py::init<>())
  .def("append", &SimplicialComplex::append)
  .def("initialize", &SimplicialComplex::initialize)
  .def("extendFloat", &SimplicialComplex::extend)
  .def("extendFlag", &SimplicialComplex::extend_flag)
  .def("sortedOrder", &SimplicialComplex::sortedOrder)
  .def("dim", &SimplicialComplex::dim)
  .def("printFiltration", &SimplicialComplex::printFiltration)
  .def("printFunctionMap", &SimplicialComplex::printFunctionMap)
  .def("printCritInds", &SimplicialComplex::printCritInds)
  .def("printDims", &SimplicialComplex::printDims)
  .def("printBoundary", &SimplicialComplex::printBoundary)
  .def("numPairs", &SimplicialComplex::numPairs)
  .def("printCells", &SimplicialComplex::printComplex);
  m.def("persistenceForwardCohom", &persistence_forward);
  m.def("persistenceBackward", &persistence_backward);
  m.def("persistenceBackwardFlag", &persistence_backward_flag);
  m.def("persistenceForwardHom", &persistence_forward_hom);
}
