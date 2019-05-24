#include <torch/extension.h>

#include "cohom.h"
#include "complex.h"
// #include "demo.h"

namespace py = pybind11;



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<SimplicialComplex>(m, "SimplicialComplex")
  .def(py::init<>())
  .def("append", &SimplicialComplex::append)
  .def("initialize", &SimplicialComplex::initialize)
  .def("extendFloat", &SimplicialComplex::extend)
  .def("sortedOrder", &SimplicialComplex::sortedOrder)
  .def("printFiltration", &SimplicialComplex::printFiltration)
  .def("printFunctionMap", &SimplicialComplex::printFunctionMap)
  .def("printCritInds", &SimplicialComplex::printCritInds)
  .def("printDims", &SimplicialComplex::printDims)
  .def("printBoundary", &SimplicialComplex::printBoundary)
  .def("numPairs", &SimplicialComplex::numPairs)
  .def("printCells", &SimplicialComplex::printComplex);
  m.def("persistenceForward", &persistence_forward);
  m.def("persistenceBackward", &persistence_backward);
  // m.def("forward", &lltm_forward, "LLTM forward");
  // m.def("backward", &lltm_backward, "LLTM backward");
  // m.def("testPrint", &testPrint);
  // m.def("testDump", &testDump);
  // m.def("testLoadFloat", &testLoadFloat);
  // m.def("testLoadInt", &testLoadInt);
}

//   .def("extendDouble", &SimplicialComplex::extend<double>)
