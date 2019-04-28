#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include"cohomology.h"

namespace py = pybind11;


//int add(int i, int j) {
//    return i + j;
//}
//
//PYBIND11_MODULE(example, m) {
//    m.doc() = "pybind11 example plugin"; // optional module docstring
//
//    m.def("add", &add, "A function which adds two numbers");
//}

PYBIND11_MODULE(cohomopt, m) {
    py::class_<Cohomology>(m, "Cohomology")
	.def(py::init<>())
	.def("addSimplex",&Cohomology::addSimplex)
        .def("init",&Cohomology::initComplex)
	.def("printBars", &Cohomology::printBars,
						py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
	.def("returnBars", &Cohomology::returnBars)
	.def("printComplex", &Cohomology::printComplex,
						py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
	.def("printComplexOrder", &Cohomology::printComplexOrder,
						py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
	.def("readIn", &Cohomology::readInOFF)
	.def("extend",&Cohomology::extend)
	.def("printFunction", &Cohomology::printFunction,
						py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
	.def("computeDiagram", &Cohomology::computeDiagram)
	.def("barcode",&Cohomology::barcode);


}
