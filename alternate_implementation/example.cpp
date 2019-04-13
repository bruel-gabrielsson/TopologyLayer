#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include"cohomology.hpp"

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

PYBIND11_MODULE(example, m) {
    py::class_<Cohomology>(m, "Cohomology")
	.def(py::init<>())
	.def("addSimplex",&Cohomology::addSimplex)
        .def("init",&Cohomology::init)
        .def("compute", &Cohomology::computeFiltration)
	.def("printBars", &Cohomology::printBars)
	.def("returnBars", &Cohomology::returnBars)
	.def("barcode",&Cohomology::barcode); 
    	
    	
}
